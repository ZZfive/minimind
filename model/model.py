import math
import struct
import inspect
import time

from .LMConfig import LMConfig
from typing import Any, Optional, Tuple, List, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    '''
    dim: 模型的隐藏维度，通常是注意力头的维度
    end: 预计算的最大序列长度，默认为32K（32,768），适应超长上下文
    theta: 基频参数，默认为1e6，控制旋转的尺度，影响模型的外推性能
    '''
    # 计算频率基向量
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore  生成位置索引序列
    freqs = torch.outer(t, freqs).float()  # type: ignore  计算外积，生成一个形状为[end, dim//2]的矩阵，freqs[pos, i] = pos * freqs[i]，表示位置pos在维度i上的旋转角度
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64  将频率转换为复数形式，表示旋转因子
    return pos_cis


def apply_rotary_emb(xq, xk, pos_cis):  # 给query和key添加位置编码；pos_cis是预先计算的复数旋转因子
    def unite_shape(pos_cis, x):  # 将pos_cis的形状调整为与x的形状一致，便于广播
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert pos_cis.shape == (x.shape[1], x.shape[-1])  # 序列的长度和每个头的维度要保持一致
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)
    #通过对xq和xk进行reshape，将其转换为复数形式，即将最后一个维度的每两个相邻数视为一个复试的实部和虚部
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)  # 将复数乘以旋转因子，在复数域中乘法相当于旋转
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:  # GQA减少参数和计算量的方法就是重复key和value
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads  # 当未设置n_kv_heads时，默认与n_heads相同，表示不使用GQA
        assert args.n_heads % self.n_kv_heads == 0  # 确保n_heads可以被n_kv_heads整除；定义n_kv_heads的目的是实现GQA
        self.n_local_heads = args.n_heads
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads  # 每个头的维度
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)  # 将mask注册为缓存

    def forward(self,
                x: torch.Tensor,
                pos_cis: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # 新增多头头数这个维度
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)  # 给query和key添加位置编码
        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )
        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0  # 如果训练，dropout_p为dropout，否则为0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )  # 使用flash attention加速计算
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 计算注意力分数   
            scores += self.mask[:, :, :seq_len, :seq_len]  # 添加掩码
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # 对注意力分数进行softmax归一化
            scores = self.attn_dropout(scores)  # 使用dropout
            output = scores @ xv  # 计算加权和

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.wo(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))  # 使用silu激活函数


class MoEGate(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 路由专家数量

        self.scoring_func = config.scoring_func  # 评分函数
        self.alpha = config.aux_loss_alpha  # 辅助损失系数
        self.seq_aux = config.seq_aux  # 序列辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否标准化top-k概率
        self.gating_dim = config.dim  # 门控维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  # 用于计算每个token对每个专家的路由得分
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # 使用kaiming初始化

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)  # [bsz * seq_len, h]
        logits = F.linear(hidden_states, self.weight, None)  # [bsz * seq_len, n_routed_experts]，计算每个token对每个专家的路由得分
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  # 选择得分前top-k个专家

        if self.top_k > 1 and self.norm_topk_prob:  # 如果top-k大于1且标准化top-k概率，就对前top-k个专家的概率进行归一化
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:  # 如果训练且辅助损失系数大于0需要计算辅助损失；辅助损失的计算是MoE中的关键技术，用于解决"专家崩溃"问题（某些专家接收过多token，而其他专家几乎不被使用）
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:  # 序列辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)  # [bsz, seq_len, n_routed_experts]
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)  # 计算每个batch中每个专家被选择的频率
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha  # 将实际选频率和理想均匀分布1/n_experts）的偏差作为损失
            else:  # 批次级别辅助损失
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])  # 创建路由专家列表
        self.gate = MoEGate(config)  # 创建门控机制
        if config.n_shared_experts is not None:
            self.shared_experts = FeedForward(config)  # 创建一个共享专家

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)  # 返回每个token选择的路由专家索引、权重和辅助损失；前两个对象的shape为[bsz*seq_len, n_routed_experts]
        x = x.view(-1, x.shape[-1])  # [bsz*seq_len, dim]
        flat_topk_idx = topk_idx.view(-1)  # [bsz*seq_len, n_routed_experts]
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)  # 重复每个token的输入，以便每个token都能被路由到多个专家，[bsz*seq_len*num_experts_per_tok, dim]
            y = torch.empty_like(x, dtype=torch.float16)  # 创建一个与x形状相同，数据类型为float16的空张量
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 只处理路由到该专家的token
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)  # 将每个token的权重乘以专家的输出，然后对每个token的输出进行求和
            y = y.view(*orig_shape)  # [bsz, seq_len, dim]
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)  # 使用推理模式
        if self.config.n_shared_experts is not None:  # 如果使用共享专家
            y = y + self.shared_experts(identity)  # 将共享专家的输出与路由专家的输出相加
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()  # 对专家索引进行排序
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)  # 计算每个专家处理的token数量
        token_idxs = idxs // self.config.num_experts_per_tok  # 计算每个token属于哪个专家
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: LMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        self.attention = Attention(config)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, x, pos_cis, past_key_value=None, use_cache=False):
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, past_kv


class MiniMindLM(PreTrainedModel):
    config_class = LMConfig

    def __init__(self, params: LMConfig = None):
        self.params = params or LMConfig()
        super().__init__(self.params)
        self.vocab_size, self.n_layers = params.vocab_size, params.n_layers
        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, params) for l in range(self.n_layers)])
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight
        self.register_buffer("pos_cis",
                             precompute_pos_cis(dim=params.dim // params.n_heads, theta=params.rope_theta),
                             persistent=False)
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)
        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []
        for l, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[l],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.output(self.norm(h)[:, slice_indices, :])
        aux_loss = sum(l.feed_forward.aux_loss for l in self.layers if isinstance(l.feed_forward, MOEFeedForward))
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT

    @torch.inference_mode()
    def generate(self, input_ids, eos_token_id=2, max_new_tokens=1024, temperature=0.75, top_p=0.90,
                 stream=False, rp=1., use_cache=True, pad_token_id=0, num_return_sequences=1, **args):
        # 流式生成
        if stream:
            return self._stream(input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)

        # 直接生成
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            for _ in range(num_return_sequences):
                out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
                tokens_list = [tokens[:, -1:] for tokens in out]
                gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
                full_sequence = torch.cat([non_pad, gen], dim=-1)
                generated.append(full_sequence)

        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        output = torch.cat(generated, dim=0)
        res = output.view(input_ids.size(0) * num_return_sequences, -1)
        return res

    def _stream(self, input_ids, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args):
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
            logits /= (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break
