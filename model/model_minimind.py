# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):   # PretrainedConfig是transformers提供的一个用于存储和管理预训练模型配置的基类
    model_type = "minimind"  # 通过model_type表示模型架构类型

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,  # 最大上下文长度32K
            num_attention_heads: int = 8,  # 注意力头数量
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,  # GQA中一个组中包含的head数量
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
            use_moe: bool = False,  # 是否使用MoE
            num_experts_per_tok: int = 2,  # 每个token选择的专家数量
            n_routed_experts: int = 4,  # 路由专家数量
            n_shared_experts: int = 1,  # 共享专家数量
            scoring_func: str = 'softmax',  # 评分函数，默认为'softmax'
            aux_loss_alpha: float = 0.1,  # 辅助损失的alpha参数
            seq_aux: bool = True,  # 是否在序列级别上计算辅助损失
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 路由专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家数量
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

import math
import torch
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    '''
    dim: 模型的隐藏维度，通常是注意力头的维度
    end: 预计算的最大序列长度，默认为32K（32,768），适应超长上下文
    theta: 基频参数，默认为1e6，控制旋转的尺度，影响模型的外推性能
    '''
    # 计算频率基向量
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # 生成位置索引序列
    freqs = torch.outer(t, freqs).float()  # 计算外积，生成一个形状为[end, dim//2]的矩阵，freqs[pos, i] = pos * freqs[i]，表示位置pos在维度i上的旋转角度
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:  # GQA减少参数和计算量的方法就是重复key和value
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads  # 当未设置num_key_value_heads时，默认与num_attention_heads相同，表示不使用GQA
        assert args.num_attention_heads % self.num_key_value_heads == 0  # 确保num_attention_heads可以被num_key_value_heads整除；定义num_key_value_heads的目的是实现GQA
        self.n_local_heads = args.num_attention_heads  # 理论上每个组中的头数
        self.n_local_kv_heads = self.num_key_value_heads  # 实际每个组中的头数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 每个组中重复key和value的次数
        self.head_dim = args.hidden_size // args.num_attention_heads  # 每个头的维度
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)  # 新增注意力头数这个维度
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])  # 给q和k应用rope

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)  # 将之前缓存的key和当前的key拼接
            xv = torch.cat([past_key_value[1], xv], dim=1)  # 将之前缓存的value和当前的value拼接
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len != 1:
            dropout_p = self.dropout if self.training else 0.0  # 如果训练，dropout_p为dropout，否则为0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)  # 使用flash attention加速计算
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)  # 计算注意力分数
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask  添加掩码

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)  # 对注意力分数进行softmax归一化
            scores = self.attn_dropout(scores)  # 使用dropout
            output = scores @ xv  # 计算加权和

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = config.n_routed_experts  # 路由专家数量

        self.scoring_func = config.scoring_func  # 评分函数
        self.alpha = config.aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = config.seq_aux  # 是否在序列级别上计算辅助损失

        self.norm_topk_prob = config.norm_topk_prob  # 是否标准化top-k概率
        self.gating_dim = config.hidden_size  # 门控维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))  # 用于计算每个token对每个专家的路由得分
        self.reset_parameters()  # 初始化权重

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # 使用kaiming初始化方法初始化权重

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
            if self.seq_aux:  # 如果序列级别上计算辅助损失
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
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
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])  # 创建路由专家列表
        self.gate = MoEGate(config)  # 创建门控机制
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])  # 创建共享专家列表

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)  # 返回每个token选择的路由专家索引、权重和辅助损失；topk_idx的shape为[bsz*seq_len, topk]
        x = x.view(-1, x.shape[-1])  # [bsz*seq_len, dim]
        flat_topk_idx = topk_idx.view(-1)  # [bsz*seq_len*topk]
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)  # 将输入的每个token重复num_experts_per_tok次，以便每个token都能被路由到多个专家，[bsz*seq_len*topk, dim]
            y = torch.empty_like(x, dtype=torch.float16)  # 创建一个与x形状相同，数据类型为float16的空张量
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致；只处理路由到该专家的token
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)  # 将每个token的权重乘以专家的输出，然后将多个专家对每个token的输出求和
            y = y.view(*orig_shape)  # [bsz, seq_len, dim]
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)  # 使用推理模式
        if self.config.n_shared_experts > 0:  # 如果使用共享专家
            for expert in self.shared_experts:
                y = y + expert(identity)  # 将共享专家的输出与路由专家的输出相加
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()  # 对每个token对应的专家的索引值按升序排序，返回的是专家索引值列表的索引
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)  # 计算每个专家索引出现的次数，相当于路由到每个专家的token数量
        token_idxs = idxs // self.config.num_experts_per_tok  # 因为flat_expert_indices中包含了每个token对应的topk的专家索引后完全拉平，要除以topk才是每个专家对应的token的索引
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]  # 获取第i个专家
            exp_token_idx = token_idxs[start_idx:end_idx]  # 获取属于第i个专家的token索引
            expert_tokens = x[exp_token_idx]  # 获取属于第i个专家的token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)  # 计算专家的输出
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])  # 将专家的输出乘以权重
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)  # 将专家的输出添加到expert_cache中

        return expert_cache


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)  # 如果use_moe为True，则使用MOEFeedForward，否则使用FeedForward

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # 词嵌入层
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)  # 预计算rope所需的cos和sin值
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)  # 注册缓存，避免在反向传播时被优化器更新
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)  # 注册缓存，避免在反向传播时被优化器更新

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)  # KV Cache
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0  # 本地输入对应的起始位置

        hidden_states = self.dropout(self.embed_tokens(input_ids))  # [bsz, seq_len] ->[bsz, seq_len, dim]

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )  # 计算辅助损失

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):  # 构建符合transformers的模型
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight  # 将内部词嵌入层的权重与线性层的权重共享；线性层的weight的shape与定义时的shape刚好转置，故两者的shape都是[vocab_size, hidden_size]，即可以共享
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])  # 获取隐变量h中指定位置的logits；[bsz, seq_len, vocab_size]
        self.OUT.__setitem__('last_hidden_state', h)  # 将隐变量h设置为last_hidden_state；包含所有层的隐变量
        self.OUT.__setitem__('logits', logits)  # 将logits设置为logits
        self.OUT.__setitem__('aux_loss', aux_loss)  # 将辅助损失设置为aux_loss
        self.OUT.__setitem__('past_key_values', past_kvs)  # 将past_key_values设置为past_kvs；包含所有层的KV Cache
        return self.OUT
