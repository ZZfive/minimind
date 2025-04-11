import torch
from torch import optim, nn


# 定义Lora网络结构
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:  # 对模型架构中输入、输出尺寸相等的线形成设置Lora，相当于只对自注意力模块中的转换矩阵进行LoRA处理
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            setattr(module, "lora", lora)  # 将LoRA模块设置为module的属性
            original_forward = module.forward

            # 显式绑定
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)  # 将原始模块的输出与LoRA模块的输出相加

            module.forward = forward_with_lora  # 将新的forward方法设置为module的forward方法


def load_lora(model, path):
    state_dict = torch.load(path, map_location=model.device)  # 从指定路径加载Lora模型参数
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):  # 检查module是否具有lora属性
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)  # 将Lora模型参数加载到module的lora属性中


def save_lora(model, path):
    state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):  # 检查module是否具有lora属性
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)  # 将state_dict保存到指定路径
