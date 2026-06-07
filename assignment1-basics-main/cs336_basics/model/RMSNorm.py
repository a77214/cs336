import torch
import torch.nn as nn
import torch.nn.init as init


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)

    RMSNorm normalizes the input by the root mean square of the activations,
    without centering (no subtraction of mean).

    Formula: y = (x / RMS(x)) * γ, where RMS(x) = sqrt(mean(x^2) + eps)
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        RMSNorm 模块

        Args:
            d_model: int - 模型的隐藏维度
            eps: float - 数值稳定性的 epsilon 值
            device: torch.device | None - 参数存储设备
            dtype: torch.dtype | None - 参数数据类型
        """
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        # 可学习的缩放参数 γ (gamma)
        # 形状为 (d_model,)，初始化为 1
        self.weight = nn.Parameter(
            torch.empty((d_model,),  device=device, dtype=dtype)
        )
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: torch.Tensor - 输入张量，形状为 (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor - 归一化后的输出，形状为 (batch_size, sequence_length, d_model)
        """
        # 保存原始 dtype 用于后续转换
        original_dtype = x.dtype

        # 上采样到 float32 以提高数值稳定性
        x = x.to(torch.float32)

        # 计算 RMS: sqrt(mean(x^2) + eps)
        # 在最后一个维度上计算 (d_model 维度)
        # 保持维度以便广播: (batch, seq, 1)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # 归一化并缩放: (x / rms) * weight
        # weight 是 (d_model,)，会自动广播到 (batch, seq, d_model)
        output = (x / rms) * self.weight.to(torch.float32)

        # 下采样回原始 dtype
        return output.to(original_dtype)

