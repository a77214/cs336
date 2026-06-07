import math

import torch
import torch.nn as nn
import torch.nn.init as init


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        线性变换模块（无 bias）

        Args:
            in_features: int - 输入特征维度
            out_features: int - 输出特征维度
            device: torch.device | None - 参数存储设备
            dtype: torch.dtype | None - 参数数据类型
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 创建权重参数 W，形状为 (out_features, in_features)
        # 注意：存储的是 W，不是 W^T

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features),  device=device, dtype=dtype)
        )
        sigma=math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, 0.0, sigma,a=-3*sigma,b=3*sigma)





    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：y = x @ W^T

        Args:
            x: torch.Tensor - 输入张量，最后一维必须是 in_features

        Returns:
            torch.Tensor - 输出张量，最后一维是 out_features
        """
        # 矩阵乘法：x shape (*, in_features)
        # W shape (out_features, in_features)
        # 需要计算 x @ W^T 得到 (*, out_features)

        return  x @ self.weight.T



