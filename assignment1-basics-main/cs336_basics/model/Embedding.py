import math
import torch
import torch.nn as nn
import torch.nn.init as init


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        嵌入查找模块

        Args:
            num_embeddings: int - 词汇表大小
            embedding_dim: int - 嵌入向量维度 (d_model)
            device: torch.device | None - 参数存储设备
            dtype: torch.dtype | None - 参数数据类型
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 创建嵌入矩阵，形状为 (num_embeddings, embedding_dim)
        # 注意：embedding_dim (d_model) 是最后一维

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim),  device=device, dtype=dtype)
        )
        torch.nn.init.trunc_normal_(self.weight, 0.0,std=1.0,a=-3,b=3)



    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        查找 token IDs 对应的嵌入向量

        Args:
            token_ids: torch.Tensor - 任意形状的张量，包含 token 索引

        Returns:
            torch.Tensor - 嵌入向量，形状为 (*, embedding_dim)
                          即输入形状后面加上 embedding_dim 维度
        """
        # 简单的查找：根据索引从 weight 矩阵中选取行
        return self.weight[token_ids]
