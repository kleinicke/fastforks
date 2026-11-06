# Modified by fastforks: translated from MegEngine to PyTorch (see net/__init__.py).
# Original: https://github.com/megvii-research/CREStereo/blob/ad3a1613bdedd88b93247e5f002cb7c80799762d/nets/attention/linear_attention.py
"""
Linear Transformer proposed in "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
Modified from: https://github.com/idiap/fast-transformers/blob/master/fast_transformers/attention/linear_attention.py
"""

import torch
import torch.nn as nn


def elu(x, alpha=1.0):
    return torch.clamp(x, min=0) + torch.clamp(alpha * (torch.exp(x) - 1), max=0)


def elu_feature_map(x):
    return elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]  # [:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]  # [:, :, None, None]
            values = values * kv_mask[:, :, None, None]  # [:, :, None, None]

        v_length = values.shape[1]
        values = values / v_length  # prevent fp16 overflow
        KV = torch.sum(torch.unsqueeze(K, -1) * torch.unsqueeze(values, 3), dim=1)
        Z = 1 / (torch.sum(Q * torch.sum(K, dim=1, keepdim=True), dim=-1) + self.eps)
        queried_values = (
            torch.sum(
                torch.unsqueeze(Q, -1) * torch.unsqueeze(KV, 1) * Z[:, :, :, None, None],
                dim=3,
            )
            * v_length
        )

        return queried_values


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(p=attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.sum(torch.unsqueeze(queries, 2) * torch.unsqueeze(keys, 1), dim=-1)
        if kv_mask is not None:
            assert q_mask.dtype == torch.bool
            assert kv_mask.dtype == torch.bool
            QK[
                ~(q_mask[:, :, None, None] & kv_mask[:, None, :, None])
            ] = float("-inf")

        # Compute the attention and the weighted average
        softmax_temp = 1.0 / queries.shape[3] ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.sum(torch.unsqueeze(A, -1) * torch.unsqueeze(values, 1), dim=2)

        return queried_values
