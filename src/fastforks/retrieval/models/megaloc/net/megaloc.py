"""MegaLoc: One Retrieval to Place Them All

This module implements the MegaLoc model for visual place recognition.
The model combines a Vision Transformer backbone with an optimal transport-based
feature aggregation module.

Paper: https://arxiv.org/abs/2502.17237
License: MIT
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# fastforks: torchvision import dropped (only MegaLoc.forward's resize used it); the DINOv2 backbone and MegaLoc
# wrapper are not vendored: fastforks.shared.depth_anything_v2's ViT-B/14 + this aggregator (see model.py)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(log_a, log_b, M, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


class FeatureAggregator(nn.Module):
    """Optimal transport-based aggregation of local features into global descriptor.

    This module aggregates local patch features into a compact global representation
    using differentiable optimal transport.

    Args:
        num_channels: Number of input feature channels (from backbone)
        num_clusters: Number of cluster centers
        cluster_dim: Dimensionality of cluster descriptors
        token_dim: Dimensionality of global scene token
        mlp_dim: Hidden dimension for MLPs
        dropout: Dropout probability (0 to disable)
    """

    def __init__(
        self,
        num_channels=1536,
        num_clusters=64,
        cluster_dim=128,
        token_dim=256,
        mlp_dim=512,
        dropout=0.3,
    ) -> None:
        super().__init__()

        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        self.mlp_dim = mlp_dim

        if dropout > 0:
            dropout = nn.Dropout(dropout)
        else:
            dropout = nn.Identity()

        # MLP for global scene token
        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, self.mlp_dim), nn.ReLU(), nn.Linear(self.mlp_dim, self.token_dim)
        )
        # MLP for local features
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, self.mlp_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.mlp_dim, self.cluster_dim, 1),
        )
        # MLP for score matrix
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, self.mlp_dim, 1),
            dropout,
            nn.ReLU(),
            nn.Conv2d(self.mlp_dim, self.num_clusters, 1),
        )
        # Dustbin parameter
        self.dust_bin = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        Args:
            x: Tuple of (features, token)
                features: [B, C, H, W] spatial feature map
                token: [B, C] global CLS token

        Returns:
            Global descriptor [B, num_clusters * cluster_dim + token_dim]
        """
        x, t = x

        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        p = get_matching_probs(p, self.dust_bin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]

        f = torch.cat(
            [
                F.normalize(t, p=2, dim=-1),
                F.normalize(torch.einsum("bdn,bkn->bdk", f, p), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )

        return F.normalize(f, p=2, dim=-1)


class Aggregator(nn.Module):
    def __init__(self, feat_dim, agg_config, salad_out_dim):
        super().__init__()
        self.agg = FeatureAggregator(**agg_config)
        self.linear = nn.Linear(salad_out_dim, feat_dim)

    def forward(self, x):
        x = self.agg(x)
        return self.linear(x)


