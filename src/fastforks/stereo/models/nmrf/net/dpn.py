import torch
from torch import nn
import torch.nn.functional as F

from .nmp import MLP, Propagation, PropagationLayer


def topk_indices(x, k):
    """fastforks: `torch.topk(x, k, dim=-1)` indices with ties to the lower index on every device.

    The proposal scores tie often (all non-maximal bins are set to `eps`: 58-68 % of the rows tie at the k-th value on
    real pairs). CUDA's topk keeps the lowest tied indices; the CPU's (and MPS') picks any, which moved the CPU float32
    disparity 0.21 px (NMRF) / 0.034 px (BridgeDepth) mean off CUDA's. A stable sort gives CUDA's set elsewhere,
    also in traced CPU graphs so they match eager.
    CUDA keeps topk (same set).
    """
    if x.device.type != "cuda":  # CPU, MPS
        return torch.sort(x, dim=-1, descending=True, stable=True)[1][..., :k]
    return torch.topk(x, k, dim=-1)[1]


class DPN(nn.Module):
    """Disparity proposal seed extraction network.

    Args:
        cost_group (int): group number of groupwise cost volume
        num_proposals (int): number of proposals for each pixel
        feat_dim (int): dimension of backbone feature map
        context_dim (int): dimension of visual context
        prop_embed_dim (int): dimension of label seed embedding
        split_size (int): width of stripe
        prop_n_heads: head of attention
    """
    def __init__(self, cost_group, num_proposals, feat_dim, context_dim, num_prop_layers,
                 prop_embed_dim, mlp_ratio,  split_size, prop_n_heads, activation="gelu",
                 attn_drop=0.,  proj_drop=0., dropout=0., normalize_before=False,
    ):
        super().__init__()

        # 1D convolutions sliding along the disparity dimension.
        # Intuition: high-pass filter to make the disparity modal prominent
        self.mlp = nn.Sequential(
            nn.Conv1d(cost_group, 8, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, 1, kernel_size=5, stride=1, padding=2),
        )
        self.eps = 1e-3
        self.num_proposals = num_proposals
        self.cost_group = cost_group

        # ---- label seed propagation ---- #
        # to obtain visual context
        self.proj = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, context_dim, 1, 1, 0, bias=False))

        prop_layers = nn.ModuleList([
            PropagationLayer(
                prop_embed_dim, mlp_ratio=mlp_ratio, context_dim=context_dim,
                split_size=split_size, n_heads=prop_n_heads, activation=activation,
                attn_drop=attn_drop, proj_drop=proj_drop, dropout=dropout,
                normalize_before=normalize_before)
            for i in range(num_prop_layers)]
        )
        prop_norm = nn.LayerNorm(prop_embed_dim)
        self.propagation = Propagation(prop_embed_dim, cost_group, layers=prop_layers, norm=prop_norm)

        self.prop_head = MLP(prop_embed_dim, prop_embed_dim, 1, 3)

        # fastforks: no weight initialisation (`_init_weights`, zeroed prop head): the checkpoint sets every parameter

    def forward(self, cost_volume, fmap1_list):
        """
        cost_volume: [B,G,D,H,W]
        returns:
            labels: [1, B*H*W, num_proposals], per-pixel disparity candidates (fastforks: upstream also returns the
                filtered cost volume, the probability and the label seeds for its losses)
        """
        # ---- step 1: extract disparity modals as label seeds ---- #
        bs, group, nd, ht, wd = cost_volume.shape
        cost_volume = cost_volume.permute(0, 3, 4, 1, 2).reshape(bs*ht*wd, group, nd)
        cost = self.mlp(cost_volume).squeeze(-2)  # [B*H*W,D]
        prob = torch.nn.functional.softmax(cost, dim=-1)
        out = torch.nn.functional.max_pool1d(prob.unsqueeze(-2), kernel_size=3, stride=1, padding=1).squeeze(-2)
        non_local_max = (prob != out) & (prob > self.eps)

        prob_ = torch.where(non_local_max, self.eps, prob.detach())  # fastforks: prob_[non_local_max] = eps on a copy
        label_seeds = topk_indices(prob_, self.num_proposals)

        # ---- step 2: label seed propagation ---- #
        context = self.proj(fmap1_list[0])  # visual context is used in affinity computation
        context = context.permute(0, 2, 3, 1)  # fastforks: rearrange 'b c h w -> b h w c'
        memory, label_seeds = self.propagation(cost_volume, label_seeds, context)
        outputs = self.prop_head(memory).view(-1, *label_seeds.shape)
        labels = F.relu(outputs + label_seeds[None])  # candidate labels

        return labels
