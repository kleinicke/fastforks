"""Track-On2 / Track-On-R goldens from upstream's `model.trackon_predictor.Predictor` (gorkaydemir/track_on).

Reference: upstream `Predictor.forward` at `COMMIT` (DINOv3 ViT-S+ adapter, inference memory 72, 20 x 20 support
grid, visibility threshold 0.8), CPU float32. Upstream imports `mmcv.ops.MultiScaleDeformableAttention`; `shim()`
provides it as mmcv 2.2's own PyTorch path (`multi_scale_deformable_attn_pytorch`, the code mmcv runs off CUDA),
without mmcv. The DINOv3 ViT is `transformers` `AutoModel` from the local `facebook/dinov3-vits16plus-pretrain-lvd1689m`
files (upstream downloads it by name). `utils.train_utils` (training imports) is replaced by its
`load_pretrained_weights` (strict except the DINOv3 keys, as upstream). Input: tests/synth.py `clip()` (6 frames,
100 x 150) in [0, 255] as upstream expects; queries `(t, x + 0.5, y + 0.5)` (upstream's pixel edges at integers).
Stored: the queries, the tracks in clip pixels (pixel centres at integers) `[T, Q, 2]` and the visibility
`sigmoid(logit - ln 4)` `[T, Q]` (the runner's: > 0.5 where upstream's sigmoid is > 0.8); frames before a query's
frame hold the query and 0.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/track_on.py TRACK_ON_SRC WEIGHTS_DIR DINOV3_DIR

TRACK_ON_SRC is a checkout of gorkaydemir/track_on at `COMMIT`; WEIGHTS_DIR holds the checkpoints; DINOV3_DIR the
HF files (config.json, model.safetensors). `tools/eval_tracking.py` uses `load` and `track` as the original.
"""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

UPSTREAM = "https://github.com/gorkaydemir/track_on"
COMMIT = "04efe8a137bd76e45e07382f02748794c9b61a96"
VIS_SHIFT = math.log(0.8 / 0.2)
QUERIES = np.array(
    [[0, 50, 45], [0, 120.5, 20.25], [2, 30, 80], [0, 0, 0], [5, 75, 55], [3, 149, 99], [4, 60, 50]], dtype=np.float32
)
MODELS = {"track-on2@1": "trackon2_dinov3_checkpoint.pt", "track-on-r@1": "track_on_r.pt"}
DINOV3 = "dinov3-vits16plus-pretrain-lvd1689m"  # the HF snapshot dir of the backbone (gated; see the card)


def _msda_pytorch(value: torch.Tensor, shapes: Any, loc: torch.Tensor, w: torch.Tensor) -> torch.Tensor:  # noqa: ANN401
    """Mmcv 2.2 `multi_scale_deformable_attn_pytorch`."""
    bs, _, heads, dims = value.shape
    _, nq, heads, levels, points, _ = loc.shape
    values = value.split([int(h) * int(w_) for h, w_ in shapes], dim=1)
    grids = 2 * loc - 1
    out = []
    for lvl, (h, w_) in enumerate(shapes):
        v = values[lvl].flatten(2).transpose(1, 2).reshape(bs * heads, dims, int(h), int(w_))
        g = grids[:, :, :, lvl].transpose(1, 2).flatten(0, 1)
        out.append(F.grid_sample(v, g, mode="bilinear", padding_mode="zeros", align_corners=False))
    w = w.transpose(1, 2).reshape(bs * heads, 1, nq, levels * points)
    return (torch.stack(out, dim=-2).flatten(-2) * w).sum(-1).view(bs, heads * dims, nq).transpose(1, 2).contiguous()


class MultiScaleDeformableAttention(nn.Module):
    """mmcv 2.2 `MultiScaleDeformableAttention` (eval: dropout off; 2-d reference points), PyTorch path."""

    def __init__(  # noqa: D107, PLR0913, PLR0917 - mmcv's signature
        self, embed_dims: int = 256, num_heads: int = 8, num_levels: int = 4, num_points: int = 4,
        im2col_step: int = 64, dropout: float = 0.1, batch_first: bool = False, norm_cfg: Any = None,  # noqa: ANN401, ARG002, FBT001, FBT002
        init_cfg: Any = None, value_proj_ratio: float = 1.0,  # noqa: ANN401, ARG002
    ) -> None:  # fmt: skip
        super().__init__()
        self.batch_first, self.num_heads, self.num_levels, self.num_points = (
            batch_first,
            num_heads,
            num_levels,
            num_points,
        )
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, size)
        self.output_proj = nn.Linear(size, embed_dims)

    def init_weights(self) -> None:
        """The mmcv initialisation: a no-op here (the checkpoint overwrites every weight)."""

    def forward(  # noqa: D102, PLR0913, PLR0917 - mmcv's signature
        self, query: torch.Tensor, key: Any = None, value: Any = None, identity: Any = None, query_pos: Any = None,  # noqa: ANN401, ARG002
        key_padding_mask: Any = None, reference_points: Any = None, spatial_shapes: Any = None, **kwargs: Any,  # noqa: ANN401, ARG002
    ) -> torch.Tensor:  # fmt: skip
        value = query if value is None else value
        identity = query if identity is None else identity
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query, value = query.permute(1, 0, 2), value.permute(1, 0, 2)
        bs, nq, _ = query.shape
        nv = value.shape[1]
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, nv, self.num_heads, -1)
        h, lv, p = self.num_heads, self.num_levels, self.num_points
        off = self.sampling_offsets(query).view(bs, nq, h, lv, p, 2)
        w = self.attention_weights(query).view(bs, nq, h, lv * p).softmax(-1).view(bs, nq, h, lv, p)
        norm = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        loc = reference_points[:, :, None, :, None, :] + off / norm[None, None, None, :, None, :]
        out = self.output_proj(_msda_pytorch(value, spatial_shapes, loc, w))
        if not self.batch_first:
            out = out.permute(1, 0, 2)
        return out + identity


def shim(dinov3: Path) -> None:
    """`mmcv.ops` and `utils.train_utils` stand-ins; `AutoModel.from_pretrained` reads the local DINOv3 files."""
    mmcv, ops = types.ModuleType("mmcv"), types.ModuleType("mmcv.ops")
    ops.MultiScaleDeformableAttention = MultiScaleDeformableAttention
    mmcv.ops = ops
    sys.modules["mmcv"], sys.modules["mmcv.ops"] = mmcv, ops
    tu = types.ModuleType("utils.train_utils")

    def load_pretrained_weights(model: nn.Module, path: str) -> None:
        sd = torch.load(path, map_location="cpu", weights_only=True)
        res = model.load_state_dict(sd, strict=False)
        missing = [k for k in res.missing_keys if ".dinov3." not in k]
        if res.unexpected_keys or missing:
            raise RuntimeError((res.unexpected_keys, missing))

    tu.load_pretrained_weights = load_pretrained_weights
    sys.modules["utils.train_utils"] = tu
    import transformers  # noqa: PLC0415

    orig = transformers.AutoModel.from_pretrained

    def from_pretrained(_name: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        kwargs.pop("local_files_only", None)
        model = orig(str(dinov3), *args, local_files_only=True, **kwargs)
        if not hasattr(model, "layer"):  # transformers 5 moved the blocks to `model.model.layer`; not re-registered
            model.__dict__["layer"] = model.model.layer
        return model

    transformers.AutoModel.from_pretrained = from_pretrained


def load(src: Path, file: Path, dinov3: Path) -> Any:  # noqa: ANN401 - upstream Predictor
    """Upstream `Predictor(checkpoint_path=file)` (default args: DINOv3 S+, memory 72, grid 20), in eval mode."""
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    shim(dinov3)
    from model.trackon_predictor import Predictor  # noqa: PLC0415

    return Predictor(checkpoint_path=str(file)).eval().requires_grad_(False)  # noqa: FBT003 - torch's signature


def track(
    model: Any,  # noqa: ANN401 - upstream Predictor
    video: torch.Tensor,
    queries: torch.Tensor,
    amp: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tracks `[T, Q, 2]` and visibility `[T, Q]` (see the module docstring) of `video` `[T, 3, H, W]` in [0, 1].

    Upstream's `Predictor.forward` returns thresholded visibility; this runs its loop (`forward_frame`) with the
    sigmoid kept.
    """
    q = queries.to(video.device).float()
    n, (h, w) = video.shape[0], video.shape[-2:]
    up = torch.cat([q[:, :1], q[:, 1:] + 0.5], 1)
    model.delta_v = 0.0  # keep the logits: forward_frame thresholds `sigmoid >= delta_v`
    m = model.model
    tracks = q[None, :, 1:].repeat(n, 1, 1)
    vis = q.new_zeros(n, len(q))
    with torch.no_grad(), torch.autocast(video.device.type, amp or torch.float32, enabled=amp is not None):
        from utils.coord_utils import get_points_on_a_grid  # noqa: PLC0415

        extra = get_points_on_a_grid(model.support_grid_size, (h, w), video.device)[0]
        allq = torch.cat([up, torch.cat([extra.new_zeros(len(extra), 1), extra], 1)])
        times = allq[:, 0].long()
        model.reset()
        model.initial_capacity = len(allq)
        order: list[int] = []
        logits: dict[int, torch.Tensor] = {}
        orig_track = m.track_frame

        def track_frame(*a: Any) -> Any:  # noqa: ANN401 - keeps the visibility logits
            p, v, qn = orig_track(*a)
            logits["v"] = v
            return p, v, qn

        m.track_frame = track_frame
        x = video.float() * 255
        for t in range(n):
            sel = (times == t).nonzero()[:, 0]
            new = allq[sel, 1:] if len(sel) else None
            order.extend(sel.tolist())
            if model.N > 0 or new is not None:
                p, _ = model.forward_frame(x[t : t + 1], new_queries=new)
                idx = torch.tensor(order, device=video.device)
                keep = idx < len(q)
                tracks[t, idx[keep]] = p[keep].float() - 0.5
                vis[t, idx[keep]] = (logits["v"][keep].float() - VIS_SHIFT).sigmoid()
        m.track_frame = orig_track
    return tracks, vis


def main() -> None:
    from _common import write  # noqa: PLC0415

    from synth import clip  # noqa: PLC0415

    src, weights, dinov3 = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    video = torch.from_numpy(np.stack(clip())).permute(0, 3, 1, 2).float() / 255
    for ref, file in MODELS.items():
        tracks, vis = track(load(src, weights / file, dinov3), video, torch.from_numpy(QUERIES))
        write(
            ref,
            {"queries": QUERIES, "tracks": tracks.numpy(), "visibility": vis.numpy()},
            family="track_on",
            upstream=f"{UPSTREAM}@{COMMIT}",
            patch=None,
            settings=f"CPU float32, upstream Predictor.forward loop (DINOv3 ViT-S+ adapter at 384 x 512, memory 72, "
            f"20 x 20 support grid), mmcv MultiScaleDeformableAttention as its PyTorch path, tests/synth.py clip() "
            f"6 x 100 x 150, torch {torch.__version__}",
            task="tracking",
        )


if __name__ == "__main__":
    main()
