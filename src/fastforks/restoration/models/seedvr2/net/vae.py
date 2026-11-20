"""The SeedVR2 video VAE (`models/video_vae_v3`, `s8_c16_t4_inflation_sd3`): causal 3-D convolutions, 8x / 4x.

Module names follow upstream so the checkpoint loads as is. Upstream's `InflatedCausalConv3d` drops the temporal
padding and instead repeats the first frame `2 * pad` times (no memory) or prepends the last `k_t - s_t` frames of the
previous chunk (`MemoryState.ACTIVE`). The encoder runs in chunks of 4 frames after the first `1 + 4`, the decoder in
chunks of 1 latent frame after the first 2 (`slicing_encode` / `slicing_decode`); both are exact sliding windows, so
the chunk length only trades memory for launch count. Upstream's `memory_limit` splitting of large convolutions and
group norms along H / channel groups is the same arithmetic and is not reproduced.

Dtypes: the weights are bf16. Upstream encodes in plain bf16 and decodes under bf16 autocast, where the group norms
run in float32 (`upcast`) before `causal_norm_wrapper` rounds them back to bf16.
"""

from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

CHANNELS = (128, 256, 512, 512)
LATENT = 16
GROUPS = 32
EPS = 1e-6
SCALE = 0.9152  # latent scaling factor
TEMPORAL_DOWN = (False, True, True)  # encoder downsamplers 0..2
TEMPORAL_UP = (True, True, False)  # decoder upsamplers 0..2
_GROUP_NORM_MIN_BYTES = 384 * 1024**2
_GROUP_NORM_CHUNK_BYTES = 256 * 1024**2


class State(Enum):
    """Upstream's `MemoryState`: whole clip, first chunk (keeps memory), later chunk (uses memory)."""

    DISABLED = 0
    INITIALIZING = 1
    ACTIVE = 2


class CausalConv3d(nn.Conv3d):
    """`InflatedCausalConv3d`: spatial padding in the convolution, temporal padding by repetition or memory."""

    def __init__(
        self,
        cin: int,
        cout: int,
        kernel: int | tuple[int, int, int],
        stride: int | tuple = 1,
        padding: int | tuple = 0,
        *,
        bias: bool = True,
    ) -> None:
        k = (kernel,) * 3 if isinstance(kernel, int) else kernel
        s = (stride,) * 3 if isinstance(stride, int) else stride
        p = (padding,) * 3 if isinstance(padding, int) else padding
        super().__init__(cin, cout, k, s, (0, p[1], p[2]), bias=bias, device="meta")
        self.head = 2 * p[0]
        self.keep = k[0] - s[0]
        self.memory: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, state: State) -> torch.Tensor:
        kt, st = self.kernel_size[0], self.stride[0]
        if kt == 1:
            return super().forward(x[:, :, ::st])
        if state is State.ACTIVE and self.memory is not None:
            frames = list(self.memory.unbind(2)) + list(x.unbind(2))
        else:
            frames = [x[:, :, 0]] * self.head + list(x.unbind(2))
        if self.keep and state is not State.DISABLED:
            self.memory = torch.stack(frames[len(frames) - self.keep :], 2)  # a copy, not a view of the clip
        else:
            self.memory = None
        # one output frame at a time from its kt input frames: no padded copy of the whole clip
        n = (len(frames) - kt) // st + 1
        if (
            not (torch.compiler.is_compiling() or torch.jit.is_tracing())
            and torch.is_inference_mode_enabled()
            and n == 1
            and state is State.DISABLED
            and self.kernel_size == (3, 3, 3)
            and self.stride == (1, 1, 1)
            and self.padding == (0, 1, 1)
            and x.is_cuda
            and x.numel() * x.element_size() * kt >= 512 * 1024**2
        ):
            return self._spatial_strips(frames)
        out = None
        for j in range(n):
            y = super().forward(torch.stack(frames[j * st : j * st + kt], 2))
            if out is None:
                out = y.new_empty((*y.shape[:2], n, *y.shape[3:]))
            out[:, :, j : j + 1] = y
        return out

    def _spatial_strips(self, frames: list[torch.Tensor]) -> torch.Tensor:
        """Apply a stride-one 3x3 convolution to rows with one input halo on either side."""
        b, c, h, w = frames[0].shape
        bytes_per_row = b * c * w * frames[0].element_size() * len(frames)
        rows = max(1, 256 * 1024**2 // bytes_per_row)
        out: torch.Tensor | None = None
        for start in range(0, h, rows):
            end = min(h, start + rows)
            top, bottom = max(0, start - 1), min(h, end + 1)
            y = super().forward(torch.stack([frame[:, :, top:bottom] for frame in frames], dim=2))
            if out is None:
                out = y.new_empty((b, self.out_channels, 1, h, w))
            out[:, :, :, start:end] = y[:, :, :, start - top : end - top]
        assert out is not None
        return out


def group_norm(norm: nn.GroupNorm, x: torch.Tensor, *, upcast: bool, silu: bool = False) -> torch.Tensor:
    """`causal_norm_wrapper`: per-frame group norm of `[B, C, T, H, W]`, rounded to the input dtype; `silu` then
    applies SiLU in place.

    Write one frame at a time into the output. For large CUDA decoder frames, split the float32 temporary at whole
    channel-group boundaries, because each group's normalization depends only on its own channels.
    """
    w, b = (norm.weight.float(), norm.bias.float()) if upcast else (norm.weight, norm.bias)
    out = torch.empty_like(x)
    groups = norm.num_groups
    group_chunk = groups
    if (
        upcast
        and x.is_cuda
        and not (torch.compiler.is_compiling() or torch.jit.is_tracing())
        and torch.is_inference_mode_enabled()
    ):
        batch, channels, _, height, width = x.shape
        frame_bytes = batch * channels * height * width * torch.float32.itemsize
        if frame_bytes >= _GROUP_NORM_MIN_BYTES:
            group_chunk = max(1, _GROUP_NORM_CHUNK_BYTES // (frame_bytes // groups))
    channels_per_group = x.shape[1] // groups
    for i in range(x.shape[2]):
        f = x[:, :, i]
        if group_chunk >= groups:
            out[:, :, i] = F.group_norm(f.float() if upcast else f.contiguous(), groups, w, b, norm.eps)
        else:
            for start in range(0, groups, group_chunk):
                end = min(start + group_chunk, groups)
                channel = slice(start * channels_per_group, end * channels_per_group)
                out[:, channel, i] = F.group_norm(f[:, channel].float(), end - start, w[channel], b[channel], norm.eps)
    return F.silu(out, inplace=True) if silu else out


class ResnetBlock3D(nn.Module):
    """norm -> SiLU -> 3x3x3 -> norm -> SiLU -> 3x3x3, plus the (1x1x1) shortcut."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(GROUPS, cin, EPS, device="meta")
        self.conv1 = CausalConv3d(cin, cout, 3, padding=1)
        self.norm2 = nn.GroupNorm(GROUPS, cout, EPS, device="meta")
        self.conv2 = CausalConv3d(cout, cout, 3, padding=1)
        self.conv_shortcut = CausalConv3d(cin, cout, 1) if cin != cout else None

    def forward(self, x: torch.Tensor, state: State, upcast: bool) -> torch.Tensor:  # noqa: FBT001
        h = self.conv1(group_norm(self.norm1, x, upcast=upcast, silu=True), state)
        h = self.conv2(group_norm(self.norm2, h, upcast=upcast, silu=True), state)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x, state)
        return h.add_(x)


class Attention(nn.Module):
    """diffusers' single-head `Attention` (`_from_deprecated_attn_block`) per frame, with the residual."""

    def __init__(self, c: int) -> None:
        super().__init__()
        self.group_norm = nn.GroupNorm(GROUPS, c, EPS, device="meta")
        self.to_q = nn.Linear(c, c, device="meta")
        self.to_k = nn.Linear(c, c, device="meta")
        self.to_v = nn.Linear(c, c, device="meta")
        self.to_out = nn.ModuleList([nn.Linear(c, c, device="meta")])

    def forward(self, x: torch.Tensor, upcast: bool) -> torch.Tensor:  # noqa: FBT001
        b, c, t, h, w = x.shape
        y = group_norm(self.group_norm, x, upcast=upcast).permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
        q, k, v = (m(y)[:, None] for m in (self.to_q, self.to_k, self.to_v))
        y = self.to_out[0](F.scaled_dot_product_attention(q, k, v)[:, 0])
        return x + y.view(b, t, h, w, c).permute(0, 4, 1, 2, 3)


class MidBlock(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock3D(c, c), ResnetBlock3D(c, c)])
        self.attentions = nn.ModuleList([Attention(c)])

    def forward(self, x: torch.Tensor, state: State, upcast: bool) -> torch.Tensor:  # noqa: FBT001
        x = self.resnets[0](x, state, upcast)
        x = self.attentions[0](x, upcast)
        return self.resnets[1](x, state, upcast)


class Downsample3D(nn.Module):
    """Zero pad right / bottom, then a stride-2 convolution (3x3x3 with temporal stride 2, or 1x3x3)."""

    def __init__(self, c: int, *, temporal: bool) -> None:
        super().__init__()
        kt = 3 if temporal else 1
        self.conv = CausalConv3d(c, c, (kt, 3, 3), (2 if temporal else 1, 2, 2), (1 if temporal else 0, 0, 0))

    def forward(self, x: torch.Tensor, state: State) -> torch.Tensor:
        return self.conv(F.pad(x, (0, 1, 0, 1)), state)


class Upsample3D(nn.Module):
    """1x1x1 conv to `c * 4 (* 2)` channels, depth-to-space (and time), drop the duplicated first frame, 3x3x3."""

    def __init__(self, c: int, *, temporal: bool) -> None:
        super().__init__()
        self.t = 2 if temporal else 1
        self.upscale_conv = nn.Conv3d(c, c * 4 * self.t, 1, device="meta")
        self.conv = CausalConv3d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor, state: State) -> torch.Tensor:
        b, c, f, h, w = x.shape
        x = self.upscale_conv(x).view(b, 2, 2, self.t, c, f, h, w)  # "b (x y z c) f h w -> b c (f z) (h x) (w y)"
        x = x.permute(0, 4, 5, 3, 6, 1, 7, 2).reshape(b, c, f * self.t, 2 * h, 2 * w)
        if self.t == 2 and state is not State.ACTIVE:
            x = torch.cat((x[:, :, :1], x[:, :, 2:]), 2)
        return self.conv(x, state)


class DownBlock(nn.Module):
    def __init__(self, cin: int, cout: int, temporal: bool | None) -> None:  # noqa: FBT001
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock3D(cin, cout), ResnetBlock3D(cout, cout)])
        self.downsamplers = None if temporal is None else nn.ModuleList([Downsample3D(cout, temporal=temporal)])

    def forward(self, x: torch.Tensor, state: State) -> torch.Tensor:
        for r in self.resnets:
            x = r(x, state, False)  # noqa: FBT003
        return x if self.downsamplers is None else self.downsamplers[0](x, state)


class UpBlock(nn.Module):
    def __init__(self, cin: int, cout: int, temporal: bool | None) -> None:  # noqa: FBT001
        super().__init__()
        self.resnets = nn.ModuleList([ResnetBlock3D(cin if i == 0 else cout, cout) for i in range(3)])
        self.upsamplers = None if temporal is None else nn.ModuleList([Upsample3D(cout, temporal=temporal)])

    def forward(self, x: torch.Tensor, state: State) -> torch.Tensor:
        for r in self.resnets:
            x = r(x, state, True)  # noqa: FBT003
        return x if self.upsamplers is None else self.upsamplers[0](x, state)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c = CHANNELS
        self.conv_in = CausalConv3d(3, c[0], 3, padding=1)
        temporal = (*TEMPORAL_DOWN, None)
        self.down_blocks = nn.ModuleList([DownBlock(c[max(i - 1, 0)], c[i], temporal[i]) for i in range(4)])
        self.mid_block = MidBlock(c[-1])
        self.conv_norm_out = nn.GroupNorm(GROUPS, c[-1], EPS, device="meta")
        self.conv_out = CausalConv3d(c[-1], 2 * LATENT, 3, padding=1)

    def forward(self, x: torch.Tensor, state: State) -> torch.Tensor:
        x = self.conv_in(x, state)
        for blk in self.down_blocks:
            x = blk(x, state)
        x = self.mid_block(x, state, False)  # noqa: FBT003
        return self.conv_out(group_norm(self.conv_norm_out, x, upcast=False, silu=True), state)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        c = CHANNELS[::-1]
        self.conv_in = CausalConv3d(LATENT, c[0], 3, padding=1)
        self.mid_block = MidBlock(c[0])
        temporal = (*TEMPORAL_UP, None)
        self.up_blocks = nn.ModuleList([UpBlock(c[max(i - 1, 0)], c[i], temporal[i]) for i in range(4)])
        self.conv_norm_out = nn.GroupNorm(GROUPS, c[-1], EPS, device="meta")
        self.conv_out = CausalConv3d(c[-1], 3, 3, padding=1)

    def forward(self, z: torch.Tensor, state: State) -> torch.Tensor:
        x = self.mid_block(self.conv_in(z, state), state, True)  # noqa: FBT003
        for blk in self.up_blocks:
            x = blk(x, state)
        return self.conv_out(group_norm(self.conv_norm_out, x, upcast=True, silu=True), state)


def _chunks(n: int, first: int, size: int) -> list[tuple[int, int, State]]:
    """`slicing_encode` / `slicing_decode`: `[0, first)` then `size`-long chunks; one pass if it all fits the first."""
    if n <= first:
        return [(0, n, State.DISABLED)]
    out = [(0, first, State.INITIALIZING)]
    out += [(i, min(i + size, n), State.ACTIVE) for i in range(first, n, size)]
    return out


class VAE(nn.Module):
    """`encode`: `[1, 3, T, H, W]` in [-1, 1] (T = 4n + 1) -> scaled latent `[t, h, w, 16]`; `decode` the inverse."""

    def __init__(self, sd: dict[str, torch.Tensor], *, encode_chunk: int = 4, decode_chunk: int = 1) -> None:
        super().__init__()
        self.encoder, self.decoder = Encoder(), Decoder()
        self.load_state_dict(sd, assign=True)
        self.encode_chunk, self.decode_chunk = encode_chunk, decode_chunk

    def encode(self, x: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """Posterior sample `mean + std * eps` (`eps` `[1, 16, t, h, w]`), times the scaling factor."""
        n = x.shape[2]
        h = torch.cat(
            [self.encoder(x[:, :, a:b], s) for a, b, s in _chunks(n, 1 + self.encode_chunk, self.encode_chunk)], 2
        )
        self._forget()
        mean, logvar = h.chunk(2, 1)
        z = mean + torch.exp(0.5 * logvar.clamp(-30.0, 20.0)) * eps.to(mean)
        return (z * SCALE)[0].permute(1, 2, 3, 0)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latent `[t, h, w, 16]` -> `[1, 3, 4t - 3, 8h, 8w]`."""
        z = (z / SCALE).permute(3, 0, 1, 2)[None]
        n = z.shape[2]
        out = torch.cat(
            [self.decoder(z[:, :, a:b], s) for a, b, s in _chunks(n, 1 + self.decode_chunk, self.decode_chunk)], 2
        )
        self._forget()
        return out

    def _forget(self) -> None:
        for m in self.modules():
            if isinstance(m, CausalConv3d):
                m.memory = None
