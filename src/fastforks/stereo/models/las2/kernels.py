"""LAS2 pieces on the `stereo_ops` kernels: the correlation cost volume and the attention modules' depthwise convs.

Both replace PyTorch ops that dominated the pass at KITTI size: the correlation volume was a pad + unfold + flip +
multiply + mean (four full-volume passes), the depthwise 1xK / Kx1 convolutions ran as cuDNN grouped convs with
NCHW <-> NHWC transforms around them. The replicate-padded feature output conv gets its padding as a separate op, so
the conv itself runs on the fp16-accumulate kernels (cuDNN's generic kernel for padding modes was ~0.2 ms per image).
"""

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from fastforks._core.patch import is_base, override
from fastforks.kernels import stereo_ops

from .net.aggregation_fasternet import AttentionModule

if TYPE_CHECKING:
    from fastforks._core.kernels import CubinModule
    from fastforks.shared.cnn_kernels import Patched


def corr_volume(k: CubinModule, left: torch.Tensor, right: torch.Tensor, depth: int) -> torch.Tensor:
    """`build_correlation_volume(left, right, depth)` (mean over channels) as one kernel, fp16 channels-last."""
    n, c, h, w = left.shape
    out = torch.empty((n, depth, h, w), device=left.device, dtype=torch.float16, memory_format=stereo_ops.CL)
    return stereo_ops.corr_volume(k, left, right, out, 1.0 / c)


def _line_weights(conv: torch.nn.Conv2d) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    kh, kw = conv.kernel_size
    if (
        conv.groups != conv.in_channels
        or conv.in_channels != conv.out_channels
        or 1 not in {kh, kw}
        or (kh * kw) % 2 == 0
        or conv.padding != (kh // 2, kw // 2)
    ):
        msg = f"not a depthwise line conv with 'same' padding: {conv}"
        raise RuntimeError(msg)
    w = conv.weight.detach().float().reshape(conv.out_channels, kh * kw).t().contiguous()  # [K, C]
    b = None if conv.bias is None else conv.bias.detach().float().contiguous()
    return w, b, 0 if kh == 1 else 1


def _attention_forward(self: Patched, cost: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Upstream's `conv3(attn + sum_i convi_2(convi_1(attn))) * cost`, the three branches on the line kernels."""
    k: CubinModule = self._ops
    attn = self.conv0(x).contiguous(memory_format=stereo_ops.CL)
    s = None
    for (wa, ba, aa), (wb, bb, ab) in self._lines:
        t = stereo_ops.dw_line(k, attn, wa, ba, aa)
        s = stereo_ops.dw_line(k, t, wb, bb, ab, res=attn if s is None else s, out=s)
    return self.conv3(s) * cost


def _pre_padded(conv: nn.Conv2d) -> nn.Module:
    """A replicate-padded conv as `ReplicationPad2d` + the same conv without padding."""
    ph, pw = cast("tuple[int, int]", conv.padding)  # replicate padding: always an int pair
    d = copy.deepcopy(conv)
    d.padding, d.padding_mode = (0, 0), "zeros"
    return nn.Sequential(nn.ReplicationPad2d((pw, pw, ph, ph)), d)


def _fuse_gelu_mlps(net: torch.nn.Module) -> None:
    """`conv -> BatchNorm -> GELU` (the FasterNet MLPs) as one `FusedConvBN` with the erf-GELU epilogue.

    The generic `fuse_sequentials` only fuses the activations every cnn_kernels user validated; LAS2's goldens cover
    GELU.
    """
    from fastforks.shared import cnn_kernels  # noqa: PLC0415 - optional kernel dependency, as in model.py

    for seq in [m for m in net.modules() if isinstance(m, nn.Sequential)]:
        children = list(seq._modules.items())  # noqa: SLF001 - in-place slot replacement
        for (a, conv), (b, bn), (c, act) in zip(children, children[1:], children[2:], strict=False):
            if (
                is_base(conv, nn.Conv2d)
                and conv.padding_mode == "zeros"
                and is_base(bn, nn.BatchNorm2d)
                and is_base(act, nn.GELU)
                and act.approximate == "none"
            ):
                seq._modules[a] = cnn_kernels.FusedConvBN(conv, bn, cnn_kernels.ACT["gelu"])  # noqa: SLF001
                seq._modules[b], seq._modules[c] = nn.Identity(), nn.Identity()  # noqa: SLF001


class _GlobalMax(nn.Module):
    """`AdaptiveMaxPool2d(1)` as one reduction (the pooling kernel was ~1.6 ms per call at 994x1482)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.amax(dim=(2, 3), keepdim=True)


class _ConvAct(nn.Module):
    """`act(conv(x))` as one `conv_ep` call: bias and activation in the conv's epilogue pass."""

    def __init__(self, conv: nn.Conv2d, act: int) -> None:
        super().__init__()
        self.conv = conv
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from fastforks.shared import cnn_kernels  # noqa: PLC0415 - optional kernel dependency, as in model.py

        return cnn_kernels.conv_ep(self.conv, x, act=self.act)


def _fuse_conv_acts(net: torch.nn.Module) -> None:
    """`conv -> ReLU / LeakyReLU(0.01) / sigmoid / tanh` runs in `Sequential`s (no norm between) as `_ConvAct`.

    Same math: the epilogue applies the activation to the fp32 conv + bias before the fp16 store.
    """
    from fastforks.shared import cnn_kernels  # noqa: PLC0415 - optional kernel dependency, as in model.py

    for seq in [m for m in net.modules() if isinstance(m, nn.Sequential)]:
        children = list(seq._modules.items())  # noqa: SLF001 - in-place slot replacement
        for (a, conv), (b, act) in itertools.pairwise(children):
            code = None if isinstance(act, nn.Identity) else cnn_kernels.act_code(act)
            if is_base(conv, nn.Conv2d) and conv.padding_mode == "zeros" and code:
                seq._modules[a] = _ConvAct(conv, code)  # noqa: SLF001
                seq._modules[b] = nn.Identity()  # noqa: SLF001


def prepare_update(block: Patched) -> None:
    """LAS2-H update block for `update_step` (call while fp32, before `prepare`).

    The z and r gate convs of each ConvGRU read the same input: one conv with both weight sets. The q convs take their
    input channels as `[x, r * h]` (the conv1 input buffer with its `h` part overwritten). The mask head's 0.25 scale
    moves into its last conv (0.25 * relu(y) = relu(0.25 * y)).
    """
    for gru in (block.gru04.small_gru, block.gru04.large_gru):
        z, r = gru.convz, gru.convr
        n = z.out_channels
        with torch.no_grad():
            w = gru.convq.weight
            w.copy_(torch.cat([w[:, n:], w[:, :n]], dim=1))
        zr = nn.Conv2d(z.in_channels, 2 * z.out_channels, z.kernel_size, padding=z.padding)
        with torch.no_grad():
            zr.weight.copy_(torch.cat([z.weight, r.weight]))
            cast("torch.Tensor", zr.bias).copy_(torch.cat([z.bias, r.bias]))
        gru.convzr = zr.to(z.weight.device)
        del gru.convz, gru.convr
    last = block.mask[2]
    with torch.no_grad():
        last.weight.mul_(0.25)
        last.bias.mul_(0.25)


def _gru(g: Patched, h: torch.Tensor, xh: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
    """`RaftConvGRU(h, x, hx)` with `xh = cat([x, h])`: sigmoid / tanh in the conv epilogues, the update as one lerp.

    Writes `r * h` over the `h` channels of `xh` (the q conv's input, weights reordered by `prepare_update`).
    """
    from fastforks.shared.cnn_kernels import ACT, conv_ep  # noqa: PLC0415 - optional kernel dependency

    n = h.shape[1]
    zr = conv_ep(g.convzr, hx, act=ACT["sigmoid"])
    torch.mul(zr[:, n:], h, out=xh[:, xh.shape[1] - n :])
    q = conv_ep(g.convq, xh, act=ACT["tanh"])
    return torch.lerp(h, q, zr[:, :n])  # (1 - z) * h + z * q


def update_step(  # noqa: PLR0913, PLR0917 - the update block's inputs
    block: Patched,
    h: torch.Tensor,
    inp: torch.Tensor,
    corr: torch.Tensor,
    disp: torch.Tensor,
    att: torch.Tensor,
    *,
    mask: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """`BasicSelectiveUpdateBlock` (one level) on the prepared block: new hidden state, mask features, disparity delta.

    ReLUs in the conv epilogues, one concatenation for the motion features, the selective mix as one lerp; the mask
    features only when `mask` (the pass needs them after the last iteration only).
    """
    from fastforks.shared.cnn_kernels import ACT, conv_ep  # noqa: PLC0415 - optional kernel dependency

    relu = ACT["relu"]
    e = block.encoder
    c = conv_ep(e.convc2, conv_ep(e.convc1, corr, act=relu), act=relu)
    d = conv_ep(e.convd2, conv_ep(e.convd1, disp, act=relu), act=relu)
    motion = torch.cat([inp, conv_ep(e.conv, torch.cat([c, d], dim=1), act=relu), disp], dim=1)
    g = block.gru04
    xh = torch.cat([g.conv0(motion), h], dim=1)
    hx = g.conv1(xh)
    large = _gru(g.large_gru, h, xh, hx)
    h = torch.lerp(large, _gru(g.small_gru, h, xh, hx), att)  # small * att + large * (1 - att)
    return h, block.mask(h) if mask else None, block.disp_head(h)


def prepare(net: torch.nn.Module, device: torch.device) -> CubinModule:
    """Load `stereo_ops`, route the `AttentionModule`s through it, split replicate padding off, fuse the GELU MLPs.

    Also keeps the image mean / std as float32 `[2, 3]` for `stereo_ops.norm_pad_pair`.

    Call while `net` is fp32.

    Raises:
        KernelUnavailableError: The library has no usable binary for `device`.
    """
    k = stereo_ops.load(device)
    m: Patched = net
    if hasattr(m, "image_mean"):  # the whole network (tests prepare single modules)
        stats = (m.image_mean, m.image_std)  # a plain attribute: the fp16 cast must not reach it
        m._norm_stats = torch.stack([t.detach().reshape(3) for t in stats]).float().to(device)  # noqa: SLF001
    _fuse_gelu_mlps(net)
    _fuse_conv_acts(net)
    for parent in list(net.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d) and child.padding_mode == "replicate":
                setattr(parent, name, _pre_padded(child))
            elif (
                isinstance(child, nn.AdaptiveMaxPool2d)
                and child.output_size in {1, (1, 1)}
                and not child.return_indices
            ):
                setattr(parent, name, _GlobalMax())
    for mod in net.modules():
        if not isinstance(mod, AttentionModule):
            continue
        m: Patched = mod  # carries our attributes and a replaced forward
        pairs = ((m.conv0_1, m.conv0_2), (m.conv1_1, m.conv1_2), (m.conv2_1, m.conv2_2))
        m._ops = k  # noqa: SLF001 - plain attributes: the fp16 cast must not reach the fp32 weights
        m._lines = [(_line_weights(a), _line_weights(b)) for a, b in pairs]  # noqa: SLF001
        override(m, _attention_forward)
    return k
