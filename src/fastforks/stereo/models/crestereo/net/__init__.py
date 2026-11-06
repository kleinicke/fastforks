"""CREStereo network, translated from the MegEngine original (megvii-research/CREStereo `nets/`, Apache-2.0).

Each file keeps upstream's name, classes, functions, variable names and statement order; only the framework calls
change. The upstream package `__init__` (`from .crestereo import CREStereo as Model`) is kept below. Translations:

- `megengine.module as M` -> `torch.nn as nn`; `megengine.functional as F` -> `torch` / `torch.nn.functional`;
  `M.Module`, `M.Conv2d`, `M.Linear`, `M.ReLU`, `M.Sequential`, `M.LayerNorm`, `M.GroupNorm`, `M.BatchNorm2d` -> the
  `nn` classes of the same name (same parameter names and shapes, except MegEngine's conv bias `[1, C, 1, 1]`, which
  the loader in `model.py` reshapes to `[C]`). `M.InstanceNorm(c, affine=False)` -> `nn.InstanceNorm2d(c,
  affine=False)` (eps 1e-5, biased variance). `M.Dropout(drop_prob=p)` -> `nn.Dropout(p=p)`.
- `F.concat(..., axis)` -> `torch.cat(..., dim)`; `F.sum` / `F.mean` / `F.softmax` / `F.cumsum` with `axis`,
  `keepdims` -> `dim`, `keepdim`; `F.transpose(x, perm)` -> `x.permute(perm)`; `F.reshape` -> `torch.reshape`;
  `F.expand_dims(x, a)` -> `torch.unsqueeze` (one axis) or `None` indexing (several axes); `F.sigmoid`, `F.tanh`,
  `F.exp`, `F.sin`, `F.cos`, `F.zeros`, `F.ones`, `F.arange`, `F.stack` -> the `torch` functions;
  `F.repeat(x, n, axis)` -> `torch.repeat_interleave(x, n, dim)`; `F.maximum(0, x)` / `F.minimum(0, x)` ->
  `torch.clamp(x, min=0)` / `torch.clamp(x, max=0)`; `x.astype("float32")` -> `x.float()`;
  `mge.tensor(a, ...)` -> `torch.tensor(a, ...)`; `F.logical_not` -> `torch.logical_not`; `p.ndim` -> `p.dim()`.
- `F.split(x, [k], axis)` (split before index k) -> `torch.split(x, [k, rest], dim)`; `F.split(x, n, axis)` (n equal
  parts) -> `torch.chunk(x, n, dim)`.
- `F.pad(x, pad_witdth=((0, 0), (0, 0), (py, py), (px, px)), mode="replicate")` -> `F.pad(x, (px, px, py, py),
  mode="replicate")`.
- `F.sliding_window(x, kernel_size=(H, W), stride=(sy, sx))` (output `[N, C, OH, OW, H, W]`) in `AGCL.get_correlation`
  -> `x.unfold(2, H, sy).unfold(3, W, sx)` (same layout). In `CREStereo.unfold`, `F.sliding_window` followed by the
  reshape / transpose / reshape to `[N, C * k1 * k2, L]` -> `F.unfold` (which yields that layout directly).
- `F.remap(img, coords, border_mode="constant")` (pixel coordinates, bilinear, zero outside) in `bilinear_sampler`
  -> `F.grid_sample(img, coords normalised to [-1, 1], padding_mode="zeros", align_corners=True)`.
- `F.nn.interpolate(..., mode="bilinear", align_corners=True)` -> `F.interpolate` (same arguments);
  `F.avg_pool2d` -> `F.avg_pool2d` (no padding, so MegEngine's count mode does not matter).
- `amp.autocast(enabled=self.mixed_precision)` -> `torch.autocast(device_type, enabled=self.mixed_precision)`.
- `M.init.msra_normal_(w, mode="fan_out", nonlinearity="relu")` -> `nn.init.kaiming_normal_` (same arguments);
  `M.init.fill_` -> `nn.init.constant_`; `M.init.xavier_uniform_` -> `nn.init.xavier_uniform_`.
- `LocalFeatureTransformer.layers`: a Python list of modules (MegEngine registers those) -> `nn.ModuleList` (same
  parameter names, `layers.0.*`).
- `FullAttention` masks: `np.bool_` dtype checks -> `torch.bool`.

Kept as upstream, quirks included: `PositionEncodingSine`'s `div_term` exponent `-math.log(10000.0) / d_model // 2`
(= -1.0), the positions starting at 1, and `AGCL.get_correlation`'s `reshape(1, -1, H, W)`, which only supports
batch 1 (the forward passes in `portable.py` and `max/` have their own correlation). Every modified file carries a notice naming its original.
"""

from .crestereo import CREStereo as Model
