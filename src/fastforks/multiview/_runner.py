"""The interface a multi-view depth family implements: one `Runner` per backend, listed in `model.py` as `RUNNERS`.

```python
class Portable(Runner):
    dtypes = ("float32", "float16")
    Params = Params
    intrinsics = "required"  # calibrated MVS: K and cam_to_world are needed, depth_range is optional
    cam_to_world = "required"
    depth_range = "optional"
    per_view = False  # predicts the reference view (images[0]) only
    min_views = 2

    def __call__(self, images, intrinsics, cam_to_world, depth_range):
        x = scale(images, self.ctx.io)  # family-specific resizing / normalisation
        return {"depth": upsample(self.net(x, ...), images.shape[-2:])}  # [1, H, W]: the reference view
```

The core converts the images (one scene: `N` views of the same size), checks the view count and the cameras the
runner declares, moves the reference view to the front, loops over several references (reference-only runners),
restores the input order of per-view outputs, checks output names and shapes, casts, passes given cameras through and
records the backend. Runners resize themselves (`fastforks._core.preprocess` has the shared helpers) and return maps
at the input size.

Cameras: `intrinsics` are OpenCV pixel matrices of the input images (pixel centres at integer coordinates);
`cam_to_world` are camera-to-world matrices with OpenCV camera axes (x right, y down, z forward). Runners that estimate
cameras return them in the frame of `images[0]` (the reference view).

Portable runners also implement `module()`: the plain network with the same inputs (`None` for unused cameras)
returning the same dict, for export; kernels runners may omit it.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch

from fastforks._core import runner as core
from fastforks._core.preprocess import scale
from fastforks._core.runner import Context, Forward, NoParams

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["Context", "Forward", "NoParams", "Runner", "Use", "invert", "scale", "take"]

Use = Literal["unused", "optional", "required"]
"""Whether a runner ignores an input (passing it is an error), uses it if given, or needs it."""


class Runner(core.Runner):
    """One backend of a multi-view depth family.

    Class attributes (besides those of `fastforks._core.runner.Runner`; `batch` is unused, a call is one scene):
        intrinsics: Use of the camera intrinsics.
        cam_to_world: Use of the camera poses.
        depth_range: Use of a `(near, far)` depth search range.
        per_view: `True`: predicts every view; `False`: the reference view (`images[0]`) only.
        min_views: Fewest views per call.
        max_views: Most views per call (`None`: no limit).
    """

    intrinsics: ClassVar[Use] = "unused"
    cam_to_world: ClassVar[Use] = "unused"
    depth_range: ClassVar[Use] = "unused"
    per_view: ClassVar[bool] = True
    min_views: ClassVar[int] = 1
    max_views: ClassVar[int | None] = None

    @classmethod
    def input_use(cls, params: Any) -> dict[str, Any]:  # noqa: ANN401, ARG003 - the family's `Params`
        """The class attributes: camera use, `per_view`, `min_views`, `max_views`.

        Args:
            params: The variant's `Params`.
        """
        names = ("intrinsics", "cam_to_world", "depth_range", "per_view", "min_views", "max_views")
        return {k: getattr(cls, k) for k in names}

    @abc.abstractmethod
    def __call__(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Run one scene.

        Args:
            images: `[N, 3, H, W]` float32 RGB in `[0, 1]` on `ctx.device`, the reference view first.
            intrinsics: `[N, 3, 3]` float32 pixel matrices of `images`, or `None`.
            cam_to_world: `[N, 4, 4]` camera-to-world matrices (float32; float64 if given as float64, on MPS float32
                relative to the first camera), or `None`.
            depth_range: `[2]` float32 `(near, far)` in the units of `cam_to_world`, or `None`.

        Returns:
            Maps at the input size with leading dim `V` (`N` if `per_view`, else 1), among the card's `outputs`:
            `"depth"`, `"confidence"` (`[V, H, W]`), `"mask"` (`[V, H, W]` bool), `"world_points"` (`[V, H, W, 3]`,
            world frame), `"intrinsics"` (`[V, 3, 3]`), `"cam_to_world"` (`[V, 4, 4]`) and extras (`[V, ...]`).
        """

    def scene(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor | None,
        cam_to_world: torch.Tensor | None,
        depth_range: torch.Tensor | None,
    ) -> Callable[[torch.Tensor | None], dict[str, torch.Tensor]]:
        """Internal: one scene in input order -> its prediction per view order (`None`: the input order).

        The core calls `scene` once per call and the returned function once per reference view with `order`, the view
        indices with the reference first; it returns what `self(*inputs[order])` returns. Runners that predict several
        references override it to do per-image work (features of every view) once per scene.
        """

        def predict(order: torch.Tensor | None) -> dict[str, torch.Tensor]:
            if order is None:
                return self(images, intrinsics, cam_to_world, depth_range)
            k, c2w = (None if t is None else t[order] for t in (intrinsics, cam_to_world))
            return self(images[order], k, c2w, depth_range)

        return predict


def take(t: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
    """`t[order]` (views, dim 0) in the memory format of `t`: channels-last stays channels-last."""
    out = t[order]
    if t.dim() == 4 and not t.is_contiguous() and t.is_contiguous(memory_format=torch.channels_last):  # noqa: PLR2004
        return out.contiguous(memory_format=torch.channels_last)
    return out


def _cross(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cross product over the last dim (written out: `aten.linalg_cross` has no ONNX conversion)."""
    a0, a1, a2 = a.unbind(-1)
    b0, b1, b2 = b.unbind(-1)
    return torch.stack((a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0), -1)


def f64(t: torch.Tensor) -> torch.Tensor:
    """`t` in float64 for composing poses; MPS has no float64 (its poses arrive with the common origin removed)."""
    return t if t.device.type == "mps" else t.double()


def invert(m: torch.Tensor) -> torch.Tensor:
    """Inverse of `[..., 3, 3]` matrices, or of `[..., 4, 4]` affine ones (last row `0 0 0 1`), in float32 (or float64).

    Closed forms (adjugate / determinant; `[R^-1, -R^-1 t]`) instead of `torch.linalg.inv`, which has no ONNX
    conversion; they agree with it to float32 rounding for the well-conditioned camera matrices of this task.
    """
    m = m if m.dtype == torch.float64 else m.float()
    if m.shape[-1] == 4:  # noqa: PLR2004 - affine 4 x 4
        r = invert(m[..., :3, :3])
        return torch.cat((torch.cat((r, -r @ m[..., :3, 3:]), -1), m[..., 3:, :]), -2)
    a, b, c = m.unbind(-2)  # rows
    bc = _cross(b, c)
    adj = torch.stack((bc, _cross(c, a), _cross(a, b)), -1)  # columns: M @ (b x c) = (det, 0, 0)
    return adj / (a * bc).sum(-1)[..., None, None]
