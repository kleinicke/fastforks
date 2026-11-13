"""3D Gaussians predicted by feed-forward splatting models (`MultiViewResult.gaussians`) and their PLY export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from fastforks._core.errors import InputError

if TYPE_CHECKING:
    import os
    from collections.abc import Mapping

__all__ = ["FIELDS", "PREFIX", "Gaussians", "pop_gaussians"]

PREFIX = "gaussians/"
"""Runner output keys `gaussians/<field>` (leading dim `G`, not `V`) are collected into a `Gaussians`."""
FIELDS = ("means", "scales", "rotations", "harmonics", "opacities")


@dataclass(frozen=True)
class Gaussians:
    """A 3D Gaussian splatting scene: `G` Gaussians in the result's world frame (OpenCV axes), on the model's device.

    Attributes:
        means: Centres `[G, 3]` float32, in the units of the result's `depth` / `cam_to_world`.
        scales: Standard deviations along the principal axes `[G, 3]` float32 (positive, not log).
        rotations: Unit quaternions `[G, 4]` float32, `(w, x, y, z)` (the 3DGS / gsplat order).
        harmonics: Spherical-harmonics colour coefficients `[G, 3, K]` float32 (`K = (degree + 1) ** 2`, RGB first,
            the 3DGS convention: colour = `0.5 + SH(direction)`).
        opacities: Opacities `[G]` float32 in `[0, 1]` (not logit).
    """

    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    harmonics: torch.Tensor
    opacities: torch.Tensor

    def __len__(self) -> int:
        """Number of Gaussians `G`."""
        return self.means.shape[0]

    @property
    def sh_degree(self) -> int:
        """Spherical-harmonics degree of `harmonics`."""
        return round(self.harmonics.shape[-1] ** 0.5) - 1

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write a binary little-endian `.ply` in the standard 3DGS layout (3DGS viewers, gsplat, SuperSplat).

        Per vertex: `x y z nx ny nz` (normals 0), `f_dc_0..2`, `f_rest_*` (channel-major: all coefficients of red,
        then green, then blue), `opacity` as a logit, `scale_0..2` as logs, `rot_0..3` as `w x y z`.

        Raises:
            InputError: The suffix is not `.ply`.
        """
        p = Path(path)
        if p.suffix.lower() != ".ply":
            msg = f"Gaussians are written as .ply, got {p.name!r}"
            raise InputError(msg)
        g = len(self)
        sh = self.harmonics.detach().float().cpu()
        rest = sh[..., 1:].flatten(1)
        eps = torch.finfo(torch.float32).eps
        op = self.opacities.detach().float().cpu().clamp(eps, 1 - eps)
        cols = [
            self.means.detach().float().cpu(),
            torch.zeros(g, 3),
            sh[..., 0],
            rest,
            torch.logit(op)[:, None],
            self.scales.detach().float().cpu().clamp_min(1e-30).log(),
            self.rotations.detach().float().cpu(),
        ]
        names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
        names += [f"f_rest_{i}" for i in range(rest.shape[1])]
        names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        data = torch.cat(cols, 1).numpy().astype("<f4")
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {g}"]
        header += [f"property float {n}" for n in names] + ["end_header", ""]
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            f.write("\n".join(header).encode("ascii"))
            f.write(np.ascontiguousarray(data).tobytes())


def pop_gaussians(maps: dict[str, torch.Tensor]) -> Gaussians | None:
    """Remove the `gaussians/<field>` entries from a runner's output and build `Gaussians` (`None` if there are none).

    Raises:
        RuntimeError: Some but not all fields, or inconsistent shapes (a runner bug).
    """
    got: Mapping[str, torch.Tensor] = {k[len(PREFIX) :]: maps.pop(k) for k in [k for k in maps if k.startswith(PREFIX)]}
    if not got:
        return None
    g = got["means"].shape[0] if "means" in got else -1
    shapes = {"means": (g, 3), "scales": (g, 3), "rotations": (g, 4), "opacities": (g,)}
    if set(got) != set(FIELDS) or any(tuple(got[k].shape) != s for k, s in shapes.items()):
        found = {k: tuple(v.shape) for k, v in got.items()}
        msg = f"runner gaussians: expected {FIELDS} with leading dim G, got {found}"
        raise RuntimeError(msg)
    h = got["harmonics"]
    if h.dim() != 3 or h.shape[:2] != (g, 3):  # noqa: PLR2004 - [G, 3, K]
        msg = f"runner gaussians: harmonics must be [G, 3, K], got {tuple(h.shape)}"
        raise RuntimeError(msg)
    return Gaussians(**{k: got[k].float() for k in FIELDS})


torch.serialization.add_safe_globals([Gaussians])  # `torch.load(..., weights_only=True)` of a saved result
