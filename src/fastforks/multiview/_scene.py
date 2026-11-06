"""`MultiViewResult.save_scene`: the result's cameras, images and points as a COLMAP model or a nerfstudio scene."""

from __future__ import annotations

import json
import posixpath
from pathlib import Path, PurePath
from typing import TYPE_CHECKING, Literal

import numpy as np

from fastforks._core.errors import InputError, OptionsError
from fastforks._core.outputs import _array

if TYPE_CHECKING:
    import os
    from collections.abc import Sequence

    from fastforks.multiview._api import MultiViewResult

SceneFormat = Literal["colmap", "nerfstudio"]
_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV <-> OpenGL camera axes


def save_scene(
    result: MultiViewResult,
    folder: str | os.PathLike[str],
    format: SceneFormat,  # noqa: A002 - as `read_calib`
    image_names: Sequence[str | os.PathLike[str]] | None,
    max_points: int | None,
) -> None:
    """See `MultiViewResult.save_scene`."""
    if format not in ("colmap", "nerfstudio"):
        msg = f"format must be 'colmap' or 'nerfstudio', got {format!r}"
        raise OptionsError(msg)
    if result.intrinsics is None or result.cam_to_world is None:
        msg = f"{result.model} returned no intrinsics / cam_to_world: a scene needs both (pass them to the model call)"
        raise InputError(msg)
    if max_points is not None and (isinstance(max_points, bool) or not isinstance(max_points, int) or max_points < 1):
        msg = f"max_points must be a positive int or None, got {max_points!r}"
        raise InputError(msg)
    from fastforks.stereo.geometry.ply import _finite, write_ply  # noqa: PLC0415 - as `_save_cloud`

    image = result.__dict__.get("_image")  # the input images of the views, as for cloud colours
    names = _names(result.views, image is not None, image_names)
    points, units = result._cloud_points(slice(None))  # noqa: SLF001 - the class's own helper
    xyz, rgb = _finite(points, None if image is None else image.movedim(-3, -1), dtype=np.float64)
    if rgb is None:
        rgb = np.full_like(xyz, 128, np.uint8)
    if max_points is not None and len(xyz) > max_points:
        keep = np.sort(np.random.default_rng(0).choice(len(xyz), max_points, replace=False))
        xyz, rgb = xyz[keep], rgb[keep]

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if image_names is None and image is not None:
        from PIL import Image  # noqa: PLC0415 - only for writing images

        (folder / "images").mkdir(exist_ok=True)
        for name, px in zip(names, _array((image.clamp(0, 1) * 255).round().byte().movedim(-3, -1)), strict=True):
            Image.fromarray(px).save(folder / name)
    k = _array(result.intrinsics).astype(np.float64)
    c2w = _array(result.cam_to_world).astype(np.float64)
    size = (points.shape[2], points.shape[1])
    if format == "nerfstudio":
        write_ply(folder / "sparse_pc.ply", xyz, rgb, frame="world", units=units)
        _nerfstudio(folder, names, k, c2w, size)
    else:
        _colmap(folder / "sparse" / "0", names, k, c2w, size, xyz, rgb)


def _names(views: tuple[int, ...], images: bool, given: Sequence[str | os.PathLike[str]] | None) -> list[str]:  # noqa: FBT001
    """The image file of each view, relative to the scene folder."""
    if given is None:
        if not images:
            msg = "the result holds no input images to write: pass image_names= (the image files, relative to folder)"
            raise InputError(msg)
        return [f"images/frame_{i + 1:05d}.png" for i in views]
    names = [PurePath(n).as_posix() for n in given]
    if len(names) != len(views) or any(PurePath(n).is_absolute() for n in names):
        msg = f"image_names must be {len(views)} paths relative to folder (one per view), got {names}"
        raise InputError(msg)
    return names


def _nerfstudio(folder: Path, names: list[str], k: np.ndarray, c2w: np.ndarray, size: tuple[int, int]) -> None:
    frames = [
        {"file_path": name, "transform_matrix": (c2w[i] @ _OPENGL).tolist(), "fl_x": k[i, 0, 0], "fl_y": k[i, 1, 1],
         "cx": k[i, 0, 2] + 0.5, "cy": k[i, 1, 2] + 0.5, "w": size[0], "h": size[1]}
        for i, name in enumerate(names)
    ]  # fmt: skip  # nerfstudio puts pixel centres at +0.5, as COLMAP
    scene = {"camera_model": "PINHOLE", "frames": frames, "ply_file_path": "sparse_pc.ply"}
    (folder / "transforms.json").write_text(json.dumps(scene, indent=2) + "\n")


def _colmap(  # noqa: PLR0913, PLR0917
    sparse: Path, names: list[str], k: np.ndarray, c2w: np.ndarray, size: tuple[int, int], xyz: np.ndarray,
    rgb: np.ndarray,
) -> None:  # fmt: skip
    cameras: dict[tuple[float, ...], int] = {}  # identical intrinsics share one camera
    images = []
    for i, name in enumerate(names):
        params = (k[i, 0, 0], k[i, 1, 1], k[i, 0, 2] + 0.5, k[i, 1, 2] + 0.5)  # COLMAP pixel centres at +0.5
        cid = cameras.setdefault(params, len(cameras) + 1)
        r = c2w[i, :3, :3].T  # world-to-camera
        pose = (*_quaternion(r), *(-r @ c2w[i, :3, 3]))
        images.append(f"{i + 1} {_num(pose)} {cid} {posixpath.relpath(name, 'images')}\n")  # COLMAP: under images/
    sparse.mkdir(parents=True, exist_ok=True)
    _write(sparse / "cameras.txt", "CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]",
           [f"{cid} PINHOLE {size[0]} {size[1]} {_num(p)}\n" for p, cid in cameras.items()])  # fmt: skip
    _write(sparse / "images.txt", "IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[]", images, "\n")
    pts = [f"{j + 1} {_num(p)} {c[0]} {c[1]} {c[2]} 0\n" for j, (p, c) in enumerate(zip(xyz, rgb, strict=True))]
    _write(sparse / "points3D.txt", "POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]", pts)


def _num(values: Sequence[float] | np.ndarray) -> str:
    return " ".join(f"{float(x):.17g}" for x in values)  # COLMAP's own precision


def _write(path: Path, columns: str, lines: list[str], after: str = "") -> None:
    """A COLMAP text file: the header comments, then each line (followed by `after`: an empty 2D-point line)."""
    head = f"# written by fastforks\n#   {columns}\n# Number of entries: {len(lines)}\n"
    path.write_text(head + "".join(line + after for line in lines))


def _quaternion(r: np.ndarray) -> np.ndarray:
    """`(qw, qx, qy, qz)` with `qw >= 0` of a rotation matrix (the nearest one, Bar-Itzhack's eigenvector method)."""
    (xx, xy, xz), (yx, yy, yz), (zx, zy, zz) = r
    m = np.array([
        [xx - yy - zz, yx + xy, zx + xz, zy - yz],
        [yx + xy, yy - xx - zz, zy + yz, xz - zx],
        [zx + xz, zy + yz, zz - xx - yy, yx - xy],
        [zy - yz, xz - zx, yx - xy, xx + yy + zz],
    ]) / 3  # fmt: skip
    q = np.linalg.eigh(m)[1][[3, 0, 1, 2], -1]
    return q if q[0] >= 0 else -q
