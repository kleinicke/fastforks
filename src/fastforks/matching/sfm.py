"""Structure-from-motion front end: an image folder -> a COLMAP database for `colmap mapper` / `glomap mapper`.

```python
from fastforks.matching import sfm

stats = sfm.colmap_database("scene/images", "scene/database.db")  # MegaLoc pairs, ALIKED + LightGlue, GPU RANSAC
# then: colmap mapper --database_path scene/database.db --image_path scene/images --output_path scene/sparse
```

Stages: retrieval (each image with its `k` most similar others, `retrieval.pairs`), sparse matching of those pairs
(`MatchingModel.match_pairs`: each image's keypoints are computed once), geometric verification (batched
`estimate_fundamental` on the GPU) and the database: cameras, images, keypoints, raw matches and the verified
two-view geometries (fundamental matrix + inlier matches), in the COLMAP 3 schema (COLMAP 4 reads it too).

A dense matcher (RoMa, LoFTR) gets keypoints the way hloc's `match_dense` does: per pair, the `dense_matches` most
certain pixels of a `DENSE_STRIDE` grid of the first image and where the warp puts them in the second; per image, the
endpoints of all its pairs merged within `DENSE_CELL` px cells (one keypoint per cell, at the mean) so matches of
different pairs share keypoints (tracks); one match per keypoint and pair (the most certain).

`reconstruct` goes on to the reconstruction with pycolmap's mapper (`pip install 'fastforks-core[sfm]'`).
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from PIL import Image

from fastforks._core.errors import InputError, MissingDependencyError, OptionsError, install_hint
from fastforks._core.image import read_images

if TYPE_CHECKING:
    import os
    from collections.abc import Generator, Iterable, Mapping, Sequence

    from fastforks._core.cards import Dtype
    from fastforks.matching._api import SparseMatchingResult

__all__ = ["DatabaseStats", "ReconstructionStats", "colmap_database", "reconstruct"]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
"""Files of the image folder that are used (case-insensitive, subfolders included)."""
MIN_INLIERS = 15  # COLMAP's `TwoViewGeometryOptions.min_num_inliers`: fewer and the pair is not verified
_VERIFY_BATCH = 64  # pairs per batched RANSAC call
_UNCALIBRATED = 3  # COLMAP `TwoViewGeometry::UNCALIBRATED` (2 is CALIBRATED): a fundamental matrix
_SIMPLE_RADIAL = 2  # COLMAP camera model id: f, cx, cy, k
_PAIR_FACTOR = 2147483647  # COLMAP `kMaxNumImages`: pair_id = id1 * it + id2
DENSE_STRIDE = 2  # dense matchers: grid of candidate pixels in the first image
DENSE_CELL = 1.0  # dense matchers: keypoints closer than this (px) merge into one (hloc `max_error` 1)
_DENSE_MIN_CERTAINTY = 0.1  # dense matchers: candidates below it are never used


@dataclass(frozen=True)
class DatabaseStats:
    """What `colmap_database` wrote and how long each stage took.

    Attributes:
        images: Images in the database.
        pairs: Image pairs matched.
        verified: Pairs with a verified fundamental matrix (at least `MIN_INLIERS` inliers).
        seconds: Wall time per stage: `retrieval`, `matching` (incl. decoding and verification), `database`.
        peak_mb: Peak allocated device memory per stage in MiB (CUDA only, else empty).
    """

    images: int
    pairs: int
    verified: int
    seconds: dict[str, float] = field(default_factory=dict)
    peak_mb: dict[str, float] = field(default_factory=dict)


def colmap_database(  # noqa: PLR0913 - the stages' options
    images: str | os.PathLike[str],
    database: str | os.PathLike[str],
    *,
    matcher: str = "aliked-lightglue",
    matcher_options: Mapping[str, Any] | None = None,
    retrieval: str | None = "megaloc",
    k: int = 20,
    threshold: float = 4.0,
    single_camera: bool = False,
    dense_matches: int = 4096,
    mode: Literal["auto", "portable", "kernels"] = "auto",
    device: str | torch.device | None = None,
    dtype: Dtype | torch.dtype | None = None,
) -> DatabaseStats:
    """Match an image folder and write a COLMAP database ready for the mapper.

    Args:
        images: Folder of the images (`IMAGE_SUFFIXES`, subfolders included); the database names them by their path
            relative to it, so pass the same folder as the mapper's `--image_path`.
        database: The COLMAP database file to write (must not exist).
        matcher: A matching model id (`matching.list_models()`): sparse (`aliked-lightglue`, `disk-lightglue`, ...)
            or dense (`romav2`, ...: keypoints from the warps, see the module docstring).
        matcher_options: Load options of the matcher (`options` of `load`), e.g. `{"long_side": 1600}`.
        retrieval: The retrieval model that picks the pairs (`retrieval.list_models()`); `None`: all pairs.
        k: Pairs per image (each image with its `k` most similar others); all pairs if `k >= images - 1`.
        threshold: Inlier threshold of the fundamental-matrix RANSAC in pixels (Sampson distance; 4 as COLMAP).
        single_camera: One camera (intrinsics) for all images of the same size; default one per image, as COLMAP.
        dense_matches: Dense matchers: matches sampled per pair (the most certain grid pixels of the first image).
        mode: Backend of the models and of the RANSAC.
        device: Of the models (default as `load`).
        dtype: Compute dtype of the matcher (as `load`; default the card's).

    Returns:
        The counts and the time per stage.

    Raises:
        InputError: No images, an unreadable image or an existing database.
    """
    from fastforks import matching  # noqa: PLC0415 - the task modules import this one lazily
    from fastforks import retrieval as retrieval_task  # noqa: PLC0415

    root = Path(images)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file())
    if not paths:
        msg = f"no images ({', '.join(IMAGE_SUFFIXES)}) in {root}"
        raise InputError(msg)
    if Path(database).exists():
        msg = f"database {database} exists; remove it or choose another file"
        raise InputError(msg)
    seconds: dict[str, float] = {}
    peak: dict[str, float] = {}
    n = len(paths)
    with _stage("retrieval", seconds, peak):
        if retrieval is None or k >= n - 1:
            pairs = torch.triu_indices(n, n, 1).T
        else:
            model = retrieval_task.load(retrieval, mode=mode, device=device)
            desc = torch.stack([model(image).descriptor for image in read_images(paths)])
            pairs = retrieval_task.pairs(desc, k).cpu()
            del model, desc
    pair_list = [(int(i), int(j)) for i, j in pairs]
    sizes = [_size(p) for p in paths]
    with _stage("matching", seconds, peak):
        model = matching.load(matcher, mode=mode, device=device, dtype=dtype, options=matcher_options)
        results = model.match_pairs(paths, pair_list)
        if model.sparse:
            keypoints, raw, verified = _sparse(pair_list, results, threshold, mode)
        else:
            keypoints, raw, verified = _dense(pair_list, results, sizes, dense_matches, (threshold, mode))
        del model
    with _stage("database", seconds, peak):
        _write(
            Path(database),
            [p.relative_to(root).as_posix() for p in paths],
            sizes,
            keypoints,
            raw,
            verified,
            single_camera=single_camera,
        )
    return DatabaseStats(n, len(pair_list), len(verified), seconds, peak)


@dataclass(frozen=True)
class ReconstructionStats:
    """What `reconstruct` built.

    Attributes:
        database: The `DatabaseStats` of the front end.
        mapper: `incremental` or `global`.
        seconds: Wall time of the mapper.
        models: Reconstructions the mapper produced (`<out>/sparse/0`, `/1`, ...; ideally one).
        registered: Images with a pose in the largest model.
        points: 3D points of the largest model.
        model: Folder of the largest model, `None` if the mapper produced none.
    """

    database: DatabaseStats
    mapper: str
    seconds: float
    models: int
    registered: int
    points: int
    model: Path | None


def reconstruct(
    images: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    mapper: Literal["incremental", "global"] = "incremental",
    **options: Any,  # noqa: ANN401 - `colmap_database`'s
) -> ReconstructionStats:
    """`colmap_database` into `<out>/database.db`, then pycolmap's mapper into `<out>/sparse` (COLMAP models).

    Args:
        images: Folder of the images (as `colmap_database`).
        out: Output folder (created; `database.db` must not exist in it).
        mapper: `incremental` (COLMAP's mapper) or `global` (GLOMAP, pycolmap >= 3.12).
        **options: Options of `colmap_database` (`matcher`, `retrieval`, `k`, `threshold`, ...).

    Returns:
        The front-end stats, the mapper time and what the largest model registered.

    Raises:
        MissingDependencyError: pycolmap is not installed (`pip install 'fastforks-core[sfm]'`).
        OptionsError: `mapper` is not `incremental` or `global`.
        InputError: As `colmap_database`.
    """
    try:
        import pycolmap  # noqa: PLC0415 - optional
    except ImportError as err:
        msg = f"sfm.reconstruct needs pycolmap: {install_hint('sfm')}"
        raise MissingDependencyError(msg) from err
    mappers = {"incremental": pycolmap.incremental_mapping, "global": getattr(pycolmap, "global_mapping", None)}
    if mappers.get(mapper) is None:
        msg = f"mapper must be one of {', '.join(mappers)} (global needs pycolmap >= 3.12), not {mapper!r}"
        raise OptionsError(msg)
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    database = folder / "database.db"
    stats = colmap_database(images, database, **options)
    sparse = folder / "sparse"
    sparse.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    models = mappers[mapper](database, Path(images), sparse)
    seconds = time.perf_counter() - t0
    if not models:
        return ReconstructionStats(stats, mapper, seconds, 0, 0, 0, None)
    best, rec = max(models.items(), key=lambda kv: kv[1].num_reg_images())
    registered, points = rec.num_reg_images(), rec.num_points3D()
    return ReconstructionStats(stats, mapper, seconds, len(models), registered, points, sparse / str(best))


Keypoints = dict[int, np.ndarray]
Raw = dict[tuple[int, int], np.ndarray]
Verified = dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
_Batch = list[tuple[tuple[int, int], torch.Tensor, torch.Tensor, torch.Tensor]]


def _sparse(
    pairs: list[tuple[int, int]], results: Iterable[Any], threshold: float, mode: str
) -> tuple[Keypoints, Raw, Verified]:
    """Keypoints, raw matches and verified geometries of a sparse matcher's `match_pairs` results."""
    keypoints: Keypoints = {}
    raw: Raw = {}
    verified: Verified = {}
    batch: _Batch = []
    for pair, result in zip(pairs, results, strict=True):
        assert isinstance(result, _sparse_type())  # noqa: S101 - `model.sparse`
        for image, kp in zip(pair, (result.keypoints1, result.keypoints2), strict=True):
            if image not in keypoints:
                keypoints[image] = kp.cpu().numpy()
        raw[pair] = result.pairs.cpu().numpy()
        if len(result.pairs) >= MIN_INLIERS:
            p1, p2 = result.keypoints1[result.pairs[:, 0]], result.keypoints2[result.pairs[:, 1]]
            batch.append((pair, p1, p2, result.pairs))
        if len(batch) == _VERIFY_BATCH:
            verified.update(_verify(batch, threshold, mode))
            batch = []
    verified.update(_verify(batch, threshold, mode))
    return keypoints, raw, verified


def _dense(
    pairs: list[tuple[int, int]],
    results: Iterable[Any],
    sizes: Sequence[tuple[int, int]],
    n: int,
    verify: tuple[float, str],
) -> tuple[Keypoints, Raw, Verified]:
    """Keypoints (merged match endpoints), raw matches and verified geometries from a dense matcher's warps.

    `sizes`: (w, h) per image; `n`: matches sampled per pair; `verify`: the RANSAC threshold and backend.
    """
    threshold, mode = verify
    sampled = []  # per pair: points in the first image, in the second, certainty (CPU)
    device = None
    for (_, j), result in zip(pairs, results, strict=True):
        p1, p2, conf = _dense_matches(result.warp, result.certainty, sizes[j], n)
        device = p1.device
        sampled.append((p1.cpu(), p2.cpu(), conf.cpu()))
    points: dict[int, list[torch.Tensor]] = {}
    for (i, j), (p1, p2, _) in zip(pairs, sampled, strict=True):
        points.setdefault(i, []).append(p1)
        points.setdefault(j, []).append(p2)
    keypoints: Keypoints = {}
    index: dict[int, list[torch.Tensor]] = {}  # per image: the keypoint index of each appended point list
    for image, lists in points.items():
        kp, idx = _merge(torch.cat(lists), DENSE_CELL)
        keypoints[image] = kp.numpy()
        index[image] = list(idx.split([len(t) for t in lists]))
    raw: Raw = {}
    verified: Verified = {}
    batch: _Batch = []
    for (i, j), (_, _, conf) in zip(pairs, sampled, strict=True):
        ids = torch.stack([index[i].pop(0), index[j].pop(0)], 1)
        ids = _one_per_keypoint(ids, conf)
        raw[i, j] = ids.numpy()
        if len(ids) >= MIN_INLIERS:
            kp1, kp2 = torch.from_numpy(keypoints[i]), torch.from_numpy(keypoints[j])
            batch.append(((i, j), kp1[ids[:, 0]].to(device), kp2[ids[:, 1]].to(device), ids))
        if len(batch) == _VERIFY_BATCH:
            verified.update(_verify(batch, threshold, mode))
            batch = []
    verified.update(_verify(batch, threshold, mode))
    return keypoints, raw, verified


def _dense_matches(
    warp: torch.Tensor, certainty: torch.Tensor | None, size2: tuple[int, int], n: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The `n` most certain pixels of the `DENSE_STRIDE` grid of the first image and where the warp puts them.

    Pixels below the minimum certainty or warped outside the second image (`size2`: w, h) are skipped.
    Returns `(x, y)` in the first image, in the second, and the certainty.
    """
    h, w = warp.shape[-2:]
    ys = torch.arange(0, h, DENSE_STRIDE, device=warp.device)
    xs = torch.arange(0, w, DENSE_STRIDE, device=warp.device)
    target = warp[:, ys][:, :, xs].reshape(2, -1).T  # [G, 2]
    conf = torch.ones(len(target), device=warp.device) if certainty is None else certainty[ys][:, xs].flatten()
    inside = (target >= 0).all(1) & (target[:, 0] <= size2[0] - 1) & (target[:, 1] <= size2[1] - 1)
    conf = conf.masked_fill(~inside | (conf < _DENSE_MIN_CERTAINTY), -1.0)
    conf, pick = conf.topk(min(n, len(conf)))
    keep = conf >= 0
    pick, conf = pick[keep], conf[keep]
    source = torch.stack([xs[pick % len(xs)], ys[pick // len(xs)]], 1).to(target.dtype)
    return source, target[pick], conf


def _merge(points: torch.Tensor, cell: float) -> tuple[torch.Tensor, torch.Tensor]:
    """One keypoint per `cell` px cell holding points (their mean): keypoints `[K, 2]`, the keypoint of each point."""
    q = torch.floor(points / cell).long()
    key = q[:, 1] * (int(q[:, 0].max()) + 2) + q[:, 0]
    _, index, counts = key.unique(return_inverse=True, return_counts=True)
    keypoints = torch.zeros(len(counts), 2).index_add_(0, index, points) / counts[:, None]
    return keypoints, index


def _one_per_keypoint(ids: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    """`ids` `[M, 2]` keypoint index pairs, at most one per first and per second keypoint: the most certain."""
    order = conf.argsort(descending=True)
    ids = ids[order]
    keep = torch.ones(len(ids), dtype=torch.bool)
    for col in (0, 1):
        _, inv = ids[:, col].unique(return_inverse=True)
        first = torch.full((int(inv.max()) + 1,), len(ids)).scatter_reduce_(0, inv, torch.arange(len(ids)), "amin")
        keep &= first[inv] == torch.arange(len(ids))
    return ids[keep]


def _size(path: Path) -> tuple[int, int]:
    """Width and height from the file header."""
    with Image.open(path) as image:
        return image.size


def _sparse_type() -> type[SparseMatchingResult]:
    from fastforks.matching._api import SparseMatchingResult  # noqa: PLC0415

    return SparseMatchingResult


@contextmanager
def _stage(name: str, seconds: dict[str, float], peak: dict[str, float]) -> Generator[None]:
    """Time a stage (with a device sync) and record its peak allocated CUDA memory."""
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    yield
    if cuda:
        torch.cuda.synchronize()
        peak[name] = torch.cuda.max_memory_allocated() / 2**20
    seconds[name] = time.perf_counter() - t0


def _verify(
    batch: Sequence[tuple[tuple[int, int], torch.Tensor, torch.Tensor, torch.Tensor]], threshold: float, mode: str
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
    """One batched fundamental-matrix RANSAC over the pairs: `{pair: (F, inlier index pairs)}` of the verified ones."""
    from fastforks.matching.relative_pose import estimate_fundamental  # noqa: PLC0415

    if not batch:
        return {}
    m = max(len(p1) for _, p1, _, _ in batch)
    device = batch[0][1].device
    p1 = torch.zeros(len(batch), m, 2, device=device)
    p2 = torch.zeros_like(p1)
    mask = torch.zeros(len(batch), m, dtype=torch.bool, device=device)
    for b, (_, a1, a2, _) in enumerate(batch):
        p1[b, : len(a1)], p2[b, : len(a2)], mask[b, : len(a1)] = a1, a2, True
    fm = estimate_fundamental(p1, p2, threshold=threshold, mask=mask, mode=mode)  # ty: ignore[invalid-argument-type]
    inliers, ok, f = (fm.inliers & mask).cpu(), fm.ok.cpu(), fm.F.double().cpu().numpy()
    out = {}
    for b, (pair, _, _, index) in enumerate(batch):
        keep = inliers[b, : len(index)]
        if ok[b] and int(keep.sum()) >= MIN_INLIERS:
            out[pair] = f[b], index.cpu()[keep].numpy()
    return out


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cameras (camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, model INTEGER NOT NULL,
    width INTEGER NOT NULL, height INTEGER NOT NULL, params BLOB, prior_focal_length INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS images (image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL, prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL, prior_tx REAL,
    prior_ty REAL, prior_tz REAL, CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < 2147483647),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id));
CREATE TABLE IF NOT EXISTS keypoints (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
    cols INTEGER NOT NULL, data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS descriptors (image_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
    cols INTEGER NOT NULL, data BLOB, FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS matches (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
    cols INTEGER NOT NULL, data BLOB);
CREATE TABLE IF NOT EXISTS two_view_geometries (pair_id INTEGER PRIMARY KEY NOT NULL, rows INTEGER NOT NULL,
    cols INTEGER NOT NULL, data BLOB, config INTEGER NOT NULL, F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB);
CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);
"""


def _write(  # noqa: PLR0913, PLR0917 - the database contents
    path: Path,
    names: Sequence[str],
    sizes: Sequence[tuple[int, int]],
    keypoints: dict[int, np.ndarray],
    raw: dict[tuple[int, int], np.ndarray],
    verified: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
    *,
    single_camera: bool,
) -> None:
    """The COLMAP database (image ids = index + 1, keypoints with COLMAP's pixel-corner origin)."""
    con = sqlite3.connect(path)
    try:
        con.executescript(_SCHEMA)
        cameras: dict[object, int] = {}
        for i, (name, (w, h)) in enumerate(zip(names, sizes, strict=True)):
            key = (w, h) if single_camera else i
            if key not in cameras:
                params = np.array([1.2 * max(w, h), w / 2, h / 2, 0.0])  # COLMAP's default focal guess
                cur = con.execute(
                    "INSERT INTO cameras VALUES (NULL, ?, ?, ?, ?, 0)", (_SIMPLE_RADIAL, w, h, params.tobytes())
                )
                cameras[key] = int(cur.lastrowid or 0)
            con.execute("INSERT INTO images (image_id, name, camera_id) VALUES (?, ?, ?)", (i + 1, name, cameras[key]))
            kp = keypoints.get(i, np.zeros((0, 2), np.float32)).astype(np.float32) + 0.5
            con.execute("INSERT INTO keypoints VALUES (?, ?, 2, ?)", (i + 1, len(kp), kp.tobytes()))
        eye, zero = np.eye(3).tobytes(), np.zeros(3).tobytes()
        shift = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]])  # our pixel -> COLMAP's: x = shift @ x_colmap
        for (i, j), index in raw.items():
            pair_id = (i + 1) * _PAIR_FACTOR + j + 1  # i < j
            con.execute(
                "INSERT INTO matches VALUES (?, ?, 2, ?)", (pair_id, len(index), index.astype(np.uint32).tobytes())
            )
            if (i, j) in verified:
                fm, inl = verified[i, j]
                con.execute(
                    "INSERT INTO two_view_geometries VALUES (?, ?, 2, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        pair_id,
                        len(inl),
                        inl.astype(np.uint32).tobytes(),
                        _UNCALIBRATED,
                        (shift.T @ fm @ shift).tobytes(),
                        eye,
                        eye,
                        np.array([1.0, 0, 0, 0]).tobytes(),
                        zero,
                    ),
                )
        con.commit()
    finally:
        con.close()
