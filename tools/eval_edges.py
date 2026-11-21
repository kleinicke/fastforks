"""Score saved boundary maps against the BSDS500 test annotations without using CUDA.

    python tools/eval_edges.py --bsr ~/data/BSR --work out/hed.kernels --workers 4 > hed.jsonl

The work directory holds one native-size, 8-bit `.npy` confidence map per test image and an `inference.json`
record with the model, mode, dtype, and sample count. Use all 200 images or the first 75 by filename. The
Berkeley matcher is built from the supplied BSDS500 archive with GNU C++.
Pass the same `--cache` directory across modes to reuse counts for byte-identical predictions and labels.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shlex
import subprocess
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

SOURCE = ("csa.cc", "kofn.cc", "match.cc", "Exception.cc", "Matrix.cc", "Random.cc", "String.cc", "Timer.cc")
FULL_SPLIT, FIRST_SUBSET = 200, 75
MAP_DIMENSIONS = 2
CUTOFFS = np.searchsorted(np.arange(256, dtype=np.float64) / 255, np.arange(1, 100, dtype=np.float64) / 100)
PROTOCOL = "BSDS500 official correspondence, tolerance 0.0075 of diagonal, no thinning, 99 thresholds, 8-bit scores"


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _build(source: Path, library: Path) -> None:
    command = [
        *shlex.split(os.environ.get("CXX", "g++")),
        "-O3",
        "-std=gnu++11",
        "-fpermissive",  # the Berkeley archive declares default arguments on friend declarations
        "-DNOBLAS",
        "-fPIC",
        "-shared",
        "-I",
        str(source),
        str(Path(__file__).with_name("bsds_correspond.cc")),
        *(str(source / name) for name in SOURCE),
        "-o",
        str(library),
    ]
    subprocess.run(command, check=True)


def _matcher(library: Path):  # noqa: ANN202 - ctypes function pointer
    match = ctypes.CDLL(str(library)).bsds_correspond
    match.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        *(np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags="F_CONTIGUOUS"),) * 2,
        *(np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="F_CONTIGUOUS"),) * 2,
    ]
    match.restype = ctypes.c_int
    return match


def _save_counts(path: Path, values: NDArray[np.int64]) -> None:
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            np.save(stream, values, allow_pickle=False)
            stream.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def _counts_at(
    confidence: NDArray[np.uint8],
    threshold: int,
    annotations: list[NDArray[np.float64]],
    match,  # noqa: ANN001 - ctypes function pointer
) -> tuple[int, int]:
    binary = np.asfortranarray(confidence >= threshold, dtype=np.float64)
    pred_match = np.empty(binary.shape, dtype=np.uint8, order="F")
    ann_match = np.empty_like(pred_match, order="F")
    accumulated = np.zeros_like(pred_match, order="F")
    recall = 0
    for annotation in annotations:
        if match(*binary.shape, binary, annotation, pred_match, ann_match):
            msg = "The BSDS500 matcher failed"
            raise RuntimeError(msg)
        recall += int(ann_match.sum())
        accumulated |= pred_match
    return recall, int(accumulated.sum())


def score_image(paths: tuple[Path, Path, Path, Path], scorer_id: str) -> NDArray[np.int64]:
    """Return matched reference, reference total, matched prediction and prediction total per threshold."""
    prediction, labels, cache, library = paths
    key = hashlib.sha256(f"{_sha(prediction)}:{_sha(labels)}:{scorer_id}".encode()).hexdigest()
    target = cache / f"{key}.npy"
    progress = cache / f"{key}.partial.npy"
    if target.is_file():
        return np.load(target, allow_pickle=False)

    from scipy.io import loadmat  # noqa: PLC0415 - needed only when the counts are missing

    confidence = np.load(prediction, allow_pickle=False)
    gt = loadmat(labels, simplify_cells=True)["groundTruth"]
    annotations = [
        np.asfortranarray(row["Boundaries"], dtype=np.float64) for row in ([gt] if isinstance(gt, dict) else gt)
    ]
    if (
        confidence.ndim != MAP_DIMENSIONS
        or confidence.dtype != np.uint8
        or not annotations
        or any(ann.shape != confidence.shape for ann in annotations)
    ):
        msg = f"{prediction} must be a native-size 8-bit boundary map matching {labels}"
        raise ValueError(msg)
    total = sum(int(ann.sum()) for ann in annotations)
    positives = confidence.size - np.cumsum(np.bincount(confidence.ravel(), minlength=256))[CUTOFFS - 1]
    counts = np.empty((len(CUTOFFS), 4), dtype=np.int64)
    start = 0
    try:
        previous = np.load(progress, allow_pickle=False)
    except FileNotFoundError:
        # A concurrent scorer can publish the result and remove its progress file.
        with suppress(FileNotFoundError):
            return np.load(target, allow_pickle=False)
    else:
        start = len(previous)
        counts[:start] = previous
    match = _matcher(library)
    for i in range(start, len(CUTOFFS)):
        num_positive = int(positives[i])
        if i and positives[i - 1] == num_positive:
            counts[i] = counts[i - 1]
        elif num_positive:
            recall, precision = _counts_at(confidence, int(CUTOFFS[i]), annotations, match)
            if recall > total or precision > num_positive:
                msg = f"The boundary counts exceed the available pixels in {prediction}"
                raise ValueError(msg)
            counts[i] = recall, total, precision, num_positive
        else:
            counts[i] = 0, total, 0, 0
        if (i + 1) % 5 == 0:
            _save_counts(progress, counts[: i + 1])
    _save_counts(target, counts)
    progress.unlink(missing_ok=True)
    return counts


def summarize(values: NDArray[np.int64]) -> dict[str, float]:
    """Calculate the same interpolated ODS, OIS and AP as the FastForks BSDS500 rows."""

    def rates(counts: NDArray[np.int64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        match_ref, total_ref, match_pred, total_pred = np.moveaxis(counts, -1, 0)
        return match_ref / np.maximum(total_ref, 1), match_pred / np.maximum(total_pred, 1)

    def fmeasure(recall: NDArray[np.float64], precision: NDArray[np.float64]) -> NDArray[np.float64]:
        total = recall + precision
        return 2 * recall * precision / np.where(total > 0, total, 1)

    recall, precision = rates(values)
    best = np.argmax(fmeasure(recall, precision), axis=1)
    ois = fmeasure(*rates(values[np.arange(len(values)), best].sum(axis=0)))
    recall, precision = rates(values.sum(axis=0))
    mix = np.linspace(0, 1, 100)
    recall_grid = recall[:-1, None] * (1 - mix) + recall[1:, None] * mix
    precision_grid = precision[:-1, None] * (1 - mix) + precision[1:, None] * mix
    ods = fmeasure(recall_grid, precision_grid).max()
    unique, first = np.unique(recall, return_index=True)
    ap = (
        np.interp(np.arange(101) / 100, unique, precision[first], left=0, right=0).sum() / 100 if len(unique) > 1 else 0
    )
    return {"ods": round(float(ods * 100), 2), "ois": round(float(ois * 100), 2), "ap": round(float(ap * 100), 2)}


def main() -> None:
    """Build the Berkeley matcher, score independent images and emit one JSON line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bsr", type=Path, required=True, help="extracted BSDS500 BSR directory")
    parser.add_argument("--work", type=Path, required=True, help="directory with inference.json and the predicted maps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", type=Path, help="resume the same score or reuse identical inputs across modes")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = json.loads((args.work / "inference.json").read_text(encoding="utf-8"))
    labels = sorted((args.bsr / "BSDS500/data/groundTruth/test").glob("*.mat"))
    predictions = sorted(args.work.glob("*.npy"))
    n = len(predictions)
    names = [path.stem for path in predictions]
    if (
        len(labels) != FULL_SPLIT
        or n not in (FIRST_SUBSET, FULL_SPLIT)
        or n != manifest.get("n")
        or names != [path.stem for path in labels[:n]]
    ):
        parser.error("expected the complete 200-image test split or its first 75 images by filename")
    cache = args.cache or args.work / "score-cache"
    cache.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="fastforks-bsds500-") as temp:
        library = Path(temp) / "correspond.so"
        _build(args.bsr / "bench/source", library)
        scorer_id = hashlib.sha256((PROTOCOL + _sha(library)).encode() + CUTOFFS.tobytes()).hexdigest()
        pairs = [(p, y, cache, library) for p, y in zip(predictions, labels[:n], strict=True)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            values = np.stack(list(pool.map(score_image, pairs, [scorer_id] * n)))
    manifest.update(
        dataset="bsds500-test-no-thinning" if n == FULL_SPLIT else "bsds500-test-first75-no-thinning",
        metrics=summarize(values),
        dataset_manifest_sha256=hashlib.sha256("\n".join(names).encode()).hexdigest(),
        case_ids=names,
        protocol=PROTOCOL,
    )
    archive = args.bsr.parent / "BSR_bsds500.tgz"
    if archive.is_file():
        manifest["dataset_archive_sha256"] = _sha(archive)
    print(json.dumps(manifest, allow_nan=False))


if __name__ == "__main__":
    main()
