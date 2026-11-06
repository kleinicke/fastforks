"""ImageNet-1k top-1 / top-5 of fastforks classifiers and of the libraries they extend, on the same images.

    FASTFORKS_KERNEL_PATH=build/kernels python tools/eval_classification.py --models resnet50.tv2_in1k \
        --tiers upstream,portable32,portable,max --every 5 --out raw.jsonl

Images: the ImageNet val folders (`<val>/<synset>/*.JPEG`, labels = sorted synset order), every `--every`-th file of the
sorted list, then `[--start:--stop]` (chunks of big models fit one GPU lock; the rows add up). fastforks tiers take the
decoded uint8 image through the model's own transform (`resize_crop` + `scale` on the GPU, as
`ClassificationModel.__call__`) and run the runner in batches. `upstream` is the library itself as a user runs it: its
pretrained loading and its PIL eval transform (torchvision `weights.transforms()`, timm `create_transform`), fp32.
Tiers: `portable32` = portable with dtype float32, `portable` / `kernels` = the validated dtype. `--subset N --seed S`
picks a random subset instead of every k-th file. One JSON line per model and tier: correct counts, images, ms per
image, peak allocated / reserved MB.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from _data import DATA

sys.path.insert(0, str(Path(__file__).parent / "golden"))

VAL = str(DATA / "classification/imagenet1k_val")


def _files(  # noqa: PLR0913, PLR0917
    val: str, every: int, start: int, stop: int | None, subset: int = 0, seed: int = 0
) -> list[tuple[Path, int]]:
    """Labelled files of the sorted list; `subset` > 0 picks that many at random (fixed `seed`) instead of `every`."""
    classes = sorted(p for p in Path(val).iterdir() if p.is_dir())
    files = [(f, i) for i, c in enumerate(classes) for f in sorted(c.iterdir())]
    if subset:
        keep = sorted(np.random.default_rng(seed).choice(len(files), subset, replace=False).tolist())
        return [files[i] for i in keep][start:stop]
    return files[::every][start:stop]


class _Set(torch.utils.data.Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, files: list[tuple[Path, int]], transform: Any = None) -> None:  # noqa: ANN401
        self.files, self.transform = files, transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        f, label = self.files[i]
        im = Image.open(f).convert("RGB")
        if self.transform is not None:
            return self.transform(im), label
        return torch.from_numpy(__import__("numpy").asarray(im).copy()).permute(2, 0, 1), label


def _loader(files: list[tuple[Path, int]], transform: Any, batch: int) -> torch.utils.data.DataLoader:  # noqa: ANN401
    collate = None if transform is not None else (lambda b: ([x for x, _ in b], torch.tensor([y for _, y in b])))
    return torch.utils.data.DataLoader(
        _Set(files, transform), batch_size=batch, num_workers=12, collate_fn=collate, prefetch_factor=4
    )


def _upstream(ref: str) -> tuple[torch.nn.Module, Any]:
    """The library's network (fp32, CUDA) and its PIL eval transform."""
    import libnet_classification as lc  # noqa: PLC0415
    import timm  # noqa: PLC0415
    import timm.data  # noqa: PLC0415
    import torchvision  # noqa: PLC0415

    from fastforks._core import registry  # noqa: PLC0415

    card, variant = next(cv for k, cv in registry.cards("classification").items() if k.split("@")[0] == ref)
    params = {**card.params, **variant.params}
    io = variant.io
    if card.family == "dinov2_linear_cls":  # torch.hub eval: resize 256 bicubic, crop 224
        net = lc._dinov2(params["arch"], io.crop)  # noqa: SLF001
        tf = timm.data.create_transform(
            input_size=(3, io.crop, io.crop), interpolation="bicubic", crop_pct=io.crop / io.resize
        )
    elif params["library"] == "torchvision":
        arch, url = params["arch"], variant.weights[0].url
        weights = next(w for w in torchvision.models.get_model_weights(arch) if w.url == url)
        net, tf = torchvision.models.get_model(arch, weights=weights), weights.transforms()
    else:
        net = timm.create_model(params["arch"], pretrained=True)
        tf = timm.data.create_transform(**timm.data.resolve_data_config({}, model=net))
    return net.eval().cuda(), tf


def _run(ref: str, tier: str, files: list[tuple[Path, int]], batch: int) -> dict[str, Any]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    if tier.startswith("upstream"):
        net, tf = _upstream(ref)
        how = {"upstream": "fp32", "upstream16": "autocast", "upstreamac": "autocast", "upstreamhalf": "half"}[tier]
        rec: dict[str, Any] = {"backend": f"upstream {how} (library PIL transform)", "how": how,
                               "dtype": "float32" if how == "fp32" else "float16"}  # fmt: skip
        if how == "half":
            net = net.half()

        def fwd(xs: Any) -> torch.Tensor:  # noqa: ANN401
            xs = xs.cuda(non_blocking=True)
            if how == "half":
                return net(xs.half())
            with torch.autocast("cuda", torch.float16, enabled=how == "autocast"):
                return net(xs)
    else:
        from fastforks import classification  # noqa: PLC0415
        from fastforks._core.preprocess import resize_crop, scale  # noqa: PLC0415

        mode, dtype = ("portable", "float32") if tier == "portable32" else (tier, None)
        model = classification.load(ref, mode=mode, dtype=dtype, device="cuda")
        io, runner, tf = model.variant.io, model._runner, None  # noqa: SLF001 - batches of differently sized images
        rec = {"backend": f"fastforks {tier} {model.backend.dtype}", "kernels": list(model.backend.kernels),
               "dtype": str(model.backend.dtype).removeprefix("torch.")}  # fmt: skip

        def fwd(xs: Any) -> torch.Tensor:  # noqa: ANN401
            ims = [scale(resize_crop(x.cuda(non_blocking=True)[None].float().div_(255), io), io) for x in xs]
            return runner(torch.cat(ims))["logits"]

    rec["build_s"] = round(time.perf_counter() - t0, 1)
    top1 = top5 = n = 0
    t_run = 0.0
    with torch.inference_mode():
        for xs, ys in _loader(files, tf, batch):
            torch.cuda.synchronize()
            t = time.perf_counter()
            logits = fwd(xs).float()
            torch.cuda.synchronize()
            t_run += time.perf_counter() - t
            pred = logits.topk(5, dim=1).indices.cpu()
            top1 += int((pred[:, 0] == ys).sum())
            top5 += int((pred == ys[:, None]).any(1).sum())
            n += len(ys)
    rec.update(
        top1_correct=top1,
        top5_correct=top5,
        images=n,
        top1=round(100 * top1 / n, 3),
        top5=round(100 * top5 / n, 3),
        ms_per_image=round(1e3 * t_run / n, 3),
        alloc_mb=torch.cuda.max_memory_allocated() >> 20,
        res_mb=torch.cuda.max_memory_reserved() >> 20,
    )
    return rec


def _chunked(a: argparse.Namespace) -> int:
    """`--chunk`: every model x tier x chunk of the selection not yet in `--out`, one record each; 3 = budget hit."""
    t0 = time.perf_counter()
    files = _files(a.val, a.every, 0, None, a.subset, a.seed)
    stop = a.stop or len(files)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)  # before the first model runs, not at its first write
    done = set()
    if out.exists():
        for r in map(json.loads, out.read_text().splitlines()):
            if "error" not in r:  # failed chunks are retried by the next call
                done.add((r["model"], r["tier"], r["start"], r["stop"]))
    for ref in a.models.split(","):
        for tier in a.tiers.split(","):
            for s0 in range(a.start, stop, a.chunk):
                s1 = min(s0 + a.chunk, stop)
                if (ref, tier, s0, s1) in done:
                    continue
                if time.perf_counter() - t0 > a.max_seconds:
                    return 3
                rec: dict[str, Any] = {"model": ref, "tier": tier, "every": a.every, "subset": a.subset,
                                       "seed": a.seed, "start": s0, "stop": s1}  # fmt: skip
                try:
                    rec.update(_run(ref, tier, files[s0:s1], a.batch))
                except Exception as e:  # noqa: BLE001 - recorded; the queue goes on
                    rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                rec.update(gpu=torch.cuda.get_device_name(), torch=torch.__version__, commit=_commit(),
                           date=time.strftime("%Y-%m-%d %H:%M"), command=" ".join(["python", *sys.argv]))  # fmt: skip
                with out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(ref, tier, s0, s1, rec.get("top1", rec.get("error")), flush=True)
    return 0


def _commit() -> str:
    """The last library commit (src/) of this checkout."""
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(root), "log", "-1", "--format=%h", "--", "src"], capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> None:
    """Command line entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tiers", default="upstream,portable,max")
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--subset", type=int, default=0, help="random subset of this size (overrides --every)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=0, help="run [start:stop] in chunks of N, skip chunks in --out")
    ap.add_argument("--max-seconds", type=float, default=1e9, help="with --chunk: exit 3 (call again) after this")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.chunk:
        sys.exit(_chunked(a))
    torch.multiprocessing.set_sharing_strategy("file_system")  # lists of decoded images: no fd per tensor
    files = _files(a.val, a.every, a.start, a.stop, a.subset, a.seed)
    rev = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    for ref in a.models.split(","):
        for tier in a.tiers.split(","):
            rec: dict[str, Any] = {
                "model": ref,
                "tier": tier,
                "every": a.every,
                "subset": a.subset,
                "seed": a.seed,
                "start": a.start,
                "stop": a.stop,
            }
            try:
                rec.update(_run(ref, tier, files, a.batch))
            except Exception as e:  # noqa: BLE001 - recorded; the queue goes on
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rec.update(
                gpu=torch.cuda.get_device_name(),
                torch=torch.__version__,
                commit=rev,
                date=time.strftime("%Y-%m-%d %H:%M"),
                command=" ".join(["python", *sys.argv]),
            )
            line = json.dumps(rec)
            print(line, flush=True)
            if a.out:
                with Path(a.out).open("a") as f:
                    f.write(line + "\n")


if __name__ == "__main__":
    main()
