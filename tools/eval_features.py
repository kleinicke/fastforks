"""kNN sanity of fastforks feature backbones (DINOv2 / DINOv3) on ImageNet-1k, kernels vs portable.

    FASTFORKS_KERNEL_PATH=build/kernels python tools/eval_features.py \
        --models dinov2-vits14,dinov3-vits16 --out raw.jsonl

Train / test: two disjoint seeded random subsets of the ImageNet val folders (the data dir has no train split):
`--test` images to classify and `--train` labelled ones as the memory. Every image is resized (shorter side 256,
bicubic) and centre-cropped to 224 on the CPU, so all tiers see the same pixels; features = `pooler_output` (CLS token
after the final norm) of a fastforks model called with `size=(224, 224)`. kNN: cosine similarity, k=20, vote weighted
by similarity (plain vote with `--weighted 0`). Tiers: `portable` (fp32) and `kernels` (default dtype). One JSON line
per model: top-1 per tier, mean / min cosine between kernels and portable features of the test images, img/s, peak MB.

Tier mode (2026-11-05 campaign; `--tiers T,... --chunk N --max-seconds S --cache FILE --out DIR`): one record per model
and tier in `DIR/features.jsonl` (top-1, img/s, peak MB, mode / dtype / how). The decoded 224 crops of train + test are
cached once in `--cache` (`.npy`, uint8, made on the CPU by `--make-cache`); features are extracted in chunks of N
images into `DIR/feat/MODEL__TIER__START-STOP.npz` (resumable, exit 3 = call again), the kNN runs once all chunks
exist. Tiers: fastforks
`portable32`, `portable`, `kernels`; the authors' code (`upstream` fp32, `upstream16` autocast float16, `upstreamhalf`
`.half()` + float16 input; none has an own option): DINOv2 = facebookresearch/dinov2 `forward_features` CLS
(`x_norm_clstoken`, bench adapter, 006 venv), DINOv3 ViT / ConvNeXt = transformers `DINOv3ViTModel` /
`DINOv3ConvNextModel` `pooler_output`, ConvNeXt = timm `num_classes=0` (pooled, head norm); input normalised
with the ImageNet mean / std. TIPSv2 = google-deepmind/tips `image_encoder` first cls token, input in [0, 1].
CLIP / SigLIP 2 / Perception Encoder = timm by the card's `arch` (`dynamic_img_size=True`, the card's weight file),
`forward_head(pre_logits=True)`, the card's mean / std; C-RADIO = NVIDIA's `AutoModel` (`trust_remote_code`, the
card's revision) `summary`, input in [0, 1]. Every tier runs at the 224 crop (position embeddings resampled where the
native size differs), so a row is a like-for-like original vs fastforks check, not the model's best kNN score.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from _data import DATA

VAL = str(DATA / "classification/imagenet1k_val")


def split(val: str, n_train: int, n_test: int, seed: int) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]]]:
    """Disjoint seeded (train, test) subsets of the val files with their class index."""
    classes = sorted(p for p in Path(val).iterdir() if p.is_dir())
    files = [(f, i) for i, c in enumerate(classes) for f in sorted(c.iterdir())]
    idx = np.random.default_rng(seed).choice(len(files), n_train + n_test, replace=False).tolist()
    return [files[i] for i in sorted(idx[:n_train])], [files[i] for i in sorted(idx[n_train:])]


class _Set(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(self, files: list[tuple[Path, int]]) -> None:
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        im = Image.open(self.files[i][0]).convert("RGB")
        w, h = im.size
        s = 256 / min(w, h)
        im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BICUBIC)
        w, h = im.size
        x0, y0 = (w - 224) // 2, (h - 224) // 2
        return torch.from_numpy(np.asarray(im.crop((x0, y0, x0 + 224, y0 + 224))).copy())  # HWC uint8


def load_images(files: list[tuple[Path, int]]) -> torch.Tensor:
    """All files as one uint8 `[N, 224, 224, 3]` tensor (resize + crop on the CPU)."""
    dl = torch.utils.data.DataLoader(_Set(files), batch_size=250, num_workers=12)
    return torch.cat(list(dl))


@torch.inference_mode()
def extract(model: Any, imgs: torch.Tensor, batch: int) -> tuple[torch.Tensor, float]:  # noqa: ANN401
    """L2-normalised fp32 CLS features `[N, C]` on the GPU and the model seconds."""
    out, t_run = [], 0.0
    for i in range(0, len(imgs), batch):
        x = imgs[i : i + batch].cuda().permute(0, 3, 1, 2)
        torch.cuda.synchronize()
        t = time.perf_counter()
        f = model(x, size=(224, 224)).pooler_output
        torch.cuda.synchronize()
        t_run += time.perf_counter() - t
        out.append(torch.nn.functional.normalize(f.float(), dim=1))
    return torch.cat(out), t_run


def knn(  # noqa: PLR0913
    train: torch.Tensor, ytr: torch.Tensor, test: torch.Tensor, yte: torch.Tensor, k: int, *, weighted: bool
) -> float:
    """Top-1 (percent) of the cosine kNN vote."""
    sim, ix = (test @ train.T).topk(k, dim=1)
    votes = torch.zeros(len(test), 1000, device=test.device)
    votes.scatter_add_(1, ytr[ix], sim.clamp(min=0) if weighted else torch.ones_like(sim))
    return float((votes.argmax(1) == yte).float().mean() * 100)


MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
GATED = Path(os.environ.get("FASTFORKS_DATA", "data")) / "gated_models/dinov3"  # the licence-gated checkpoints


def _original(ref: str, tier: str) -> tuple[Any, dict[str, Any]]:  # noqa: C901, PLR0915 - one branch per family
    """The authors' feature extractor: fn(uint8 [B, 3, 224, 224] CUDA) -> CLS / pooled features [B, C]."""
    import _original as orig  # noqa: PLC0415

    bu = orig.bench()
    full, card, variant = orig.full_ref("features", ref)
    if card.family == "dinov2":
        a = bu.FAM["dinov2"]
        a.setup()
        fn0 = a.build(full, False)  # noqa: FBT003
        net = next(c.cell_contents for c in fn0.__closure__ or () if isinstance(c.cell_contents, torch.nn.Module))

        def feat(x: torch.Tensor) -> torch.Tensor:
            return net.forward_features(x)["x_norm_clstoken"]

    elif (
        card.family == "tipsv2"
    ):  # TIPS_REPO = a google-deepmind/tips checkout, TIPS_WEIGHTS = the golden's WEIGHTS_DIR
        g = orig._golden("tipsv2")  # noqa: SLF001 - the golden's builder
        net = g.build(Path(os.environ["TIPS_REPO"]), Path(os.environ["TIPS_WEIGHTS"]), ref.removeprefix("tipsv2-"))
        net = net.cuda()

        def feat(x: torch.Tensor) -> torch.Tensor:
            return net(x)[0][:, 0]

    elif card.family == "convnext_features":
        import timm  # noqa: PLC0415

        pooled: Any = timm.create_model(full.split("@")[0], pretrained=True, num_classes=0).cuda().eval()
        net = feat = pooled  # head norm of the mean stage-4 map (`forward_head(pre_logits=True)`)

    elif "arch" in {**card.params, **variant.params}:  # CLIP / SigLIP 2 / Perception Encoder: timm by name
        import timm  # noqa: PLC0415

        from fastforks._core import registry  # noqa: PLC0415

        arch = {**card.params, **variant.params}["arch"]
        file = str(registry.weights(card, variant)[0])  # the timm hub file the card pins
        vit: Any = (
            timm.create_model(arch, pretrained=True, pretrained_cfg_overlay={"file": file}, dynamic_img_size=True)
            .cuda()
            .eval()
        )
        net = vit

        def feat(x: torch.Tensor) -> torch.Tensor:
            return vit.forward_head(vit.forward_features(x), pre_logits=True)

    elif card.family == "c_radio":  # NVIDIA's code (trust_remote_code) at the card's revision, input in [0, 1]
        import re  # noqa: PLC0415

        import transformers  # noqa: PLC0415

        hub = re.match(r"https://huggingface.co/([^/]+/[^/]+)/resolve/([0-9a-f]+)/", variant.weights[0].url)
        if hub is None:
            msg = f"{ref}: unexpected weight url {variant.weights[0].url}"
            raise RuntimeError(msg)
        radio: Any = transformers.AutoModel.from_pretrained(hub[1], revision=hub[2], trust_remote_code=True)
        net = radio = radio.float().cuda().eval()

        def feat(x: torch.Tensor) -> torch.Tensor:
            return radio(x).summary

    else:
        import transformers  # noqa: PLC0415

        cls: Any = transformers.DINOv3ConvNextModel if card.family == "dinov3_convnext" else transformers.DINOv3ViTModel
        arch = ref.split("@", maxsplit=1)[0].removeprefix("dinov3-")
        net = cls.from_pretrained(GATED / f"dinov3-{arch}-pretrain-lvd1689m", dtype=torch.float32).cuda().eval()

        def feat(x: torch.Tensor) -> torch.Tensor:
            return net(pixel_values=x).pooler_output

    how = orig.TIERS[tier].replace("autocast-plain", "autocast")
    if how == "half":
        net.half()
    m = torch.tensor(MEAN, device="cuda").view(1, 3, 1, 1)
    sd = torch.tensor(STD, device="cuda").view(1, 3, 1, 1)

    if card.family in {"tipsv2", "c_radio"}:  # the encoders take [0, 1] images
        m, sd = torch.zeros_like(m), torch.ones_like(sd)
    elif card.family in {"clip", "siglip2", "perception_encoder"}:  # the card's mean / std
        m = torch.tensor(variant.io.mean, device="cuda").view(1, 3, 1, 1)
        sd = torch.tensor(variant.io.std, device="cuda").view(1, 3, 1, 1)

    def fn(x: torch.Tensor) -> torch.Tensor:
        xx = (x.float() / 255 - m) / sd
        with torch.autocast("cuda", torch.float16, enabled=how == "autocast"):
            return feat(xx.half() if how == "half" else xx)

    dt = "float32" if how == "fp32" else "float16"
    return fn, {"mode": "upstream", "dtype": dt, "how": how}


def _fastforks(ref: str, tier: str) -> tuple[Any, dict[str, Any]]:
    from fastforks import features  # noqa: PLC0415

    port32 = tier == "portable32"
    m = features.load(ref, mode="portable" if port32 else tier, device="cuda", dtype="float32" if port32 else None)
    rec = {"mode": m.backend.mode, "dtype": str(m.backend.dtype).removeprefix("torch."), "how": None,
           "kernels": sorted(m.backend.kernels)}  # fmt: skip
    return (lambda x: m(x, size=(224, 224)).pooler_output), rec


def _part(stem: Path, s0: int, chunk: int, n: int) -> Path:
    """Feature file of images [s0, s1): start and stop in the name, so a rerun with another --chunk never reuses it."""
    return Path(f"{stem}__{s0:05d}-{min(s0 + chunk, n):05d}.npz")


def _tiers(a: argparse.Namespace) -> int:  # noqa: C901, PLR0912, PLR0915 - resumable chunk loop
    """Tier mode: chunks of features per model / tier, then one kNN record each; 3 = budget hit."""
    t0 = time.perf_counter()
    out = Path(a.out)
    (out / "feat").mkdir(parents=True, exist_ok=True)
    tr, te = split(a.val, a.train, a.test, a.seed)
    labels = torch.tensor([y for _, y in tr + te])
    imgs = np.load(a.cache, mmap_mode="r")
    if len(imgs) != len(tr) + len(te):
        msg = f"cache {a.cache} has {len(imgs)} images, expected {len(tr) + len(te)}"
        raise SystemExit(msg)
    jl = out / "features.jsonl"
    done = {(r["model"], r["tier"]) for r in map(json.loads, jl.read_text().splitlines())} if jl.exists() else set()
    for ref in a.models.split(","):
        for tier in a.tiers.split(","):
            if (ref, tier) in done:
                continue
            stem = out / "feat" / f"{ref}__{tier}"
            built = None
            meta: dict[str, Any] = {}
            err = None
            for s0 in range(0, len(imgs), a.chunk):
                f = _part(stem, s0, a.chunk, len(imgs))
                if f.exists():
                    continue
                if time.perf_counter() - t0 > a.max_seconds:
                    return 3
                try:
                    if built is None:
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        tb = time.perf_counter()
                        built = _original(ref, tier) if tier.startswith("upstream") else _fastforks(ref, tier)
                        meta = built[1] | {"build_s": round(time.perf_counter() - tb, 1)}
                    fn = built[0]
                    feats, t_run = [], 0.0
                    with torch.inference_mode():
                        for i in range(s0, min(s0 + a.chunk, len(imgs)), a.batch):
                            x = torch.from_numpy(np.ascontiguousarray(imgs[i : min(i + a.batch, s0 + a.chunk)]))
                            x = x.cuda().permute(0, 3, 1, 2)
                            torch.cuda.synchronize()
                            t = time.perf_counter()
                            y = fn(x)
                            torch.cuda.synchronize()
                            t_run += time.perf_counter() - t
                            feats.append(torch.nn.functional.normalize(y.float(), dim=1).cpu())
                    ff = torch.cat(feats)
                    np.savez(f, feat=ff.half().numpy(), t_run=t_run, n=len(ff),
                             nonfinite=int((~torch.isfinite(ff)).any(1).sum()),
                             alloc=torch.cuda.max_memory_allocated() >> 20, res=torch.cuda.max_memory_reserved() >> 20,
                             meta=json.dumps(meta))  # fmt: skip
                    print(ref, tier, s0, round(len(ff) / t_run, 1), "img/s", flush=True)
                except Exception as e:  # noqa: BLE001 - recorded; the queue goes on
                    err = f"{type(e).__name__}: {str(e)[:300]}"
                    if "out of memory" in str(e).lower():
                        err = f"OOM, {err}"
                    break
            del built
            rec: dict[str, Any] = {"model": ref, "tier": tier, "train": a.train, "test": a.test, "seed": a.seed,
                                   "k": a.k, "weighted": a.weighted}  # fmt: skip
            if err is None:
                parts = [np.load(_part(stem, s0, a.chunk, len(imgs))) for s0 in range(0, len(imgs), a.chunk)]
                feats = torch.from_numpy(np.concatenate([p["feat"] for p in parts])).float().cuda()
                feats = torch.nn.functional.normalize(feats.nan_to_num(), dim=1)
                ntr = len(tr)
                lab = labels.cuda()
                top1 = knn(feats[:ntr], lab[:ntr], feats[ntr:], lab[ntr:], a.k, weighted=bool(a.weighted))
                meta = json.loads(str(parts[0]["meta"]))
                rec |= meta | {
                    "top1": round(top1, 3),
                    "img_s": round(sum(int(p["n"]) for p in parts) / sum(float(p["t_run"]) for p in parts), 1),
                    "alloc_mb": max(int(p["alloc"]) for p in parts), "res_mb": max(int(p["res"]) for p in parts),
                    "nonfinite": sum(int(p["nonfinite"]) for p in parts),
                }  # fmt: skip
            else:
                rec["error"] = err
            rec |= {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "commit": _commit(),
                    "date": time.strftime("%F %H:%M"), "command": " ".join(["python", *sys.argv])}  # fmt: skip
            with jl.open("a") as fo:
                fo.write(json.dumps(rec) + "\n")
            print(json.dumps(rec), flush=True)
    return 0


def _probe(a: argparse.Namespace) -> int:
    """Blocker probe: every model / tier built and run on one batch of the cache; prints one line each."""
    x = torch.from_numpy(np.ascontiguousarray(np.load(a.cache, mmap_mode="r")[: a.batch])).cuda().permute(0, 3, 1, 2)
    for ref in a.models.split(","):
        for tier in a.tiers.split(","):
            try:
                fn, meta = _original(ref, tier) if tier.startswith("upstream") else _fastforks(ref, tier)
                with torch.inference_mode():
                    y = fn(x).float()
                print("PROBE", ref, tier, meta, tuple(y.shape), bool(torch.isfinite(y).all()), flush=True)
                del fn
            except Exception as e:  # noqa: BLE001 - the probe reports every failure
                print("PROBE", ref, tier, "ERROR", f"{type(e).__name__}: {str(e)[:300]}", flush=True)
            torch.cuda.empty_cache()
    return 0


def _commit() -> str:
    import subprocess  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(root), "log", "-1", "--format=%h", "--", "src"], capture_output=True, text=True, check=False
    ).stdout.strip()


def main() -> None:  # noqa: PLR0915
    """Command line entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--val", default=VAL)
    ap.add_argument("--train", type=int, default=10000)
    ap.add_argument("--test", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--weighted", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out")
    ap.add_argument("--tiers", default="", help="tier mode (see the module doc)")
    ap.add_argument("--chunk", type=int, default=3000)
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--cache", default="")
    ap.add_argument("--probe", action="store_true", help="with --tiers: build + one batch per model / tier, print")
    ap.add_argument("--make-cache", action="store_true", help="decode + crop train and test into --cache (CPU)")
    a = ap.parse_args()
    if a.make_cache:
        tr, te = split(a.val, a.train, a.test, a.seed)
        np.save(a.cache, load_images(tr + te).numpy())
        return
    if a.tiers and a.probe:
        sys.exit(_probe(a))
    if a.tiers:
        sys.exit(_tiers(a))
    from fastforks import features  # noqa: PLC0415

    tr, te = split(a.val, a.train, a.test, a.seed)
    msg = "train and test overlap"
    if {f for f, _ in tr} & {f for f, _ in te}:
        raise SystemExit(msg)
    itr, ite = load_images(tr), load_images(te)
    ytr, yte = (torch.tensor([y for _, y in s]).cuda() for s in (tr, te))
    for ref in a.models.split(","):
        rec: dict[str, Any] = {"model": ref, "train": a.train, "test": a.test, "seed": a.seed, "k": a.k}
        try:
            feats = {}
            for tier in ("portable", "kernels"):
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                m = features.load(ref, mode=tier, **({"dtype": "float32"} if tier == "portable" else {}), device="cuda")
                ftr, t1 = extract(m, itr, a.batch)
                fte, t2 = extract(m, ite, a.batch)
                feats[tier] = fte
                rec[f"{tier}_top1"] = round(knn(ftr, ytr, fte, yte, a.k, weighted=bool(a.weighted)), 3)
                rec[f"{tier}_img_s"] = round((len(itr) + len(ite)) / (t1 + t2), 1)
                rec[f"{tier}_alloc_mb"] = torch.cuda.max_memory_allocated() >> 20
                rec[f"{tier}_res_mb"] = torch.cuda.max_memory_reserved() >> 20
                rec[f"{tier}_dtype"] = str(m.backend.dtype)
                del m, ftr
            cos = (feats["kernels"] * feats["portable"]).sum(1)
            rec |= {"cos_mean": round(float(cos.mean()), 5), "cos_min": round(float(cos.min()), 5)}
        except Exception as e:  # noqa: BLE001 - recorded; the queue goes on
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec |= {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "date": time.strftime("%F %H:%M")}
        rec["command"] = " ".join(["python", *sys.argv])
        print(json.dumps(rec), flush=True)
        if a.out:
            with Path(a.out).open("a") as f:
                f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
