"""Goldens of the segmentation families built by Hugging Face transformers (cards with `uses = ["seginfer"]`).

Reference: transformers itself at the version in the card's `[upstream]`, loading the authors' checkpoint the way a
user would: `<arch>.from_pretrained(repo, revision=...)` (the Hugging Face cache; set `HF_HUB_OFFLINE=1` to use it as
is), the repo and revision taken from the variant's weight URL. CPU float32, on `tests/synth.py`'s image with the
upstream test-time protocol written out here independently of `fastforks.shared`:

- `["mmseg", long, short]` (SegFormer, mmseg): keep-ratio resize (bilinear, half-pixel centres), ImageNet
  normalisation, one whole-image forward pass, logits resized bilinearly to the input size, then to the image size.
- `["short", short, max]` (Mask2Former / OneFormer, detectron2): `ResizeShortestEdge` (PIL bilinear: antialiased
  when downscaling), ImageNet normalisation, zero padding bottom / right to `pad`, one forward pass; mask logits
  resized to the padded size;
  `semantic_inference` (class softmax without "no object" times mask sigmoid). Mask2Former forms it at the padded size,
  then crops and resizes (`SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False`); OneFormer crops and resizes the mask
  logits first.
  OneFormer's `task_inputs` come from the checkpoint repo's own CLIP tokenizer ("the task is semantic", padded to
  `task_seq_len`) and must equal the card's `task_tokens`.
- `["fixed", h, w]` (fine-tuned Hugging Face SegFormer checkpoints): the checkpoint repo's own
  `SegformerImageProcessor` on the PIL image (its size must be the card's), logits resized bilinearly to the image size
  (the model cards' code).
- `window` set (EoMT, `window_imgs_semantic` / `revert_window_logits_semantic` of tue-mps/eomt): shorter side to
  `window` (PIL bilinear, `round`), ImageNet normalisation, `ceil(long / window)` square windows along the longer side
  (overlap spread evenly, `int` starts), one batch; per window `semantic_inference` at the window size; window logits
  averaged where they overlap and resized to the image size.

Also checks that the checkpoint's `id2label` is the card's label set, in order: the same names, or (OneFormer uses
the dataset's synonym lists) names that are all synonyms of the label in ADE20K's `objectInfo150.txt` (or their last
words).

    HF_HUB_OFFLINE=1 FASTFORKS_GOLDEN_PATH=... python tools/golden/seginfer_segmentation.py [FAMILY ...]
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import transformers
from _common import image, write
from PIL import Image

from fastforks._core import registry
from fastforks._core.labels import label_set
from fastforks._core.preprocess import IMAGENET_MEAN, IMAGENET_STD

TASK = "segmentation"
DATA = Path(os.environ.get("FASTFORKS_DATA", "~/.cache/fastforks/data")).expanduser()
ADE_INFO = str(DATA / "segmentation/ADE20K/ADEChallengeData2016/objectInfo150.txt")


def _synonym(name: str, synonyms: set[str]) -> bool:
    """`name` is one of `synonyms` or the last word(s) of one (detectron2 calls "swimming pool" "pool")."""
    return any(x == name or x.endswith(" " + name) for x in synonyms)


def _check_labels(ref: str, names: list[str], labels: str) -> None:
    want = list(label_set(labels))
    if names == want:
        return
    assert labels == "ade20k-150", f"{ref}: id2label differs from {labels}"  # noqa: S101
    rows = Path(ADE_INFO).read_text().splitlines()[1:]
    syn = [{n.strip() for n in r.split("\t")[-1].split(",")} for r in rows]
    bad = [
        (i, n, w)
        for i, (n, w, s) in enumerate(zip(names, want, syn, strict=True))
        if n != w and not (w in s and all(_synonym(p.strip(), s) for p in n.split(",")))
    ]
    assert not bad, f"{ref}: id2label differs from {labels}: {bad[:5]}"  # noqa: S101


def _mmseg_size(h: int, w: int, long: int, short: int) -> tuple[int, int]:
    """Size of mmcv `rescale(keep_ratio=True)` to fit `long x short`."""
    s = min(long / max(h, w), short / min(h, w))
    return int(h * s + 0.5), int(w * s + 0.5)


def _short_size(h: int, w: int, short: int, max_size: int) -> tuple[int, int]:
    """detectron2 `ResizeShortestEdge.get_output_shape`."""
    s = short / min(h, w)
    nh, nw = (short, s * w) if h < w else (s * h, short)
    if max(nh, nw) > max_size:
        s = max_size / max(nh, nw)
        nh, nw = nh * s, nw * s
    return int(nh + 0.5), int(nw + 0.5)


def _normalise(y: torch.Tensor) -> torch.Tensor:
    mean, std = (torch.tensor(v, device=y.device)[:, None, None] for v in (IMAGENET_MEAN, IMAGENET_STD))
    return (y - mean) / std


def _semantic(out: Any, size: tuple[int, int]) -> torch.Tensor:  # noqa: ANN401 - a transformers output
    m = F.interpolate(out.masks_queries_logits, size, mode="bilinear", align_corners=False)
    probs = out.class_queries_logits.softmax(-1)[..., :-1]
    return torch.einsum("bqc,bqhw->bchw", probs, m.sigmoid())


def _pil(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """PIL `BILINEAR` (detectron2 `ResizeShortestEdge`, EoMT): antialiased when downscaling."""
    down = size[0] < x.shape[-2] or size[1] < x.shape[-1]
    return F.interpolate(x, size, mode="bilinear", align_corners=False, antialias=down)


def _windows(net: torch.nn.Module, x: torch.Tensor, side: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    y = _normalise(_pil(x, (round(h * side / min(h, w)), round(w * side / min(h, w)))))
    rh, rw = y.shape[-2:]
    long = max(rh, rw)
    n = math.ceil(long / side)
    step = side - ((n * side - long) / (n - 1) if n > 1 else 0)
    starts = [int(j * step) for j in range(n)]
    crops = [y[..., s : s + side, :] if rh > rw else y[..., :, s : s + side] for s in starts]
    logits = _semantic(net(pixel_values=torch.cat(crops)), (side, side))
    acc = torch.zeros((1, logits.shape[1], rh, rw), device=x.device)
    cnt = torch.zeros((1, 1, rh, rw), device=x.device)
    for s, lg in zip(starts, logits, strict=True):
        sl = (..., slice(s, s + side), slice(None)) if rh > rw else (..., slice(None), slice(s, s + side))
        acc[sl] += lg
        cnt[sl] += 1
    return F.interpolate(acc / cnt, (h, w), mode="bilinear", align_corners=False)


def _whole(net: torch.nn.Module, x: torch.Tensor, params: dict, task: torch.Tensor | None) -> torch.Tensor:
    h, w = x.shape[-2:]
    if "window" in params:
        return _windows(net, x, params["window"])
    rule = params["resize"]
    if rule[0] == "mmseg":
        y = _normalise(F.interpolate(x, _mmseg_size(h, w, rule[1], rule[2]), mode="bilinear", align_corners=False))
        s = F.interpolate(net(pixel_values=y).logits, y.shape[-2:], mode="bilinear", align_corners=False)
        return F.interpolate(s, (h, w), mode="bilinear", align_corners=False)
    if rule[0] != "short":
        msg = f"no reference protocol for {rule}"
        raise NotImplementedError(msg)
    rh, rw = _short_size(h, w, rule[1], rule[2])
    y = _normalise(_pil(x, (rh, rw)))
    p = params["pad"]
    y = F.pad(y, (0, -rw % p, 0, -rh % p))
    out = net(pixel_values=y) if task is None else net(pixel_values=y, task_inputs=task)
    m = F.interpolate(out.masks_queries_logits, y.shape[-2:], mode="bilinear", align_corners=False)
    probs = out.class_queries_logits.softmax(-1)[..., :-1]
    if params["arch"].startswith("Mask2Former"):
        s = torch.einsum("bqc,bqhw->bchw", probs, m.sigmoid())[..., :rh, :rw]
        return F.interpolate(s, (h, w), mode="bilinear", align_corners=False)
    m = F.interpolate(m[..., :rh, :rw], (h, w), mode="bilinear", align_corners=False)
    return torch.einsum("bqc,bqhw->bchw", probs, m.sigmoid())


def _processor(repo: str, revision: str, x: torch.Tensor, rule: list) -> torch.Tensor:
    """The repo's `SegformerImageProcessor` applied to the uint8 image, logits as its model card resizes them."""
    proc = transformers.SegformerImageProcessor.from_pretrained(repo, revision=revision)
    assert [proc.size["height"], proc.size["width"]] == rule[1:], f"processor size {proc.size} != card {rule}"  # noqa: S101
    pil = Image.fromarray((x[0] * 255).round().byte().permute(1, 2, 0).numpy())
    return proc(images=pil, return_tensors="pt").pixel_values


def _task(repo: str, revision: str, net: Any, params: dict) -> torch.Tensor:  # noqa: ANN401 - a transformers model
    n = net.config.task_seq_len
    tok = transformers.CLIPTokenizer.from_pretrained(repo, revision=revision)
    t = tok("the task is semantic", padding="max_length", max_length=n, truncation=True)
    ids = [
        i * a for i, a in zip(t["input_ids"], t["attention_mask"], strict=True)
    ]  # OneFormerProcessor._preprocess_text
    card = list(params["task_tokens"])
    assert ids == card + [0] * (n - len(card)), f"task tokens {ids[:12]} != card {card}"  # noqa: S101
    return torch.tensor([ids])


def main() -> None:
    families = set(sys.argv[1:])
    for card, variant in registry.cards(TASK).values():
        if "seginfer" not in card.uses or (families and card.family not in families):
            continue
        params = {**card.params, **variant.params}
        url = variant.weights[-1].url or ""
        m = re.match(r"https://huggingface.co/(.+?)/resolve/([0-9a-f]{40})/", url)
        assert m, url  # noqa: S101
        net = getattr(transformers, params["arch"]).from_pretrained(m[1], revision=m[2]).float().eval()
        names = [net.config.id2label[i].strip() for i in range(net.config.num_labels)]  # b0 has "bed "
        _check_labels(variant.ref, names, variant.output_info["label_set"])
        task = _task(m[1], m[2], net, params) if "task_tokens" in params else None
        x = torch.from_numpy(image()).permute(2, 0, 1)[None].float() / 255
        with torch.inference_mode():
            if params.get("resize", [""])[0] == "fixed":
                y = net(pixel_values=_processor(m[1], m[2], x, list(params["resize"]))).logits
                logits = F.interpolate(y, x.shape[-2:], mode="bilinear", align_corners=False)[0].numpy()
            else:
                logits = _whole(net, x, params, task)[0].numpy()
        write(
            variant.ref,
            {"logits": logits},
            family=card.family,
            upstream=f"{card.upstream}@{card.upstream_commit}",
            patch=None,
            settings=f"CPU float32, transformers {transformers.__version__}, torch {torch.__version__}",
            task=TASK,
        )
        print(f"  {variant.ref}: classes {sorted(set(logits.argmax(0).ravel().tolist()))[:8]}")


if __name__ == "__main__":
    main()
