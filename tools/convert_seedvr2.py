"""SeedVR2 (3B / 7B) -> safetensors in the dtypes the original runs them in, in plain torch.

    python -I tools/convert_seedvr2.py dit seedvr2_ema_3b.pth pos_emb.pt --out seedvr2-3b.safetensors
    python -I tools/convert_seedvr2.py vae ema_vae.pth --out seedvr2-vae.safetensors

The released checkpoints are float32. `projects/inference_seedvr2_*.py` runs the transformer under
`torch.autocast("cuda", torch.bfloat16)`, which casts every linear's weight and bias and (apex `FusedRMSNorm`) every
norm weight to bf16 on use, and the VAE with `vae.dtype = bfloat16`. The converted transformer therefore stores those
tensors in bf16 (the same values autocast makes) and keeps the AdaSingle vectors (`*_shift`, `*_scale`, `*_gate`) in
float32: upstream adds them to the bf16 time embedding in float32 outside autocast. The RoPE `freqs` buffers (fixed
by the config) are dropped. `txt` is the fixed positive prompt embedding (`pos_emb.pt`, bf16 `[58, 5120]`; the
negative one is unused at cfg 1). The VAE is stored in bf16 with upstream's keys.

The checkpoints are read memory-mapped (`torch.load(weights_only=True, mmap=True)`) and the output is written tensor
by tensor, so the 33 GB 7B checkpoint converts in a few GB of RAM.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch

BF = torch.bfloat16
ADA = ("_shift", "_scale", "_gate")
DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16"}


def write(sd: dict[str, torch.Tensor], dtype_of, out: Path, meta: dict[str, str]) -> None:  # noqa: ANN001
    """Header from the shapes, then each tensor converted and written in turn."""
    keys = sorted(sd)
    header: dict[str, object] = {"__metadata__": meta}
    pos = 0
    for k in keys:
        v = sd[k]
        dt = dtype_of(k)
        n = v.numel() * dt.itemsize
        header[k] = {"dtype": DTYPES[dt], "shape": list(v.shape), "data_offsets": [pos, pos + n]}
        pos += n
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    tmp = out.with_name(out.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for k in keys:
            v = sd[k].to(dtype_of(k)).contiguous()
            f.write(v.view(torch.uint8).numpy().tobytes())
    tmp.rename(out)
    print(f"{out}: {len(keys)} tensors, {pos / 1e9:.2f} GB")


def dit(ckpt: Path, pos_emb: Path, out: Path) -> None:
    """The transformer: bf16 except the AdaSingle vectors (float32); `txt` = the positive prompt embedding."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    sd = {k: v for k, v in sd.items() if not k.endswith("rope.freqs")}
    txt = torch.load(pos_emb, map_location="cpu", weights_only=True)
    if not (isinstance(txt, torch.Tensor) and txt.ndim == 2):  # noqa: PLR2004
        msg = f"{pos_emb}: expected a [tokens, 5120] tensor"
        raise SystemExit(msg)
    sd["txt"] = txt

    def dtype_of(k: str) -> torch.dtype:
        return torch.float32 if k.rsplit(".", 1)[-1].endswith(ADA) else BF

    write(sd, dtype_of, out, {"source": ckpt.name, "txt": pos_emb.name})


def vae(ckpt: Path, out: Path) -> None:
    """The VAE in bf16."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=True, mmap=True)
    write(sd, lambda _k: BF, out, {"source": ckpt.name})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dit")
    d.add_argument("ckpt", type=Path)
    d.add_argument("pos_emb", type=Path)
    d.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("vae")
    v.add_argument("ckpt", type=Path)
    v.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "dit":
        dit(a.ckpt, a.pos_emb, a.out)
    else:
        vae(a.ckpt, a.out)


if __name__ == "__main__":
    main()
