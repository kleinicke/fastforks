"""Marigold V2 (depth) -> one lean safetensors file, in plain torch (no diffusers / bitsandbytes / peft).

    python tools/convert_marigold_v2.py pack QWEN_DIR MARIGOLD_DIR --out lean.safetensors --fold fold.safetensors
    python tools/convert_marigold_v2.py fold lean.safetensors fold.safetensors [--device cuda]
    python tools/convert_marigold_v2.py check lean.safetensors reference.safetensors

`QWEN_DIR` is Qwen/Qwen-Image-Edit-2509 (`transformer/*.safetensors` bf16 shards, `vae/`), `MARIGOLD_DIR` is
prs-eth/Marigold-V2 (`depth/Log-stage2/trainables.safetensors`, `qwen_text_embeddings/`). The authors' stack loads
the transformer through bitsandbytes NF4 (`transformer_blocks.0.img_mod` kept in bf16, `proj_out` dequantized back
to bf16), adds the rank-128 LoRA of `trainables.safetensors` (alpha 128, so scaling 1) and runs one rectified-flow
step at t = 499/1000 (bf16) with fixed prompt embeddings. `pack` (CPU, reads the 41 GB of shards once) redoes the
NF4 packing (per 64 weights: absmax as bf16, nearest of the 16 codes, two codes per byte, high nibble first), copies
the LoRA factors, the biases and the norms, and writes the Qwen VAE as a single-frame 2D network (last temporal tap
of every causal 3D conv; `time_conv` never runs for one frame). Everything the fixed timestep and prompt make constant
goes to a second file for `fold`, which computes the 120 modulation vectors, the output-norm vector and the text
states and adds them to the lean file. `fold` emulates the bitsandbytes 4-bit GEMV (bf16 products, one fp32
accumulator per lane, warp tree) so the vectors equal the authors' bit for bit on the same GEMM library; the
cuBLAS parts (block-0 `img_mod`, the LoRA GEMVs, `txt_in`) can differ in the last bf16 ulp between GPUs.
`check` compares two lean files tensor by tensor.

Lean file (`fastforks.depth.models.marigold_v2`): `<linear>.q` (uint8 NF4, flat), `.absmax` (bf16, per 64),
`.bias`, `.lora_A`, `.lora_B` (bf16); `nf4_code` [16] fp32; `transformer_blocks.{i}.img_mod` / `txt_mod` [6 * dim]
fp32; `norm_out.linear` [2 * dim] fp32; `txt` [tokens, dim] fp32; `proj_out.weight` / `.bias` bf16; the last
block's text branch (`attn.to_add_out`, `txt_mlp`) is dropped (its output is never read); `vae.*` bf16 2D.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

BF = torch.bfloat16
BLOCK = 64  # NF4 block size (bitsandbytes default)
DIM = 3072
# bitsandbytes NF4 codes (`create_normal_map`), index = 4-bit value
NF4_CODE = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
]  # fmt: skip
# decision thresholds of bitsandbytes' `dQuantizeNF4` (x > threshold -> the upper code): the midpoints of the codes
NF4_THRESHOLDS = [
    -0.8480964004993439, -0.6106329262256622, -0.4599952697753906, -0.33967943489551544,
    -0.23460740596055984, -0.13791173323988914, -0.045525018125772476, 0.03979014977812767,
    0.1202552504837513, 0.2035212516784668, 0.2920137718319893, 0.3893125355243683,
    0.5016634166240692, 0.6427869200706482, 0.8614784181118011,
]  # fmt: skip
BLOCK_LINEARS = (
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "attn.add_q_proj", "attn.add_k_proj",
    "attn.add_v_proj", "attn.to_add_out", "img_mlp.net.0.proj", "img_mlp.net.2", "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
)  # fmt: skip
BLOCK_NORMS = ("attn.norm_q", "attn.norm_k", "attn.norm_added_q", "attn.norm_added_k")
LAST_DROPPED = ("attn.to_add_out", "txt_mlp.net.0.proj", "txt_mlp.net.2")
FOLD_LINEARS = (
    "time_text_embed.timestep_embedder.linear_1",
    "time_text_embed.timestep_embedder.linear_2",
    "txt_in",
    "norm_out.linear",
)
TIMESTEP = 499.0  # the authors' single denoising step: t = 499 / 1000 in bf16
TXT_EMBED = "qwen_edit_2509_qwen_depth_realimg512_prompt"


def header(path: Path) -> dict[str, dict]:
    """The safetensors header (name -> dtype / shape / data_offsets), without `__metadata__`."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h


def quantize(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 `[N, K]` -> (packed uint8 `[N * K / 2]`, absmax bf16 `[N * K / 64]`), as bitsandbytes `quantize_4bit`.

    The kernel scales each block by the reciprocal of its absmax (fp32) and walks a comparison tree whose
    thresholds are the code midpoints; `bucketize` with those thresholds is the same decision.
    """
    x = w.reshape(-1, BLOCK).float()
    absmax = x.abs().amax(dim=1)
    x = x * (1.0 / absmax)[:, None]
    idx = torch.bucketize(x.reshape(-1), torch.tensor(NF4_THRESHOLDS, dtype=torch.float32)).to(torch.uint8)
    idx = idx.view(-1, 2)
    return (idx[:, 0] << 4) | idx[:, 1], absmax.to(BF)


def dequantize(q: torch.Tensor, absmax: torch.Tensor, n_out: int, code: torch.Tensor) -> torch.Tensor:
    """Packed NF4 -> bf16 `[n_out, K]`, as bitsandbytes `dequantize_4bit` (fp32 code x absmax, rounded once)."""
    idx = torch.stack([q >> 4, q & 15], dim=1).reshape(-1).long()
    return (code[idx].view(-1, BLOCK) * absmax.float()[:, None]).to(BF).view(n_out, -1)


class Packer:
    """Routes the Qwen / Marigold tensors to the lean file (`lean`) or the inputs of `fold` (`fold`)."""

    def __init__(self, last_block: int) -> None:
        """`last_block`: index of the last transformer block."""
        self.code = torch.tensor(NF4_CODE, dtype=torch.float32)
        self.lean: dict[str, torch.Tensor] = {"nf4_code": self.code}
        self.fold: dict[str, torch.Tensor] = {"nf4_code": self.code}
        self.last = last_block

    @staticmethod
    def put(name: str, v: torch.Tensor, dst: dict[str, torch.Tensor]) -> None:
        """Store `v` as bf16 (it must be bf16-exact)."""
        vb = v.to(BF)
        if not torch.equal(vb.float(), v.float()):
            msg = f"{name}: not representable in bf16"
            raise ValueError(msg)
        dst[name] = vb.contiguous()

    @staticmethod
    def put_nf4(name: str, w: torch.Tensor, dst: dict[str, torch.Tensor]) -> None:
        """Store `w` NF4-packed as `.q` + `.absmax`."""
        dst[name + ".q"], dst[name + ".absmax"] = quantize(w)

    def put_linear(self, mod: str, leaf: str, v: torch.Tensor, dst: dict[str, torch.Tensor]) -> None:
        """An NF4 linear: packed weight, bf16 bias."""
        self.put_nf4(mod, v, dst) if leaf == "weight" else self.put(mod + "." + leaf, v, dst)

    def dropped(self, mod: str) -> bool:
        """The last block's text branch feeds nothing."""
        return mod.startswith(f"transformer_blocks.{self.last}.") and mod.split(".", 2)[2] in LAST_DROPPED

    def block(self, k: str, mod: str, leaf: str, v: torch.Tensor) -> None:
        """A `transformer_blocks.*` tensor."""
        sub = mod.split(".", 2)[2]
        if sub in BLOCK_LINEARS:
            if not self.dropped(mod):
                self.put_linear(mod, leaf, v, self.lean)
        elif sub in BLOCK_NORMS:
            self.put(k, v, self.lean)
        elif sub in ("img_mod.1", "txt_mod.1"):
            # block 0 img_mod stays bf16 (the authors skip it in the quantization)
            bf16 = k.startswith("transformer_blocks.0.img_mod.")
            self.put(k, v, self.fold) if bf16 else self.put_linear(mod, leaf, v, self.fold)
        else:
            msg = f"unexpected tensor {k}"
            raise ValueError(msg)

    def transformer(self, k: str, v: torch.Tensor) -> None:
        """A tensor of the Qwen transformer shards."""
        mod, _, leaf = k.rpartition(".")
        if k.startswith("transformer_blocks."):
            self.block(k, mod, leaf, v)
        elif mod == "img_in":
            self.put_linear(mod, leaf, v, self.lean)
        elif mod == "proj_out":  # quantized by the loader, then dequantized back to bf16
            if leaf == "weight":
                q, am = quantize(v)
                self.lean[k] = dequantize(q, am, v.shape[0], self.code)
            else:
                self.put(k, v, self.lean)
        elif mod in FOLD_LINEARS:
            self.put_linear(mod, leaf, v, self.fold)
        elif k == "txt_norm.weight":
            self.put(k, v, self.fold)
        else:
            msg = f"unexpected tensor {k}"
            raise ValueError(msg)

    def lora(self, k: str, v: torch.Tensor) -> None:
        """A LoRA factor from `trainables.safetensors` (keys without the `Diffuser.` prefix)."""
        name = k.replace(".lora_A.default.weight", ".lora_A").replace(".lora_B.default.weight", ".lora_B")
        mod = name.rsplit(".", 1)[0]
        if not self.dropped(mod):
            self.put(name, v, self.fold if mod in FOLD_LINEARS else self.lean)

    def vae(self, k: str, v: torch.Tensor) -> None:
        """One frame: the last temporal tap of every causal 3D conv; `time_conv` never runs."""
        if "time_conv" in k:
            return
        if v.ndim == 5:  # noqa: PLR2004
            v = v[:, :, -1]
        elif k.endswith("gamma") and v.ndim == 4:  # noqa: PLR2004
            v = v[:, 0]
        self.lean["vae." + k] = v.to(BF).contiguous()


def pack(qwen: Path, marigold: Path, out: Path, fold: Path) -> None:
    """Write the lean file (without the folded constants) and the inputs of `fold`."""
    index = json.loads(
        (qwen / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shards = sorted(set(index["weight_map"].values()))
    blocks = [int(k.split(".")[1]) for k in index["weight_map"] if k.startswith("transformer_blocks.")]
    pk = Packer(max(blocks))
    for shard in shards:
        path = qwen / "transformer" / shard
        order = sorted(header(path).items(), key=lambda kv: kv[1]["data_offsets"][0])  # sequential reads
        print(f"[pack] {shard}: {len(order)} tensors", flush=True)
        with safe_open(path, "pt", device="cpu") as f:
            for k, _ in order:
                pk.transformer(k, f.get_tensor(k))
    # LoRA factors (bf16) and the fine-tuned VAE decoder; the iREPA projector heads are training-only
    vae_tuned = {}
    with safe_open(marigold / "depth" / "Log-stage2" / "trainables.safetensors", "pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118 - safe_open has no __iter__
            if k.startswith("Diffuser."):
                pk.lora(k.removeprefix("Diffuser."), f.get_tensor(k))
            elif k.startswith("VAE."):
                vae_tuned[k.removeprefix("VAE.")] = f.get_tensor(k)
    with safe_open(qwen / "vae" / "diffusion_pytorch_model.safetensors", "pt", device="cpu") as f:
        vae = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    unexpected = set(vae_tuned) - set(vae)
    if unexpected:
        msg = f"trainables VAE keys not in the Qwen VAE: {sorted(unexpected)[:5]}"
        raise ValueError(msg)
    vae.update(vae_tuned)
    for k, v in vae.items():
        pk.vae(k, v)
    cfg = json.loads((qwen / "vae" / "config.json").read_text(encoding="utf-8"))
    pk.lean["vae.latents_mean"] = torch.tensor(cfg["latents_mean"], dtype=torch.float32)
    pk.lean["vae.latents_std"] = torch.tensor(cfg["latents_std"], dtype=torch.float32)
    # the prompt embeddings (fixed text condition)
    emb = marigold / "qwen_text_embeddings"
    embeds = torch.load(emb / f"{TXT_EMBED}_embeds.pt", map_location="cpu", weights_only=True)
    mask = torch.load(emb / f"{TXT_EMBED}_mask.pt", map_location="cpu", weights_only=True)
    pk.fold["prompt_embeds"] = embeds[:1].contiguous()
    pk.fold["prompt_mask"] = (mask[:1] if mask.dtype == torch.bool else mask[:1] > 0).to(torch.uint8).contiguous()
    for p, d in ((out, pk.lean), (fold, pk.fold)):
        size = sum(v.numel() * v.element_size() for v in d.values()) / 2**30
        print(f"[pack] {p}: {len(d)} tensors, {size:.2f} GiB", flush=True)
        save_file(d, str(p))


def gemv_nf4(
    x: torch.Tensor, q: torch.Tensor, absmax: torch.Tensor, bias: torch.Tensor, code: torch.Tensor
) -> torch.Tensor:
    """The bitsandbytes `gemv_4bit` (bf16, one row): `x` bf16 `[K]` -> bf16 `[N]`, then `+ bias` in bf16.

    The kernel's `kgemm_4bit_inference_naive`: the code table and absmax in bf16, weight = bf16(code * absmax),
    each product rounded to bf16, lane l of a warp accumulates elements `[l * 32, l * 32 + 32)` of every 1024-chunk
    in order into one fp32 sum, then a shuffle-down tree (offsets 1, 2, 4, 8, 16) sums the 32 lanes.
    """
    n = bias.numel()
    k = x.numel()
    idx = torch.stack([q >> 4, q & 15], dim=1).reshape(-1).long()
    w = (code.to(BF)[idx].view(-1, BLOCK) * absmax[:, None]).view(n, k)  # bf16 x bf16 -> bf16
    p = (x[None, :] * w).float()  # bf16 products
    chunks = math.ceil(k / 1024)
    p = F.pad(p, (0, chunks * 1024 - k)).view(n, chunks, 32, 32)
    acc = torch.zeros(n, 32, dtype=torch.float32, device=x.device)
    for c in range(chunks):
        for j in range(32):
            acc += p[:, c, :, j]
    for off in (1, 2, 4, 8, 16):
        acc = acc.clone()
        acc[:, : 32 - off] += acc[:, off:].clone()
    return acc[:, 0].to(BF) + bias


def timestep_embedding(t: torch.Tensor, channels: int = 256) -> torch.Tensor:
    """The diffusers `get_timestep_embedding(t, 256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)`."""
    half = channels // 2
    exponent = -math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
    emb = t[:, None].float() * torch.exp(exponent)[None, :]
    emb = 1000 * emb
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


def fold_constants(lean: Path, fold: Path, device: str) -> None:
    """Add the timestep- and prompt-constant vectors to the lean file (the authors' forward at t = 499 / 1000)."""
    with safe_open(fold, "pt", device=device) as f:
        fd = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    code = fd["nf4_code"]

    def gemv(name: str, x: torch.Tensor) -> torch.Tensor:
        return gemv_nf4(x, fd[name + ".q"], fd[name + ".absmax"], fd[name + ".bias"], code)

    t = torch.full((1,), TIMESTEP, dtype=BF, device=device) / 1000.0
    proj = timestep_embedding(t).to(BF)[0]
    temb = gemv("time_text_embed.timestep_embedder.linear_1", proj)
    temb = gemv("time_text_embed.timestep_embedder.linear_2", F.silu(temb))
    s = F.silu(temb)  # every modulation is Sequential(SiLU, Linear)(temb)
    new: dict[str, torch.Tensor] = {}
    blocks = sorted({int(k.split(".")[1]) for k in fd if k.startswith("transformer_blocks.")})
    for i in blocks:
        p = f"transformer_blocks.{i}."
        if p + "img_mod.1.weight" in fd:  # bf16 linear (cuBLAS)
            new[p + "img_mod"] = F.linear(s[None], fd[p + "img_mod.1.weight"], fd[p + "img_mod.1.bias"])[0].float()
        else:
            new[p + "img_mod"] = gemv(p + "img_mod.1", s).float()
        new[p + "txt_mod"] = gemv(p + "txt_mod.1", s).float()
    # norm_out.linear: NF4 base (GEMV) in fp32 + bf16 LoRA term (PEFT adds the two in the input's dtype, fp32)
    lora = F.linear(F.linear(s[None], fd["norm_out.linear.lora_A"]), fd["norm_out.linear.lora_B"])[0]
    new["norm_out.linear"] = gemv("norm_out.linear", s).float() + lora.float()
    # text states: RMSNorm (fp32 weight after `prepare_model_for_kbit_training`), NF4 GEMM in bf16 + bf16 LoRA, fp32
    mask = fd["prompt_mask"][0].bool()
    n_tok = int(mask.sum())
    if not bool(mask[:n_tok].all()) or bool(mask[n_tok:].any()):
        msg = "the prompt mask is not a prefix"
        raise ValueError(msg)
    x = fd["prompt_embeds"][0].to(BF)
    var = x.float().pow(2).mean(-1, keepdim=True)
    x = (x * torch.rsqrt(var + 1e-6)) * fd["txt_norm.weight"].float()
    w = dequantize(fd["txt_in.q"], fd["txt_in.absmax"], fd["txt_in.bias"].numel(), code)
    y = F.linear(x.to(BF), w, fd["txt_in.bias"]).float()
    y += F.linear(F.linear(x.to(BF), fd["txt_in.lora_A"]), fd["txt_in.lora_B"]).float()
    new["txt"] = y[:n_tok]
    print(f"[fold] {len(new)} constants ({n_tok} text tokens)", flush=True)
    with safe_open(lean, "pt", device="cpu") as f:
        lean_t = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118
    lean_t.update({k: v.cpu().contiguous() for k, v in new.items()})
    save_file(lean_t, str(lean))


def check(a: Path, b: Path) -> int:
    """Compare two lean files tensor by tensor; returns the number of differing or missing tensors."""
    ha, hb = header(a), header(b)
    bad = 0
    for k in sorted(set(ha) | set(hb)):
        if k not in ha or k not in hb:
            print(f"only in {'b' if k not in ha else 'a'}: {k}")
            bad += 1
    with safe_open(a, "pt", device="cpu") as fa, safe_open(b, "pt", device="cpu") as fb:
        for k in sorted(set(ha) & set(hb)):
            va, vb = fa.get_tensor(k), fb.get_tensor(k)
            if va.dtype != vb.dtype or va.shape != vb.shape:
                print(f"{k}: {va.dtype} {tuple(va.shape)} vs {vb.dtype} {tuple(vb.shape)}")
                bad += 1
            elif not torch.equal(va, vb):
                d = (va.float() - vb.float()).abs()
                n = int((va != vb).sum())
                print(f"{k}: {n} of {va.numel()} differ, max |d| {d.max().item():.3g}, mean |d| {d.mean().item():.3g}")
                bad += 1
    print(f"[check] {len(set(ha) & set(hb))} tensors compared, {bad} differ")
    return bad


def main() -> int:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("qwen", type=Path)
    p.add_argument("marigold", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fold", type=Path, required=True, help="where to put the inputs of `fold`")
    p = sub.add_parser("fold")
    p.add_argument("lean", type=Path)
    p.add_argument("fold", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p = sub.add_parser("check")
    p.add_argument("a", type=Path)
    p.add_argument("b", type=Path)
    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.qwen, args.marigold, args.out, args.fold)
    elif args.cmd == "fold":
        with torch.no_grad():
            fold_constants(args.lean, args.fold, args.device)
    else:
        return 1 if check(args.a, args.b) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
