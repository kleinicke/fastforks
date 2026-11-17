# ruff: noqa: ANN001, ANN003, ANN201, ANN204, C901, D102, D107, E741, FBT002, FBT003, PLC0415, PLR0913, PLR0917, PLR2004, RUF059
"""SeedVR2 reference run: upstream ByteDance-Seed/SeedVR code on the GPU, as `projects/inference_seedvr2_*.py` runs it.

    python tools/seedvr2_reference.py SEEDVR_CHECKOUT WEIGHTS_DIR --model 3b --input in.npz --scale 2 --out ref.npz

Run with the reference venv (torch, diffusers 0.29, einops, omegaconf, rotary-embedding-torch). `in.npz` holds `video`
uint8 `[T, H, W, 3]`; the output (`scale` x the input size, as upstream with `res_h * res_w = sH * sW`) is resized,
cropped to /16 and run through `generation_loop` / `generation_step` with upstream's modules. Deviations, all
value-preserving except the attention backend:

- weights: the converted safetensors (`tools/convert_seedvr2.py`: bf16 as autocast casts them, AdaSingle in fp32);
  RoPE `freqs` are read from the original checkpoint.
- `flash_attn_varlen_func` (not installed) -> SDPA per run of equal-length segments (bf16, as flash-attn).
- `apex.normalization.FusedRMSNorm` (not installed) -> its CUDA kernel's arithmetic: inputs cast to the autocast
  dtype, fp32 statistics, `gamma * bf16(invvar * x)`.
- `NaMMRotaryEmbedding3d.get_freqs` (3B) builds the axial table only as large as the windows need instead of
  `[1024, 128, 128, 126]` fp32 (8.5 GB); positions are the same aranges, so the values are identical.
- noise: the posterior eps (`[1, 16, t, h, w]`) and the DiT start noise (`[t, h, w, 16]`) come from a CPU float32
  generator seeded with `--seed` (in that order), cast to bf16, instead of `randn_like` on the GPU.
- `--offload` (7B): transformer blocks are loaded from the safetensors file onto the GPU one at a time.

The output npz holds `out` (uint8 `[T, sH', sW', 3]`), `sample` (float32 `[T, 3, sH', sW']` before clipping), `latent`
(the encoded condition `[t, h, w, 16]`), `pred` (the DiT output), `x0` (the sampler's endpoint), `time_s`,
`peak_alloc`, `peak_reserved`.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TXT = 58


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kw):  # noqa: ARG001
    """Varlen attention as SDPA, batched over runs of equal-length segments (self-attention: cu_q == cu_k)."""
    if kw.get("causal") or kw.get("dropout_p"):
        raise NotImplementedError(kw)
    cu = cu_seqlens_q.tolist()
    out = torch.empty_like(q)
    i = 0
    while i < len(cu) - 1:
        n = cu[i + 1] - cu[i]
        j = i
        while j < len(cu) - 1 and cu[j + 1] - cu[j] == n:
            j += 1
        s, e = cu[i], cu[j]
        qq, kk, vv = (x[s:e].unflatten(0, (j - i, n)).transpose(1, 2) for x in (q, k, v))
        o = F.scaled_dot_product_attention(qq, kk, vv, scale=kw.get("softmax_scale"))
        out[s:e] = o.transpose(1, 2).flatten(0, 1)
        i = j
    return out


class FusedRMSNorm(nn.Module):
    """apex `FusedRMSNorm` (affine): autocast casts input and weight; fp32 statistics; `gamma * bf16(invvar * x)`."""

    def __init__(self, normalized_shape, elementwise_affine=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else x.dtype
        x = x.to(dt)
        with torch.autocast("cuda", enabled=False):
            xf = x.float()
            y = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.eps)).to(dt)
            return y if self.weight is None else self.weight.to(dt) * y


class FusedLayerNorm(nn.LayerNorm):
    """apex `FusedLayerNorm` (unused by SeedVR2's checkpoints)."""


def shims() -> None:
    fa = types.ModuleType("flash_attn")
    fa.flash_attn_varlen_func = flash_attn_varlen_func
    apex = types.ModuleType("apex")
    norm = types.ModuleType("apex.normalization")
    norm.FusedRMSNorm, norm.FusedLayerNorm = FusedRMSNorm, FusedLayerNorm
    apex.normalization = norm
    sys.modules |= {"flash_attn": fa, "apex": apex, "apex.normalization": norm}


def patch_rope() -> None:
    from models.dit_v2.rope import NaMMRotaryEmbedding3d  # ty: ignore[unresolved-import]

    def get_freqs(self, vid_shape: torch.Tensor, txt_shape: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ls = txt_shape[:, 0].tolist()
        fhw = vid_shape.tolist()
        lf = max(l + f for (f, _, _), l in zip(fhw, ls, strict=True))
        vid_freqs = self.get_axial_freqs(lf, max(s[1] for s in fhw), max(s[2] for s in fhw))
        txt_freqs = self.get_axial_freqs(max(ls))
        vl, tl = [], []
        for (f, h, w), l in zip(fhw, ls, strict=True):
            vl.append(vid_freqs[l : l + f, :h, :w].reshape(-1, vid_freqs.size(-1)))
            tl.append(txt_freqs[:l].repeat(1, 3).reshape(-1, vid_freqs.size(-1)))
        return torch.cat(vl), torch.cat(tl)

    NaMMRotaryEmbedding3d.get_freqs = get_freqs


EPS: dict[str, torch.Tensor] = {}


def patch_posterior() -> None:
    """The VAE samples its posterior with `diffusers`' unseeded `randn`: draw the given `EPS` instead."""
    from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution  # ty: ignore[unresolved-import]

    def sample(self, generator=None) -> torch.Tensor:  # noqa: ARG001
        eps = EPS["posterior"]
        if eps.shape != self.mean.shape:
            raise ValueError((eps.shape, self.mean.shape))
        return self.mean + self.std * eps.to(self.mean)

    DiagonalGaussianDistribution.sample = sample


def build_dit(config, model: str, weights: Path, src: Path, *, offload: bool) -> tuple[nn.Module, torch.Tensor]:
    from common.config import create_object  # ty: ignore[unresolved-import]
    from common.distributed.meta_init_utils import meta_non_persistent_buffer_init_fn  # ty: ignore[unresolved-import]
    from safetensors import safe_open

    with torch.device("meta"):
        dit = create_object(config.dit.model)
    orig = torch.load(src / f"seedvr2_ema_{model}.pth", map_location="cpu", mmap=True, weights_only=True)
    freqs = {k: v.clone() for k, v in orig.items() if k.endswith("rope.freqs")}
    del orig
    dit.load_state_dict(freqs, strict=False, assign=True)
    path = weights / f"seedvr2-{model}.safetensors"
    with safe_open(path, "pt", device="cuda") as f:
        keys = [k for k in f.keys() if k != "txt"]  # noqa: SIM118
        txt = f.get_tensor("txt")
        resident = [k for k in keys if not (offload and k.startswith("blocks."))]
        info = dit.load_state_dict({k: f.get_tensor(k) for k in resident}, strict=False, assign=True)
    missing = [k for k in info.missing_keys if not k.endswith("rope.freqs") and k not in keys]
    if missing or info.unexpected_keys:
        raise RuntimeError((missing, info.unexpected_keys))
    dit = meta_non_persistent_buffer_init_fn(dit)
    for b in dit.modules():  # the RoPE buffers (freqs, dummy) to the GPU; parameters stay where they are
        for n, buf in list(b.named_buffers(recurse=False)):
            if not buf.is_meta:
                b.register_buffer(n, buf.cuda(), persistent=n in b.state_dict())
    if offload:
        for i, block in enumerate(dit.blocks):
            prefix = f"blocks.{i}."

            def pre(mod: nn.Module, _args, prefix=prefix) -> None:
                with safe_open(path, "pt", device="cuda") as f:
                    sd = {k[len(prefix) :]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}  # noqa: SIM118
                mod.load_state_dict(sd, strict=False, assign=True)

            def post(mod: nn.Module, _args, _out) -> None:
                for m in mod.modules():
                    for n, p in list(m.named_parameters(recurse=False)):
                        setattr(m, n, nn.Parameter(torch.empty_like(p, device="meta"), requires_grad=False))

            block.register_forward_pre_hook(pre)
            block.register_forward_hook(post)
    return dit.eval().requires_grad_(False), txt


def build_vae(config, weights: Path) -> nn.Module:
    from common.config import create_object  # ty: ignore[unresolved-import]
    from safetensors.torch import load_file

    vae = create_object(config.vae.model).requires_grad_(False).eval()
    vae.to(device="cuda", dtype=torch.bfloat16)
    vae.load_state_dict(load_file(weights / "seedvr2-vae.safetensors", device="cuda"))
    vae.set_causal_slicing(**config.vae.slicing)
    vae.set_memory_limit(**config.vae.memory_limit)
    return vae


def cut_videos(videos: torch.Tensor, sp_size: int = 1) -> torch.Tensor:
    """`generation_loop.cut_videos`."""
    t = videos.size(1)
    if t == 1:
        return videos
    if t <= 4 * sp_size:
        padding = torch.cat([videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1), dim=1)
        return torch.cat([videos, padding], dim=1)
    if (t - 1) % (4 * sp_size) == 0:
        return videos
    padding = torch.cat([videos[:, -1].unsqueeze(1)] * (4 * sp_size - ((t - 1) % (4 * sp_size))), dim=1)
    return torch.cat([videos, padding], dim=1)


def main() -> None:  # noqa: PLR0915
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkout", type=Path)
    ap.add_argument("weights", type=Path, help="dir with seedvr2-{3b,7b,vae}.safetensors")
    ap.add_argument("--src", type=Path, help="dir with the original .pth files (RoPE freqs)")
    ap.add_argument("--model", choices=["3b", "7b"], default="3b")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=666)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--runs", type=int, default=1, help="timed calls (time_s: the median after the first)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    os.chdir(a.checkout)
    sys.path.insert(0, str(a.checkout))
    shims()
    from common.config import load_config  # ty: ignore[unresolved-import]
    from data.image.transforms.divisible_crop import DivisibleCrop  # ty: ignore[unresolved-import]
    from data.image.transforms.na_resize import NaResize  # ty: ignore[unresolved-import]
    from data.video.transforms.rearrange import Rearrange  # ty: ignore[unresolved-import]
    from einops import rearrange  # ty: ignore[unresolved-import]
    from projects.video_diffusion_sr.infer import VideoDiffusionInfer  # ty: ignore[unresolved-import]
    from torchvision.transforms import Compose, Lambda, Normalize

    patch_rope()
    patch_posterior()
    from common.distributed import init_torch  # ty: ignore[unresolved-import]

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
    init_torch(cudnn_benchmark=False)  # as configure_runner: TF32 on, a one-process group (VAE slicing asks it)
    config = load_config(f"configs_{a.model}/main.yaml")
    runner = VideoDiffusionInfer(config)
    from omegaconf import OmegaConf  # ty: ignore[unresolved-import]

    OmegaConf.set_readonly(runner.config, False)
    runner.config.diffusion.cfg.scale = 1.0
    runner.config.diffusion.cfg.rescale = 0.0
    runner.config.diffusion.timesteps.sampling.steps = 1
    runner.configure_diffusion()

    video = np.load(a.input)["video"]
    t_in, h_in, w_in = video.shape[:3]
    res_h, res_w = round(h_in * a.scale), round(w_in * a.scale)
    transform = Compose([
        NaResize(resolution=(res_h * res_w) ** 0.5, mode="area", downsample_only=False),
        Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
        DivisibleCrop((16, 16)),
        Normalize(0.5, 0.5),
        Rearrange("t c h w -> c t h w"),
    ])  # fmt: skip

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dit, txt = build_dit(config, a.model, a.weights, a.src, offload=a.offload)
    vae = build_vae(config, a.weights)
    runner.dit, runner.vae = dit, vae
    neg = torch.zeros(64, txt.shape[1], dtype=txt.dtype, device="cuda")  # unused at cfg 1
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0

    x = torch.from_numpy(video).cuda().permute(0, 3, 1, 2) / 255.0
    cond = transform(x)
    ori_length = cond.size(1)
    cond = cut_videos(cond)
    g = torch.Generator().manual_seed(a.seed)
    tl = (cond.size(1) - 1) // 4 + 1
    hl, wl = cond.size(2) // 8, cond.size(3) // 8
    EPS["posterior"] = torch.randn(1, 16, tl, hl, wl, generator=g).to("cuda", torch.bfloat16)
    noise = torch.randn(tl, hl, wl, 16, generator=g).to("cuda", torch.bfloat16)

    stash: dict[str, torch.Tensor] = {}
    dit.register_forward_hook(lambda _m, _a, out: stash.__setitem__("pred", out.vid_sample))
    decode = runner.vae_decode

    def vae_decode(latents: list[torch.Tensor]) -> list[torch.Tensor]:
        stash["x0"] = latents[0]
        return decode(latents)

    runner.vae_decode = vae_decode
    torch.cuda.synchronize()
    cond_noise_scale = 0.0

    def _add_noise(x: torch.Tensor, aug_noise: torch.Tensor) -> torch.Tensor:
        t = torch.tensor([1000.0], device="cuda") * cond_noise_scale
        shape = torch.tensor(x.shape[1:], device="cuda")[None]
        t = runner.timestep_transform(t, shape)
        return runner.schedule.forward(x, aug_noise, t)

    times = []
    for _ in range(a.runs):
        t1 = time.perf_counter()
        if not a.offload:  # as generation_loop: the DiT leaves the GPU while the VAE encodes / decodes
            dit.to("cpu")
        latent = runner.vae_encode([cond])[0]
        if not a.offload:
            dit.to("cuda")
        condition = runner.get_condition(noise, task="sr", latent_blur=_add_noise(latent, torch.zeros_like(latent)))
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
            video_tensors = runner.inference(
                noises=[noise], conditions=[condition], texts_pos=[txt], texts_neg=[neg], dit_offload=not a.offload
            )
        v = video_tensors[0]
        sample = rearrange(v[:, None], "c t h w -> t c h w") if v.ndim == 3 else rearrange(v, "c t h w -> t c h w")
        sample = sample[:ori_length]
        torch.cuda.synchronize()
        t_run = time.perf_counter() - t1
        times.append(t_run)
    t_first, t_run = times[0], statistics.median(times[1:] or times)
    out = sample.clip(-1, 1).mul(0.5).add(0.5).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        a.out,
        out=out.cpu().numpy(),
        sample=sample.float().cpu().numpy(),
        latent=latent.float().cpu().numpy(),
        pred=stash["pred"].float().cpu().numpy(),
        x0=stash["x0"].float().cpu().numpy(),
        load_s=t_load,
        time_s=t_run,
        first_s=t_first,
        peak_alloc=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
    )
    print(
        f"{a.model} {tuple(video.shape)} -> {tuple(out.shape)}: load {t_load:.1f} s, first {t_first:.2f} s,"
        f" run {t_run:.2f} s (median of {max(a.runs - 1, 1)}), peak"
        f" {torch.cuda.max_memory_allocated() / 2**30:.2f} / {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB"
    )


if __name__ == "__main__":
    main()
