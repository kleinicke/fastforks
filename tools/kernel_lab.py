"""Kernel lab: variants of one model's `kernels` forward at sustained load (ms and J per forward, clocks, error).

    uv run python tools/kernel_lab.py --spec lab.json [--out result.json] [--image im0.png]
    uvx modal run tools/modal_survey.py::lab --gpu L4 --spec lab.json --out result.json   # on a Modal GPU

The spec is JSON: `{"model": "depth-pro", "variants": [...], "sources": {"name": "path of a dpt_fused.cu"}}`.
A variant is `{"name", "gemm": {...}, "conv": {...}, "attn": ..., "lib": "base" | <source>, "streams": true}`, every
key but the name optional; `attn`: "flash64" (the model's, checked), "sdpa", or a fixed flash64 variant "hh" / "fh" /
"ff". `gemm` maps `"MxN"` (exact problem) / `"gelu"` / `"default"`
to a tile configuration (`dpt_fused.cu` cases), `conv` maps `"MxK"` / `"xK"` (any M) / `"default"`; what no key
covers keeps the library's rule (`gemm_cfg`, `conv_cfg`). `lib` names a source built here for this GPU (needs
nvcc + CUTLASS: `FASTFORKS_LAB_CUTLASS`). The first variant is the base the others' outputs are compared with.

Timing: a warm-up, then rounds; each round runs every variant (shuffled) for a few unmeasured transition forwards
and `PER` timed forwards back to back, so all variants see the same thermal / power-cap state. `nvidia-smi` samples
SM clock and power every 100 ms; J per forward = mean power in a variant's windows x its time per forward.
Error: max / mean |inverse depth - reference| over the reference's mean, against `portable` fp32 and the base.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import tomllib
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

import fastforks
import fastforks.depth.models.depth_anything_v2.fused as da2_fused
import fastforks.kernels.dpt_fused as dpt
from fastforks.kernels.dpt_fused import ACT_GELU, DptFused
from fastforks.kernels.fastbb import FLASH_VARIANTS, Flash64

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

ROOT = Path(__file__).resolve().parents[1]
WARM_S, ROUNDS, PER, TRANSITION = 20.0, 4, 8, 2


class Smi:
    """`nvidia-smi` samples `(monotonic s, SM MHz, W, C)` every 100 ms in a reader thread."""

    def __init__(self) -> None:
        """Start polling."""
        q = "--query-gpu=clocks.sm,power.draw,temperature.gpu"
        cmd = ["nvidia-smi", q, "--format=csv,noheader,nounits", "-lms", "100"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        self.rows: list[tuple[float, float, float, float]] = []
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self) -> None:
        assert self.p.stdout is not None  # noqa: S101 - Popen(stdout=PIPE)
        for ln in self.p.stdout:
            with contextlib.suppress(ValueError):
                mhz, w, c = (float(v) for v in ln.split(","))
                self.rows.append((time.monotonic(), mhz, w, c))

    def window(self, t0: float, t1: float) -> list[tuple[float, float, float, float]]:
        """Samples taken in `[t0, t1]`."""
        return [r for r in self.rows if t0 <= r[0] <= t1]

    def stop(self) -> None:
        """Stop polling."""
        self.p.terminate()


def build(name: str, src: Path, out_dir: Path) -> Path:
    """`dpt_fused.cu` variant `src` -> a shared library for this GPU (the release build's flags, one target)."""
    import build_kernels as bk  # noqa: PLC0415 - tools/, needs the kernels dependency group

    lib_dir = ROOT / "src/fastforks/kernels/dpt_fused"
    cfg = tomllib.loads((lib_dir / "build.toml").read_text())
    cap = torch.cuda.get_device_capability()
    target = f"sm_{cap[0]}{cap[1]}"
    cuda = (
        Path(os.environ.get("FASTFORKS_LAB_CUDA", "")) if os.environ.get("FASTFORKS_LAB_CUDA") else bk.pip_cuda_home()
    )
    cutlass = Path(os.environ["FASTFORKS_LAB_CUTLASS"])
    stage = out_dir / name
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "dpt_fused.cu").write_bytes(src.read_bytes())
    exports = stage / "exports.map"
    exports.write_text(f"{{ global: {'; '.join(bk.c_exports(stage, cfg))}; local: *; }};\n")
    so = stage / f"{name}-{target}.so"
    tools = os.pathsep.join(str(d) for d in (cuda / "bin", cuda / "nvvm" / "bin"))
    env = dict(os.environ, PATH=f"{tools}{os.pathsep}{os.environ.get('PATH', '')}")
    t0 = time.monotonic()
    cmd = bk.nvcc_cmd(cuda, stage, cfg, target, so, cutlass, "g++", exports)
    subprocess.run(cmd, check=True, env=env)
    print(f"built {name} for {target} in {time.monotonic() - t0:.0f} s", flush=True)
    return so


def load_lib(so: Path, device: torch.device) -> ctypes.CDLL:
    """A `dpt_fused` library from `so`, with the argument types `DptFused` sets."""
    orig = dpt.shared_library
    dpt.shared_library = lambda _m, _d: (ctypes.CDLL(str(so)), f"lab:{so.name}")
    try:
        return DptFused(device).lib
    finally:
        dpt.shared_library = orig


def _sdpa(qkv: torch.Tensor, scale: float, _attn: Any) -> torch.Tensor:  # noqa: ANN401 - Flash64, unused
    b, n, _, h, d = qkv.shape
    q, k, v = (t.transpose(1, 2) for t in qkv.unbind(2))
    return F.scaled_dot_product_attention(q, k, v, scale=scale).transpose(1, 2).reshape(b, n, h * d)


class Lab:
    """One loaded `kernels` model whose tile rule, attention, library and streams a variant sets."""

    def __init__(self, model: str, libs: dict[str, ctypes.CDLL]) -> None:
        """Load `model` in `kernels` mode, fp16."""
        self.model = fastforks.load(model, mode="kernels", device="cuda", dtype=torch.float16)
        self.net = self.model._runner.net  # noqa: SLF001 - the kernels network
        self.k: DptFused = self.net.k
        self.side = getattr(self.net, "streams", None)
        self.libs = {"base": self.k.lib} | libs
        self.flash = da2_fused._flash  # noqa: SLF001
        self.rules = (DptFused.gemm_cfg.__get__(self.k), DptFused.conv_cfg.__get__(self.k))
        self.seen: Counter[str] = Counter()
        self.fixed: dict[str, Flash64] = {}

    def apply(self, v: dict) -> None:
        """Set variant `v` (unset keys: the library's defaults)."""
        g, c = v.get("gemm", {}), v.get("conv", {})
        rule_g, rule_c = self.rules

        def gemm_cfg(m: int, n: int, act: int) -> int:
            key = f"{m}x{n}"
            self.seen[f"gemm {key}{' gelu' if act == ACT_GELU else ''} rule {rule_g(m, n, act)}"] += 1
            if key in g:
                return g[key]
            if act == ACT_GELU and "gelu" in g:
                return g["gelu"]
            return g.get("default", rule_g(m, n, act))

        def conv_cfg(m: int, k: int) -> int:
            self.seen[f"conv {m}x{k} rule {rule_c(m, k)}"] += 1
            return c.get(f"{m}x{k}", c.get(f"x{k}", c.get("default", rule_c(m, k))))

        self.k.gemm_cfg = gemm_cfg  # ty: ignore[invalid-assignment]
        self.k.conv_cfg = conv_cfg  # ty: ignore[invalid-assignment]
        self.k.lib = self.libs[v.get("lib", "base")]
        a = v.get("attn", "flash64")
        if a in FLASH_VARIANTS:  # one fixed flash64 variant, unchecked
            fl = self.fixed.setdefault(a, Flash64(a, probe=0))
            da2_fused._flash = lambda qkv, scale, _attn: self.flash(qkv, scale, fl)  # noqa: SLF001
        else:
            da2_fused._flash = _sdpa if a == "sdpa" else self.flash  # noqa: SLF001
        if hasattr(self.net, "streams"):
            self.net.streams = self.side if v.get("streams", True) else None


def _ms(f: Callable[[], Any]) -> float:
    a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    f()
    z.record()
    torch.cuda.synchronize()
    return a.elapsed_time(z)


def _err(a: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    d = (a.float() - ref.float()).abs()
    s = ref.float().abs().mean().item()
    return {"max": round(d.max().item() / s, 6), "mean": round(d.mean().item() / s, 7)}


def _depth(out: Any) -> torch.Tensor:  # noqa: ANN401 - a task result (DepthResult)
    """The inverse depth of a result (the network's own output; depth also carries the FOV estimate)."""
    inv = getattr(out, "inverse_depth", None)
    return inv if inv is not None else out.depth


@contextlib.contextmanager
def _seeded() -> Iterator[None]:
    st = random.getstate()
    random.seed(0)
    try:
        yield
    finally:
        random.setstate(st)


@torch.inference_mode()
def run(spec: dict, img: torch.Tensor, libs: dict[str, ctypes.CDLL]) -> dict:
    model = spec.get("model", "depth-pro")
    variants = spec["variants"]
    ref32 = fastforks.load(model, mode="portable", device="cuda", dtype=torch.float32)
    ref = _depth(ref32(img)).clone()
    del ref32
    torch.cuda.empty_cache()
    lab = Lab(model, libs)
    f = lambda: lab.model(img)  # noqa: E731
    out: dict[str, Any] = {}
    base = None
    for v in variants:  # outputs (also: every variant runs once before timing)
        lab.apply(v)
        d = _depth(f()).clone()
        base = d if base is None else base
        out[v["name"]] = {"spec": v, "err_fp32": _err(d, ref), "err_base": _err(d, base)}
        del d
    lab.apply(variants[0])
    lab.seen.clear()
    f()
    seen = dict(lab.seen)  # the problems one forward runs (keys for `gemm` / `conv`), with today's rule
    smi = Smi()
    lab.apply(variants[0])
    t_end = time.monotonic() + WARM_S
    while time.monotonic() < t_end:
        f()
    torch.cuda.synchronize()
    times: dict[str, list[float]] = {v["name"]: [] for v in variants}
    windows: dict[str, list[tuple[float, float, int]]] = {v["name"]: [] for v in variants}
    with _seeded():
        for r in range(ROUNDS):
            order = list(variants)
            random.shuffle(order)
            for v in order:
                lab.apply(v)
                for _ in range(TRANSITION):
                    f()
                torch.cuda.synchronize()
                t0 = time.monotonic()
                ts = [_ms(f) for _ in range(PER)]
                t1 = time.monotonic()
                times[v["name"]] += ts
                windows[v["name"]].append((t0, t1, PER))
            print(f"round {r + 1}/{ROUNDS} done", flush=True)
    smi.stop()
    lab.apply(variants[0])
    for name, ws in windows.items():
        samples = [s for t0, t1, _ in ws for s in smi.window(t0, t1)]
        wall_s = sum(t1 - t0 for t0, t1, _ in ws)
        n = sum(c for *_, c in ws)
        w = statistics.mean(s[2] for s in samples) if samples else float("nan")
        ms = statistics.median(times[name])
        out[name] |= {
            "ms": round(ms, 1),
            "ms_mean": round(statistics.mean(times[name]), 1),
            "ms_min": round(min(times[name]), 1),
            "sm_mhz": round(statistics.mean(s[1] for s in samples), 0) if samples else None,
            "power_w": round(w, 1),
            "temp_c": round(statistics.mean(s[3] for s in samples), 1) if samples else None,
            "j_per_fwd": round(w * wall_s / n, 2),
            "samples": len(samples),
        }
    p = torch.cuda.get_device_properties(0)
    head = {"gpu": p.name, "sms": p.multi_processor_count, "torch": str(torch.__version__), "model": model}
    return head | {"problems": seen, "variants": out}


def _image(path: str | None) -> torch.Tensor:
    if path and Path(path).is_file():
        from torchvision.io import read_image  # noqa: PLC0415 - the vision group

        return (read_image(path)[:3].float() / 255).cuda()
    print("no image: a fixed random one", file=sys.stderr)
    return torch.rand(3, 994, 1482, device="cuda", generator=torch.Generator("cuda").manual_seed(0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--spec", required=True, help="variants JSON (see the module docstring)")
    ap.add_argument("--image", help="input image (default: a fixed random one)")
    ap.add_argument("--out", help="write the result as JSON here")
    ap.add_argument("--build-dir", default="/tmp/fastforks-lab")  # noqa: S108 - scratch builds
    a = ap.parse_args()
    spec_path = Path(a.spec)
    spec = json.loads(spec_path.read_text())
    dev = torch.device("cuda")
    libs = {}
    for name, src in spec.get("sources", {}).items():
        libs[name] = load_lib(build(name, (spec_path.parent / src).resolve(), Path(a.build_dir)), dev)
    r = run(spec, _image(a.image), libs)
    print(json.dumps(r["problems"], indent=0), flush=True)
    for name, v in r["variants"].items():
        print(f"{name:24s} {v['ms']:7.1f} ms {v['j_per_fwd']:6.2f} J {v['sm_mhz']} MHz {v['power_w']} W "
              f"err fp32 {v['err_fp32']} base {v['err_base']}", flush=True)  # fmt: skip
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
