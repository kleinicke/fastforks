# ruff: noqa: ANN001, ANN002, ANN201, ANN202, ANN204, D101, D102, D107, C901, PLR0912, PLR0915, PLR2004, SLF001
"""CPU speed harness: fastforks portable vs the pinned upstream original, plus plain-PyTorch CPU variants.

One process = one (side, model, threads); the model is loaded once, then the variants run in sequence (a variant that
changes the network works on a copy or is undone). Appends one JSON line per (variant, kind) to `--out`.

    CUDA_VISIBLE_DEVICES= python tools/bench_cpu.py --side ff|up --model REF --threads 8 --variants base,cl \
        --out x.jsonl

Upstream pins: classification = the library itself (torchvision / timm, the card's arch + the card's weights),
DAv2 = DepthAnything/Depth-Anything-V2 a561b84 (`infer_image`), S2M2 = junhong-3dv/s2m2 8f3bd78 + lookup-fix patch.
Kinds: `fwd` = the network on a prepared tensor (ff: `Model.module()`), `e2e` = image (uint8 array) in, result out.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _repo(env: str) -> Path:
    """The upstream checkout named by environment variable `env` (`reference_tree` applies the pinned commit)."""
    if not os.environ.get(env):
        sys.exit(f"set {env} to a checkout of the upstream repository")
    return Path(os.environ[env])


ap = argparse.ArgumentParser()
ap.add_argument("--side", choices=["ff", "up"], required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--variants", default="base")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--size", default="")  # HxW of the e2e input image
ap.add_argument("--out", required=True)
ap.add_argument("--tag", default="")
ap.add_argument(
    "--ab", action="store_true", help="interleave the variants round by round on the forward pass (paired ratios)"
)
args = ap.parse_args()
os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(args.threads)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools"), str(ROOT / "tools" / "golden")]
torch.set_num_threads(args.threads)
torch.manual_seed(0)
import synth  # noqa: E402

# model -> (task, e2e size, upstream kind)
MODELS = {
    "resnet50.tv2_in1k": ("classification", (480, 640), "lib"),
    "deit_base_patch16_224.fb_in1k": ("classification", (480, 640), "lib"),
    "mobilenet_v2.tv2_in1k": ("classification", (480, 640), "lib"),
    "efficientnet_b0.tv_in1k": ("classification", (480, 640), "lib"),
    "regnety_160.tv2_in1k": ("classification", (480, 640), "lib"),
    "mobilenet_v3_large.tv2_in1k": ("classification", (480, 640), "lib"),
    "mobilenetv4_conv_medium.e500_r256_in1k": ("classification", (480, 640), "lib"),
    "mobilenetv1_100.ra4_e3600_r224_in1k": ("classification", (480, 640), "lib"),
    "efficientnet_v2_s.tv_in1k": ("classification", (480, 640), "lib"),
    "resnet18.tv_in1k": ("classification", (480, 640), "lib"),
    "densenet121.tv_in1k": ("classification", (480, 640), "lib"),
    "convnextv2_base.fcmae_ft_in22k_in1k": ("classification", (480, 640), "lib"),
    "depth-anything-v2-s": ("depth", (480, 640), "dav2"),
    "s2m2-s": ("stereo", (384, 512), "s2m2"),
    "segformer-b0-ade20k": ("segmentation", (512, 512), None),
    "dinov2-vits14": ("features", (518, 518), None),
    "hitnet-eth3d": ("stereo", (384, 512), None),
}


def vm_rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return 0.0


def busy_jiffies() -> float:
    f = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    v = list(map(int, f))
    return sum(v) - v[3] - v[4]  # minus idle, iowait


def wait_quiet(limit_s: float = 240, thresh: float = 0.7) -> float:
    """Wait until other processes use < `thresh` cores over a 3 s window (CPU may be shared); returns the wait."""
    t0 = time.perf_counter()
    while True:
        b0, w0 = busy_jiffies(), time.perf_counter()
        time.sleep(3)
        cores = (busy_jiffies() - b0) / os.sysconf("SC_CLK_TCK") / (time.perf_counter() - w0)
        if cores < thresh or time.perf_counter() - t0 > limit_s:
            return time.perf_counter() - t0


def timeit(f, warm: int, n: int):
    t0 = time.perf_counter()
    out = f()
    first = time.perf_counter() - t0
    b0, p0, w0 = busy_jiffies(), time.process_time(), time.perf_counter()
    for _ in range(warm):
        f()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    wall = time.perf_counter() - w0
    foreign = ((busy_jiffies() - b0) / os.sysconf("SC_CLK_TCK") - (time.process_time() - p0)) / wall
    return out, first, ts, foreign


def tensors(out):
    if isinstance(out, torch.Tensor):
        return [out.float()]
    if isinstance(out, dict):
        return [t for v in out.values() for t in tensors(v)]
    if isinstance(out, (list, tuple)):
        return [t for v in out for t in tensors(v)]
    return []


def weights(task: str, ref: str):
    from fastforks._core import registry  # noqa: PLC0415

    card, variant = registry.resolve(task, ref)[:2]
    return card, variant, registry.weights(card, variant)


# ---------------------------------------------------------------- setups: -> (net, inputs, e2e, extra)
def setup_ff(ref, task, size):
    import importlib  # noqa: PLC0415

    m = importlib.import_module(f"fastforks.{task}").load(ref, mode="portable", device="cpu")
    h, w = size
    if task == "stereo":
        img = tuple(torch.from_numpy(a).clone() for a in synth.pair(h, w))
        img = tuple(a.numpy() for a in img)
        e2e = lambda: m(*img)  # noqa: E731
        x = m.example_inputs(h, w)
    else:
        img = synth.image(h, w)
        e2e = lambda: m(img)  # noqa: E731
        if task == "classification":
            c = m.card.io.crop
            x = m.example_inputs(c, c)
        elif task == "depth":  # DAv2's runner resizes to a multiple of 14 with the shorter side >= 518
            x = m.example_inputs(518, 686)
        else:
            x = m.example_inputs(h, w)
    net = m.module()
    extra = [net, getattr(m._runner, "net", None)]
    return net, x, e2e, [e for e in extra if isinstance(e, torch.nn.Module)]


class Wrap(torch.nn.Module):
    def __init__(self, net, fn):
        super().__init__()
        self.net, self.fn = net, fn

    def forward(self, *a):
        return self.fn(self.net, *a)


def setup_up(ref, task, size, kind):
    card, variant, wfiles = weights(task, ref)
    from fastforks._core.checkpoint import load_state_dict  # noqa: PLC0415

    h, w = size
    if kind == "lib":
        sd = load_state_dict(wfiles[0], strip="")
        if card.params["library"] == "torchvision":
            import torchvision  # noqa: PLC0415

            net = torchvision.models.get_model(variant.params["arch"], weights=None)
            net.load_state_dict(sd, strict=False)  # num_batches_tracked
        else:
            import timm  # noqa: PLC0415

            net = timm.create_model(variant.params["arch"], pretrained=False)
            net.load_state_dict(sd)
        net.eval()
        io = card.io
        from torchvision.transforms.v2 import functional as TF  # noqa: PLC0415

        mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        mode = TF.InterpolationMode.BICUBIC if io.interpolation == "bicubic" else TF.InterpolationMode.BILINEAR
        img = synth.image(h, w)

        def e2e():
            x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
            x = TF.resize(x, [io.resize], interpolation=mode, antialias=True)
            if io.interpolation == "bicubic":
                x = x.clamp(0, 1)
            x = TF.normalize(TF.center_crop(x, [io.crop]), mean, std)
            return net(x)

        x = (torch.randn(1, 3, io.crop, io.crop),)
        return net, x, e2e, [net]
    if kind == "dav2":
        from _common import reference_tree  # noqa: PLC0415

        sys.path.insert(
            0,
            str(
                reference_tree(
                    _repo("FASTFORKS_DAV2_REPO"),
                    "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf",
                )
            ),
        )
        sys.path.remove(str(ROOT / "tools" / "golden"))
        from depth_anything_v2.dpt import DepthAnythingV2  # noqa: PLC0415

        net = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        net.load_state_dict(torch.load(wfiles[0], map_location="cpu", weights_only=True))
        net.eval()
        bgr = np.ascontiguousarray(synth.image(h, w)[..., ::-1])
        e2e = lambda: net.infer_image(bgr, 518)  # noqa: E731
        return net, (torch.rand(1, 3, 518, 686),), e2e, [net]
    if kind == "s2m2":
        import types  # noqa: PLC0415

        from _common import reference_tree  # noqa: PLC0415

        sys.modules["cv2"] = types.ModuleType("cv2")
        sys.path.insert(
            0,
            str(
                reference_tree(
                    _repo("FASTFORKS_S2M2_REPO"),
                    "8f3bd785df593619d955203c6f2c741bef6e31df",
                    "s2m2-lookup-fix.patch",
                )
                / "src"
            ),
        )
        from s2m2.core.model.s2m2 import S2M2  # noqa: PLC0415
        from s2m2.core.utils.image_utils import image_crop, image_pad  # noqa: PLC0415

        net = S2M2(feature_channels=128, dim_expansion=1, num_transformer=1, use_positivity=True, refine_iter=3)
        net.my_load_state_dict(torch.load(wfiles[0], map_location="cpu", weights_only=True, mmap=True)["state_dict"])
        net.eval()

        def fwd(n, left, right):
            d = n(image_pad(left, 32), image_pad(right, 32))[0]
            return image_crop(d, left.shape[-2:])

        left, right = synth.pair(h, w)
        lt, rt = (torch.from_numpy(a).permute(2, 0, 1)[None].float() for a in (left, right))
        wrap = Wrap(net, fwd)
        return wrap, (lt, rt), (lambda: wrap(lt, rt)), [wrap]
    raise SystemExit(kind)


def to_cl(mods, x):
    for m in mods:
        m.to(memory_format=torch.channels_last)
    return tuple(t.contiguous(memory_format=torch.channels_last) if t.dim() == 4 else t for t in x)


def to_nchw(mods, x):
    for m in mods:
        m.to(memory_format=torch.contiguous_format)
    return tuple(t.contiguous() for t in x)


def main() -> None:
    task, esize, kind = MODELS[args.model]
    if args.size:
        esize = tuple(map(int, args.size.split("x")))
    la0 = os.getloadavg()
    t0 = time.perf_counter()
    rss0 = vm_rss_mb()
    with torch.inference_mode():
        if args.side == "ff":
            net, x, e2e, mods = setup_ff(args.model, task, esize)
        else:
            net, x, e2e, mods = setup_up(args.model, task, esize, kind)
        load_s = time.perf_counter() - t0
        head = {
            "side": args.side,
            "model": args.model,
            "threads": args.threads,
            "tag": args.tag,
            "date": time.strftime("%F %T"),
            "torch": torch.__version__,
            "num_threads": torch.get_num_threads(),
            "e2e_size": list(esize),
            "cpu": platform.processor() or "x86_64",
            "affinity": len(os.sched_getaffinity(0)),
            "commit": subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip(),
            "load_s": round(load_s, 2),
            "rss_load_mb": round(vm_rss_mb() - rss0 if False else vm_rss_mb(), 1),
        }
        ref_out = None
        n = args.reps
        if args.ab:
            ab(head, net, x, mods)
            return
        for var in args.variants.split(","):
            xv, fnet, fe2e = x, net, e2e
            extra = {}
            if var in {"denorm", "cl+denorm"}:
                torch.set_flush_denormal(True)
            if var in {"cl", "cl+denorm"}:
                xv = to_cl(mods, x)
            elif var == "bf16":
                fnet = lambda *a, _n=net: _autocast(_n, *a)  # noqa: E731
                fe2e = None
            elif var == "int8":
                fnet = torch.ao.quantization.quantize_dynamic(copy.deepcopy(net), {torch.nn.Linear}, dtype=torch.qint8)
                fe2e = None
            elif var == "compile":
                fnet = torch.compile(net)
                fe2e = None
            for kind_, f in (("fwd", lambda: fnet(*xv)), ("e2e", fe2e)):  # noqa: B023  # f is called within this iteration
                if f is None:
                    continue
                if var == "compile" and kind_ == "fwd":
                    pass
                waited = wait_quiet()
                out, first, ts, foreign = timeit(f, 2, n)
                rec = dict(
                    head,
                    variant=var,
                    kind=kind_,
                    first_s=round(first, 4),
                    median_s=round(statistics.median(ts), 4),
                    min_s=round(min(ts), 4),
                    times=[round(t, 4) for t in ts],
                    n=len(ts),
                    rss_mb=round(vm_rss_mb(), 1),
                    maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
                    load_before=la0[0],
                    load_after=os.getloadavg()[0],
                    foreign_cores=round(foreign, 2),
                    quiet_wait_s=round(waited),
                    **extra,
                )
                if kind_ == "fwd":
                    ts_ = tensors(out)
                    if var == "base":
                        ref_out = [t.clone() for t in ts_]
                    elif ref_out is not None and len(ts_) == len(ref_out):
                        errs = [(a.float() - b.float()).abs().max().item() for a, b in zip(ts_, ref_out, strict=True)]
                        scale = [b.abs().max().item() + 1e-12 for b in ref_out]
                        rec["err_abs"] = max(errs)
                        rec["err_rel"] = max(e / s for e, s in zip(errs, scale, strict=True))
                with Path(args.out).open("a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                print(
                    var,
                    kind_,
                    rec["median_s"],
                    "first",
                    rec["first_s"],
                    "fc",
                    rec["foreign_cores"],
                    rec.get("err_rel"),
                    flush=True,
                )
            if var in {"denorm", "cl+denorm"}:
                torch.set_flush_denormal(False)
            if var in {"cl", "cl+denorm"}:
                to_nchw(mods, x)
            la0 = os.getloadavg()


def ab(head, net, x, mods) -> None:
    """Interleaved variants of the forward pass.

    Every round times each variant once (order rotated), so load drift hits all of them alike; reports median / min
    and the median of the per-round ratio to `base`.
    """
    names = args.variants.split(",")
    fns, xs, extra, inplace = {}, {}, {}, set()
    for v in names:
        xs[v] = x
        if v in {"base", "denorm"}:
            fns[v] = net
        elif v in {"cl", "cl+denorm"}:
            try:
                n2 = copy.deepcopy(net)
                m2 = [n2]
                xs[v] = to_cl(m2, x)
                fns[v] = n2
            except TypeError:  # not copyable (stereo runners): convert the shared module in place before each call
                inplace.add(v)
                fns[v] = net
        elif v == "bf16":
            fns[v] = lambda *a: _autocast(net, *a)
        elif v == "int8":
            fns[v] = torch.ao.quantization.quantize_dynamic(copy.deepcopy(net), {torch.nn.Linear}, dtype=torch.qint8)
        elif v == "compile":
            t0 = time.perf_counter()
            fns[v] = torch.compile(copy.deepcopy(net))
            fns[v](*x)
            extra[v] = {"compile_s": round(time.perf_counter() - t0, 1)}
        else:
            raise SystemExit(v)

    def call(v):
        if inplace:
            xs[v] = (to_cl if v in inplace else to_nchw)(mods, x)
        torch.set_flush_denormal(v.endswith("denorm"))
        t = time.perf_counter()
        out = fns[v](*xs[v])
        dt = time.perf_counter() - t
        torch.set_flush_denormal(False)
        return out, dt

    outs = {}
    for v in names:  # first call + warm-up, outside the timed rounds
        t0 = time.perf_counter()
        outs[v], _ = call(v)
        extra.setdefault(v, {})["first_s"] = round(time.perf_counter() - t0, 4)
        call(v)
        call(v)
    times = {v: [] for v in names}
    b0, p0, w0 = busy_jiffies(), time.process_time(), time.perf_counter()
    waited = wait_quiet()
    b0, p0, w0 = busy_jiffies(), time.process_time(), time.perf_counter()
    for r in range(args.reps):
        for v in names[r % len(names) :] + names[: r % len(names)]:
            times[v].append(call(v)[1])
    foreign = ((busy_jiffies() - b0) / os.sysconf("SC_CLK_TCK") - (time.process_time() - p0)) / (
        time.perf_counter() - w0
    )
    base = times.get("base")
    ref = [t.float() for t in tensors(outs["base"])] if "base" in outs else None
    for v in names:
        rec = dict(
            head,
            variant=v,
            kind="fwd",
            mode="ab",
            median_s=round(statistics.median(times[v]), 4),
            min_s=round(min(times[v]), 4),
            times=[round(t, 4) for t in times[v]],
            n=args.reps,
            foreign_cores=round(foreign, 2),
            quiet_wait_s=round(waited),
            rss_mb=round(vm_rss_mb(), 1),
            maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            load_after=os.getloadavg()[0],
            **extra.get(v, {}),
        )
        if base and v != "base":
            rec["ratio_vs_base"] = round(statistics.median(a / b for a, b in zip(times[v], base, strict=True)), 3)
        if ref and v != "base":
            o = [t.float() for t in tensors(outs[v])]
            errs = [(a - b).abs().max().item() for a, b in zip(o, ref, strict=True)]
            rec["err_abs"] = max(errs)
            rec["err_rel"] = max(e / (b.abs().max().item() + 1e-12) for e, b in zip(errs, ref, strict=True))
        with Path(args.out).open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(
            v,
            rec["median_s"],
            rec["min_s"],
            rec.get("ratio_vs_base"),
            rec.get("err_rel"),
            "fc",
            rec["foreign_cores"],
            flush=True,
        )


def _autocast(net, *a):
    with torch.autocast("cpu", dtype=torch.bfloat16):
        return net(*a)


if __name__ == "__main__":
    main()
