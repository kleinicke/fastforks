"""Stereo accuracy of the originals that are not PyTorch: HITNet (TensorFlow graphs) and CREStereo (MegEngine).

    VENV_TF/python tools/eval_foreign.py hitnet-eth3d upstream --out raw/        # TensorFlow 2.x, GPU
    VENV_MGE/python tools/eval_foreign.py crestereo-eth3d upstream16 --out raw/  # Python 3.10 + megengine + cv2

Same items, masks and metrics as `eval_stereo_depth.py` (whose loaders it imports; it needs no PyTorch), and the same
per-item file `<out>/stereo/<ref>__<tier>.jsonl`, so `eval_stereo_depth.py rows` summarises these runs too.

- HITNet: the authors' frozen graphs `eth3d.pb` (gray, BT.601 luma) / `middlebury_d400.pb` (RGB), images in 0..1 as
  `tools/golden/hitnet.py`; the graph pads and scales itself. `upstream` = float32; `upstream16` = TensorFlow's
  automatic mixed precision graph rewrite (`auto_mixed_precision`, float16; a frozen graph has no own option).
- crestereo-eth3d-pt: the same through the community PyTorch port ibaiGorordo/CREStereo-Pytorch (GPU-capable here; the
  MegEngine build has no sm_120 kernels), weights from its convert_weights.py; `upstream16` = torch.autocast fp16.
- CREStereo: upstream `test.py` `inference` (half resolution, then full resolution from its flow, 20 iterations each)
  on BGR float32 images padded to a multiple of 32 (edge), as `tools/golden/crestereo_mge.py`. `upstream` =
  `Model(mixed_precision=False)`, `upstream16` = the authors' `mixed_precision=True`.

ms = wall time of the inference call per pair (host arrays in, host disparity out, as the authors' scripts run it);
peak = the framework's peak GPU memory where it reports one.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _data import DATA
from eval_stereo_depth import BAD, DATASETS, NONFINITE_MAX, stereo_items

HITNET = {"hitnet-eth3d": ("eth3d.pb", 1), "hitnet-middlebury-d400": ("middlebury_d400.pb", 3)}
CRE_COMMIT = "ad3a1613bdedd88b93247e5f002cb7c80799762d"
CRE_REPO = DATA / "crestereo-src/CREStereo"
CRE_WEIGHTS = DATA / "crestereo-mge/crestereo_eth3d.mge"
CREPT_COMMIT = "b6c7a9fe8dc2e9e56ba7b96f4677312309282d15"
CREPT_REPO = Path(__file__).resolve().parents[1].parent / "ext/CREStereo-Pytorch"  # next to the worktrees
CREPT_WEIGHTS = DATA / "crestereo-pytorch/crestereo_eth3d.pth"  # the port's convert_weights.py output
BUDGET_S = float(os.environ.get("FF_BUDGET_S", "120"))  # a chunk stops with exit 3 after this; the next resumes
_ = NONFINITE_MAX


def _metrics(pred: np.ndarray, gt: np.ndarray, ds: str) -> dict[str, Any]:
    fin = np.isfinite(pred)
    row: dict[str, Any] = {"nonfinite": round(1 - float(fin.mean()), 6)}
    p = np.where(fin, pred, 0.0)
    thr, vmax = BAD[ds]
    v = np.isfinite(gt) & (gt < vmax)
    e = np.abs(p - gt)[v]
    return row | {"epe": float(e.mean()), "bad": float((e > thr).mean()), "n_px": int(v.sum())}


def _preload_pip_cuda() -> None:
    """Load the pip CUDA libraries (site-packages/nvidia/*/lib) by path before TensorFlow looks them up by soname.

    TensorFlow's rpath misses some of them (libcusolver.so.11 in TF 2.21): it then silently runs on the CPU.
    """
    import ctypes  # noqa: PLC0415
    import site  # noqa: PLC0415

    todo = sorted({p for sp in site.getsitepackages() for p in Path(sp).glob("nvidia/*/lib/lib*.so*")})
    while todo:  # retry until no library loads: a few depend on others of the set
        left = []
        for p in todo:
            try:
                ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                left.append(p)
        if len(left) == len(todo):
            return
        todo = left


class _Hitnet:
    def __init__(self, ref: str, tier: str) -> None:
        _preload_pip_cuda()
        import tensorflow as tf  # noqa: PLC0415 - reference-only dependency (its own venv)  # ty: ignore[unresolved-import]

        self.tf = tf
        file, self.channels = HITNET[ref]
        gd = tf.compat.v1.GraphDef()
        gd.ParseFromString((DATA / "hitnet-pb" / file).read_bytes())
        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(gd, name="graph")
        cfg = tf.compat.v1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        if tier == "upstream16":
            from tensorflow.core.protobuf import rewriter_config_pb2  # noqa: PLC0415  # ty: ignore[unresolved-import]

            cfg.graph_options.rewrite_options.auto_mixed_precision = rewriter_config_pb2.RewriterConfig.ON
        self.sess = tf.compat.v1.Session(graph=graph, config=cfg)
        self.inp = graph.get_tensor_by_name("graph/input:0")
        self.out = graph.get_tensor_by_name("graph/reference_output_disparity:0")
        self.dtype = "float16" if tier == "upstream16" else "float32"
        self.how = "autocast" if tier == "upstream16" else "fp32"
        self.note = f"TensorFlow {tf.__version__} frozen graph {file}" + (
            ", auto_mixed_precision graph rewrite (float16)" if tier == "upstream16" else ", float32"
        )
        self.gpu = bool(tf.config.list_physical_devices("GPU"))

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        ims = [im.astype(np.float32) for im in (left, right)]
        if self.channels == 1:
            ims = [(im @ np.array([0.299, 0.587, 0.114], np.float32))[..., None] for im in ims]
        x = np.concatenate(ims, -1)[None] / 255.0
        return self.sess.run(self.out, {self.inp: x.astype(np.float32)})[0, :, :, 0]

    def reset_peak(self) -> None:
        if self.gpu:
            self.tf.config.experimental.reset_memory_stats("GPU:0")

    def peak(self) -> tuple[int | None, int | None]:
        if not self.gpu:
            return None, None
        return round(self.tf.config.experimental.get_memory_info("GPU:0")["peak"] / 2**20), None


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> object:
        allowed = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,  # noqa: SLF001
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        try:
            return allowed[module, name]
        except KeyError:
            msg = f"refused global {module}.{name}"
            raise pickle.UnpicklingError(msg) from None


class _Crestereo:
    def __init__(self, _ref: str, tier: str) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="ff-cre-")
        archive = subprocess.run(["git", "-C", CRE_REPO, "archive", CRE_COMMIT], check=True, capture_output=True)
        with tarfile.open(fileobj=__import__("io").BytesIO(archive.stdout)) as t:
            t.extractall(self.tmp.name, filter="data")
        sys.path.insert(0, self.tmp.name)
        import megengine as mge  # noqa: PLC0415 - reference-only dependency (its own venv)  # ty: ignore[unresolved-import]
        import megengine.functional as mf  # noqa: PLC0415  # ty: ignore[unresolved-import]

        pad = mf.pad

        def pad_compat(src: object, pad_witdth: object = None, **kw: object) -> object:  # upstream's 1.8 keyword
            if pad_witdth is not None:
                kw["pad_width"] = pad_witdth
            return pad(src, **kw)

        mf.pad = pad_compat
        import test  # noqa: PLC0415  # upstream test.py  # ty: ignore[unresolved-import]
        from nets import Model  # noqa: PLC0415  # ty: ignore[unresolved-import]

        self.mge, self.test = mge, test
        with CRE_WEIGHTS.open("rb") as f:
            sd = _Unpickler(f).load()
        self.model = Model(max_disp=256, mixed_precision=tier == "upstream16", test_mode=True)
        self.model.load_state_dict(sd["state_dict"], strict=True)
        self.model.eval()
        self.dtype = "float16" if tier == "upstream16" else "float32"
        self.how = "authors flag" if tier == "upstream16" else "fp32"
        self.note = f"MegEngine {mge.__version__}, test.py inference (2 x 20 iters), BGR, pad 32 (edge)" + (
            ", mixed_precision=True" if tier == "upstream16" else ""
        )
        self.gpu = mge.is_cuda_available()

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lb, rb = (x[..., ::-1].astype(np.float32) for x in (left, right))
        h, w = lb.shape[:2]
        ph, pw = (-h) % 32, (-w) % 32
        pads = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0))
        disp = self.test.inference(np.pad(lb, pads, mode="edge"), np.pad(rb, pads, mode="edge"), self.model, n_iter=20)
        disp = np.asarray(disp, np.float32)
        return disp[pads[0][0] : pads[0][0] + h, pads[1][0] : pads[1][0] + w]

    def reset_peak(self) -> None:
        dev = getattr(self.mge, "device", None)
        if dev is not None and hasattr(dev, "reset_max_memory_stats"):
            dev.reset_max_memory_stats()

    def peak(self) -> tuple[int | None, int | None]:
        dev = getattr(self.mge, "device", None)
        try:
            return round(dev.get_max_allocated_memory() / 2**20), round(dev.get_max_reserved_memory() / 2**20)
        except AttributeError:
            return None, None


class _CrestereoPt:
    """ibaiGorordo/CREStereo-Pytorch (community PyTorch port of the original; runs on current GPUs): test_model.py."""

    commit = CREPT_COMMIT

    def __init__(self, _ref: str, tier: str) -> None:
        import torch  # noqa: PLC0415

        sys.path.insert(0, str(CREPT_REPO))
        from nets import Model  # noqa: PLC0415  # ty: ignore[unresolved-import]

        self.torch = torch
        self.model = Model(max_disp=256, mixed_precision=False, test_mode=True)
        self.model.load_state_dict(torch.load(CREPT_WEIGHTS, weights_only=True), strict=True)
        self.model.cuda().eval()
        self.fp16 = tier == "upstream16"
        self.dtype = "float16" if self.fp16 else "float32"
        self.how = "torch.autocast fp16" if self.fp16 else "fp32"
        self.note = (
            f"PyTorch {torch.__version__}, community port ibaiGorordo/CREStereo-Pytorch @ {CREPT_COMMIT[:7]}, "
            "test_model.py inference (2 x 20 iters), BGR, pad 32 (edge)"
            + (", torch.autocast fp16" if self.fp16 else "")
        )
        self.gpu = True

    def __call__(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        torch = self.torch
        lb, rb = (x[..., ::-1].astype(np.float32) for x in (left, right))
        h, w = lb.shape[:2]
        ph, pw = (-h) % 32, (-w) % 32
        pads = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0))
        img = [torch.from_numpy(np.ascontiguousarray(np.pad(x, pads, mode="edge").transpose(2, 0, 1)[None])).cuda()
               for x in (lb, rb)]  # fmt: skip
        half = [torch.nn.functional.interpolate(x, size=(x.shape[2] // 2, x.shape[3] // 2), mode="bilinear",
                                                align_corners=True) for x in img]  # fmt: skip
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            flow = self.model(half[0], half[1], iters=20, flow_init=None)
            flow = self.model(img[0], img[1], iters=20, flow_init=flow)
        disp = flow[0, 0].float().cpu().numpy()
        return disp[pads[0][0] : pads[0][0] + h, pads[1][0] : pads[1][0] + w]

    def reset_peak(self) -> None:
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def peak(self) -> tuple[int | None, int | None]:
        c = self.torch.cuda
        return round(c.max_memory_allocated() / 2**20), round(c.max_memory_reserved() / 2**20)


def _net(ref: str) -> type:
    return _Hitnet if ref in HITNET else _CrestereoPt if ref.endswith("-pt") else _Crestereo


def main() -> None:
    """Command line entry."""
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", choices=[*HITNET, "crestereo-eth3d", "crestereo-eth3d-pt"])
    ap.add_argument("tiers")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speed", action="store_true", help="speed rows (bench_upstream inputs and protocol) to --out")
    a = ap.parse_args()
    t_start = time.perf_counter()
    if a.speed:
        _speed(a)
        return
    out = Path(a.out) / "stereo"
    out.mkdir(parents=True, exist_ok=True)
    items = [(ds, k, ld) for ds in DATASETS["stereo"] for k, ld in stereo_items(ds)[: a.limit or None]]
    for tier in a.tiers.split(","):
        per = out / f"{a.ref}__{tier}.jsonl"
        rows = [json.loads(x) for x in per.read_text().splitlines()] if per.exists() else []
        done = {(r["ds"], r["key"]) for r in rows if "key" in r}
        todo = [x for x in items if (x[0], x[1]) not in done]
        if not todo or any("error" in r for r in rows):
            continue
        t0 = time.perf_counter()
        net = _net(a.ref)(a.ref, tier)
        meta = {"meta": True, "ref": a.ref, "tier": tier, "mode": "upstream", "dtype": net.dtype, "how": net.how,
                "note": net.note + ("" if net.gpu else ", CPU (no GPU device)"), "backend": f"original {net.how}",
                "kind": "", "build_s": round(time.perf_counter() - t0, 1), "commit": _commit(),
                "gpu": os.environ.get("FF_GPU_NAME", "NVIDIA GeForce RTX 5060 Ti"), "torch": None,
                "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "command": " ".join(["python", "tools/eval_foreign.py", *sys.argv[1:]])}  # fmt: skip
        with per.open("a") as f:
            f.write(json.dumps(meta) + "\n")
            first = True
            for ds, key, ld in todo:
                if time.perf_counter() - t_start > BUDGET_S:
                    sys.exit(3)  # budget hit: the caller starts the next chunk, which resumes from this file
                left, right, gt = ld()
                net.reset_peak()
                tc = time.perf_counter()
                try:
                    pred = net(left, right)
                except Exception as e:  # noqa: BLE001 - recorded
                    f.write(json.dumps({"ds": ds, "key": key, "error": f"call: {type(e).__name__}: {e}"[:400]}) + "\n")
                    break
                ms = 1e3 * (time.perf_counter() - tc)
                pa, pr = net.peak()
                row = {"ds": ds, "key": key, "ms": round(ms, 2), "first": first, "peak_alloc_mb": pa,
                       "peak_reserved_mb": pr} | _metrics(pred, gt, ds)  # fmt: skip
                first = False
                f.write(json.dumps(row) + "\n")
                f.flush()
                print(a.ref, tier, ds, key, round(ms, 1), round(row["bad"] * 100, 2), flush=True)


SPEED_INPUTS = (
    ("375x1242", "KITTI 2015 000000_10", DATA / "stereo/KITTI/training/image_{}/000000_10.png", ("2", "3")),
    ("994x1482", "Middlebury 2014 trainingH Motorcycle", DATA / "stereo/Middlebury/MiddEval3/trainingH/Motorcycle/im{}.png",  # noqa: E501
     ("0", "1")),
)  # fmt: skip


def _speed(a: argparse.Namespace, warmup: int = 3, runs: int = 5) -> None:
    """`--speed`: median of 5 after 3 warm-up per tier and input, as bench/bench_upstream.py; rows appended to --out."""
    from PIL import Image  # noqa: PLC0415

    out = Path(a.out)
    have = (
        {(r["ref"], r["tier"], r["input"]) for r in map(json.loads, out.read_text().splitlines())}
        if out.exists()
        else set()
    )
    for tier in a.tiers.split(","):
        t0 = time.perf_counter()
        net = _net(a.ref)(a.ref, tier)
        load_s = time.perf_counter() - t0
        for label, scene, pat, sides in SPEED_INPUTS:
            if (a.ref, tier, label) in have:
                continue
            left, right = (np.asarray(Image.open(str(pat).format(s)).convert("RGB")) for s in sides)
            net.reset_peak()
            tc = time.perf_counter()
            net(left, right)
            first = time.perf_counter() - tc
            pf = net.peak()
            for _ in range(warmup):
                net(left, right)
            ts = []
            for _ in range(runs):
                tc = time.perf_counter()
                net(left, right)
                ts.append(time.perf_counter() - tc)
            pa, pr = net.peak()
            row = {"ref": a.ref, "task": "stereo", "mode": "upstream", "tier": tier, "dtype": net.dtype, "how": net.how,
                   "note": net.note + ("" if net.gpu else ", CPU (no GPU device)"), "input": label, "scene": scene,
                   "ms": round(1e3 * sorted(ts)[len(ts) // 2], 2), "ms_min": round(1e3 * min(ts), 2), "runs": runs,
                   "load_s": round(load_s, 2), "first_call_s": round(first, 3), "peak_first_alloc_mb": pf[0],
                   "peak_first_reserved_mb": pf[1], "peak_alloc_mb": pa, "peak_reserved_mb": pr,
                   "protocol": f"median of {runs} after {warmup} warm-up, host uint8 images in, host disparity out "
                   "(as the authors' scripts), original code at the pinned commit; peak = the framework's peak",
                   "gpu": os.environ.get("FF_GPU_NAME", "NVIDIA GeForce RTX 5060 Ti"), "torch": None,
                   "date": time.strftime("%Y-%m-%d"),
                   "commit": getattr(net, "commit", CRE_COMMIT if a.ref == "crestereo-eth3d" else ""),
                   "ff_commit": _commit(), "source": "",
                   "command": " ".join(["python", "tools/eval_foreign.py", *sys.argv[1:]])}  # fmt: skip
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


def _commit() -> str:
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        ["git", "-C", str(root), "log", "-1", "--format=%h", "--", "src"], capture_output=True, text=True, check=False
    ).stdout.strip()


if __name__ == "__main__":
    main()
