"""SeedVR2 accuracy rows: lib (portable, kernels) and the authors' code (reference npz from tools/seedvr2_reference.py, bf16 autocast)
on (1) DIV2K crops: PSNR / SSIM (Y, `scale` px border) to the HR ground truth + PSNR to the original's output, (2) frame 0 of the
DAVIS / synthetic clips: PSNR to the original's output (RGB, uint8).
python seedvr2_eval.py MODEL(3b|7b) OUT.jsonl --crops d0801 .. --clips synth car-turn .. [--refonly]"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/florian/projects/experiments/017-fastforks-lib/wt-accrows/tools")
from eval_restoration import score

D = Path("/home/florian/.cache/ff_r5_tmp/accrows_sv")
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--crops", nargs="*", default=[]); ap.add_argument("--clips", nargs="*", default=[])
ap.add_argument("--modes", default="portable,kernels")
a = ap.parse_args()
ref = f"seedvr2-{a.model}"
W = Path("/home/florian/projects/experiments/017-fastforks-lib/wt-accrows")
commit = subprocess.run(["git", "-C", str(W), "log", "-1", "--format=%h", "--", "src"], capture_output=True, text=True).stdout.strip()
sets = {}
n = len(a.crops)
if a.crops: sets[f"div2k-val-x2-crop512-n{n}"] = [(c, D / "in" / f"{c}.npz", D / "ref" / f"{a.model}_{c}.npz", np.load(D / "in" / f"{c}_hr.npy")) for c in a.crops]
if a.clips: sets[f"seedvr2-clips-n{len(a.clips)}"] = [(c, D / "in" / f"f1_{c}.npz", D / "ref" / f"{a.model}_f1_{c}.npz", None) for c in a.clips]
def t8(x): return torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1).float().cuda() / 255
def p_orig(o8, r8): return float(10 * np.log10(255**2 / ((o8.astype(np.float64) - r8) ** 2).mean()))
base = {"ref": ref, "task": "restoration", "commit": commit, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "command": "python " + " ".join(["seedvr2_eval.py", *sys.argv[1:]])}
def emit(mode, tier, column, dtype, how, ds, rows, extra=None):
    m = {k: round(float(np.mean([r[k] for r in rows])), 4 if k == "ssim" else 2) for k in rows[0]}
    row = {**base, "tier": tier, "mode": mode, "dtype": dtype, "how": how, "dataset": ds, "n": len(rows), **m, "metrics": m, "column": column, **(extra or {})}
    print(json.dumps(row), flush=True)
    open(a.out, "a").write(json.dumps(row) + "\n")
# the original: against the ground truth only (the clips have none; the original is the reference there)
for ds, items in sets.items():
    if items[0][3] is None: continue
    rows, ms, pk = [], [], []
    for c, _, rf, hr in items:
        z = np.load(rf); o8 = z["out"][0]
        p, s = score(t8(o8), t8(hr), 2)
        rows.append({"psnr": p, "ssim": s}); ms.append(float(z["time_s"]) * 1e3); pk.append(int(z["peak_reserved"]) >> 20)
    emit("upstream-fp16", "upstream16", "original fp16", "bfloat16", "authors' code, bf16 autocast (tools/seedvr2_reference.py" + (", --offload" if a.model == "7b" else "") + ")", ds, rows,
         {"ms_per_image": round(float(np.median(ms)), 1), "peak_reserved_mb": max(pk)})
import fastforks.restoration as R
for mode in a.modes.split(","):
    m = R.load(ref, mode=mode, device="cuda")
    dtype = str(m.backend.dtype).removeprefix("torch.")
    for ds, items in sets.items():
        rows, ms = [], []
        torch.cuda.reset_peak_memory_stats()
        for c, inp, rf, hr in items:
            img = np.load(inp)["video"][0]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = m(img).image
            torch.cuda.synchronize(); ms.append(1e3 * (time.perf_counter() - t0))
            o8 = (out.float().clamp(0, 1) * 255).round().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            r8 = np.load(rf)["out"][0]
            row = {}
            if hr is not None:
                row["psnr"], row["ssim"] = score(t8(o8), t8(hr), 2)
            row["psnr_orig"] = p_orig(o8, r8)
            rows.append(row)
        emit(mode, mode, mode, dtype, None, ds, rows, {"ms_per_image": round(float(np.median(ms[1:] or ms)), 1), "peak_reserved_mb": torch.cuda.max_memory_reserved() >> 20})
    del m; torch.cuda.empty_cache()
