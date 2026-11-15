"""Video Depth Anything speed rows in the format of the VDA-s rows: one 88-frame video per call (TUM fr1_room), ms per video.

    python vda_speed.py REF TIER INPUT OUT.jsonl      TIER upstream|upstream16|portable|kernels, INPUT 480x640|1080x1920
Protocol: first call, 1 warm-up, peak reset, median of 3. FF_VDA_REPO = upstream clone at the card's pin.
"""
import datetime as dt, json, statistics, sys, time
from pathlib import Path
import cv2, numpy as np, torch
sys.path.insert(0, str(Path.cwd() / "tools"))
import eval_video_depth as e
ref, tier, inp, out = sys.argv[1:5]
frames, _ = e.load_sequence("freiburg1_room", 88)
scene = "TUM RGB-D fr1_room, first 88 frames"
if inp == "1080x1920":
    frames = np.stack([cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_CUBIC) for f in frames]); scene += " (bicubic upscaled)"
t0 = time.perf_counter(); fn = e.predictor(ref, tier); torch.cuda.synchronize(); load_s = time.perf_counter() - t0
def call():
    torch.cuda.synchronize(); t = time.perf_counter(); fn(frames); torch.cuda.synchronize(); return time.perf_counter() - t
torch.cuda.reset_peak_memory_stats(); first = call(); call(); torch.cuda.reset_peak_memory_stats()
ts = [call() for _ in range(3)]
mode, dtype, how, _ = e.TIERS[tier]
row = {"ref": ref, "task": "depth", "mode": "upstream" if tier.startswith("upstream") else mode, "tier": tier, "dtype": dtype, "how": how,
       "input": inp, "frames": len(frames), "scene": scene, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
       "date": dt.date.today().isoformat(), "protocol": "one video per call (4 windows of 32 frames), first call, 1 warm-up, peak reset, median of 3; ms per video",
       "command": "results/raw/2026-11-15_gaps/solve/vda_speed.py " + " ".join(sys.argv[1:]), "load_s": round(load_s, 2), "first_call_s": round(first, 3),
       "ms": round(1e3 * statistics.median(ts), 1), "ms_min": round(1e3 * min(ts), 1), "runs": 3, "ms_per_frame": round(1e3 * statistics.median(ts) / len(frames), 2),
       "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20), "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)}
if tier.startswith("upstream"):
    row["note"] = f"upstream infer_video_depth(input_size=518, fp32={tier=='upstream'}): uint8 numpy frames in, numpy out (its API)" + ("; the authors' autocast fp16" if tier == "upstream16" else "")
    row["commit"] = "4f5ae23172ba60fd7bc11ef671cca678842c7072"
print(json.dumps(row)); open(out, "a").write(json.dumps(row) + "\n")
