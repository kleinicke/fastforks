"""Append the gap-solve rows to docs/data (new (ref, mode, side) keys only, never an existing row) and the
`unsupported:` rows for TRT export/build failures that are not fixable in the harness. Run from the worktree root."""
import json, sys
from pathlib import Path
SV = Path("results/raw/2026-11-15_gaps/solve")
rd = lambda p: [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
UNSUP = {
 "xfeat": "torch.export: data-dependent guard in the NMS / keypoint selection",
 "aliked-lightglue": "torch.export: data-dependent guard (masks.sum() == 0 branch)",
 "loftr-outdoor": "torch.export: data-dependent guard in loftr.py:190",
 "geocalib-pinhole": "torch.export: data-dependent guard ((norm_x == 0).any())",
 "track-on2": "torch.export: data-dependent guard in model.py:107",
 "grounding-dino-tiny": "ONNX export: aten._is_all_true has no ONNX decomposition",
 "gmflow-scale2-sintel": "TensorRT 10 Myelin compiler error CHECK(is_const()) on the attention graph",
 "las2-h": "TensorRT 10 Myelin autotuner exception (all tactics skipped)",
 "bridgedepth-sf": "TensorRT 10 Myelin tactic check fails during the engine build",
 "dedode-v2-b": "TensorRT ITopK supports K <= 3840 only; the keypoint top-k exceeds it",
 "raft-small": "engine build needs more than the 16 GB of this GPU at 1080x1920 (CUDA OOM in the builder)",
 "memfof-kitti": "engine build needs more than the 16 GB of this GPU (CUDA OOM in the builder)",
 "dfine-x-coco": "fp16 engine output is all NaN (strongly-typed fp16 overflow in the decoder)",
 "c-foundationstereo-s": None,
}
def merge(dst, rows, src, cmd):
    d = Path(dst); have = {(r["ref"], r["mode"], r.get("side")) for r in rd(d)}; new = {}
    for r in rows:
        k = (r["ref"], r["mode"], r.get("side"))
        if k in have or str(r.get("fails", "")).startswith(("load: Weights", "load: Licence", "OSError")): continue
        r["source"] = src; r["command"] = cmd; new[k] = r
    with d.open("a") as f:
        for r in new.values(): f.write(json.dumps(r) + "\n")
    print(dst, "appended", len(new))
C = rd(SV / "rows_compile.jsonl"); P = rd(SV / "rows_portable.jsonl")
# a successful retry replaces a diagnosis failure: keep only the last row per key
diag = [r for r in rd(SV / "diag1.jsonl") if r["mode"] == "tensorrt" and r["ref"] in UNSUP and UNSUP[r["ref"]]]
good = {(r["ref"], r["mode"]) for r in C if not r.get("fails")}
for r in diag:
    if (r["ref"], r["mode"]) in good: continue
    r["fails"] = "unsupported: " + UNSUP[r["ref"]]; C.append(r)
cmd = "tools/speed_check.py run --sides lib --no-profile --prefix 'gpulock-verylow timeout 175' (results/raw/2026-11-15_gaps/solve/master.sh)"
merge("docs/data/speed_compile.jsonl", C, "results/raw/2026-11-15_gaps/solve/rows_compile.jsonl", cmd)
merge("docs/data/speed.jsonl", P, "results/raw/2026-11-15_gaps/solve/rows_portable.jsonl", cmd)
