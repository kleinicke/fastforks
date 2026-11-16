"""Raw files of this run -> docs/data eval rows (appended; replaces rows of the same ref, mode, dataset)."""
import json, sys
from pathlib import Path
R = Path(__file__).parent
D = R.parents[2] / "docs/data"
COL = {"upstream": "original fp32", "upstream-fp16": "original fp16", "portable": "portable", "kernels": "kernels"}
TIER = {"upstream": "upstream", "upstream-fp16": "upstream16", "portable": "portable", "kernels": "kernels"}
G = dict(commit="33b2432c", gpu="NVIDIA GeForce RTX 5060 Ti", torch="2.14.1+cu130", date="2026-11-16")

def put(task, rows):
    f = D / f"{task}_eval.jsonl"
    old = [json.loads(l) for l in open(f)]
    keys = {(r["ref"], r["mode"], r["dataset"]) for r in rows}
    old = [r for r in old if (r["ref"], r["mode"], r["dataset"]) not in keys]
    with open(f, "w") as o:
        for r in old + rows:
            o.write(json.dumps(r) + "\n")
    print(task, len(rows))

def jl(n):
    p = R / n
    return [json.loads(l) for l in open(p)] if p.exists() else []

rows = []
for r in jl("promptda.jsonl"):
    r["tier"] = TIER[r["mode"]]; r["column"] = COL[r["mode"]]; r["how"] = "autocast" if r["mode"] == "upstream-fp16" else None
    r["note"] = f"prompt = 192x256 cell means of the GT; the same prompt upsampled alone: AbsRel {r['prompt_only']['absrel']}, d1 {r['prompt_only']['d1']}"
    del r["prompt_only"]
    rows.append(r)
if rows: put("depth", rows)

rows = []
for r in jl("wd_agree.jsonl"):
    m = r["mode"]
    rows.append({"ref": r["ref"], "task": "classification", "tier": TIER[m], "mode": m, "dtype": r["dtype"],
                 "how": "autocast" if m == "upstream-fp16" else None, "dataset": r["dataset"], "n": r["n"],
                 "metrics": {"tag_f1": r["tag_f1"], "top1_agree": r["top1_agree"]}, "ms_per_img": r["ms_per_img"],
                 "peak_reserved_mb": r["peak_reserved_mb"], "column": COL[m], **G,
                 "note": f"agreement with the original fp32 (max |dp| {r['max_abs_prob']}, mean {r['mean_abs_prob']}, {r['tags_per_image_original']} tags per image)",
                 "command": "results/raw/2026-11-16_accuracy_rows/agree_wd.py (q2.sh), 500 first COCO val2017 images, threshold 0.35"})
if rows: put("classification", rows)

rows = []
for r in jl("seedvr2.jsonl"):
    m = r["mode"]; r.setdefault("column", COL[m]); r.setdefault("tier", TIER[m])
    if "crop512" in r["dataset"]:
        b = {4: "29.01 dB / SSIM 0.8075", 2: "31.03 dB / SSIM 0.8003"}[r["n"]]
        r["note"] = f"reference: plain bicubic x2 of the same LR crops = PSNR {b} (a generative restorer scores lower: it re-draws detail and shifts tone); LPIPS not computed (package not installed in the test venv)"
    rows.append(r)
if rows: put("restoration", rows)
