"""Build volume eval rows of nnfoundation-cnn/vit (feature parity) and nninteractive (Dice) from the 2026-11-14/15 raw files."""
import json, sys
from pathlib import Path
R = Path(__file__).parent
COMMIT = sys.argv[1]
OUT = Path(sys.argv[2])
base = {"task": "volume", "gpu": "NVIDIA GeForce RTX 5060 Ti", "torch": "2.14.1+cu130", "commit": COMMIT}
rows = []
cols = {"upstream": ("upstream", "original fp32", "float32"), "upstream16": ("upstream-fp16", "original fp16", "float16"),
        "portable": ("portable", "portable", "float16"), "kernels": ("kernels", "kernels", "float16")}
for arch, ref in (("cnn", "nnfoundation-cnn"), ("vit", "nnfoundation-vit")):
    d = json.load(open(R / f"featparity_{arch}.json"))
    s = d["summary"]
    for who, tier in (("original-fp16", "upstream16"), ("portable", "portable"), ("kernels", "kernels")):
        ms = [v for k, v in s.items() if k.startswith(who + " map")]
        m = {"cos_min": min(v["cos_min"] for v in ms), "cos_mean": sum(v["cos_mean"] for v in ms) / len(ms),
             "rel_l2": sum(v["rel_l2_mean"] for v in ms) / len(ms)}
        mode, col, dt = cols[tier]
        rows.append({**base, "date": d["date"], "source": f"results/raw/2026-11-16_accuracy_rows/featparity_{arch}.json",
                     "ref": ref, "tier": tier, "mode": mode, "column": col, "dtype": dt, "dataset": "totalseg-featparity-8",
                     "n": ms[0]["cases"], "metrics": {k: round(v, 6) for k, v in m.items()},
                     "note": f"encoder features vs the original fp32 ({d['reference']}), {len(ms)} feature maps, cosine (min over cases) and relative L2 (mean)"})
    rows.append({**base, "date": d["date"], "source": "reference", "ref": ref, "tier": "upstream", "mode": "upstream",
                 "column": "original fp32", "dtype": "float32", "dataset": "totalseg-featparity-8", "n": ms[0]["cases"],
                 "metrics": {"cos_min": 1.0, "cos_mean": 1.0, "rel_l2": 0.0}, "note": "the reference itself"})
d = json.load(open(R / "dice.json"))["summary"]["all"]
for key, tier in (("up_gt", "upstream16"), ("portable_gt", "portable"), ("kernels_gt", "kernels")):
    mode, col, dt = cols[tier]
    m = {"dice": round(d[key]["mean"], 5)}
    if tier != "upstream16":
        m["agree"] = round(d[key.replace("_gt", "_vs_up")]["mean"], 5)
    rows.append({**base, "date": "2026-11-15", "source": "results/raw/2026-11-16_accuracy_rows/dice.json (= 2026-11-14_nninteractive)",
                 "ref": "nninteractive", "tier": tier, "mode": mode, "column": col, "dtype": dt,
                 "dataset": "totalseg-nninteractive-prompts-96", "n": d[key]["n"], "metrics": m,
                 "note": "8 TotalSegmentator cases x 12 prompts (point, box, scribble, point+box x 3 classes), Dice vs the ground-truth mask; agree = Dice vs the original's mask; original = fp16 autocast"})
with open(OUT, "a") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(len(rows))
