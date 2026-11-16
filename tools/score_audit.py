"""Score audit: completeness and plausibility of every ref's numbers (CPU only, stdlib + tomllib).

    python tools/score_audit.py [--out out/score_audit.md] [--csv DIR]

Per ref (card variant): original / portable / kernels / compile / TensorRT speed + peak memory, accuracy rows of the task
(original, portable, kernels), each present or explained by a card reason (`[upstream] no_original_reason`,
`[modes.kernels] status = "none" + reason`, `[[variants]] planned`, OOM / `fails` rows). Plausibility: portable vs
original, kernels vs portable (reserved memory only above +25 % and +256 MB: allocator workspace),  compile / TensorRT vs portable, accuracy drift, sizes across modes, stale rows (older than
the last source change of the family), unit sanity, card text vs numbers. Known exceptions: docs/data/speed_allowlist.json.
Exit code 1 when a problem remains.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import math
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
ORIG_MODES = ("upstream", "upstream-default")
PRIM = {
    "stereo": "bad", "depth": "rel_absrel", "flow": "epe", "classification": "top1", "segmentation": "miou",
    "detection": "box_map", "pose": "keypoint_ap", "features": "top1", "promptable_segmentation": "miou_best",
    "interpolation": "psnr", "restoration": "psnr", "matching": "auc5", "matting": "mad", "tracking": "aj",
    "multiview": "absrel", "retrieval": "recall@1", "calibration": "vfov_auc1", "volume": "dice",
}
# accuracy drifts explained in the numbers audit (one step of a small-n metric / fp16 effect)
EXPLAINED = {("anycalib-pinhole", "portable"): 1, ("anycalib-pinhole", "kernels"): 1, ("flowseek-t", "portable"): 1,
             # n=150 every10 subset: loftr-outdoor on all 1500 pairs portable32 50.75 vs original 50.14; dedode-v2-b auc10/auc20 are
             # +0.2/+0.5 above the original on the same pairs, only auc5 (the 5 degree bin, n=150) differs
             ("loftr-outdoor", "portable"): 1, ("dedode-v2-b", "portable"): 1,
             ("vggt-omega-1b-512", "kernels"): 1}
LOWER_BETTER = {"bad", "rel_absrel", "epe", "abs_rel", "absrel", "fl_all", "mae", "mad", "error", "met_absrel"}


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def load_cards() -> dict[str, dict]:
    """ref -> {task, family, card (dict), variant (dict)}."""
    out = {}
    for p in sorted(glob.glob(str(ROOT / "src/fastforks/*/models/*/card.toml"))):
        t = tomllib.loads(Path(p).read_text())
        parts = Path(p).parts
        task, fam = parts[-4], parts[-2]
        for v in t.get("variants", []):
            out[v["id"]] = dict(task=task, family=fam, card=t, variant=v, path=p, text=Path(p).read_text())
    return out


def family_dates() -> dict[tuple[str, str], str]:
    """(task, family) -> date of the last commit touching its .py / .cu / .cpp source, shared code included separately."""
    dates = {}
    for d in glob.glob(str(ROOT / "src/fastforks/*/models/*")):
        task, fam = Path(d).parts[-3], Path(d).parts[-1]
        try:
            o = subprocess.run(
                ["git", "log", "-1", "--format=%cs", "--", f"{d}/*.py", f"{d}/**/*.py", f"{d}/*.cu", f"{d}/**/*.cu"],
                cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            o = ""
        dates[task, fam] = o
    return dates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "out/score_audit.md"))
    ap.add_argument("--csv", default="")
    ap.add_argument("--no-stale", action="store_true", help="skip the git-based stale-row check")
    a = ap.parse_args()
    cards = load_cards()
    allow = json.loads((DATA / "speed_allowlist.json").read_text())
    speed = jl(DATA / "speed.jsonl")
    comp = jl(DATA / "speed_compile.jsonl")
    fdates = {} if a.no_stale else family_dates()

    stale: dict = collections.defaultdict(list)
    P: list[dict] = []  # problems: ref, family, kind, mode, issue
    cnt = collections.defaultdict(lambda: collections.Counter())  # family -> counter

    def prob(ref, kind, mode, issue):
        c = cards.get(ref, {})
        P.append(dict(ref=ref, family=f"{c.get('task', '?')}/{c.get('family', '?')}", kind=kind, mode=mode, issue=issue))

    # ---- speed rows per ref
    S = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in speed:
        S[r["ref"]][r["mode"]].append(r)
    C = collections.defaultdict(dict)
    for r in comp:
        if r["mode"] in ("compile", "tensorrt"):
            C[r["ref"]][r["mode"]] = r  # latest wins (as gen_docs)
    # ---- eval rows per ref
    E = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in sorted(DATA.glob("*_eval.jsonl")):
        for r in jl(p):
            if "ref" in r:
                E[r["ref"]][r.get("mode")].append(r)

    def timed(rows):
        return [r for r in rows if r.get("ms") is not None and not r.get("status") and not r.get("fails")]

    # convention (docs: accuracy n/m per family): a variant with no eval rows at all is covered by an evaluated sibling
    # variant of the same family (same code path, other weights/size); only a family without any eval row is missing.
    fam_eval = {f"{c['task']}/{c['family']}" for r_, c in cards.items() if E.get(r_)}
    for ref, c in sorted(cards.items(), key=lambda kv: (kv[1]["task"], kv[1]["family"], kv[0])):
        fam = f"{c['task']}/{c['family']}"
        cc = cnt[fam]
        cc["refs"] += 1
        v, card = c["variant"], c["card"]
        planned = v.get("planned", {})
        no_orig = card.get("upstream", {}).get("no_original_reason", "")
        kmode = card.get("modes", {}).get("kernels", {})
        k_none = kmode.get("status") == "none"
        text = c["text"]
        s = S.get(ref, {})
        st = {m: timed(s.get(m, [])) for m in ("upstream", "upstream-default", "portable", "kernels")}
        orig_t = st["upstream"] + st["upstream-default"]
        orig_oom = [r for m in ORIG_MODES for r in s.get(m, []) if r.get("status") == "oom"]
        # --- presence
        def present(label, ok, explained, issue):
            cc[f"{label} present"] += bool(ok)
            if not ok:
                if explained:
                    cc[f"{label} explained"] += 1
                else:
                    cc[f"{label} MISSING"] += 1
                    prob(ref, "missing", label, issue)
        present("original speed", orig_t or orig_oom, no_orig, "no timed original row, no OOM row, no [upstream] no_original_reason")
        present("portable speed", st["portable"], False, "no timed portable row")
        present("kernels speed", st["kernels"], k_none and kmode.get("reason"), "no timed kernels row and no [modes.kernels] status none + reason")
        if k_none and st["kernels"]:
            prob(ref, "contradiction", "kernels", "card says kernels status none but kernels speed rows exist")
        if not k_none and "kernels" in planned:
            pass
        cr = C.get(ref, {})
        for m in ("compile", "tensorrt"):
            r = cr.get(m)
            ok = r is not None and ("ms" in r and not r.get("fails"))
            deferred = r is not None and str(r.get("fails", "")).startswith("deferred")
            expl = (r is not None and r.get("fails") and not deferred) or m in planned or (m == "tensorrt" and "onnx" in planned)
            present(m, ok, expl, f"no {m} row ({"deferred build, never measured" if deferred else "none, no fails reason, not planned on the card"})")
            if r is not None and r.get("fails") and not (m in planned or (m == "tensorrt" and "onnx" in planned)):
                cc[f"{m} fails"] += 1
        # memory presence
        for m, rows in (("original", orig_t), ("portable", st["portable"]), ("kernels", st["kernels"])):
            if rows:
                last = sorted(rows, key=lambda r: r.get("date", ""))[-1]
                ok = last.get("peak_alloc_mb") is not None and (last.get("peak_reserved_mb") is not None or (m == "original" and last.get("torch") is None))  # TensorFlow originals have no reserved counter
                cc[f"{m} memory present"] += ok
                if not ok:
                    cc[f"{m} memory MISSING"] += 1
                    prob(ref, "missing", f"{m} memory", f"latest {m} row {last.get('input')} ({last.get('date')}) lacks peak_alloc_mb / peak_reserved_mb")
        # --- accuracy presence
        e = E.get(ref, {})
        eval_task = (DATA / f"{c['task']}_eval.jsonl").is_file()
        if not eval_task:
            cc["accuracy: task has no eval file"] += 1
        else:
            prim = PRIM.get(c["task"])
            def has(modes):
                return [r for mm in modes for r in e.get(mm, []) if not r.get("error") and r.get("metrics")]
            o_e, p_e, k_e = has(("upstream", "upstream-fp16", "upstream16")), has(("portable", "portable32")), has(("kernels",))
            o_err = [r for mm in ("upstream", "upstream-fp16") for r in e.get(mm, []) if r.get("error")]
            sib = (not e) and fam in fam_eval
            present("accuracy original", o_e or o_err, no_orig or not e, "no original accuracy row")
            no_acc = re.search(r"No accuracy measurement", " ".join(card.get("limitations", [])))  # the card names why (no public evaluation set)
            present("accuracy portable", p_e, sib or no_acc, "no portable accuracy row")
            present("accuracy kernels", k_e, (k_none and kmode.get("reason")) or sib or no_acc, "no kernels accuracy row")
        # --- plausibility: speed per input
        by_in = collections.defaultdict(dict)
        for m in ("upstream", "upstream-default", "portable", "kernels"):
            for r in st[m]:
                by_in[r["input"]].setdefault(m, []).append(r)
        sizes_by_mode = {m: {r["input"] for r in st[m]} for m in st if st[m]}
        if st["portable"] and orig_t and not (sizes_by_mode.get("portable", set()) & (sizes_by_mode.get("upstream", set()) | sizes_by_mode.get("upstream-default", set()))):
            prob(ref, "sizes", "portable/original", f"no common input size: portable {sorted(sizes_by_mode['portable'])} vs original {sorted(sizes_by_mode.get('upstream', set()) | sizes_by_mode.get('upstream-default', set()))}")
        if st["portable"] and st["kernels"] and not (sizes_by_mode["portable"] & sizes_by_mode["kernels"]):
            prob(ref, "sizes", "portable/kernels", f"no common input size: {sorted(sizes_by_mode['portable'])} vs {sorted(sizes_by_mode['kernels'])}")
        for inp, d in by_in.items():
            lat = lambda l: sorted(l, key=lambda r: r.get("date", ""))[-1]
            Pr = lat(d["portable"]) if "portable" in d else None
            K = lat(d["kernels"]) if "kernels" in d else None
            O = [lat(d[m]) for m in ORIG_MODES if m in d]
            for x in [Pr, K, *O]:
                if x is None:
                    continue
                if not (0.005 < x["ms"] < 1e6):
                    prob(ref, "units", x["mode"], f"{inp}: ms {x['ms']} out of range")
                al, rs = x.get("peak_alloc_mb"), x.get("peak_reserved_mb")
                if al is not None and rs is not None and rs < al * 0.97 - 2:
                    prob(ref, "units", x["mode"], f"{inp}: reserved {rs} < allocated {al}")
                if al is not None and al > 200000:
                    prob(ref, "units", x["mode"], f"{inp}: peak alloc {al} MB absurd")
            key = lambda mode: f"{ref} {inp} {mode} vs upstream"
            if Pr and O:
                om = min(x["ms"] for x in O)
                oa = [x["peak_alloc_mb"] for x in O if x.get("peak_alloc_mb")]
                orr = [x["peak_reserved_mb"] for x in O if x.get("peak_reserved_mb")]
                ex = any(k.startswith(f"{ref} {inp} portable vs ") for k in allow) or bool(
                    re.search(r"portable.{0,200}slower than", " ".join(card.get("limitations", [])), re.I))
                if Pr["ms"] > om * 1.03 and not ex:
                    prob(ref, "implausible", "portable", f"{inp}: portable {Pr['ms']} ms slower than original {om} ms ({100 * (Pr['ms'] / om - 1):.0f} %)")
                    cc["portable slower than original"] += 1
                if oa and Pr.get("peak_alloc_mb") and Pr["peak_alloc_mb"] > min(oa) * 1.05 + 64 and not ex:
                    prob(ref, "implausible", "portable", f"{inp}: portable alloc {Pr['peak_alloc_mb']} MB above original {min(oa)} MB")
                    cc["portable memory above original"] += 1
                if orr and Pr.get("peak_reserved_mb") and Pr["peak_reserved_mb"] > min(orr) * 1.05 + 64 and not ex:
                    prob(ref, "implausible", "portable", f"{inp}: portable reserved {Pr['peak_reserved_mb']} MB above original {min(orr)} MB")
                    cc["portable memory above original"] += 1
            if K and O:
                om = min(x["ms"] for x in O)
                if K["ms"] > om * 1.03 and not any(k.startswith(f"{ref} {inp} kernels vs ") for k in allow):
                    prob(ref, "implausible", "kernels", f"{inp}: kernels {K['ms']} ms slower than original {om} ms")
                    cc["kernels slower than original"] += 1
            if K and Pr:
                ex = any(k.startswith(f"{ref} {inp} kernels vs ") for k in allow)
                if K["ms"] > Pr["ms"] * 1.03 and not ex:
                    prob(ref, "implausible", "kernels", f"{inp}: kernels {K['ms']} ms slower than portable {Pr['ms']} ms")
                    cc["kernels slower than portable"] += 1
                for kind, kk, pp in (("alloc", K.get("peak_alloc_mb"), Pr.get("peak_alloc_mb")), ("reserved", K.get("peak_reserved_mb"), Pr.get("peak_reserved_mb"))):
                    if kk and pp and kk > pp * (1.05 if kind == "alloc" else 1.25) + (64 if kind == "alloc" else 256):
                        prob(ref, "implausible", "kernels", f"{inp}: kernels {kind} {kk} MB above portable {pp} MB")
                        cc["kernels memory above portable"] += 1
            # stale rows
            fd = fdates.get((c["task"], c["family"]), "")
            if fd:
                for x in [Pr, K, *O]:
                    if x and x.get("date", "9999")[:10] < fd and x["mode"] in ("portable", "kernels"):
                        cc["stale rows"] += 1
                        stale[ref, x["mode"]].append(f"{inp} {x.get('date', '')[:10]}")
        # compile / TRT plausibility
        std = None
        for m in ("compile", "tensorrt"):
            r = cr.get(m)
            if r and r.get("ms") and not r.get("fails"):
                at = [x for x in st["portable"] if x["input"] == r["input"]]
                if at:
                    pm = sorted(at, key=lambda x: x.get("date", ""))[-1]["ms"]
                    if r["ms"] > pm * 3 + 0.3:
                        prob(ref, "implausible", m, f"{r['input']}: {m} {r['ms']} ms vs portable {pm} ms (> 3x)")
                        cc[f"{m} slower than portable"] += 1
                else:
                    prob(ref, "sizes", m, f"{m} input {r['input']} has no portable row at that size")
                if r.get("peak_alloc_mb") is None and m == "compile":
                    prob(ref, "missing", "compile memory", "compile row without peak_alloc_mb")
        # card text vs numbers
        for mm in re.finditer(r"[^.\n]*(not measured|unmeasured|to be measured|TODO)[^.\n]*", re.sub(r"(?m)^\s*#.*$|^.*fp32_accumulate.*$", "", text), re.I):
            prob(ref, "card text", "-", f"card says: {mm.group(0).strip()[:120]}")
            cc["card text problems"] += 1
        if no_orig and orig_t:
            prob(ref, "contradiction", "original", "card no_original_reason but a timed original row exists")
        if not card.get("limitations") and (any(r.get("status") == "oom" for m in ("portable", "kernels") for r in s.get(m, []))):
            prob(ref, "card reason", "portable/kernels", "portable/kernels OOM row but the card has no `limitations`")
        # OOM cells only for original fp32
        for m in ("portable", "kernels"):
            for r in s.get(m, []):
                if r.get("status") == "oom":
                    newer = [x for x in st[m] if x["input"] == r["input"] and x.get("date", "") >= r.get("date", "")]
                    if not newer and not re.search(r"OOM|exceed\w* .*memory|out of memory", " ".join(card.get("limitations", [])), re.I):
                        prob(ref, "oom", m, f"{r['input']}: {m} OOM row (card must say why: limitations)")
        # --- accuracy plausibility
        if eval_task and e:
            prim = PRIM.get(c["task"])
            by_ds = collections.defaultdict(dict)
            for mm, rows in e.items():
                for r in rows:
                    if not r.get("error") and r.get("metrics"):
                        by_ds[r["dataset"]].setdefault(mm, r)  # first = ok? take last below
            for ds in by_ds:
                for mm in by_ds[ds]:
                    by_ds[ds][mm] = [r for r in e[mm] if r.get("dataset") == ds and not r.get("error") and r.get("metrics")][-1]
            for ds, d in by_ds.items():
                o = d.get("upstream") or d.get("upstream-fp16")
                if not o:
                    continue
                om = o["metrics"]
                key = prim if prim in om else next((k for k, x in om.items() if isinstance(x, (int, float))), None)
                for mm in ("portable", "kernels"):
                    t = d.get(mm)
                    if not t or key not in t["metrics"] or not isinstance(om.get(key), (int, float)) or not om[key]:
                        continue
                    rel = (t["metrics"][key] - om[key]) / abs(om[key])
                    cc["accuracy comparisons"] += 1
                    worse = (rel > 0) == (key in LOWER_BETTER)
                    if abs(rel) > 0.02 and worse and (ref, mm) not in EXPLAINED:
                        cc["accuracy drift > 2 %"] += 1
                        prob(ref, "implausible", mm, f"{ds}: {key} {t['metrics'][key]} vs original {om[key]} ({100 * rel:+.1f} %, worse)")
                    for rr in (o, t):
                        if rr.get("n") and t.get("n") and rr["n"] != t["n"] and rr is t:
                            prob(ref, "sizes", mm, f"{ds}: n {t['n']} vs original n {o['n']}")
                            break
    for (ref, m), l in stale.items():
        prob(ref, "stale?", m, f"{len(l)} row(s) older than the family's last .py/.cu commit {fdates[cards[ref]['task'], cards[ref]['family']]}: {'; '.join(l[:3])}")
    # ---- tables
    fams = sorted(cnt)
    labels = ["original speed", "portable speed", "kernels speed", "compile", "tensorrt", "accuracy original", "accuracy portable", "accuracy kernels"]
    lines = ["# Score audit, 2026-11-16", "",
             f"Generated by `tools/score_audit.py` (CPU only). Data: docs/data/*.jsonl at `{subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()}`; "
             "cards `src/fastforks/*/models/*/card.toml`. Cell: present / explained by a card reason or fails row / MISSING. "
             "Implausible = rule violated (portable vs original beyond 3 % / 5 % + 64 MB, kernels vs portable (reserved memory only above +25 % and +256 MB: allocator workspace),  compile/TRT > 3x portable, accuracy > 2 % worse, stale, units, sizes).", "",
             "| family | refs | " + " | ".join(labels) + " | implausible | other problems |", "|" + "---|" * (len(labels) + 4)]
    tot = collections.Counter()
    pf = collections.Counter(); pf_other = collections.Counter()
    for p in P:
        (pf if p["kind"] in ("implausible", "stale?", "sizes", "units") else pf_other)[p["family"]] += 1
    for f in fams:
        c = cnt[f]
        cells = []
        for l in labels:
            pr, ex, mi = c[f"{l} present"], c[f"{l} explained"], c[f"{l} MISSING"]
            tot[l, "p"] += pr; tot[l, "e"] += ex; tot[l, "m"] += mi
            cells.append(f"{pr} / {ex} / {mi}")
        tot["refs", ""] += c["refs"]; tot["imp", ""] += pf[f]; tot["oth", ""] += pf_other[f]
        lines.append(f"| {f} | {c['refs']} | " + " | ".join(cells) + f" | {pf[f]} | {pf_other[f]} |")
    lines.append(f"| **total** | {tot['refs', '']} | " + " | ".join(f"{tot[l, 'p']} / {tot[l, 'e']} / {tot[l, 'm']}" for l in labels) + f" | {tot['imp', '']} | {tot['oth', '']} |")
    lines += ["", "## Notes (2026-11-16 run)", "",
              "- The `implausible` column counts `stale?` rows too (informational: row older than the family's last source commit; `--no-stale` skips them). Real violations of the rules: 0 after the run.",
              "- Gap-fill measured 2026-11-16 on RTX 5060 Ti (a GPU job, one process at a time) with `results/raw/2026-11-16_score_audit/kernels_speed.py` via `queue_runner*.sh` / `eval_queue.sh`; rows merged by `merge.py`, `merge_fails.py` (placeholders 'deferred: measure later' replaced by the real failure). Base commit `5ea6bb9d`.",
              "- Dropped as superseded: stale video-depth-anything-s compile 1080x1920 row (compile/TRT jobs run the standard 480x640 input only).",
              "- Explained, not fixed: loftr-outdoor / dedode-v2-b auc5 on the n=150 subset (EXPLAINED in this script), track-on2 reserved memory (speed_allowlist.json).",
              "- Open: accuracy rows for portable/kernels of families without any eval row (listed below), original accuracy of marigold-v2-depth (no runnable upstream harness). `tests/test_policy.py::test_no_architecture_constants_in_model_code[matting]` fails on origin/work/ports (ben2 `_TINY`), unrelated.", ""]
    lines += ["", f"## Problems ({len(P)})", "", "| ref | family | kind | mode | issue |", "|---|---|---|---|---|"]
    for p in sorted(P, key=lambda p: (p["kind"], p["family"], p["ref"])):
        lines.append(f"| {p['ref']} | {p['family']} | {p['kind']} | {p['mode']} | {p['issue'].replace('|', '/')} |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    if a.csv:
        os.makedirs(a.csv, exist_ok=True)
        with open(f"{a.csv}/problems.csv", "w", newline="") as f:
            w = csv.DictWriter(f, ["ref", "family", "kind", "mode", "issue"])
            w.writeheader()
            w.writerows(P)
    kinds = collections.Counter(p["kind"] for p in P)
    print(dict(kinds), "total", len(P))
    return 1 if P else 0


if __name__ == "__main__":
    sys.exit(main())
