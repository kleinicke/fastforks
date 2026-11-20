"""Score audit: completeness and plausibility of every ref's numbers (CPU only, stdlib + tomllib).

    python tools/score_audit.py [--out out/score_audit.md] [--csv DIR]

Per ref (card variant): original / portable / kernels / compile / TensorRT speed + peak memory, accuracy rows of the
task
(original, portable, kernels), each present or explained by a card reason (`[upstream] no_original_reason`,
`[modes.kernels] status = "none" + reason`, `[[variants]] planned`, OOM / `fails` rows). Plausibility: portable vs
original, kernels vs portable (reserved memory only above +25 % and +256 MB: allocator workspace), compile / TensorRT
vs portable, accuracy drift, sizes across modes, stale rows (older than the last source change of the family), unit
sanity, card text vs numbers. Known exceptions: docs/data/speed_allowlist.json.
Exit code 1 when a problem remains.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data"
ORIG_MODES = ("upstream", "upstream-default")
ORIG_EVAL_MODES = (*ORIG_MODES, "upstream-fp16", "upstream-bf16", "upstreamhalf", "upstream16")
PRIM = {
    "stereo": "bad",
    "depth": "rel_absrel",
    "flow": "epe",
    "classification": "top1",
    "segmentation": "miou",
    "detection": "box_map",
    "pose": "keypoint_ap",
    "features": "top1",
    "promptable_segmentation": "miou_best",
    "interpolation": "psnr",
    "restoration": "psnr",
    "matching": "auc5",
    "matting": "mad",
    "tracking": "aj",
    "multiview": "absrel",
    "retrieval": "recall@1",
    "calibration": "vfov_auc1",
    "volume": "dice",
}
# accuracy drifts explained in the numbers audit (one step of a small-n metric / fp16 effect)
EXPLAINED = {
    ("anycalib-pinhole", "portable"): 1,
    ("anycalib-pinhole", "kernels"): 1,
    # n=150 every10 subset: loftr-outdoor on all 1500 pairs portable32 50.75 vs original 50.14; dedode-v2-b auc10/auc20
    # are
    # +0.2/+0.5 above the original on the same pairs, only auc5 (the 5 degree bin, n=150) differs
    ("loftr-outdoor", "portable"): 1,
    ("dedode-v2-b", "portable"): 1,
    ("vggt-omega-1b-512", "kernels"): 1,
}
LOWER_BETTER = {"bad", "rel_absrel", "epe", "abs_rel", "absrel", "fl_all", "mae", "mad", "error", "met_absrel"}


_MS_RANGE = (0.005, 1e6)  # plausible ms per call
_ABSURD_ALLOC_MB = 200000
_DRIFT = 0.02  # accuracy drift (relative) that counts as worse


def _latest(rows: list[dict]) -> dict:
    """The newest row by date (the last of equals, as `sorted(...)[-1]`)."""
    return max(reversed(rows), key=lambda r: r.get("date", ""))


def _portable_reference(compiled: dict, portable: list[dict]) -> float | None:
    """Use the measured portable arm of a scoped backend comparison when present."""
    paired = compiled.get("paired_portable_ms")
    if isinstance(paired, (int, float)) and paired > 0:
        return paired
    same_input = [row for row in portable if row["input"] == compiled["input"]]
    return _latest(same_input)["ms"] if same_input else None


def _accuracy_pairs(evals: dict[str, list[dict]]) -> Iterator[tuple[str, dict, dict]]:
    """Pair reported scores on the same dataset and precision, and match cohort metadata when available."""
    originals = collections.defaultdict(list)
    for mode in ORIG_EVAL_MODES:
        for row in evals.get(mode, ()):
            if not row.get("error") and row.get("metrics"):
                originals[row["dataset"], row.get("dtype")].append(row)
    keys = ("n", "dataset_manifest_sha256")
    for mode in ("portable", "kernels"):
        latest = {
            (row["dataset"], row.get("dtype"), *(row.get(key) for key in keys)): row
            for row in evals.get(mode, ())
            if not row.get("error") and row.get("metrics")
        }
        for (dataset, dtype, *_), row in latest.items():
            candidates = [
                original
                for original in originals[dataset, dtype]
                if all(not original.get(key) or not row.get(key) or original[key] == row[key] for key in keys)
            ]
            if candidates:
                yield mode, row, _latest(candidates)


def jl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def load_cards() -> dict[str, dict]:
    """Ref -> {task, family, card (dict), variant (dict)}."""
    out = {}
    for p in sorted((ROOT / "src/fastforks").glob("*/models/*/card.toml")):
        t = tomllib.loads(p.read_text())
        parts = p.parts
        task, fam = parts[-4], parts[-2]
        for v in t.get("variants", []):
            out[v["id"]] = {"task": task, "family": fam, "card": t, "variant": v, "path": str(p), "text": p.read_text()}
    return out


def family_dates() -> dict[tuple[str, str], str]:
    """(task, family) -> date of the last commit touching its .py / .cu / .cpp source (shared code separately)."""
    dates = {}
    for d in (ROOT / "src/fastforks").glob("*/models/*"):
        task, fam = d.parts[-3], d.parts[-1]
        try:
            o = subprocess.run(
                ["git", "log", "-1", "--format=%cs", "--", f"{d}/*.py", f"{d}/**/*.py", f"{d}/*.cu", f"{d}/**/*.cu"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
        except OSError:
            o = ""
        dates[task, fam] = o
    return dates


def main() -> int:  # noqa: C901, PLR0912, PLR0915 - one linear pass over every audit check and the report
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
    probs: list[dict] = []  # problems: ref, family, kind, mode, issue
    cnt = collections.defaultdict(collections.Counter)  # family -> counter

    def prob(ref: str, kind: str, mode: str, issue: str) -> None:
        c = cards.get(ref, {})
        family = f"{c.get('task', '?')}/{c.get('family', '?')}"
        probs.append({"ref": ref, "family": family, "kind": kind, "mode": mode, "issue": issue})

    def present(
        where: tuple[collections.Counter, str], label: str, ok: object, issue: str, explained: object = None
    ) -> None:
        cc, ref = where
        cc[f"{label} present"] += bool(ok)
        if not ok:
            if explained:
                cc[f"{label} explained"] += 1
            else:
                cc[f"{label} MISSING"] += 1
                prob(ref, "missing", label, issue)

    # ---- speed rows per ref
    spd = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in speed:
        spd[r["ref"]][r["mode"]].append(r)
    comp_rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in comp:
        if r["mode"] in ("compile", "tensorrt"):
            comp_rows[r["ref"]][r["mode"]][r["input"]] = r  # latest for each measured input
    # ---- eval rows per ref
    evals = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in sorted(DATA.glob("*_eval.jsonl")):
        for r in jl(p):
            if "ref" in r:
                evals[r["ref"]][r.get("mode")].append(r)

    def timed(rows: list[dict]) -> list[dict]:
        return [r for r in rows if r.get("ms") is not None and not r.get("status") and not r.get("fails")]

    # convention (docs: accuracy n/m per family): a variant with no eval rows at all is covered by an evaluated sibling
    # variant of the same family (same code path, other weights/size); only a family without any eval row is missing.
    fam_eval = {f"{c['task']}/{c['family']}" for r_, c in cards.items() if evals.get(r_)}
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
        s = spd.get(ref, {})
        st = {m: timed(s.get(m, [])) for m in ("upstream", "upstream-default", "portable", "kernels")}
        orig_t = st["upstream"] + st["upstream-default"]
        orig_oom = [r for m in ORIG_MODES for r in s.get(m, []) if r.get("status") == "oom"]

        # --- presence
        present(
            (cc, ref),
            "original speed",
            orig_t or orig_oom,
            "no timed original row, no OOM row, no [upstream] no_original_reason",
            no_orig,
        )
        present((cc, ref), "portable speed", st["portable"], "no timed portable row")
        present(
            (cc, ref),
            "kernels speed",
            st["kernels"],
            "no timed kernels row and no [modes.kernels] status none + reason",
            k_none and kmode.get("reason"),
        )
        if k_none and st["kernels"]:
            prob(ref, "contradiction", "kernels", "card says kernels status none but kernels speed rows exist")
        if not k_none and "kernels" in planned:
            pass
        portable_sizes = {r["input"] for r in st["portable"]}
        checked = {}
        for m in ("compile", "tensorrt"):
            profiles = comp_rows[ref][m].values()
            gated = m in planned or (m == "tensorrt" and "onnx" in planned)
            ok = [
                r
                for r in profiles
                if (r.get("ms") or 0) > 0
                and not r.get("fails")
                and (not gated or (m == "tensorrt" and r.get("standalone_build")))
            ]
            failures = [r["fails"] for r in profiles if r.get("fails")]
            expl = gated or any(not str(reason).startswith("deferred") for reason in failures)
            deferred = failures and not expl
            why = "deferred build, never measured" if deferred else "none, no fails reason, not planned on the card"
            present((cc, ref), m, ok, f"no {m} row ({why})", expl)
            if ok:
                checked[m] = _latest([r for r in ok if r["input"] in portable_sizes] or ok)
            if failures and not ok and not gated:
                cc[f"{m} fails"] += 1
        # memory presence
        for m, rows in (("original", orig_t), ("portable", st["portable"]), ("kernels", st["kernels"])):
            if rows:
                last = _latest(rows)
                ok = last.get("peak_alloc_mb") is not None and (
                    last.get("peak_reserved_mb") is not None or (m == "original" and last.get("torch") is None)
                )  # TensorFlow originals have no reserved counter
                cc[f"{m} memory present"] += ok
                if not ok:
                    cc[f"{m} memory MISSING"] += 1
                    prob(
                        ref,
                        "missing",
                        f"{m} memory",
                        f"latest {m} row {last.get('input')} ({last.get('date')}) lacks peak_alloc_mb / "
                        "peak_reserved_mb",
                    )
        # --- accuracy presence
        e = evals.get(ref, {})
        eval_task = (DATA / f"{c['task']}_eval.jsonl").is_file()
        if not eval_task:
            cc["accuracy: task has no eval file"] += 1
        else:
            prim = PRIM.get(c["task"])

            def has(modes: tuple[str, ...], e: dict = e) -> list[dict]:
                return [r for mm in modes for r in e.get(mm, []) if not r.get("error") and r.get("metrics")]

            o_e, p_e, k_e = (
                has(("upstream", "upstream-fp16", "upstream16")),
                has(("portable", "portable32")),
                has(("kernels",)),
            )
            o_err = [r for mm in ("upstream", "upstream-fp16") for r in e.get(mm, []) if r.get("error")]
            sib = (not e) and fam in fam_eval
            present((cc, ref), "accuracy original", o_e or o_err, "no original accuracy row", no_orig or not e)
            # An old dataset limitation cannot explain a missing tier once the checkpoint has measured accuracy.
            no_acc = not (o_e or p_e or k_e) and re.search(
                r"No accuracy measurement", " ".join(card.get("limitations", []))
            )
            present((cc, ref), "accuracy portable", p_e, "no portable accuracy row", sib or no_acc)
            present(
                (cc, ref),
                "accuracy kernels",
                k_e,
                "no kernels accuracy row",
                (k_none and kmode.get("reason")) or sib or no_acc,
            )
        # --- plausibility: speed per input
        by_in = collections.defaultdict(dict)
        for m in ("upstream", "upstream-default", "portable", "kernels"):
            for r in st[m]:
                by_in[r["input"]].setdefault(m, []).append(r)
        sizes_by_mode = {m: {r["input"] for r in st[m]} for m in st if st[m]}
        if (
            st["portable"]
            and orig_t
            and not (
                sizes_by_mode.get("portable", set())
                & (sizes_by_mode.get("upstream", set()) | sizes_by_mode.get("upstream-default", set()))
            )
        ):
            prob(
                ref,
                "sizes",
                "portable/original",
                f"no common input size: portable {sorted(sizes_by_mode['portable'])} vs original "
                f"{sorted(sizes_by_mode.get('upstream', set()) | sizes_by_mode.get('upstream-default', set()))}",
            )
        if st["portable"] and st["kernels"] and not (sizes_by_mode["portable"] & sizes_by_mode["kernels"]):
            prob(
                ref,
                "sizes",
                "portable/kernels",
                f"no common input size: {sorted(sizes_by_mode['portable'])} vs {sorted(sizes_by_mode['kernels'])}",
            )
        for inp, d in by_in.items():
            port = _latest(d["portable"]) if "portable" in d else None
            kern = _latest(d["kernels"]) if "kernels" in d else None
            origs = [_latest(d[m]) for m in ORIG_MODES if m in d]
            for x in [port, kern, *origs]:
                if x is None:
                    continue
                if not (_MS_RANGE[0] < x["ms"] < _MS_RANGE[1]):
                    prob(ref, "units", x["mode"], f"{inp}: ms {x['ms']} out of range")
                al, rs = x.get("peak_alloc_mb"), x.get("peak_reserved_mb")
                if al is not None and rs is not None and rs < al * 0.97 - 2:
                    prob(ref, "units", x["mode"], f"{inp}: reserved {rs} < allocated {al}")
                if al is not None and al > _ABSURD_ALLOC_MB:
                    prob(ref, "units", x["mode"], f"{inp}: peak alloc {al} MB absurd")
            if port and origs:
                om = min(x["ms"] for x in origs)
                oa = [x["peak_alloc_mb"] for x in origs if x.get("peak_alloc_mb")]
                orr = [x["peak_reserved_mb"] for x in origs if x.get("peak_reserved_mb")]
                ex = any(k.startswith(f"{ref} {inp} portable vs ") for k in allow) or bool(
                    re.search(r"portable.{0,200}slower than", " ".join(card.get("limitations", [])), re.IGNORECASE)
                )
                if port["ms"] > om * 1.03 and not ex:
                    prob(
                        ref,
                        "implausible",
                        "portable",
                        f"{inp}: portable {port['ms']} ms slower than original {om} ms "
                        f"({100 * (port['ms'] / om - 1):.0f} %)",
                    )
                    cc["portable slower than original"] += 1
                if oa and port.get("peak_alloc_mb") and port["peak_alloc_mb"] > min(oa) * 1.05 + 64 and not ex:
                    prob(
                        ref,
                        "implausible",
                        "portable",
                        f"{inp}: portable alloc {port['peak_alloc_mb']} MB above original {min(oa)} MB",
                    )
                    cc["portable memory above original"] += 1
                if orr and port.get("peak_reserved_mb") and port["peak_reserved_mb"] > min(orr) * 1.05 + 64 and not ex:
                    prob(
                        ref,
                        "implausible",
                        "portable",
                        f"{inp}: portable reserved {port['peak_reserved_mb']} MB above original {min(orr)} MB",
                    )
                    cc["portable memory above original"] += 1
            if kern and origs:
                om = min(x["ms"] for x in origs)
                if kern["ms"] > om * 1.03 and not any(k.startswith(f"{ref} {inp} kernels vs ") for k in allow):
                    prob(ref, "implausible", "kernels", f"{inp}: kernels {kern['ms']} ms slower than original {om} ms")
                    cc["kernels slower than original"] += 1
            if kern and port:
                ex = any(k.startswith(f"{ref} {inp} kernels vs ") for k in allow)
                if kern["ms"] > port["ms"] * 1.03 and not ex:
                    prob(
                        ref,
                        "implausible",
                        "kernels",
                        f"{inp}: kernels {kern['ms']} ms slower than portable {port['ms']} ms",
                    )
                    cc["kernels slower than portable"] += 1
                for kind, kk, pp in (
                    ("alloc", kern.get("peak_alloc_mb"), port.get("peak_alloc_mb")),
                    ("reserved", kern.get("peak_reserved_mb"), port.get("peak_reserved_mb")),
                ):
                    if kk and pp and kk > pp * (1.05 if kind == "alloc" else 1.25) + (64 if kind == "alloc" else 256):
                        prob(ref, "implausible", "kernels", f"{inp}: kernels {kind} {kk} MB above portable {pp} MB")
                        cc["kernels memory above portable"] += 1
            # stale rows
            fd = fdates.get((c["task"], c["family"]), "")
            if fd:
                for x in [port, kern, *origs]:
                    if x and x.get("date", "9999")[:10] < fd and x["mode"] in ("portable", "kernels"):
                        cc["stale rows"] += 1
                        stale[ref, x["mode"]].append(f"{inp} {x.get('date', '')[:10]}")
        # compile / TRT plausibility
        for m in ("compile", "tensorrt"):
            r = checked.get(m)
            if r and r.get("ms") and not r.get("fails"):
                pm = _portable_reference(r, st["portable"])
                if pm is not None:
                    if r["ms"] > pm * 3 + 0.3:
                        prob(ref, "implausible", m, f"{r['input']}: {m} {r['ms']} ms vs portable {pm} ms (> 3x)")
                        cc[f"{m} slower than portable"] += 1
                else:
                    prob(ref, "sizes", m, f"{m} input {r['input']} has no portable row at that size")
                if r.get("peak_alloc_mb") is None and m == "compile":
                    prob(ref, "missing", "compile memory", "compile row without peak_alloc_mb")
        # card text vs numbers
        for mm in re.finditer(
            r"[^.\n]*(not measured|unmeasured|to be measured|TODO)[^.\n]*",
            re.sub(r"(?m)^\s*#.*$|^.*fp32_accumulate.*$", "", text),
            re.IGNORECASE,
        ):
            prob(ref, "card text", "-", f"card says: {mm.group(0).strip()[:120]}")
            cc["card text problems"] += 1
        if no_orig and orig_t:
            prob(ref, "contradiction", "original", "card no_original_reason but a timed original row exists")
        if not card.get("limitations") and (
            any(r.get("status") == "oom" for m in ("portable", "kernels") for r in s.get(m, []))
        ):
            prob(ref, "card reason", "portable/kernels", "portable/kernels OOM row but the card has no `limitations`")
        # OOM cells only for original fp32
        for m in ("portable", "kernels"):
            for r in s.get(m, []):
                if r.get("status") == "oom":
                    newer = [x for x in st[m] if x["input"] == r["input"] and x.get("date", "") >= r.get("date", "")]
                    if not newer and not re.search(
                        r"OOM|exceed\w* .*memory|out of memory", " ".join(card.get("limitations", [])), re.IGNORECASE
                    ):
                        prob(ref, "oom", m, f"{r['input']}: {m} OOM row (card must say why: limitations)")
        # --- accuracy plausibility
        if eval_task and e:
            prim = PRIM.get(c["task"])
            for mm, target, original in _accuracy_pairs(e):
                om = original["metrics"]
                key = prim if prim in om else next((k for k, x in om.items() if isinstance(x, (int, float))), None)
                value = target["metrics"].get(key)
                if not isinstance(value, (int, float)) or not isinstance(om.get(key), (int, float)) or not om[key]:
                    continue
                rel = (value - om[key]) / abs(om[key])
                cc["accuracy comparisons"] += 1
                worse = (rel > 0) == (key in LOWER_BETTER)
                if abs(rel) > _DRIFT and worse and (ref, mm) not in EXPLAINED:
                    cc["accuracy drift > 2 %"] += 1
                    prob(
                        ref,
                        "implausible",
                        mm,
                        f"{target['dataset']} ({target.get('dtype', 'unknown precision')}): "
                        f"{key} {value} vs original {om[key]} ({100 * rel:+.1f} %, worse)",
                    )
    for (ref, m), old_rows in stale.items():
        last_commit = fdates[cards[ref]["task"], cards[ref]["family"]]
        prob(
            ref,
            "stale?",
            m,
            f"{len(old_rows)} row(s) older than the family's last .py/.cu commit {last_commit}: "
            f"{'; '.join(old_rows[:3])}",
        )
    # ---- tables
    fams = sorted(cnt)
    labels = [
        "original speed",
        "portable speed",
        "kernels speed",
        "compile",
        "tensorrt",
        "accuracy original",
        "accuracy portable",
        "accuracy kernels",
    ]
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    lines = [
        "# Score audit",
        "",
        (
            f"Generated by `tools/score_audit.py` (CPU only). Data: docs/data/*.jsonl at `{head}`; "
            "cards `src/fastforks/*/models/*/card.toml`. Cell: present / explained by a card reason or fails row / "
            "MISSING. Implausible = rule violated (portable vs original beyond 3 % / 5 % + 64 MB, kernels vs portable "
            "(reserved memory only above +25 % and +256 MB: allocator workspace),  compile/TRT > 3x portable, "
            "accuracy > 2 % worse at matched precision and cohort, stale, units, sizes)."
        ),
        "",
        "| family | refs | " + " | ".join(labels) + " | implausible | other problems |",
        "|" + "---|" * (len(labels) + 4),
    ]
    tot = collections.Counter()
    pf = collections.Counter()
    pf_other = collections.Counter()
    for p in probs:
        (pf if p["kind"] in ("implausible", "stale?", "sizes", "units") else pf_other)[p["family"]] += 1
    for f in fams:
        c = cnt[f]
        cells = []
        for lab in labels:
            pr, ex, mi = c[f"{lab} present"], c[f"{lab} explained"], c[f"{lab} MISSING"]
            tot[lab, "p"] += pr
            tot[lab, "e"] += ex
            tot[lab, "m"] += mi
            cells.append(f"{pr} / {ex} / {mi}")
        tot["refs", ""] += c["refs"]
        tot["imp", ""] += pf[f]
        tot["oth", ""] += pf_other[f]
        lines.append(f"| {f} | {c['refs']} | " + " | ".join(cells) + f" | {pf[f]} | {pf_other[f]} |")
    lines.append(
        f"| **total** | {tot['refs', '']} | "
        + " | ".join(f"{tot[lab, 'p']} / {tot[lab, 'e']} / {tot[lab, 'm']}" for lab in labels)
        + f" | {tot['imp', '']} | {tot['oth', '']} |"
    )
    lines += [
        "",
        "## How to read the report",
        "",
        "- A later backend attempt replaces the earlier record for the same input size.",
        "- Standalone TensorRT measurements can count when a card still gates public export.",
        "- The `implausible` column includes `stale?` notices for rows older than the family's last source commit.",
        "  Use `--no-stale` to skip the git history check.",
        "",
    ]
    lines += ["", f"## Problems ({len(probs)})", "", "| ref | family | kind | mode | issue |", "|---|---|---|---|---|"]
    for p in sorted(probs, key=lambda p: (p["kind"], p["family"], p["ref"])):
        lines.append(f"| {p['ref']} | {p['family']} | {p['kind']} | {p['mode']} | {p['issue'].replace('|', '/')} |")
    Path(a.out).write_text("\n".join(lines) + "\n")
    if a.csv:
        Path(a.csv).mkdir(parents=True, exist_ok=True)
        with (Path(a.csv) / "problems.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, ["ref", "family", "kind", "mode", "issue"])
            w.writeheader()
            w.writerows(probs)
    kinds = collections.Counter(p["kind"] for p in probs)
    print(dict(kinds), "total", len(probs))
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
