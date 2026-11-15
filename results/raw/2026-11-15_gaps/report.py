import csv, collections, sys
from pathlib import Path
D = Path(__file__).resolve().parent; ROOT = D.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fastforks._core import registry
R = list(csv.DictReader(open(D / "gaps.csv")))
fam = collections.defaultdict(list)
for r in R: fam[r["family"]].append(r)
meta = {}
for t in registry.TASKS:
    for m, (c, v) in registry.cards(t).items(): meta[c.family] = (c.kernel_libs, c.no_kernels_reason)
nocache = {l.split("\t")[0] for l in open(D / "weights_not_cached.txt")}
trtmiss = [r["ref"] for r in R if r["tensorrt"] == "missing"]
cmpmiss = [r["ref"] for r in R if r["compile"] == "missing"]
nk_by_design = [f for f, rs in fam.items() if all(x["kernels"] == "missing" for x in rs) and meta[f][1]]
nk_nodesign = [f for f, rs in fam.items() if all(x["kernels"] == "missing" for x in rs) and not meta[f][1]]
partial = [(f, [x["ref"] for x in rs if x["kernels"] == "missing"]) for f, rs in fam.items() if any(x["kernels"] == "missing" for x in rs) and any(x["kernels"] != "missing" for x in rs)]
trterr = [(r["ref"], r["tensorrt"]) for r in R if r["tensorrt"].startswith("error")]
cmperr = [(r["ref"], r["compile"]) for r in R if r["compile"].startswith("error")]
n = len(R)
L = []
L += ["# Measurement gaps, 2026-11-15", "",
 f"Matrix of every model ref of the library ({n} refs, {len(fam)} families, from `registry.cards`) against `docs/data/speed.jsonl` and `docs/data/speed_compile.jsonl` (side `lib`). Per-ref csv: `results/raw/2026-11-15_gaps/gaps.csv` (made by `gapmatrix.py`; plan by `plan.py`). Commit of the data: `git log -1 -- docs/data` at the time of writing.", "",
 "## Counts", "", "| column | ok | missing | error / OOM |", "|---|---|---|---|"]
for c in ("upstream", "upstream-default", "portable", "kernels", "compile", "tensorrt"):
    ok = sum(r[c] == "ok" for r in R); ms = sum(r[c] == "missing" for r in R)
    L.append(f"| {c} | {ok} | {ms} | {n - ok - ms} |")
L += ["", "## (a) No original (upstream / upstream-default) speed + peak-memory row", "",
 "Every ref has an `upstream` OR `upstream-default` row except 5 (and every ok original row carries peak memory). The 5 need an upstream clone that is not on disk (a download, not allowed inside the queue lock):", ""]
for r in R:
    if r["orig_ms"] == "False":
        why = {"memfof": "needs `FF_MEMFOF_REPO` (clone of msu-video-group/memfof)", "tapnext": "needs the `tapnet` repo", "video_depth_anything": "no original adapter in `tools/_original.py`"}[r["family"]]
        L.append(f"- `{r['ref']}` ({r['family']}): {why}" + ("; also no portable row yet" if r["portable"] == "missing" else ""))
L += ["", "Four of them (memfof-sintel, memfof-spring, tapnext-pp, video-depth-anything-metric-s) have no portable row either: `tools/speed_check.py` builds its jobs from the floor's portable rows, so they cannot be queued before their owner records portable / kernels rows (`tools/speed_survey.py`-style).", "",
 f"## (b) No TensorRT row: {len(trtmiss)} refs", "",
 f"Of these, {sum(r not in nocache for r in trtmiss)} have cached weights (queued), {sum(r in nocache for r in trtmiss)} do not (weights would have to be downloaded: `weights_not_cached.txt` lists all {len(nocache)} refs without cached weights). Missing:", "", ", ".join(f"`{r}`" for r in trtmiss), "",
 f"## (c) No compile row: {len(cmpmiss)} refs", "", f"{sum(r not in nocache for r in cmpmiss)} with cached weights (queued), {sum(r in nocache for r in cmpmiss)} without. Missing:", "", ", ".join(f"`{r}`" for r in cmpmiss), "",
 "## (d) Rows that are errors / OOM / NaN", "",
 "No OOM rows exist in the original columns (all 447 `upstream` rows have ms + memory). Failed rows are all in the TRT / compile files:", "",
 f"TensorRT, {len(trterr)} refs (real failures of the export / engine build, recorded as `fails`; none is a by-design reason such as original-fp32 OOM; they are TRT-export limits of the model and stay documented as failures, not retried):", "", "| ref | reason |", "|---|---|"]
for r, e in trterr: L.append(f"| `{r}` | {e[7:110]} |")
L += ["", f"Compile, {len(cmperr)} refs:", ""]
for r, e in cmperr: L.append(f"- `{r}`: {e[7:140]}" + (" (torch dynamo guard bug; real failure)" if "Guard" in e else " (cold compile exceeded the 290 s step limit; a real gap, needs a longer unlocked-lock slot)"))
L += ["", "## (e) Kernels", "",
 f"`kernels` row missing for {sum(r['kernels'] == 'missing' for r in R)} refs. {len(nk_by_design)} families ({sum(len(fam[f]) for f in nk_by_design)} refs) declare `[modes.kernels] status = \"none\"` with a reason in their card (by design); families without a declared reason: {nk_nodesign or 'none'}. Partly missing:", ""]
for f, rs in partial: L.append(f"- `{f}`: {', '.join(rs)} (new ports without a floor row)")
L += ["", "By-design families: " + ", ".join(f"`{f}`" for f in sorted(nk_by_design)), "",
 "## Start-cost columns", "",
 f"Cold start (no engine / no cache) exists for all ok rows: TRT `onnx_export_s` + `engine_build_s` ({sum(r['tensorrt_startcold'] == 'True' for r in R)} rows), compile `compile_s` = first call on empty Inductor / Triton caches ({sum(r['compile_startcold'] == 'True' for r in R)} rows). The START WITH AN ALREADY COMPILED MODEL did not exist in any row (0 of {sum(r['tensorrt'] == 'ok' for r in R)} TRT, 0 of {sum(r['compile'] == 'ok' for r in R)} compile rows). Added to `tools/speed_check.py` (138e4308): `engine_load_s` + `engine_first_call_s` (load of the cached engine file and its first call) in every TRT row, `compile_warm_s` (second start on the filled caches, `--warm-start`) in every compile row.", "",
 "## Queue (unit `gapfill-trt`)", "",
 "`results/raw/2026-11-15_gaps/queue.sh`: (1) originals skipped (above), (2) 7 TRT, (3) 6 compile gaps with cached weights into `speed_compile_new.jsonl`, merged into `docs/data/speed_compile.jsonl` (new keys only) and pushed after `tests/test_docs.py tests/test_policy.py tests/test_speed_floor.py` pass (`publish.sh`), (4) re-measure of the 193 TRT / 216 compile refs that have cached weights with the start columns, into `backfill_trt.jsonl` / `backfill_compile.jsonl` (existing rows untouched; the owner decides how to merge). Log: `queue.log`, `queue_steps.log`. Every step: `gpulock-verylow timeout 175`, a failed step is logged and the queue goes on.", ""]
(ROOT / "results/2026-11-15_measurement_gaps.md").write_text("\n".join(L))
print(len(L))
