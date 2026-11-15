import csv,collections
R=list(csv.DictReader(open('results/raw/2026-11-15_kernel_audit/audit_rows.csv')))
N=list(csv.DictReader(open('results/raw/2026-11-15_kernel_audit/no_kernels_ranking.csv')))
def tbl(flag,cols,heads):
    out=['| '+' | '.join(heads)+' |','|'+'---|'*len(heads)]
    for x in R:
        if flag in x['flags'].split(';'): out.append('| '+' | '.join(x[c] for c in cols)+' |')
    return '\n'.join(out)
cnt=collections.defaultdict(set)
for x in R:
    for f in x['flags'].split(';'):
        if f: cnt[f].add(x['ref'])
refs={x['ref'] for x in R}; r452={x['ref'] for x in R if x['in452']=='True'}
base=['ref','input','orig_ms_best','port_ms','port_speedup','orig_mem_best','port_mem','port_mem_ratio']
bh=['ref','input','orig ms','portable ms','speedup','orig MB','portable MB','mem ratio']
kc=['ref','input','port_ms','k_ms','k_vs_port','k_speedup_vs_orig','orig_mem_best','port_mem','k_mem']
kh=['ref','input','portable ms','kernels ms','k vs port','k vs orig','orig MB','portable MB','kernels MB']
md=f"""# Kernel bar audit, 2026-11-15

Bar (user, 2026-11-15): every network, in BOTH modes, faster than the original, lower peak memory than the original, same task accuracy; kernels additionally faster than portable. CPU only: `docs/data/speed.jsonl` at `6ec60fba` (origin/work/ports) + the gap matrix `results/raw/2026-11-15_gaps/gaps.csv` (452 refs; the card registry has {len(refs)+19} refs, 19 of them have no speed row yet: any2full-s/b/l, depthfm-v1, lingbot-depth(-dc), marigold-v2-depth, omni-dc, video-depth-anything-metric-s, memfof-kitti/sintel/spring, aliked-flexglue, loma-b, lingbot-map, resplat-small/base, seedvr2-7b, tapnext-pp; they are in the audit as "no row"). Script: `results/raw/2026-11-15_kernel_audit/audit.py` (rows: `audit_rows.csv` one per ref x size, `audit_families.csv`), ranking: `judge.py` -> `no_kernels_ranking.csv`.

Method: per ref and tested size, the latest row per mode. "Original" = the STRICTEST original row at that size (fastest ms over the upstream / upstream-default fp32 / fp16 rows; lowest peak_alloc over the same rows). Peak memory = `peak_alloc_mb`. Tolerance: portable "slower" if original < 0.99 x portable; memory "above" if ratio > 1.01 (noise). Accuracy is not part of this table (see the accuracy results per family).

## Counts

{len(refs)} refs have rows ({len(r452)} of the 452-gap-matrix refs); {len(R)} ref x size rows.

| failure | refs | of them in the 452 |
|---|---|---|
""" + '\n'.join(f"| {k} | {len(v)} | {len(v & r452)} |" for k,v in sorted(cnt.items()))+f"""

`no_kernels_row` = {len(cnt['no_kernels_row'])} refs in {len({x['family'] for x in R if 'no_kernels_row' in x['flags'].split(';')})} families (31 families carry `status = "none"` in their card; the user counted 29, the difference is the 2 experimental cards flexglue / loma that the 452 matrix omits). Kernels-mem-above-portable is mostly a small fixed overhead of the kernel workspace (+1..5 %), see the table.

## (a) Refs failing the bar

### Portable slower than the original ({len(cnt['portable_slower'])} refs)
{tbl('portable_slower',base,bh)}

### Portable above the original's peak memory ({len(cnt['portable_mem_above'])} refs; speed column shows whether it is also slower)
{tbl('portable_mem_above',base,bh)}

### Kernels slower than portable ({len(cnt['kernels_slower_than_portable'])}) and slower than the original ({len(cnt['kernels_slower_than_orig'])})
{tbl('kernels_slower_than_portable',kc,kh)}

{tbl('kernels_slower_than_orig',kc,kh)}

### Kernels above the original's peak memory ({len(cnt['kernels_mem_above_orig'])} refs)
{tbl('kernels_mem_above_orig',kc,kh)}

Kernels peak memory above PORTABLE: {len(cnt['kernels_mem_above_portable'])} refs, full list in `audit_rows.csv` (flag `kernels_mem_above_portable`); worst: """
w=sorted([x for x in R if 'kernels_mem_above_portable' in x['flags'].split(';')],key=lambda x:-float(x['k_mem_vs_port']))[:12]
md+=', '.join(f"{x['ref']} {x['k_mem_vs_port']}x ({x['port_mem']} -> {x['k_mem']} MB)" for x in w)+'.\n'
md+="""
## (b) Families without kernels: reason real or not, ranked

Score = users (1-5, how likely a user runs it) x (expected kernels gain vs portable - 1) x 10. Expected gain from the measured portable numbers and from what the same shared kernels gave in sibling families (fastbb + dpt_fused ViT path: 1.4-3x vs portable in depth_anything_v2; cudnn_hacc / cutlass_conv CNN runners 1.2-1.5x); an estimate until built. "REAL" = measured slower or tiny / already minimal; "PARTLY" = the claim holds for one part only.

| rank | family | task | refs | verdict | users | gain | effort | portable vs orig (speed / mem) | why |
|---|---|---|---|---|---|---|---|---|---|
"""
for i,n in enumerate(N,1):
    md+=f"| {i} | {n['family']} | {n['task']} | {n['n_refs']} | {n['verdict']} | {n['users_1_5']} | {n['expected_gain_vs_portable']}x | {n['effort']} | {n['med_port_speedup_vs_orig']}x / {n['med_port_mem_ratio']}x | {n['note']} |\n"
md+="""
Summary: REAL = bisenet, inspyrenet (measured slower), pidinet, mlsd, teed, anyline (tiny / minimal); the three other edge families are UNMEASURED (one cudnn_hacc A/B each decides); everything else has no real reason. Portable also fails the bar for flowformer (slower, +33 % memory), gmflow (ties), neuflow-v2, film, moge-3, efficientvit-b2, squeezenet, the taggers / yolox / retinaface / rtmpose / swin-family memory (see (a)); these are portable fixes, not kernels work.

Step 2 picks (effort-adjusted from the top of the ranking, see the results files of each): see README log.
"""
open('results/2026-11-15_kernel_bar_audit.md','w').write(md)
print(len(md))
