"""Numbers audit (CPU only, stdlib): run from the repo root: python3 results/raw/2026-11-15_audit/audit.py"""
import json, csv, glob, os, math, collections, tomllib, re
OUT = 'results/raw/2026-11-15_audit'
def jl(p): return [json.loads(l) for l in open(p) if l.strip()]
def wr(name, rows):
    cols = list(dict.fromkeys(k for r in rows for k in r)) or ['none']
    with open(f'{OUT}/{name}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
cards = {}
for p in glob.glob('src/fastforks/*/models/*/card.toml'):
    t = tomllib.load(open(p, 'rb')); fam = os.path.basename(os.path.dirname(p))
    for v in t.get('variants', []): cards[v['id']] = dict(family=fam, task=p.split('/')[2], card=p, text=open(p).read())
px = lambda i: (lambda h, w: int(h) * int(w))(*i.split('x')) if i and re.fullmatch(r'\d+x\d+', i) else 0
speed = jl('docs/data/speed.jsonl')
summary = collections.OrderedDict()
# ---------- (a)(b)(c) speed
D = collections.defaultdict(lambda: collections.defaultdict(list))
for r in speed:
    if r.get('ms') is None or r.get('status'): continue
    if r['mode'] in ('compile', 'tensorrt'): continue
    D[(r['ref'], r['input'])][r['mode']].append(r)
latest = lambda l: sorted(l, key=lambda r: r.get('date', ''))[-1]
def origs(d):
    best = {}
    for m in ('upstream', 'upstream-default'):
        for x in d.get(m, []):
            k = (m, x.get('tier'), x.get('dtype'))
            if k not in best or x['date'] >= best[k]['date']: best[k] = x
    return list(best.values())
A, B, C, rowsall = [], [], [], []
for (ref, inp), d in sorted(D.items()):
    if 'portable' not in d: continue
    P = latest(d['portable']); K = latest(d['kernels']) if 'kernels' in d else None
    o = origs(d)
    if not o: continue
    g = lambda x, k: x.get(k) if x else None
    o_ms = min(x['ms'] for x in o)
    o_al = [x['peak_alloc_mb'] for x in o if x.get('peak_alloc_mb')]; o_rs = [x['peak_reserved_mb'] for x in o if x.get('peak_reserved_mb')]
    o_al = min(o_al) if o_al else None; o_rs = min(o_rs) if o_rs else None
    row = dict(ref=ref, family=cards.get(ref, {}).get('family'), input=inp, orig_ms=o_ms, port_ms=P['ms'], k_ms=g(K, 'ms'),
               orig_alloc=o_al, port_alloc=g(P, 'peak_alloc_mb'), k_alloc=g(K, 'peak_alloc_mb'), orig_res=o_rs,
               port_res=g(P, 'peak_reserved_mb'), k_res=g(K, 'peak_reserved_mb'), port_note=g(P, 'note') or '')
    f = []
    # (a) memory: >1 % AND >16 MB (below that: allocator granularity)
    for kind, pm, om in (('alloc', row['port_alloc'], o_al), ('reserved', row['port_res'], o_rs)):
        if pm and om and pm > om * 1.01 and pm - om > 16: f.append(f'a_portable_{kind}_above_orig')
    if K:
        for kind, km, om in (('alloc', row['k_alloc'], o_al), ('reserved', row['k_res'], o_rs)):
            if km and om and km > om * 1.01 and km - om > 16: f.append(f'a_kernels_{kind}_above_orig')
    if o_ms < P['ms'] * 0.99: f.append('b_portable_slower_than_orig')
    if K and o_ms < K['ms'] * 0.99: f.append('b_kernels_slower_than_orig')
    if K and K['ms'] > P['ms'] * 1.01: f.append('c_kernels_slower_than_portable')
    if K:
        for kind in ('alloc', 'res'):
            kk, pp = row['k_' + kind], row['port_' + kind]
            if kk and pp and kk > pp * 1.01 and kk - pp > 16: f.append(f'c_kernels_{"reserved" if kind=="res" else kind}_above_portable')
    # cross-GPU/torch comparison (i)
    if len({x.get('torch') for x in [P, K] + o if x}) > 1: f.append('i_mixed_torch_in_comparison')
    row['flags'] = ';'.join(f); rowsall.append(row)
wr('speed_compare_all', rowsall)
for cat, name in (('a', 'a_memory_above_original'), ('b', 'b_slower_than_original'), ('c', 'c_kernels_worse_than_portable'), ('i', 'i_mixed_torch')):
    wr(name, [r for r in rowsall if any(x.startswith(cat + '_') for x in r['flags'].split(';'))])
cnt = collections.Counter(); refs_by = collections.defaultdict(set)
for r in rowsall:
    for x in r['flags'].split(';'):
        if x: cnt[x] += 1; refs_by[x].add(r['ref'])
summary['speed: ref x size comparisons'] = len(rowsall); summary['speed: refs compared'] = len({r['ref'] for r in rowsall})
for k in sorted(cnt): summary[f'{k}: rows / refs'] = f'{cnt[k]} / {len(refs_by[k])}'
json.dump({k: sorted(v) for k, v in refs_by.items()}, open(f'{OUT}/_flags_by_ref.json', 'w'))
# ---------- (e)(f) speed rows errors / fields
E_, F_ = [], []
for i, r in enumerate(speed):
    if r.get('status') or r.get('error'):
        bydesign = r['mode'] in ('upstream', 'upstream-default') and r.get('status') == 'oom'
        E_.append(dict(file='speed.jsonl', line=i + 1, ref=r['ref'], mode=r['mode'], input=r.get('input'), status=r.get('status'), error=(r.get('error') or '')[:120], by_design_orig_oom=bydesign))
    miss = [k for k in ('gpu', 'commit', 'date', 'ms', 'peak_alloc_mb', 'peak_reserved_mb', 'input', 'torch', 'command', 'protocol') if r.get(k) in (None, '') and not r.get('status')]
    if miss and r['mode'] not in ('compile', 'tensorrt'): F_.append(dict(file='speed.jsonl', line=i + 1, ref=r['ref'], mode=r['mode'], missing=';'.join(miss)))
    for k, v in r.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): E_.append(dict(file='speed.jsonl', line=i + 1, ref=r['ref'], mode=r['mode'], status='nan', error=k))
# ---------- eval files
PRIM = {'stereo': 'bad', 'depth': 'rel_absrel', 'flow': 'epe', 'classification': 'top1', 'segmentation': 'miou', 'detection': 'box_map', 'pose': 'keypoint_ap',
        'features': 'top1', 'promptable_segmentation': 'miou_best', 'matching': None, 'matting': None, 'tracking': None, 'multiview': None, 'interpolation': 'psnr', 'restoration': 'psnr', 'retrieval': None, 'calibration': None, 'relative_pose': None, 'volume': None}
LOWER_BETTER = {'bad', 'rel_absrel', 'epe', 'abs_rel', 'fl_all', 'mae', 'error'}
Dd, Dups, EV = [], [], []
for p in sorted(glob.glob('docs/data/*_eval.jsonl')):
    task = os.path.basename(p)[:-11]; rows = jl(p)
    if rows and 'ref' not in rows[0]: continue  # relative_pose: method-level schema
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for i, r in enumerate(rows):
        EV.append((task, i + 1, r))
        by[(r['ref'], r['dataset'])][r.get('mode')].append(r)
        if r.get('error'):
            bydesign = r.get('mode') in ('upstream', 'upstream-default', 'upstream-fp16', 'upstream16') and ('OOM' in r['error'] or 'out of memory' in r['error'].lower())
            E_.append(dict(file=os.path.basename(p), line=i + 1, ref=r['ref'], mode=r.get('mode'), input=r['dataset'], status='error', error=r['error'][:120], by_design_orig_oom=bydesign))
        miss = [k for k in ('gpu', 'commit', 'date', 'command', 'n', 'torch') if r.get(k) in (None, '')]
        if not r.get('error'):
            if r.get('mode') != 'upstream-cpu' and r.get('peak_alloc_mb') in (None, '') : miss.append('peak_alloc_mb')
        if miss: F_.append(dict(file=os.path.basename(p), line=i + 1, ref=r['ref'], mode=r.get('mode'), missing=';'.join(miss)))
        for k, v in (r.get('metrics') or {}).items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): E_.append(dict(file=os.path.basename(p), line=i + 1, ref=r['ref'], mode=r.get('mode'), status='nan', error=k))
    for (ref, ds), m in by.items():
        for mode, l in m.items():
            if len(l) > 1:
                Dups.append(dict(file=os.path.basename(p), ref=ref, dataset=ds, mode=mode, n_rows=len(l), dates=';'.join(sorted(str(x.get('date'))[:10] for x in l)),
                                 metrics=' | '.join(json.dumps(x.get('metrics')) for x in l)[:200], identical=len({json.dumps(x.get('metrics'), sort_keys=True) for x in l}) == 1))
        orig = (m.get('upstream') or [None])[-1]
        if not orig or orig.get('error'): continue
        om = orig.get('metrics') or {}
        keys = [PRIM.get(task)] if PRIM.get(task) in om else [k for k, v in om.items() if isinstance(v, (int, float))][:1]
        for mode in ('portable', 'kernels'):
            t = (m.get(mode) or [None])[-1]
            if not t or t.get('error'): continue
            tm = t.get('metrics') or {}
            for k in keys:
                if k in tm and isinstance(om[k], (int, float)) and om[k]:
                    rel = (tm[k] - om[k]) / abs(om[k])
                    worse = (rel > 0) == (k in LOWER_BETTER)
                    Dd.append(dict(task=task, ref=ref, dataset=ds, mode=mode, metric=k, original=om[k], value=tm[k], rel_diff_pct=round(100 * rel, 2), worse=worse, flagged=abs(rel) > 0.02,
                                   tolerance_in_card=bool(re.search(r'toler', cards.get(ref, {}).get('text', ''), re.I)), orig_dtype=orig.get('dtype'), mode_dtype=t.get('dtype'), n=t.get('n'), orig_n=orig.get('n')))
wr('d_accuracy_vs_original_all', Dd); wr('d_accuracy_flagged', [r for r in Dd if r['flagged']])
fl = [r for r in Dd if r['flagged']]
summary['(d) accuracy comparisons (primary metric, portable/kernels vs original fp32)'] = len(Dd)
summary['(d) flagged > 2 % relative, worse / better'] = f"{sum(r['worse'] for r in fl)} / {sum(not r['worse'] for r in fl)}  (refs worse: {len({r['ref'] for r in fl if r['worse']})})"
wr('e_errors_nan', E_); wr('f_missing_fields', F_); wr('h_duplicates_eval', Dups)
summary['(e) error/oom/nan rows total'] = len(E_)
summary['(e) of them by design (original OOM)'] = sum(bool(r.get('by_design_orig_oom')) for r in E_)
summary['(e) NOT by design'] = sum(not r.get('by_design_orig_oom') for r in E_)
fc = collections.Counter(k for r in F_ for k in r['missing'].split(';'))
summary['(f) rows with missing fields (by field)'] = f'{len(F_)} rows; ' + ', '.join(f'{k}={v}' for k, v in fc.most_common())
summary['(h) eval duplicates (same ref+mode+dataset), identical / differing'] = f"{sum(r['identical'] for r in Dups)} / {sum(not r['identical'] for r in Dups)}"
# (h) speed duplicates
SD = []
g = collections.defaultdict(list)
for i, r in enumerate(speed):
    if r.get('ms') is not None and not r.get('status'): g[(r['ref'], r['mode'], r['input'], r.get('tier'), r.get('dtype'), r.get('backend') and json.dumps(r['backend'], sort_keys=True))].append((i + 1, r))
for k, l in g.items():
    if len(l) > 1:
        ms = [x[1]['ms'] for x in l]; spread = (max(ms) - min(ms)) / min(ms)
        SD.append(dict(ref=k[0], mode=k[1], input=k[2], n_rows=len(l), lines=';'.join(str(x[0]) for x in l), ms=';'.join(map(str, ms)), dates=';'.join(str(x[1]['date']) for x in l), spread_pct=round(100 * spread, 1),
                       contradictory=spread > 0.10, torch_differ=len({x[1].get('torch') for x in l}) > 1))
wr('h_duplicates_speed', SD)
summary['(h) speed duplicate groups; ms spread > 10 % (contradictory)'] = f"{len(SD)}; {sum(r['contradictory'] for r in SD)}"
# (i) gpu/torch census across data
cen = collections.Counter()
for _, _, r in EV: cen[('eval', r.get('gpu'), r.get('torch'))] += 1
for r in speed: cen[('speed', r.get('gpu'), r.get('torch'))] += 1
wr('i_gpu_torch_census', [dict(kind=k[0], gpu=k[1], torch=k[2], rows=v) for k, v in sorted(cen.items(), key=lambda x: str(x))])
summary['(i) distinct (gpu, torch) combos in main tables'] = len(cen)
# (g) card text vs data
G = []
for ref, c in cards.items():
    for m in re.finditer(r'[^.\n]*(not measured|unmeasured|TODO|to be measured|pending)[^.\n]*', c['text'], re.I): G.append(dict(ref=ref, kind='stale_marker_in_card', text=m.group(0).strip()[:160]))
    for m in re.finditer(r'[^.\n]*\b(\d+(?:\.\d+)?)\s*x\s+(faster|slower)[^.\n]*', c['text'], re.I): G.append(dict(ref=ref, kind='speed_claim_in_card', text=m.group(0).strip()[:160]))
wr('g_card_text_claims', G)
summary['(g) card text hits (stale markers / speed claims, to check)'] = collections.Counter(r['kind'] for r in G)
for p in sorted(glob.glob('docs/models/*.md')):
    for n, l in enumerate(open(p), 1):
        if re.search(r'not measured|unmeasured', l, re.I) and not l.startswith('- Speed and memory'): G.append(dict(ref=os.path.basename(p), kind='md_stale', text=f'{n}: {l.strip()[:150]}'))
wr('g_card_text_claims', G)
for k, v in summary.items(): print(k, '=>', v)
json.dump({k: str(v) for k, v in summary.items()}, open(f'{OUT}/_summary.json', 'w'), indent=1)
