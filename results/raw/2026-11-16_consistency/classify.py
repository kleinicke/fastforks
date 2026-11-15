"""Classify (a)-(c) findings -> findings_speed.csv, needs_attention.txt, counts (run after audit.py, repo root)."""
import csv, collections, json
OUT = 'results/raw/2026-11-16_consistency'
R = list(csv.DictReader(open(f'{OUT}/speed_compare_all.csv')))
fl = lambda x: float(x) if x not in ('', None) else None
F = []; fam_attn = collections.defaultdict(set); refs_attn = collections.defaultdict(list)
for r in R:
    for flag in [x for x in r['flags'].split(';') if x and x[0] in 'abc']:
        # numbers behind the flag
        if flag.startswith('b_'):
            other = fl(r['port_ms']) if 'portable' in flag else fl(r['k_ms']); ratio = other / fl(r['orig_ms'])
            tag = 'slower'; mag = ratio - 1; big = mag > 0.03
        else:
            if flag.startswith('a_'):
                kind = 'alloc' if 'alloc' in flag else 'res'; who = 'port' if 'portable' in flag else 'k'; base = fl(r['orig_' + kind])
            else:
                kind = 'alloc' if 'alloc' in flag else 'res'; who = 'k'; base = fl(r['port_' + kind])
                if flag == 'c_kernels_slower_than_portable': base = fl(r['port_ms']); 
            if flag == 'c_kernels_slower_than_portable': mag = fl(r['k_ms']) / base - 1; big = mag > 0.03
            else:
                mag = fl(r[f'{who}_{kind}']) / base - 1; diff = fl(r[f'{who}_{kind}']) - base
                big = mag > (0.05 if flag.startswith('c_') else 0.03) and diff > 64
        if flag.startswith('c_') and 'above_portable' in flag and not big: cls = 'explained: kernel workspace / allocator overhead <= 5 % (or <= 64 MB) over portable'
        elif not big: cls = 'explained: within 3 % / 64 MB (timer + allocator noise)'
        elif r['port_note'] and ('memory' in r['port_note'].lower() or 'slower' in r['port_note'].lower()): cls = 'explained: named in the row note'
        else: cls = 'needs optimisation or re-measure'
        F.append(dict(ref=r['ref'], family=r['family'], input=r['input'], flag=flag, magnitude_pct=round(100 * mag, 1), classification=cls))
        if cls.startswith('needs') and flag[0] in 'ab' or (cls.startswith('needs') and flag[0] == 'c'):
            fam_attn[r['family']].add(r['ref']); refs_attn[r['ref']].append(f"{flag}@{r['input']}:{round(100*mag,1)}%")
with open(f'{OUT}/findings_speed.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, list(F[0])); w.writeheader(); w.writerows(F)
with open(f'{OUT}/needs_attention.txt', 'w') as f:
    f.write('# ref\tfamily\tfindings (flag@size:excess%)\taction\n# a=memory above original, b=slower than original, c=kernels worse than portable; only findings beyond noise (>3 % and >64 MB for memory; >3 % for time; kernels-vs-portable memory >5 %)\n')
    for ref in sorted(refs_attn):
        fam = next(x['family'] for x in F if x['ref'] == ref)
        flags = refs_attn[ref]
        act = 'optimise family (kernel agent)' if any(x[0] in 'bc' for x in flags) or any(x.startswith('a_kernels') for x in flags) else 'optimise memory (portable path)'
        f.write(f"{ref}\t{fam}\t{';'.join(flags)}\t{act}\n")
c = collections.Counter((x['flag'], x['classification'].split(':')[0]) for x in F)
for k, v in sorted(c.items()): print(k, v)
print('refs needing attention', len(refs_attn)); print(collections.Counter(next(x['family'] for x in F if x['ref'] == r) for r in refs_attn).most_common())
