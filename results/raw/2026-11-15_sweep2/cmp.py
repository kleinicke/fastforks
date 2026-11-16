import json,sys
D='results/raw/2026-11-15_sweep2/'
new=[json.loads(l) for l in open(D+'rows.jsonl')][498:]
new=[r for r in new if 'error' not in r and r['side']=='lib']
fl={}
for l in open('docs/data/speed.jsonl'):
    r=json.loads(l); fl[r['ref'],r['mode'],r['input']]=r
n=0
for r in new:
    k=(r['ref'],r['mode'],r['input']); f=fl.get(k)
    p=fl.get((r['ref'],'portable',r['input'])); o=None
    flags=[]
    if f:
        if r['ms']>f['ms']*(1.05+f.get('noise',0)): flags.append(f"ms {f['ms']:.2f}->{r['ms']:.2f}")
        if r['peak_alloc_mb']>f['peak_alloc_mb']*1.03+8: flags.append(f"alloc {f['peak_alloc_mb']}->{r['peak_alloc_mb']}")
    else: flags.append('nofloor')
    if p and r['ms']>p['ms']: flags.append(f"slower than portable {p['ms']:.2f}")
    if p and r['peak_alloc_mb']>p['peak_alloc_mb']: flags.append(f"bigger than portable {p['peak_alloc_mb']}")
    if flags: print(k, '; '.join(flags))
print(len(new),'ok rows')
