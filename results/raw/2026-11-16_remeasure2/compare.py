"""compare new rows (rows_*.jsonl in a dir) with the latest old row of docs/data/speed.jsonl; print table, flag >3 % slower or more memory."""
import json,glob,sys
old={}
for l in open('docs/data/speed.jsonl'):
    r=json.loads(l)
    if r['mode'] in('kernels','portable') and 'ms' in r:
        k=(r['ref'],r['mode'],r['input'])
        if k not in old or r['date']>=old[k]['date']: old[k]=r
new=[]
for f in sorted(glob.glob(sys.argv[1]+'/rows_*.jsonl')):
    for l in open(f):
        r=json.loads(l)
        if 'ms' in r and r['mode'] in('kernels','portable'): new.append(r)
        else: print('NOROW',f.split('/')[-1],str(r)[:150])
flag=[]
for r in new:
    o=old.get((r['ref'],r['mode'],r['input']))
    if not o: print('nonold',r['ref'],r['mode'],r['input']); continue
    ds=r['ms']/o['ms']-1; da=r['peak_alloc_mb']-o['peak_alloc_mb']; dr=r['peak_reserved_mb']-o['peak_reserved_mb']
    bad = ds>0.03 or da>max(8,0.02*o['peak_alloc_mb']) or dr>max(8,0.02*o['peak_reserved_mb'])
    line=f"{r['ref']} {r['mode'][0]} {r['input']}: {o['ms']:.2f}->{r['ms']:.2f} ({ds*100:+.1f}%) alloc {o['peak_alloc_mb']}->{r['peak_alloc_mb']} res {o['peak_reserved_mb']}->{r['peak_reserved_mb']}"
    print(('FLAG ' if bad else '     ')+line)
