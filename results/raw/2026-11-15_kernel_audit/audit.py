"""Kernel-bar audit: portable/kernels vs original from docs/data/speed.jsonl (CPU only)."""
import json, csv, tomllib, glob, os, collections
G='/home/florian/projects/experiments/017-fastforks-lib/wt-ports/results/raw/2026-11-15_gaps/gaps.csv'
g452={r['ref'] for r in csv.DictReader(open(G))}
refs=[]
for p in sorted(glob.glob('src/fastforks/*/models/*/card.toml')):
    t=tomllib.load(open(p,'rb'))
    for v in t.get('variants',[]): refs.append(dict(ref=v['id'],task=p.split('/')[2],family=os.path.basename(os.path.dirname(p))))
rows=[json.loads(l) for l in open('docs/data/speed.jsonl')]
def px(i):
    h,w=i.split('x'); return int(h)*int(w)
# latest row per (ref, mode-class, input)
D=collections.defaultdict(dict)  # ref -> (cls,input)->list
for r in rows:
    if r.get('status') or r.get('ms') is None: 
        if r.get('status')=='oom': D[r['ref']].setdefault(('oom_'+r['mode'],r.get('input')),r)
        continue
    cls=r['mode']
    if cls in ('compile','tensorrt'): continue
    D[r['ref']].setdefault((cls,r.get('input')),[]).append(r)
def latest(l): return sorted(l,key=lambda r:r.get('date',''))[-1]
cards={}
for p in glob.glob('src/fastforks/*/models/*/card.toml'):
    t=tomllib.load(open(p,'rb')); fam=os.path.basename(os.path.dirname(p))
    k=t.get('modes',{}).get('kernels',{})
    cards[fam]=(p,k.get('status'),k.get('reason',''))
out=[];fails=collections.defaultdict(list)
for r in refs:
    ref=r['ref'];d=D.get(ref,{})
    inputs=sorted({i for (c,i) in d if c=='portable' and isinstance(d[(c,i)],list)},key=px,reverse=True)
    fam=r['family']; has_k=any(c=='kernels' for (c,i) in d)
    for inp in inputs:
        P=latest(d[('portable',inp)])
        K=latest(d[('kernels',inp)]) if ('kernels',inp) in d else None
        origs=[latest(d[(c,inp)]) for c in ('upstream','upstream-default') if (c,inp) in d]
        # also all tiers
        allo=[x for c in ('upstream','upstream-default') for x in d.get((c,inp),[])]
        # per (c,tier) latest
        best={}
        for x in allo: 
            key=(x['mode'],x.get('tier'),x.get('dtype'))
            if key not in best or x.get('date','')>best[key].get('date',''): best[key]=x
        allo=[x for x in best.values() if x.get('ms')]
        if not allo: continue
        f=lambda x,k: x.get(k)
        o_ms=min(x['ms'] for x in allo)
        mems=[x['peak_alloc_mb'] for x in allo if x.get('peak_alloc_mb')]
        o_mem=min(mems) if mems else None
        d32=[x for x in allo if x['mode']=='upstream-default' or x.get('dtype')=='float32']
        o32_ms=min((x['ms'] for x in d32),default=None)
        row=dict(ref=ref,in452=ref in g452,task=r['task'],family=fam,input=inp,orig_ms_best=o_ms,orig_mem_best=o_mem,
          port_ms=P['ms'],port_mem=P.get('peak_alloc_mb'),k_ms=K and K['ms'],k_mem=K and K.get('peak_alloc_mb'))
        row['port_speedup']=round(o_ms/P['ms'],3)
        row['port_mem_ratio']=round(P['peak_alloc_mb']/o_mem,3) if o_mem and P.get('peak_alloc_mb') else ''
        if K:
            row['k_vs_port']=round(P['ms']/K['ms'],3)
            row['k_speedup_vs_orig']=round(o_ms/K['ms'],3)
            row['k_mem_vs_port']=round(K['peak_alloc_mb']/P['peak_alloc_mb'],3) if K.get('peak_alloc_mb') and P.get('peak_alloc_mb') else ''
            row['k_mem_vs_orig']=round(K['peak_alloc_mb']/o_mem,3) if o_mem and K.get('peak_alloc_mb') else ''
        fl=[]
        if o_ms<P['ms']*0.99: fl.append('portable_slower')
        if row['port_mem_ratio']!='' and row['port_mem_ratio']>1.01: fl.append('portable_mem_above')
        if K:
            if row['k_vs_port']<1.0: fl.append('kernels_slower_than_portable')
            if row['k_speedup_vs_orig']<1.0: fl.append('kernels_slower_than_orig')
            if row['k_mem_vs_port']!='' and row['k_mem_vs_port']>1.01: fl.append('kernels_mem_above_portable')
            if row['k_mem_vs_orig']!='' and row['k_mem_vs_orig']>1.01: fl.append('kernels_mem_above_orig')
        else: fl.append('no_kernels_row')
        row['flags']=';'.join(fl); out.append(row)
cols=list(dict.fromkeys(k for r in out for k in r))
w=csv.DictWriter(open('results/raw/2026-11-15_kernel_audit/audit_rows.csv','w'),cols); w.writeheader(); w.writerows(out)
# per ref summary: flags union
byref=collections.defaultdict(lambda:collections.defaultdict(list))
for r in out:
    for fl in r['flags'].split(';'):
        if fl: byref[r['ref']][fl].append(r)
json.dump({k:{f:[(x['input'],x) for x in v] for f,v in d.items()} for k,d in byref.items()},open('/tmp/x.json','w'),default=str) if False else None
# family table
fams=collections.defaultdict(list)
for r in refs: fams[r['family']].append(r['ref'])
fr=[]
for fam,rl in sorted(fams.items()):
    c=cards.get(fam,(None,None,''))
    sub=[x for x in out if x['family']==fam]
    nref=len(rl); ref_with_k=len({x['ref'] for x in sub if x['k_ms']})
    fr.append(dict(family=fam,n_refs_452=sum(1 for x in rl if x in g452),task=refs[[x['ref'] for x in refs].index(rl[0])]['task'],n_refs=nref,card_kernels_status=c[1],
      refs_with_kernels_row=ref_with_k,
      med_port_speedup=sorted(x['port_speedup'] for x in sub)[len(sub)//2] if sub else '',
      med_port_mem_ratio=(lambda v:sorted(v)[len(v)//2] if v else '')([x['port_mem_ratio'] for x in sub if x['port_mem_ratio']!='']),
      med_orig_ms=sorted(x['orig_ms_best'] for x in sub)[len(sub)//2] if sub else '',
      no_kernels_reason=c[2]))
w=csv.DictWriter(open('results/raw/2026-11-15_kernel_audit/audit_families.csv','w'),list(fr[0])); w.writeheader(); w.writerows(fr)
cnt=collections.Counter()
for ref,d in byref.items():
    for fl in d: cnt[fl]+=1
print('rows',len(out),'refs with rows',len({x['ref'] for x in out}),cnt)
print('refs total',len(refs),'refs without rows',[r['ref'] for r in refs if r['ref'] not in {x['ref'] for x in out}])
print('nonek families',sum(1 for r in fr if r['card_kernels_status']=='none'),'families w/o any k row',sum(1 for r in fr if r['refs_with_kernels_row']==0))
