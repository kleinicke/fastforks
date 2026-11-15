import os,sys,collections,filecmp
roots=["/home/florian/projects/data/ff-weights","/home/florian/projects/data/gated_models","/home/florian/.cache/fastforks/blobs","/home/florian/.cache/huggingface/hub/blobs","/home/florian/projects/data/multiview/weights","/home/florian/projects/data/mono","/home/florian/projects/data/matching","/home/florian/projects/data/classification","/home/florian/projects/data/segmentation","/home/florian/projects/data/matting","/home/florian/projects/data/restoration"]
by=collections.defaultdict(list)
for r in roots:
  for dp,dn,fn in os.walk(r):
    for f in fn:
      p=os.path.join(dp,f)
      try:
        st=os.lstat(p)
      except: continue
      if os.path.islink(p) or st.st_size<100e6: continue
      by[st.st_size].append((p,st.st_ino,st.st_dev))
freed=0
for s,l in sorted(by.items(),reverse=True):
  inos={}
  for p,i,d in l: inos.setdefault((d,i),[]).append(p)
  if len(inos)<2: continue
  keys=list(inos); base=inos[keys[0]][0]
  for k in keys[1:]:
    if k[0]!=keys[0][0]: continue
    p=inos[k][0]
    if filecmp.cmp(base,p,shallow=False):
      for q in inos[k]:
        os.link(base,q+".lnk"); os.replace(q+".lnk",q)
      freed+=s; print("dedup",s//10**6,p)
print("freed MB",freed//10**6)
