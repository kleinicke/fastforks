"""Agreement of the WD v3 taggers (original fp16, portable, kernels) with the original fp32 network on the first N COCO val2017
images (fixed order = sorted file names). tag F1 at probability 0.35 (the card's general threshold), top-1 tag agreement.
python agree_wd.py REF,REF,.. N OUT.jsonl"""
import json, sys, time, datetime, subprocess
from pathlib import Path
import numpy as np, torch
from PIL import Image
W = Path("/home/florian/projects/experiments/017-fastforks-lib/wt-accrows")
sys.path.insert(0, str(W / "tools/golden"))
import libnet_classification as g
from fastforks import classification
from fastforks._core import registry

refs, n, out = sys.argv[1].split(","), int(sys.argv[2]), sys.argv[3]
files = sorted(Path("/home/florian/projects/data/coco/val2017").glob("*.jpg"))[:n]
ims = [torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).permute(2, 0, 1).contiguous() for f in files]
commit = subprocess.run(["git", "-C", str(W), "rev-parse", "--short=8", "HEAD"], capture_output=True, text=True).stdout.strip()
cards = registry.cards("classification")
THR = 0.35
for ref in refs:
    card, variant = cards[ref]
    io = variant.io
    net = g.reference(card, variant)[0].eval().cuda()
    res = {}
    for name, half in (("upstream", False), ("upstream-fp16", True)):
        torch.cuda.reset_peak_memory_stats()
        if half:
            net = net.half()
        with torch.inference_mode():
            x0 = g.prepare(ims[0].cuda()[None].float() / 255, io)
            net(x0.half() if half else x0); torch.cuda.synchronize(); t = time.perf_counter()
            lg = torch.stack([net((lambda p: p.half() if half else p)(g.prepare(x.cuda()[None].float() / 255, io)))[0].float() for x in ims])
            torch.cuda.synchronize()
        res[name] = (lg, 1e3 * (time.perf_counter() - t) / len(ims), torch.cuda.max_memory_reserved() >> 20, "float16" if half else "float32")
    del net; torch.cuda.empty_cache()
    for mode in ("portable", "kernels"):
        torch.cuda.reset_peak_memory_stats()
        m = classification.load(ref, mode=mode, device="cuda")
        with torch.inference_mode():
            m(ims[0]); torch.cuda.synchronize(); t = time.perf_counter()
            got = torch.stack([m(x).logits.float() for x in ims]); torch.cuda.synchronize()
        res[mode] = (got, 1e3 * (time.perf_counter() - t) / len(ims), torch.cuda.max_memory_reserved() >> 20, str(m.backend.dtype).removeprefix("torch."))
        del m; torch.cuda.empty_cache()
    pw = res["upstream"][0].sigmoid(); bw = pw > THR
    for mode, (lg, ms, mem, dt) in res.items():
        pg = lg.sigmoid(); a = pg > THR
        tp = (a & bw).sum().item()
        rec = {"ref": ref, "mode": mode, "dtype": dt, "n": len(ims), "dataset": f"coco-val2017-first{len(ims)}-tags",
               "tag_f1": round(100 * 2 * tp / max(1, a.sum().item() + bw.sum().item()), 3),
               "top1_agree": round(100 * (pg.argmax(-1) == pw.argmax(-1)).float().mean().item(), 2),
               "max_abs_prob": round(float((pg - pw).abs().max()), 4), "mean_abs_prob": round(float((pg - pw).abs().mean()), 7),
               "tags_per_image_original": round(bw.sum().item() / len(ims), 2),
               "ms_per_img": round(ms, 2), "peak_reserved_mb": mem, "commit": commit, "gpu": torch.cuda.get_device_name(),
               "date": datetime.datetime.now().astimezone().isoformat(timespec="minutes"), "command": " ".join(["python", *sys.argv])}
        print(json.dumps(rec), flush=True)
        with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
