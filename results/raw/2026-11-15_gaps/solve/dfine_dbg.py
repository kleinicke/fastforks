import sys, tempfile, time
from pathlib import Path
import numpy as np, onnx, torch
import fastforks.detection as task
from fastforks._core.export import build_tensorrt, TensorRTEngine
H, W = 800, 1344
m = task.load("dfine-x-coco", mode="portable", device="cuda")
inputs = m.example_inputs(H, W, batch=1)
tmp = Path(tempfile.mkdtemp(prefix="dfdbg-", dir="/home/florian/scratch_tests"))
p = m.export_onnx(tmp / "m.onnx", H, W, batch=1)
g = onnx.load(str(p), load_external_data=False)
print("nodes", len(g.graph.node), "outputs", [o.name for o in g.graph.output], flush=True)
nodes = [n for n in g.graph.node if n.op_type not in ("Constant", "Cast", "Shape", "Gather", "Unsqueeze", "Reshape", "Transpose")]
k = int(sys.argv[1]) if len(sys.argv) > 1 else 24
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else len(nodes)
sel = nodes[lo:hi]
step = max(1, len(sel) // k)
picks = sel[::step]
print("picked", len(picks), "of", len(nodes), flush=True)
extra = []
for n in picks:
    extra.append(n.output[0])
    g.graph.output.append(onnx.ValueInfoProto(name=n.output[0]))
onnx.save(onnx.load(str(p)) if False else g, str(tmp / "m2.onnx"), save_as_external_data=False) if False else None
full = onnx.load(str(p))
for n in picks:
    full.graph.output.append(onnx.ValueInfoProto(name=n.output[0]))
import shutil
q = tmp / "m2.onnx"
onnx.save(full, str(q), save_as_external_data=True, location="m2.data")
eng = build_tensorrt(q, tmp / "m.engine")
e = TensorRTEngine(eng)
outs = e.run(*inputs)
names = e.outputs if hasattr(e, "outputs") else None
print("n outs", len(outs), flush=True)
for i, (name, o) in enumerate(outs.items()):
    t = o if torch.is_tensor(o) else torch.as_tensor(o)
    f = t.float()
    idx = next((j for j, n in enumerate(nodes) if n.output[0] == name), -1)
    print(i, idx, name, t.dtype, "nan" if torch.isnan(f).any() else "ok", "maxabs", float(torch.nan_to_num(f, 0, 0, 0).abs().max()), flush=True)
shutil.rmtree(tmp, ignore_errors=True)
