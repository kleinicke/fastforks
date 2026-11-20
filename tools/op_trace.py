"""Find the first PyTorch op whose output differs between two machines (e.g. Linux and Windows) on a golden input.

    python tools/op_trace.py record REF MODE OUT.json [--device cuda|cpu] [--sdpa BACKEND,...] [--max-ops N]
    python tools/op_trace.py compare A.json B.json [--show N]

`record` runs the golden forward of REF (as tests/test_golden.py's `run`, so FASTFORKS_GOLDEN_PATH and the weights must
be available) under a `TorchDispatchMode` and writes, for every aten op output, its name, shape, dtype, a hash of its
bytes and its float64 sum. `compare` prints the first ops whose output bytes differ and which op kinds differ most.
The first differing op names the cause: its inputs were bit-equal, so the op itself computes differently (an ATen CPU
kernel built by another compiler, another SDPA backend, ...). Constants computed at load time (RoPE tables, position
embeddings) are not traced: when the first difference is an op that reads one, compare that constant directly.
`--sdpa` restricts `torch.nn.attention.sdpa_kernel` (e.g. `efficient` reproduces, on Linux, a PyTorch build without
flash attention such as the Windows wheels). Run both machines from the same commit.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
HASH_LIMIT = 1 << 24  # bytes: larger outputs are compared by their float64 sum only (hashing them is slow)


def record(args: argparse.Namespace) -> int:
    """Trace the golden forward of `args.ref`; write the rows to `args.out`; return the number of rows."""
    ref, mode, out, sdpa, max_ops = args.ref, args.mode, args.out, args.sdpa, args.max_ops
    import torch  # noqa: PLC0415 - not needed by `compare`
    from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415
    from torch.utils._python_dispatch import TorchDispatchMode  # noqa: PLC0415 - the public API of dispatch modes
    from torch.utils._pytree import tree_flatten  # noqa: PLC0415

    sys.path[:0] = [str(ROOT / "tests"), str(ROOT / "tools")]
    import test_golden as tg  # noqa: PLC0415 - needs the sys.path entries

    rows: list[list[Any]] = []

    def row(func: object, t: torch.Tensor) -> list[Any]:
        x = t.detach()
        s = float((torch.view_as_real(x) if x.dtype.is_complex else x).double().sum())
        if x.numel() * x.element_size() > HASH_LIMIT:
            return [str(func), list(x.shape), str(x.dtype), repr(s), s]
        data = x.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return [str(func), list(x.shape), str(x.dtype), hashlib.sha1(data).hexdigest()[:16], s]  # noqa: S324 - a fingerprint

    class Trace(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # noqa: ANN001, ANN204 - torch's dispatch signature
            r = func(*args, **(kwargs or {}))
            if len(rows) < max_ops:
                flat, _ = tree_flatten(r)
                rows.extend(
                    row(func, t)
                    for t in flat
                    if isinstance(t, torch.Tensor)
                    and t.numel()
                    and t.layout == torch.strided
                    and t.device.type != "meta"
                )
            return r

    names = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "math": SDPBackend.MATH,
    }
    forward = tg.forward

    def traced(*a: Any, **kw: Any) -> Any:  # noqa: ANN401 - test_golden.forward's own signature
        backends = sdpa_kernel([names[s] for s in sdpa.split(",")]) if sdpa else contextlib.nullcontext()
        with backends, Trace():
            return forward(*a, **kw)

    tg.forward = traced
    want = tg.golden(ref)
    tg.run(ref, mode=mode, device=args.device, opts=tg.options(ref, want), want=want, built=tg.gpu_built(ref, mode))
    out.write_text(json.dumps(rows), encoding="utf-8")
    return len(rows)


def compare(a: list[list[Any]], b: list[list[Any]], show: int) -> int | None:
    """Print the first `show` differing ops of two traces and the op kinds that differ; return the first index."""
    first, shown, kinds = None, 0, Counter()
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x[:3] != y[:3]:
            print(f"structure differs at op {i}: {x[:3]} vs {y[:3]} (data-dependent shapes; stop)")
            break
        if x[3] == y[3]:
            continue
        kinds[x[0]] += 1
        if first is None:
            first = i
            for j in range(max(0, i - 3), i):
                print(f"  same {j} {a[j][0]} {a[j][1]} {a[j][2]}")
        if shown < show:
            print(f"  DIFF {i} {x[0]} {x[1]} {x[2]} sum {x[4]:.9g} vs {y[4]:.9g}")
            shown += 1
    print(f"ops {len(a)} / {len(b)}; first difference: {first}")
    print("differing op kinds:", kinds.most_common(10))
    return first


def main() -> None:
    """Command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("ref")
    r.add_argument("mode", choices=["portable", "kernels"])
    r.add_argument("out", type=Path)
    r.add_argument("--device", default="cuda")
    r.add_argument("--sdpa", help="comma list of SDPA backends allowed: flash, efficient, cudnn, math")
    r.add_argument("--max-ops", type=int, default=200_000)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--show", type=int, default=10)
    args = ap.parse_args()
    if args.cmd == "record":
        n = record(args)
        print(f"{n} op outputs -> {args.out}")
    else:
        load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731 - one-line helper
        compare(load(args.a), load(args.b), args.show)


if __name__ == "__main__":
    main()
