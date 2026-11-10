"""The first `torch._dynamo` import (kernels load, `compile()`) does not pin the caller's frames in a cycle."""

from __future__ import annotations

import subprocess
import sys

import pytest

SCRIPT = """
import gc, sys, weakref
from fastforks._core.model import _import_dynamo
assert "torch._dynamo" not in sys.modules, "import fastforks must not import dynamo"
class Held: pass
def load():
    held = Held()  # stands for the model in `Model.__init__` / `compile()`
    {call}
    return weakref.ref(held)
gc.disable()
ref = load()
assert "torch._dynamo" in sys.modules
print("freed" if ref() is None else "pinned")
"""


def _run(call: str) -> str:
    out = subprocess.run([sys.executable, "-c", SCRIPT.format(call=call)], capture_output=True, text=True, check=True)
    return out.stdout.strip()


@pytest.mark.slow
def test_first_dynamo_import_frees_the_caller():
    assert _run("_import_dynamo()") == "freed"


EAGER = """
import sys, torch
from fastforks._core import model
from fastforks.shared import seginfer
assert "torch._dynamo" not in sys.modules, "import seginfer must not import dynamo"
assert model._AFTER_DYNAMO, "the eager label step waits for dynamo"
scores, sizes = torch.randn(1, 5, 4, 6), [(8, 12)]
want = seginfer.upsample_argmax(scores, sizes)
assert "torch._dynamo" not in sys.modules, "an eager call must not import dynamo"
model._import_dynamo()
assert not model._AFTER_DYNAMO
frames = []
def count(fn):
    def run(*a, **k):
        frames.append(torch.compiler.is_compiling())
        return fn(*a, **k)
    return run
torch.nn.functional.interpolate = count(torch.nn.functional.interpolate)  # `_argmax` calls it through `F`
got = torch.compile(lambda s: seginfer.upsample_argmax(s, sizes), backend="eager")(scores)
assert torch.equal(got, want)
print("eager" if frames == [False] else f"traced {frames}")
"""


@pytest.mark.slow
def test_seginfer_import_leaves_dynamo_out_and_labels_run_eagerly_in_compile():
    out = subprocess.run([sys.executable, "-c", EAGER], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "eager"
