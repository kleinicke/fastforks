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
