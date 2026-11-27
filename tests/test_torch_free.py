"""MLX runs without torch (docs/plans/torch-free.md, step 1): torch stays installed, but never imported.

- Importing `fastforks` and every task package imports no torch (fast, no GPU).
- `TORCH_FREE`: the refs whose MLX runners compute on NumPy arrays (`Runner.arrays = "numpy"`); their model modules
  import without torch. A family joins by porting its MLX runners (see the plan) and adding a ref here.
- `test_mlx_without_torch`: each of those refs, in a subprocess that cannot import torch, with a warm MLX conversion
  cache and NumPy outputs, is within the golden's GPU tolerances (`test_golden._check`).
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fastforks._core import registry

# task -> refs (with a golden) whose every MLX runner computes on NumPy arrays: a warm MLX run imports no torch
TORCH_FREE: dict[str, tuple[str, ...]] = {
    "depth": ("depth-anything-v2-s@1",),
}

# a meta-path finder that fails every torch import and records it (an `except ImportError` must not hide one)
BLOCK = """
import importlib.abc, sys
BLOCKED = []
class _NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] in {"torch", "torchvision", "triton"}:
            BLOCKED.append(name)
            raise ImportError(f"torch-free run imported {name}")
sys.meta_path.insert(0, _NoTorch())
def done():
    assert not BLOCKED, f"imported {BLOCKED}"
    assert "torch" not in sys.modules
"""


def _blocked(code: str, *args: str) -> subprocess.CompletedProcess[str]:
    src = Path(__file__).parents[1] / "src"
    env = os.environ | {"PYTHONPATH": os.pathsep.join([str(src), os.environ.get("PYTHONPATH", "")])}
    return subprocess.run(
        [sys.executable, "-c", BLOCK + code, *args], capture_output=True, text=True, env=env, check=False
    )


def _ok(out: subprocess.CompletedProcess[str]) -> None:
    assert out.returncode == 0, out.stderr[-4000:]


def test_import_without_torch():
    code = "import fastforks\n" + "".join(f"import fastforks.{t}\n" for t in registry.TASKS) + "done()\n"
    _ok(_blocked(code))


def test_load_api_without_torch():
    """The public names a torch-free run touches: options, the outputs default, model cards, `list_models`."""
    code = (
        "import fastforks\n"
        "assert fastforks.get_outputs() == 'torch'\n"
        "fastforks.set_outputs('numpy'); assert fastforks.get_outputs() == 'numpy'\n"
        "assert fastforks.list_models(task='depth')\n"
        "fastforks.depth.model_card('depth-anything-v2-s')\n"
        "done()\n"
    )
    _ok(_blocked(code))


def _refs() -> list[str]:
    return [ref for refs in TORCH_FREE.values() for ref in refs]


@pytest.mark.parametrize("ref", _refs())
def test_model_module_without_torch(ref):
    """The model module imports torch-free and its MLX runners compute on NumPy arrays."""
    task = next(t for t, refs in TORCH_FREE.items() if ref in refs)
    card, _ = registry.resolve(task, ref)
    code = (
        "import importlib, sys\n"
        "m = importlib.import_module(sys.argv[1])\n"
        "mlx = {k: v for k, v in m.RUNNERS.items() if k.endswith('_mlx')}\n"
        "assert mlx, 'no MLX runner'\n"
        "assert all(v.arrays == 'numpy' for v in mlx.values()), {k: v.arrays for k, v in mlx.items()}\n"
        "done()\n"
    )
    _ok(_blocked(code, card.module))


# the child: load `ref` on MLX, call it on the stored inputs with NumPy outputs, save the golden's keys
CHILD = """
import json
import numpy as np
import fastforks
spec = json.loads(sys.argv[1])
with np.load(spec["inputs"]) as z:
    args = [z[f"arg{i}"] for i in range(len(z.files))]
m = getattr(fastforks, spec["task"]).load(
    spec["ref"], device="mlx", mode=spec["mode"], dtype=spec["dtype"], **spec["options"]
)
assert m.backend.device == "mlx" and m.backend.mode == spec["mode"], m.backend
r = m(*args, outputs="numpy")
extras = dict(getattr(r, "extras", {}) or {})
got = {k: getattr(r, k, None) if getattr(r, k, None) is not None else extras.get(k) for k in spec["keys"]}
got = {k: v for k, v in got.items() if v is not None}
assert all(isinstance(v, np.ndarray) for v in got.values()), {k: type(v) for k, v in got.items()}
np.savez(spec["outputs"], **got)
done()
"""


def _inputs(task: str, want: dict[str, np.ndarray]) -> list[np.ndarray]:
    from test_golden import image  # noqa: PLC0415 - test_golden imports torch: GPU runs only

    img = want["input"].round().astype(np.uint8) if "input" in want else image()
    if task in {"depth", "classification", "segmentation", "detection", "edges", "restoration"}:
        return [img]
    msg = f"no torch-free golden call for task {task!r} yet"
    raise NotImplementedError(msg)


@pytest.mark.model
@pytest.mark.mlx
@pytest.mark.parametrize("ref", _refs())
def test_mlx_without_torch(ref, tmp_path):
    import torch  # noqa: PLC0415 - the parent compares in torch, the child never imports it

    from test_golden import API, TASK, _check, golden, options, outputs  # noqa: PLC0415

    if importlib.util.find_spec("mlx") is None:
        pytest.skip("no mlx")
    task = TASK[ref]
    card = API[task].model_card(ref)
    want = golden(ref)
    opts = options(ref, want)
    keys = sorted(outputs(ref, want))
    np.savez(tmp_path / "in.npz", **{f"arg{i}": a for i, a in enumerate(_inputs(task, want))})
    for mode in ("portable", "kernels"):
        for dtype in card.mlx_dtypes(mode):
            API[task].load(ref, device="mlx", mode=mode, dtype=dtype, **opts)  # warms the conversion cache
            spec: dict[str, Any] = {
                "task": task,
                "ref": ref,
                "mode": mode,
                "dtype": dtype,
                "options": opts,
                "keys": keys,
                "inputs": str(tmp_path / "in.npz"),
                "outputs": str(tmp_path / "out.npz"),
            }
            _ok(_blocked(CHILD, json.dumps(spec)))
            with np.load(tmp_path / "out.npz") as z:
                got = {k: torch.from_numpy(z[k]).float() for k in z.files}
            _check(ref, mode, outputs(ref, want), got, f"{mode} {dtype} without torch: ")
