"""Cheap loads and first calls.

Networks build on the meta device (the loaders assign every tensor from the checkpoint, no fp32 init), and no
forward imports `torch.onnx` (~1.5 s) to ask whether it is being exported.
"""

import gc
import re
import subprocess
import sys
import weakref
from pathlib import Path

import torch

from fastforks.shared.depth_anything_v2.net.dinov2 import vit_small

SRC = Path(__file__).resolve().parents[1] / "src" / "fastforks"


def test_drop_path_schedules_are_built_on_the_cpu() -> None:
    """`.item()` on a `torch.linspace` built under `torch.device("meta")` raises: the schedule names the CPU."""
    pattern = re.compile(r"\[\s*\w+\.item\(\) for \w+ in torch\.linspace\(([^\]]*)\)\s*\]", re.DOTALL)
    bad = [
        f"{path.relative_to(SRC)}:{text[: m.start()].count(chr(10)) + 1}"
        for path in SRC.rglob("*.py")
        for text in (path.read_text(),)
        for m in pattern.finditer(text)
        if 'device="cpu"' not in m.group(1)
    ]
    assert not bad, bad


def test_shared_dinov2_builds_on_meta_with_drop_path() -> None:
    """The shared DINOv2 ViT (dinov2, tipsv2, Depth Anything) builds on the meta device with a drop-path decay."""
    with torch.device("meta"):
        net = vit_small(patch_size=14, img_size=56, drop_path_rate=0.1, block_chunks=0)
    assert all(p.is_meta for p in net.parameters())


def test_forwards_ask_for_onnx_export_without_importing_it() -> None:
    """Every forward uses `fastforks._core.tracing.in_onnx_export`, not `torch.onnx.is_in_onnx_export`."""
    bad = [
        f"{path.relative_to(SRC)}:{n}"
        for path in SRC.rglob("*.py")
        if path.relative_to(SRC).as_posix() != "_core/tracing.py"
        for n, line in enumerate(path.read_text().splitlines(), 1)
        if "is_in_onnx_export(" in line
    ]
    assert not bad, bad


def test_in_onnx_export_leaves_torch_onnx_unimported() -> None:
    code = (
        "import sys\nfrom fastforks._core.tracing import in_onnx_export\n"
        "assert not in_onnx_export()\nprint('torch.onnx' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_loaded_model_drops_the_given_state(tmp_path: Path) -> None:
    """`load_variant(..., state=)` builds from the read checkpoint, and the model does not keep it alive."""
    from fastforks._core import registry  # noqa: PLC0415
    from fastforks._core.model import load_variant  # noqa: PLC0415
    from fastforks.features._api import TASK, FeaturesModel  # noqa: PLC0415
    from fastforks.features.models.dinov3.net import CONFIGS, DINOv3ViT  # noqa: PLC0415

    card, variant = registry.resolve(TASK, "dinov3-vits16")
    state = DINOv3ViT(CONFIGS["vits16"]).state_dict()
    probe = weakref.ref(next(iter(state.values())))
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"")
    model = load_variant(TASK, card, variant, FeaturesModel, mode="portable", device="cpu", weights=path, state=state)
    del state
    gc.collect()
    assert probe() is None
    assert model._runner.ctx.user_state is None
