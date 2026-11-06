"""HITNet weights from the authors' frozen TensorFlow graphs: variable names -> PyTorch parameter names and layouts."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from .pb import Node

_KIND = {"kernel": "weight", "bias": "bias"}
_RESNET = re.compile(r"resblock(\d+)/(conv[12])$")


def _resnet(name: str) -> str:
    m = _RESNET.match(name)
    return f"res.{m.group(1)}.{m.group(2)}" if m else name


def torch_name(name: str, scales: int) -> str | None:
    """PyTorch parameter of the TF variable `name` (`None` if it is no parameter of ours); `scales` = pyramid depth."""
    if m := re.match(r"fe_shared/(conv_down_1|conv_up_0|conv_up_1|conv_up_2)_(\d+)/(kernel|bias)$", name):
        kind, i, t = m.group(1), int(m.group(2)), _KIND[m.group(3)]
        return f"fe.{kind}n.{t}" if kind in ("conv_up_1", "conv_up_2") and i == scales else f"fe.{kind}.{i}.{t}"
    if m := re.match(r"fe_shared/(conv_down|conv_upsample)_(\d+)/(kernel|bias)$", name):
        return f"fe.{m.group(1)}.{m.group(2)}.{_KIND[m.group(3)]}"
    if m := re.match(r"level(\d+)/shared/(conv_reduce|conv_embedding)/(kernel|bias)$", name):
        return f"init.{m.group(1)}.{m.group(2)}.{_KIND[m.group(3)]}"
    if m := re.match(r"level(\d+)/init_to_prop/conv_hyp/(kernel|bias)$", name):
        return f"init.{m.group(1)}.conv_hyp.{_KIND[m.group(2)]}"
    if m := re.match(r"level(\d+)/shared/prop_warp/conv2d_neighborsX_1/(kernel|bias)$", name):
        return f"prop.{m.group(1)}.conv_neighbors.{_KIND[m.group(2)]}"
    if m := re.match(r"level(\d+)/shared/prop/(.+)/(kernel|bias)$", name):
        return f"prop.{m.group(1)}.prop.{_resnet(m.group(2))}.{_KIND[m.group(3)]}"
    if m := re.match(r"shared/refinement_l(\d+)/conv1x1/(kernel|bias)$", name):
        return f"refine.{m.group(1)}.conv1x1.{_KIND[m.group(2)]}"
    if m := re.match(r"shared/refinement_l(\d+)/refiner/(.+)/(kernel|bias)$", name):
        return f"refine.{m.group(1)}.refiner.{_resnet(m.group(2))}.{_KIND[m.group(3)]}"
    return None


def state_dict(graph: dict[str, Node], scales: int) -> dict[str, torch.Tensor]:
    """Float32 state dict (PyTorch layouts) from the constants of a HITNet graph.

    Raises:
        KeyError: A kernel / bias constant that no parameter of ours corresponds to.
    """
    out = {}
    for name, node in graph.items():
        if node.value is None or not name.endswith(("/kernel", "/bias")):
            continue
        target = torch_name(name, scales)
        if target is None:
            msg = f"unmapped variable {name}"
            raise KeyError(msg)
        v = node.value
        if name.endswith("/kernel"):  # TF [kh, kw, in, out] -> [out, in, kh, kw]; transposed convs [kh, kw, out, in] -> [in, out, kh, kw]
            v = np.transpose(v, (3, 2, 0, 1))
        out[target] = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    return out

