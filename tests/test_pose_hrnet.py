"""Weight-free checks of the HRNet pose family: the heatmap decode, the BatchNorm folding and the network's shapes."""

from __future__ import annotations

import pytest
import torch

from fastforks.pose.models.hrnet import model as hrnet
from fastforks.pose.models.hrnet.net import BN_EPS, STRIDE, Arch, HRNet, convert

ARCH = Arch(channels=32, keypoints=17, height=64, width=64)


def _heatmap(x: int, y: int, right: float, below: float) -> torch.Tensor:
    """A `[1, 1, 8, 6]` heatmap, maximum at `(x, y)`, the given right / lower neighbours (left / upper: 0)."""
    hm = torch.zeros(1, 1, 8, 6)
    hm[0, 0, y, x] = 1.0
    if x + 1 < hm.shape[-1]:
        hm[0, 0, y, x + 1] = right
    if y + 1 < hm.shape[-2]:
        hm[0, 0, y + 1, x] = below
    return hm


@pytest.mark.parametrize(
    ("x", "y", "right", "below", "want"),
    [
        (3, 4, 0.5, 0.0, (3.25, 4.0)),  # towards the larger neighbour, a quarter heatmap pixel
        (3, 4, -0.5, 0.5, (2.75, 4.25)),
        (1, 4, 0.5, 0.5, (1.0, 4.25)),  # the outer two columns: no shift along x
        (4, 4, 0.5, 0.5, (4.25, 4.25)),  # x = W - 2 shifts in the reference, W - 1 = 5 does not
        (5, 4, 0.0, 0.5, (5.0, 4.25)),
        (3, 1, 0.5, 0.5, (3.25, 1.0)),
        (3, 6, 0.5, 0.5, (3.25, 6.25)),
        (3, 7, 0.5, 0.0, (3.25, 7.0)),
        (3, 0, 0.5, 0.5, (3.0, 0.0)),  # the first row: no shift at all
        (0, 4, 0.5, 0.5, (0.0, 4.0)),
    ],
)
def test_decode_moves_the_maximum_a_quarter_pixel_towards_the_larger_neighbour(x, y, right, below, want):
    coords, score = hrnet.decode(_heatmap(x, y, right, below))
    assert score.tolist() == [[1.0]]
    assert coords[0, 0].tolist() == [want[0] * STRIDE, want[1] * STRIDE]


def test_convert_folds_every_batchnorm_and_rejects_other_entries():
    torch.manual_seed(0)
    net = HRNet(ARCH)
    ours = net.state_dict()
    sd = {}
    for key in [k for k in ours if k.endswith(".weight") and k != "head.weight"]:
        name = key.removesuffix(".weight")
        stem, _, leaf = f"backbone.{name}".rpartition(".")
        bn = f"{stem}.{leaf.replace('conv', 'bn')}" if "conv" in leaf else f"{stem}.{int(leaf) + 1}"
        cout = ours[key].shape[0]
        gamma, var = torch.rand(cout) + 0.5, torch.rand(cout) + 0.5
        mean = torch.randn(cout)
        scale = gamma / (var + BN_EPS).sqrt()
        sd[f"backbone.{key}"] = ours[key] / scale[:, None, None, None]
        sd |= {f"{bn}.weight": gamma, f"{bn}.running_var": var, f"{bn}.running_mean": mean}
        sd[f"{bn}.bias"] = ours[f"{name}.bias"] + mean * scale
        sd[f"{bn}.num_batches_tracked"] = torch.tensor(1)
    sd["head.final_layer.weight"], sd["head.final_layer.bias"] = ours["head.weight"], ours["head.bias"]
    got = convert(dict(sd))
    assert got.keys() == ours.keys()
    assert all(torch.allclose(got[k], ours[k], rtol=1e-5, atol=1e-6) for k in ours)
    with pytest.raises(KeyError, match="unexpected"):
        convert(sd | {"neck.weight": torch.zeros(1)})


def test_network_returns_one_quarter_size_heatmap_per_keypoint():
    with torch.inference_mode():
        out = HRNet(ARCH).eval()(torch.zeros(1, 3, ARCH.height, ARCH.width))
    assert out.shape == (1, ARCH.keypoints, ARCH.height // STRIDE, ARCH.width // STRIDE)
