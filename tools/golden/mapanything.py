"""MapAnything (map-anything-apache) golden from the reference implementation.

Reference: upstream facebookresearch/map-anything (`REF_COMMIT`) with uniception 0.1.7 and the DINOv2 torch-hub code,
CPU float32: the steps of `MapAnything.infer` without AMP (validate / preprocess the views, images only, `forward`,
`postprocess_model_outputs_for_inference` with its defaults apply_mask + mask_edges) on the 4 views of the
`tests/synth.py` scene, with the runner's pre- and post-processing: images resized to the network size (long side 518,
multiples of 14, bicubic antialiased, clamped; upstream's loader crops instead), DINOv2-normalised; maps resized back to
the input size (bilinear; the mask nearest), intrinsics rescaled to the input pixels (pixel centres at integer
coordinates). Depth / world points are the unmasked `pts3d_cam[..., 2]` / `pts3d` (upstream zeroes masked pixels; the
runner returns the mask instead), confidence the raw `conf`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/mapanything.py MAP_ANYTHING_CHECKOUT SNAPSHOT_DIR

SNAPSHOT_DIR holds the Hugging Face facebook/map-anything-apache files (config.json, model.safetensors). Needs
uniception and the torch-hub checkout of facebookresearch/dinov2 (used offline when present).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write
from safetensors.torch import load_file

from synth import scene

UPSTREAM = "https://github.com/facebookresearch/map-anything"
REF_COMMIT = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9"
RESOLUTION = 518
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _hub_local() -> None:
    """Load DINOv2 from the local torch-hub checkout (`torch.hub.load` otherwise contacts GitHub)."""
    local = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if not local.exists():
        return
    orig = torch.hub.load

    def load_local(repo: str, *a: object, **kw: object) -> object:
        if repo == "facebookresearch/dinov2":
            for k in ("source", "trust_repo", "skip_validation", "force_reload"):
                kw.pop(k, None)
            return orig(str(local), *a, source="local", **kw)
        return orig(repo, *a, **kw)

    torch.hub.load = load_local


def _reference_net(cls: type, snap: Path) -> Any:  # noqa: ANN401 - the reference model class
    """`MapAnything.from_pretrained(snap)` with one copy of the weights in RAM.

    Its encoder is built without torch-hub weights (as `_from_pretrained`), on the meta device, and filled with the
    checkpoint tensors (assign=True) instead of copying them into initialised parameters.
    """
    cfg = json.loads((snap / "config.json").read_text())
    cfg["encoder_config"] = dict(cfg["encoder_config"], torch_hub_pretrained=False)
    with torch.device("meta"):
        net = cls(**cfg)
    # strict=False: the checkpoint stores aliased modules (dpt_feature_head = dense_head.0, scratch.layer*_rn =
    # input_process.*.1) once, as safetensors.load_model; the meta check below catches anything really missing
    bad = net.load_state_dict(load_file(snap / "model.safetensors"), strict=False, assign=True).unexpected_keys
    if bad:
        msg = f"unexpected checkpoint keys: {bad[:5]}"
        raise RuntimeError(msg)
    left = [n for n, t in [*net.named_parameters(), *net.named_buffers()] if t.is_meta]
    if left:
        msg = f"not in the checkpoint: {left[:5]}"
        raise RuntimeError(msg)
    return net.eval()


def main() -> None:
    repo, snap = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(tree))
    for m in [m for m in sys.modules if m.split(".")[0] == "mapanything"]:
        sys.modules.pop(m)
    _hub_local()
    from mapanything.models import MapAnything  # noqa: PLC0415  # reference checkout
    from mapanything.utils.inference import (  # noqa: PLC0415
        postprocess_model_outputs_for_inference,
        preprocess_input_views_for_inference,
        validate_input_views_for_inference,
    )

    if not str(sys.modules["mapanything"].__file__).startswith(str(tree)):
        msg = f"mapanything imported from {sys.modules['mapanything'].__file__}, not the reference tree"
        raise RuntimeError(msg)
    net = _reference_net(MapAnything, snap)

    s = scene()
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    h0, w0 = img.shape[-2:]
    f = RESOLUTION / max(h0, w0)
    hn, wn = (max(14, round(v * f / 14) * 14) for v in (h0, w0))
    x = F.interpolate(img, (hn, wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    x = (x - MEAN) / STD
    views = [{"img": x[i : i + 1], "data_norm_type": ["dinov2"]} for i in range(len(x))]
    views = preprocess_input_views_for_inference(validate_input_views_for_inference(views))
    net._configure_geometric_input_config(  # noqa: SLF001 - as infer()
        use_calibration=True, use_depth=True, use_pose=True, use_depth_scale=True, use_pose_scale=True
    )
    with torch.no_grad():
        raw = net.forward(views, memory_efficient_inference=False)
        keep = [{k: raw[i][k].clone() for k in ("pts3d", "pts3d_cam", "conf", "metric_scaling_factor")}
                for i in range(len(raw))]  # fmt: skip
        post = postprocess_model_outputs_for_inference(raw_outputs=raw, input_views=views)

    def up(t: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:  # [V, h, w, C] -> [V, h0, w0, C]
        kw = {"align_corners": False} if mode == "bilinear" else {}
        return F.interpolate(t.permute(0, 3, 1, 2).float(), (h0, w0), mode=mode, **kw).permute(0, 2, 3, 1)

    cat = torch.cat
    k = cat([p["intrinsics"] for p in post]).clone()
    k[:, 0, 0] *= w0 / wn
    k[:, 1, 1] *= h0 / hn
    k[:, 0, 2] = (k[:, 0, 2] + 0.5) * w0 / wn - 0.5
    k[:, 1, 2] = (k[:, 1, 2] + 0.5) * h0 / hn - 0.5
    maps = {
        "depth": up(cat([p["pts3d_cam"][..., 2:3] for p in keep]))[..., 0],
        "confidence": up(cat([p["conf"][..., None] for p in keep]))[..., 0],
        "mask": up(cat([p["mask"] for p in post]).float(), "nearest")[..., 0] > 0.5,  # noqa: PLR2004
        "world_points": up(cat([p["pts3d"] for p in keep])),
        "intrinsics": k,
        "cam_to_world": cat([p["camera_poses"] for p in post]),
        "metric_scale": cat([p["metric_scaling_factor"].reshape(1) for p in keep]),
    }
    maps = {n: v.numpy() for n, v in maps.items()}
    write(
        "mapanything-apache@1",
        maps,
        family="mapanything",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, synth scene (4 views) at {h0}x{w0} -> network {hn}x{wn}, torch {torch.__version__}",
        task="multiview",
    )
    gt = s["depth"]
    d = maps["depth"]
    print(f"  depth median {np.median(d):.4f} (GT {np.median(gt):.4f}), AbsRel {np.mean(np.abs(d - gt) / gt):.4f}, "
          f"after median scaling {np.mean(np.abs(d * np.median(gt) / np.median(d) - gt) / gt):.4f}")  # fmt: skip
    print(f"  mask {maps['mask'].mean():.3f}, metric scale {maps['metric_scale'].tolist()}")
    print(f"  fx {maps['intrinsics'][:, 0, 0].tolist()} (GT {s['intrinsics'][0, 0, 0]:.2f})")
    print(f"  cam centres {maps['cam_to_world'][:, :3, 3].round(4).tolist()}")


if __name__ == "__main__":
    main()
