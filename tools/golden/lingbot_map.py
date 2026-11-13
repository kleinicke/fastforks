"""LingBot-Map golden from the reference implementation.

Reference: upstream Robbyant/lingbot-map (`REF_COMMIT`), CPU float32, `GCTStream.inference_streaming` (SDPA backend,
3D RoPE, cross-frame special tokens, 4 camera iterations) with 2 scale frames and a 1-frame KV window, so the 4 views
of the `tests/synth.py` scene exercise the scale frames, the streamed frames and the eviction of frames from the
window (golden.toml `options`). The runner's pre- and post-processing: images resized to long side 518 (multiples of
14, bicubic antialiased, clamped), maps resized back (bilinear), intrinsics from `pose_encoding_to_extri_intri`
rescaled to the input pixels (principal point moved by -0.5), the pose encoding taken as camera-to-world.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/lingbot_map.py LINGBOT_MAP_CHECKOUT lingbot-map.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from _common import reference_tree, write

from synth import scene

UPSTREAM = "https://github.com/Robbyant/lingbot-map"
REF_COMMIT = "b9dd1b1de535e6d36e777fb42b8433c781bfd783"
LONG_SIDE = 518
SCALE_FRAMES, WINDOW = 2, 1  # golden.toml options


def main() -> None:
    repo, ckpt = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(tree))
    from lingbot_map.models.gct_stream import GCTStream  # noqa: PLC0415  # reference checkout
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri  # noqa: PLC0415

    net = GCTStream(
        img_size=518,
        patch_size=14,
        enable_3d_rope=True,
        max_frame_num=1024,
        kv_cache_sliding_window=WINDOW,
        kv_cache_scale_frames=SCALE_FRAMES,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=True,
        camera_num_iterations=4,
    )
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(sd.get("model", sd), strict=False)
    print(f"  missing {missing}, unexpected {unexpected}")
    net.eval()

    s = scene()
    img = torch.from_numpy(s["images"]).permute(0, 3, 1, 2).float() / 255
    h0, w0 = img.shape[-2:]
    f = LONG_SIDE / max(h0, w0)
    hn, wn = (max(14, round(v * f / 14) * 14) for v in (h0, w0))
    x = F.interpolate(img, (hn, wn), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
    with torch.no_grad():
        p = net.inference_streaming(x, num_scale_frames=SCALE_FRAMES)

    def up(t: torch.Tensor) -> torch.Tensor:
        return F.interpolate(t.float(), (h0, w0), mode="bilinear", align_corners=False)

    c2w3, k = pose_encoding_to_extri_intri(p["pose_enc"], (hn, wn))
    c2w3, k = c2w3[0], k[0].clone()
    k[:, 0] *= w0 / wn
    k[:, 1] *= h0 / hn
    k[:, :2, 2] -= 0.5
    c2w = torch.eye(4).repeat(c2w3.shape[0], 1, 1)
    c2w[:, :3] = c2w3
    maps = {
        "depth": up(p["depth"][0].permute(0, 3, 1, 2))[:, 0],
        "confidence": up(p["depth_conf"][0][:, None])[:, 0],
        "intrinsics": k,
        "cam_to_world": c2w,
    }
    maps = {n: v.numpy() for n, v in maps.items()}
    write(
        "lingbot-map@1",
        maps,
        family="lingbot_map",
        upstream=f"{UPSTREAM}@{REF_COMMIT}",
        patch=None,
        settings=f"CPU float32, synth scene (4 views) at {h0}x{w0} -> network {hn}x{wn}, scale_frames"
        f" {SCALE_FRAMES}, window {WINDOW}, torch {torch.__version__}",
        task="multiview",
    )
    gt = s["depth"]
    d = maps["depth"]
    sc = np.median(gt) / np.median(d)
    print(f"  depth median {np.median(d):.4f} (GT {np.median(gt):.4f}), AbsRel after median scaling "
          f"{np.mean(np.abs(d * sc - gt) / gt):.4f}")  # fmt: skip
    print(f"  fx {maps['intrinsics'][:, 0, 0].tolist()} (GT {s['intrinsics'][0, 0, 0]:.2f})")
    print(f"  cam centres {maps['cam_to_world'][:, :3, 3].round(4).tolist()}")


if __name__ == "__main__":
    main()
