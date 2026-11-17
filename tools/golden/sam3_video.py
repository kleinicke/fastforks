"""SAM 3 video-tracking goldens from the Hugging Face transformers port (`Sam3TrackerVideoModel`, `facebook/sam3`).

Reference: transformers `Sam3TrackerVideoProcessor` + `Sam3TrackerVideoModel` (`REF_VERSION`), CPU float32, on
`tests/synth.py` `clip()`; one session per `clip_prompts()` entry (one object, `add_inputs_to_inference_session` on
frame 0: point, box or mask), the prompted frame, then `propagate_in_video_iterator` forward. Stored per prompt (key
prefix) and frame: the 288 x 288 mask logits (`pred_masks`) and the object score logits. Written to
`$FASTFORKS_GOLDEN_PATH/<GOLDEN>.npz` (its own file: the image golden `sam3@1` stays as it is); its SHA-256 is
`tests/test_promptable_segmentation_video.py` `SAM3_VIDEO_SHA256`.

    FASTFORKS_GOLDEN_PATH=... python tools/golden/sam3_video.py SAM3_DIR [PIXELS_OUT.npz]

`PIXELS_OUT` (optional): frame 0 as the processor feeds it (`pixel_values`), to check the port's input against.
"""

from __future__ import annotations

import hashlib
import io
import os
import sys
from pathlib import Path
from typing import Any

import _common  # noqa: F401 - puts tests/ on the path
import numpy as np
import torch

from synth import clip, clip_prompts

REF_VERSION = "v5.19.0"
GOLDEN = "sam3-video@1"  # fixed: the file name of these goldens


def track(proc: Any, model: Any, frames: list[np.ndarray], prompt: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:  # noqa: ANN401 - reference objects
    """One object prompted on frame 0, then forward: `pred_masks [T, 288, 288]`, `object_score_logits [T]`."""
    session = proc.init_video_session(video=frames, inference_device="cpu", dtype=torch.float32)
    kw: dict[str, Any] = {}
    if "input_points" in prompt:
        kw = {"input_points": [[prompt["input_points"]]], "input_labels": [[prompt["input_labels"]]]}
    elif "input_boxes" in prompt:
        kw = {"input_boxes": [[prompt["input_boxes"]]]}
    else:
        kw = {"input_masks": prompt["input_masks"]}
    proc.add_inputs_to_inference_session(session, frame_idx=0, obj_ids=1, **kw)
    lows, objs = {}, {}
    with torch.inference_mode():
        o = model(inference_session=session, frame_idx=0)
        lows[0], objs[0] = o.pred_masks, o.object_score_logits
        for o in model.propagate_in_video_iterator(session):
            lows[o.frame_idx], objs[o.frame_idx] = o.pred_masks, o.object_score_logits
    t = len(frames)
    low = torch.stack([lows[i].reshape(-1, *lows[i].shape[-2:])[0] for i in range(t)])
    obj = torch.stack([objs[i].reshape(-1)[0] for i in range(t)])
    return low.float().numpy(), obj.float().numpy()


def main() -> None:
    weights = Path(sys.argv[1])
    import transformers  # noqa: PLC0415 - reference environment

    if f"v{transformers.__version__}" != REF_VERSION:
        msg = f"transformers {transformers.__version__}, expected {REF_VERSION}"
        raise SystemExit(msg)
    proc = transformers.Sam3TrackerVideoProcessor.from_pretrained(weights)
    model = transformers.Sam3TrackerVideoModel.from_pretrained(weights, dtype=torch.float32).eval()
    frames = clip()
    if len(sys.argv) > 2:  # noqa: PLR2004 - optional output
        session = proc.init_video_session(video=frames, inference_device="cpu", dtype=torch.float32)
        np.savez(sys.argv[2], pixel_values=session.get_frame(0).float().numpy())
    maps = {}
    for prefix, p in clip_prompts().items():
        low, obj = track(proc, model, frames, p)
        maps |= {f"{prefix}pred_masks": low, f"{prefix}object_score_logits": obj}
        print(f"  {prefix}: object scores {np.round(obj, 3)}, mask px {(low > 0).sum((1, 2))}")
    buf = io.BytesIO()
    np.savez(buf, allow_pickle=False, **{k: v.astype(np.float32) for k, v in maps.items()})
    data = buf.getvalue()
    (Path(os.environ["FASTFORKS_GOLDEN_PATH"]) / f"{GOLDEN}.npz").write_bytes(data)
    print(GOLDEN, hashlib.sha256(data).hexdigest(), f"torch {torch.__version__}")


if __name__ == "__main__":
    main()
