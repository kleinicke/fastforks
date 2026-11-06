"""HITNet goldens from the authors' TensorFlow graphs.

Reference: the released frozen graphs `eth3d.pb` and `middlebury_d400.pb` of google-research/hitnet (UPSTREAM at
`REF_COMMIT`), run by TensorFlow on the CPU in float32; the graphs do their own scaling and padding. Same input as
fastforks: the uint8 test pair, for the gray graph converted with ITU-R BT.601 luma weights (`fastforks` does the
same in float; the graph takes floats). Needs TensorFlow 2.x and numpy only (no PyTorch needed to run the graphs):

    FASTFORKS_GOLDEN_PATH=... python tools/golden/hitnet.py WEIGHTS_DIR [eth3d middlebury-d400]

`WEIGHTS_DIR` holds `eth3d.pb` / `middlebury_d400.pb` (from the PINTO 142_HITNET archive, see the card).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf  # reference-only dependency (separate venv)
from _common import pair, write

UPSTREAM = "https://github.com/google-research/google-research"
REF_COMMIT = "f200f2b89d29484bf0dd0fef0c3979ac8bed9af4"
VARIANTS = {
    "eth3d": ("hitnet-eth3d", "eth3d.pb", 1),
    "middlebury-d400": ("hitnet-middlebury-d400", "middlebury_d400.pb", 3),
}


def run_graph(path: Path, x: np.ndarray) -> np.ndarray:
    """Disparity `[H, W]` of the graph at `path` for `x` `[1, H, W, 2 C]` (images in 0..1, left then right channels)."""
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(path.read_bytes())
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="graph")
    with tf.compat.v1.Session(graph=graph) as s:
        return s.run(
            graph.get_tensor_by_name("graph/reference_output_disparity:0"),
            {graph.get_tensor_by_name("graph/input:0"): x},
        )[0, :, :, 0]


def main() -> None:
    weights = Path(sys.argv[1])
    for name in sys.argv[2:] or list(VARIANTS):
        variant, file, channels = VARIANTS[name]
        images = [im.astype(np.float32) for im in pair()]
        if channels == 1:
            images = [(im @ np.array([0.299, 0.587, 0.114], np.float32))[..., None] for im in images]
        x = np.concatenate(images, -1)[None] / 255.0
        disp = run_graph(weights / file, x.astype(np.float32))
        write(
            f"{variant}@1",
            {"disparity": disp},
            family="hitnet",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"TensorFlow {tf.__version__} CPU float32, frozen graph {file} (PINTO 142_HITNET archive)",
        )
        print(f"  median disparity {np.median(disp):.4f}")


if __name__ == "__main__":
    main()
