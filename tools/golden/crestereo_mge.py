"""Reference side of `crestereo.py`: runs upstream's MegEngine `test.py` inference in a MegEngine venv.

    MGE_PYTHON -I tools/golden/crestereo_mge.py REFERENCE_TREE WEIGHTS.mge INPUTS.npz OUTPUT.npz

`INPUTS.npz` holds `left` / `right` as `[H, W, 3]` BGR float32 (upstream reads images with `cv2.imread`), already
padded. The weights are read with a restricted unpickler (numpy arrays and dicts only) instead of `mge.load`, which
returns the same dict but can run arbitrary pickled code. The rest is upstream `test.py`: `load_model` (max_disp 256,
test mode, strict state dict) and `inference` (half-resolution pass, then full resolution from its flow, 20 iterations
each), on the CPU in float32. Needs Python <= 3.10 with `megengine` and `opencv-python-headless` (imported by test.py).

Upstream was written for MegEngine 1.8.2, which is no longer installable (PyPI has 1.13.x only); 1.13 renamed the
`F.pad` keyword `pad_witdth` (sic) to `pad_width`, so `_shim_pad` accepts the old spelling. Nothing else is changed.
"""

from __future__ import annotations

import collections
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ITERS = 20  # upstream test.py: inference(..., n_iter=20)


class _Unpickler(pickle.Unpickler):
    allowed: dict = {  # noqa: RUF012 - fixed table
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,  # noqa: SLF001 - numpy's array pickle
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
    }

    def find_class(self, module: str, name: str) -> object:
        try:
            return self.allowed[module, name]
        except KeyError:
            msg = f"refused global {module}.{name}"
            raise pickle.UnpicklingError(msg) from None


def _shim_pad(mge_functional: object) -> None:
    """Let `F.pad` take upstream's (MegEngine 1.8) keyword `pad_witdth`; same op otherwise."""
    pad = mge_functional.pad  # megengine.functional

    def pad_compat(src: object, pad_witdth: object = None, **kw: object) -> object:
        if pad_witdth is not None:
            kw["pad_width"] = pad_witdth
        return pad(src, **kw)

    mge_functional.pad = pad_compat  # megengine.functional


def main() -> None:
    tree, weights, inputs, output = (Path(a) for a in sys.argv[1:5])
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(tree))
    import megengine as mge  # noqa: PLC0415  # reference-only dependency
    import megengine.functional  # noqa: PLC0415  # reference-only dependency

    mge.set_default_device("cpux")
    _shim_pad(megengine.functional)
    import test  # noqa: PLC0415  # upstream test.py on sys.path
    from nets import Model  # noqa: PLC0415  # reference tree on sys.path

    with weights.open("rb") as f:
        pretrained_dict = _Unpickler(f).load()
    # test.load_model, with the restricted read above in place of mge.load
    model = Model(max_disp=256, mixed_precision=False, test_mode=True)
    model.load_state_dict(pretrained_dict["state_dict"], strict=True)
    model.eval()
    data = np.load(inputs)
    disp = test.inference(data["left"], data["right"], model, n_iter=ITERS)
    np.savez(output, disparity=np.asarray(disp, dtype=np.float32), megengine=np.array(mge.__version__))


if __name__ == "__main__":
    main()
