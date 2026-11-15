"""PARSeq goldens from upstream strhub (baudm/parseq `REF_COMMIT`).

Reference: `strhub.models.parseq.model.PARSeq` with the release checkpoint (the Lightning `model.` prefix dropped,
loaded strictly), `read.py`'s transform (PIL bicubic resize to 32 x 128, `ToTensor`, `Normalize(0.5, 0.5)`), decoding
with strhub's `Tokenizer` (94 characters), CPU float32. Input: upstream's `demo_images/art-01107.jpg` (stored as
`input`, RGB). Outputs: the whole image as one quad, the text as code points (`goldens.text_codes`), the score (mean
probability of the characters and the end token).

    FASTFORKS_GOLDEN_PATH=... python tools/golden/parseq.py PARSEQ_CHECKOUT WEIGHTS_DIR

Needs `timm` (strhub's ViT) and `torchvision`; the checkout's package is used from an export of `REF_COMMIT`.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from _common import reference_tree, write
from PIL import Image

from goldens import text_codes

UPSTREAM = "https://github.com/baudm/parseq"
REF_COMMIT = "1902db043c029a7e03a3818c616c06600af574be"
IMAGE = "demo_images/art-01107.jpg"
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
VARIANTS = {  # ref: checkpoint, embed dim, encoder heads, decoder heads (strhub configs/model/parseq*.yaml)
    "parseq@1": ("parseq-bb5792a6.pt", 384, 6, 12),
    "parseq-tiny@1": ("parseq_tiny-e7a21b54.pt", 192, 3, 6),
}


def main() -> None:
    repo, weights = Path(sys.argv[1]), Path(sys.argv[2])
    tree = reference_tree(repo, REF_COMMIT)
    sys.path.insert(0, str(tree))
    from strhub.data.utils import Tokenizer  # noqa: PLC0415  # reference-only dependency
    from strhub.models.parseq.model import PARSeq  # noqa: PLC0415
    from torchvision import transforms as T  # noqa: PLC0415

    blob = subprocess.run(["git", "-C", repo, "show", f"{REF_COMMIT}:{IMAGE}"], check=True, capture_output=True).stdout
    img = Image.open(io.BytesIO(blob)).convert("RGB")
    tok = Tokenizer(CHARSET)
    tf = T.Compose([T.Resize((32, 128), T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(0.5, 0.5)])
    w, h = img.size
    quad = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], np.float32)
    for ref, (name, dim, enc_heads, dec_heads) in VARIANTS.items():
        net = PARSeq(len(tok), 25, [32, 128], [4, 8], dim, enc_heads, 4, 12, dec_heads, 4, 1, True, 1, 0.1)  # noqa: FBT003 - upstream signature
        state = torch.load(weights / name, map_location="cpu", weights_only=True)
        net.load_state_dict({k.removeprefix("model."): v for k, v in state.items()}, strict=True)
        with torch.no_grad():
            probs = net.eval()(tok, tf(img)[None]).softmax(-1)
        (text,), (p,) = tok.decode(probs)
        maps = {"boxes": quad, "texts": text_codes([text]), "scores": np.array([float(p.mean())], np.float32)}
        write(
            ref,
            maps | {"input": np.asarray(img, np.float32)},
            family="parseq",
            upstream=f"{UPSTREAM}@{REF_COMMIT}",
            patch=None,
            settings=f"CPU float32, read.py transform (PIL bicubic), refine 1, {IMAGE}, torch {torch.__version__}",
            task="ocr",
        )
        print(f"  {ref}: {text!r} {float(p.mean()):.4f}")


if __name__ == "__main__":
    main()
