"""4 DIV2K validation images (0801, 0826, 0851, 0876): centre crop 512x512 of the HR image = ground truth, LR = PIL bicubic x2
(256x256) as one-frame npz `video` [1, 256, 256, 3] uint8 for the SeedVR2 reference script (scale 2) and the lib.
python mkin_div2k.py OUTDIR"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
D = Path("/home/florian/projects/data/restoration/div2k/DIV2K_valid_HR")
for n in ("0801", "0826", "0851", "0876"):
    im = Image.open(D / f"{n}.png").convert("RGB"); w, h = im.size
    hr = im.crop(((w - 512) // 2, (h - 512) // 2, (w - 512) // 2 + 512, (h - 512) // 2 + 512))
    lr = hr.resize((256, 256), Image.BICUBIC)
    np.savez(out / f"d{n}.npz", video=np.asarray(lr)[None]); np.save(out / f"d{n}_hr.npy", np.asarray(hr))
    print(n, hr.size, lr.size)
