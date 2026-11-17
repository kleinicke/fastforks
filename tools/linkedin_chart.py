"""One-off: LinkedIn image of original vs portable vs kernels runtime for four models (reference GPU).

uv run --with matplotlib python tools/linkedin_chart.py --out linkedin-speed.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = {  # ref -> label, top to bottom
    "da3-large-1.1": "Depth Anything 3 (Large)",
    "dinov3-vitl16": "DINOv3 (ViT-L/16)",
    "sam3": "SAM 3",
    "raft-stereo-middlebury": "RAFT-Stereo",
}
MODES = (("original", "original"), ("portable", "fastforks portable"), ("kernels", "fastforks kernels"))
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
COLORS = {"original": "#2a78d6", "portable": "#eb6834", "kernels": "#1baf7a"}  # categorical slots 1-3


def times() -> dict[str, dict[str, float]]:
    """Fastest original (any of the authors' settings), portable and kernels ms per model at its standard input."""
    best: dict[str, dict[str, float]] = {}
    for line in (ROOT / "docs/data/speed.jsonl").read_text().splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("ref") not in MODELS or not isinstance(r.get("ms"), int | float):
            continue
        mode = "original" if r["mode"].startswith("upstream") else r["mode"]
        cur = best.setdefault(r["ref"], {})
        cur[mode] = min(cur.get(mode, float("inf")), r["ms"])
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="linkedin-speed.png")
    a = ap.parse_args()
    t = times()

    mpl.rcParams.update({"font.family": ["Helvetica Neue", "Arial Unicode MS", "DejaVu Sans"], "text.color": INK})
    fig = plt.figure(figsize=(10.8, 10.8), dpi=200, facecolor=SURFACE)
    ax = fig.add_axes((0.05, 0.04, 0.90, 0.92), facecolor=SURFACE)
    ax.set_axis_off()
    y, h = 0.0, 0.72
    for ref, name in MODELS.items():
        ax.text(-0.5, y, name, ha="left", va="center", fontsize=21, weight="bold")
        y -= 1.05
        o = t[ref]["original"]
        for mode, label in MODES:
            v = t[ref][mode]
            ax.barh(y, v / o, height=h, color=COLORS[mode], edgecolor="none")
            ax.text(-0.03, y, label, ha="right", va="center", fontsize=16, color=INK2)
            end = f"{v:.1f} ms" if v < 100 else f"{v:.0f} ms"  # noqa: PLR2004
            ax.text(v / o + 0.02, y, end, ha="left", va="center", fontsize=16)
            if mode != "original":
                ax.text(
                    1.45,
                    y,
                    f"{o / v:.1f}\N{MULTIPLICATION SIGN} faster",
                    ha="right",
                    va="center",
                    fontsize=16,
                    weight="bold",
                )
            y -= 0.9
        y -= 0.55
    ax.set_xlim(-0.5, 1.45)
    ax.set_ylim(y + 0.4, 0.6)
    fig.text(0.05, 0.03, "Measured on an NVIDIA RTX 5060 Ti 16 GB", ha="left", va="center", fontsize=15, color=INK2)
    fig.savefig(a.out, facecolor=SURFACE)
    print(a.out)


if __name__ == "__main__":
    main()
