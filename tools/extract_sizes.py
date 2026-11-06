"""Condense the size-table queue's `sizes.jsonl` and the min-size probe into `src/fastforks/_core/sizes.json`.

`max`: per model ref the largest size (by pixels) that ran in the portable mode, with the peak allocated / reserved MiB.
`min`: per model ref the smallest square side at which `module()` ran (`tools/probe_min_size.py`; one probe per family,
applied to all its variants except `VARIANT_MIN`).
Usage: `python tools/extract_sizes.py SIZES_JSONL MIN_SIZES_JSONL`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastforks._core import registry

VARIANT_MIN = {  # variants whose own architecture needs less than the family's probe (the smallest-weights variant)
    "cas-gwcnet-gc-sceneflow@1": 1,  # probed on gc itself (the family probe ran -c: 64 x 64 pooling needs 225)
}
OUT = Path(__file__).resolve().parents[1] / "src" / "fastforks" / "_core" / "sizes.json"


def main() -> None:
    """Rewrite the JSON file from the given logs."""
    best: dict[str, list[int]] = {}
    for line in Path(sys.argv[1]).read_text().splitlines():
        r = json.loads(line)
        if r.get("status") != "ok" or r["mode"] != "portable":
            continue
        h, w = r["hw"]
        cur = best.get(r["ref"])
        if cur is None or h * w > cur[0] * cur[1]:
            best[r["ref"]] = [h, w, r["alloc_mib"], r["reserved_mib"]]
    probed = {}
    for line in Path(sys.argv[2]).read_text().splitlines():
        r = json.loads(line)
        if r.get("min"):
            probed[(r["task"], r["card"])] = r["min"]
    smallest: dict[str, int] = {}
    for task in registry.TASKS:
        for card, variant in registry.cards(task).values():
            if (task, card.name) in probed:
                smallest[variant.ref] = probed[(task, card.name)]
    smallest.update(VARIANT_MIN)
    OUT.write_text(json.dumps({"min": smallest, "max": best}, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"{len(best)} max, {len(smallest)} min -> {OUT}")


if __name__ == "__main__":
    main()
