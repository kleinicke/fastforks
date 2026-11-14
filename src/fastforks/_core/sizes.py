"""Valid input sizes of a model, from its card's `[io]` plus the largest size measured (`sizes.json`)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastforks._core.cards import ModelCard, Variant

CAP = (3000, 4000)
"""Largest size of the size-table queue: a model that ran there has no limit below it."""

ENCODER_SIDE = {"sam3": 1008}
"""Encoder input side of the promptable-segmentation families that fix it in code, not in `[params] image_size`."""

PATCH = {
    "depth_anything_v2": 14,
    "prompt_depth_anything": 14,
    "depth_anything_3": 14,
    "moge": 14,
    "lingbot_depth": 14,
    "any2full": 14,
    "omni_dc": 16,
    "unik3d": 14,
    "depth_pro": 16,
}
"""ViT patch side of the families that resize / pad inside the model (shown in the size range)."""

EXAMPLE_SIZE = "704x1280"
"""Size used in error messages and docs as the example of `--size`."""


@dataclass(frozen=True)
class SizeRange:
    """Input sizes a model accepts.

    Attributes:
        min_size: Smallest supported height / width in pixels.
        multiple: Inputs are padded to a multiple of this (1: nothing is padded by the core).
        native: `(H, W)` the model is fixed to (classification / features: the transformed crop), else `None`.
        resize: Short side the input is resized to inside the model (0: none).
        patch: ViT patch side of the family (14 / 16; 0: none); informational, the model handles any size.
        encoder: Side of the square encoder input an image is resized to inside the model (promptable segmentation; 0:
            none). It is also the default export size.
        reject_below: Shortest side the call accepts: the card minimum or the smallest square side that ran,
            not the pad multiple (inputs below `min_size` but not below this are padded, e.g. 48x640 for a multiple
            of 64); 0: `min_size`.
        tested_max: Largest `(H, W)` measured, or `None` (not measured yet).
        tested_alloc_mib: Peak allocated GPU memory at `tested_max`, MiB.
        tested_reserved_mib: Peak reserved GPU memory at `tested_max`, MiB.
    """

    min_size: int
    multiple: int
    native: tuple[int, int] | None = None
    resize: int = 0
    encoder: int = 0
    patch: int = 0
    tested_max: tuple[int, int] | None = None
    tested_alloc_mib: int = 0
    tested_reserved_mib: int = 0
    reject_below: int = 0

    def example(self) -> tuple[int, int]:
        """`EXAMPLE_SIZE` rounded up to the pad multiple and not below the minimum: a size that exports."""
        h, w = (int(n) for n in EXAMPLE_SIZE.split("x"))
        m = max(self.multiple, 1)
        return tuple(-(-max(n, self.min_size) // m) * m for n in (h, w))  # ty: ignore[invalid-return-type]

    def describe(self, *, tested: bool = True) -> str:
        """One-line text of the range, e.g. `any H x W >= 64, padded to a multiple of 32; tested up to 1080x1920`.

        Args:
            tested: Append the largest size measured (or "not measured yet").
        """
        if self.native:
            side = f"{self.native[0]}x{self.native[1]}"
            text = f"fixed {side}: the input is resized (short side {self.resize}) and centre-cropped to it"
            if not self.resize:
                text = f"fixed {side} (the model's crop)"
        elif self.encoder:
            text = (
                f"any H x W >= {self.min_size}, resized inside to the fixed encoder input "
                f"{self.encoder}x{self.encoder} (the default export size)"
            )
        elif self.multiple > 1:
            text = f"any H x W >= {self.min_size}, padded to a multiple of {self.multiple}"
        else:
            text = f"any H x W >= {self.min_size}, resized or padded inside the model"
            if self.patch:
                text += f" (ViT patch {self.patch})"
        if not tested:
            return text
        if self.tested_max:
            h, w = self.tested_max
            mem = (
                f"{self.tested_alloc_mib / 1024:.1f} GiB peak allocated, {self.tested_reserved_mib / 1024:.1f} reserved"
            )
            limit = "no limit found, largest size tried" if (h, w) == CAP else "largest size that ran"
            return f"{text}; tested up to {h}x{w} ({limit}; {mem})"
        return f"{text}; tested size: not measured yet"


@cache
def _data() -> dict[str, dict[str, Any]]:
    """`{"min": {ref: smallest square side that runs}, "max": {ref: [H, W, MiB allocated, MiB reserved]}}`."""
    return json.loads(Path(__file__).with_name("sizes.json").read_text())


def size_range(variant: Variant, card: ModelCard | None = None) -> SizeRange:
    """The `SizeRange` of `variant`.

    Args:
        variant: The model variant.
        card: The variant's family card; promptable segmentation reads its encoder side (`params.image_size`) there.
    """
    io = variant.io
    side = io.crop or io.resize
    data = _data()
    row = data["max"].get(variant.ref)
    probed = int(data["min"].get(variant.ref, 0))
    kw: dict[str, Any] = {}
    if row:
        kw = {"tested_max": (row[0], row[1]), "tested_alloc_mib": row[2], "tested_reserved_mib": row[3]}
    encoder = 0
    if card and card.task == "promptable_segmentation":
        encoder = int(card.params.get("image_size", ENCODER_SIDE.get(card.family, 0)))
    multiple = 1 if encoder or io.pad_mode == "none" else io.pad_multiple
    hard = max(io.min_size, 0 if encoder else probed)
    return SizeRange(
        min_size=max(hard, multiple),
        reject_below=hard,
        multiple=multiple,
        encoder=encoder,
        patch=0 if encoder or side else PATCH.get(card.family, 0) if card else 0,
        native=(side, side) if side else None,
        resize=io.resize,
        **kw,
    )


def card_size_lines(card: ModelCard, variants: list[Variant] | None = None, *, heading: bool = True) -> list[str]:
    """Markdown lines of a family's input size: the range (per distinct text), then the largest size tested.

    Args:
        card: The family card.
        variants: The variants to list (default: all).
        heading: Start the first line with `- Input size:` (a bullet of the model docs); else the lines stand under a
            heading of their own (the Hugging Face card).
    """
    vs = variants or card.variants
    groups: dict[str, list[str]] = {}
    for v in vs:
        groups.setdefault(size_range(v, card).describe(tested=False), []).append(f"`{v.id}`")
    if len(groups) == 1:
        text = next(iter(groups))
        lines = [f"- Input size: {text}" if heading else text]
    else:
        lines = ["- Input size:"] if heading else []
        indent = "  " if heading else ""
        lines += [f"{indent}- {', '.join(ids)}: {text}" for text, ids in groups.items()]
    tested = []
    for v in vs:
        r = size_range(v, card)
        if r.tested_max:
            mem = f"{r.tested_alloc_mib / 1024:.1f} GiB peak allocated, {r.tested_reserved_mib / 1024:.1f} reserved"
            tested.append(f"`{v.id}` {r.tested_max[0]}x{r.tested_max[1]} ({mem})")
        else:
            tested.append(f"`{v.id}` not measured yet")
    bullet = (
        "- Tested up to (portable, fp16, RTX 5060 Ti 16 GB)"
        if heading
        else "Tested up to (portable, fp16, RTX 5060 Ti 16 GB)"
    )
    return [*lines, f"{bullet}: {'; '.join(tested)}"]
