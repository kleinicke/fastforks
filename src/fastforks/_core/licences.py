"""Licence classes and the per-variant commercial status computed from a card.

A distribution ships *code*, so it is cut by the class of the code licence (`permissive`, `copyleft`,
`noncommercial`); weights always come from the authors' hosts, so their licence alone (the training data is only a note)
changes the commercial status of a variant at listing and load time, never which wheel the code is in.
"""

from __future__ import annotations

import functools
import importlib.resources
import re
import tomllib
from dataclasses import dataclass
from typing import Literal

from fastforks._core.errors import CardError

CodeClass = Literal["permissive", "copyleft", "noncommercial"]
Commercial = Literal["yes", "no", "unclear"]

# SPDX ids (and our `LicenseRef-*` ids) by class. Unknown ids are a card error: add them here with a reason.
CLASSES: dict[str, CodeClass] = {
    **dict.fromkeys(
        (
            "MIT",
            "BSD-2-Clause",
            "BSD-3-Clause",
            "Apache-2.0",
            "ISC",
            "Zlib",
            "CC0-1.0",
            "CC-BY-4.0",
            "Unlicense",
            "WTFPL",
        ),
        "permissive",
    ),
    # NVIDIA model licences: commercial use allowed. Redistributing the weights needs a NOTICE text file with
    # "Licensed by NVIDIA Corporation under the NVIDIA Open Model License" (resp. "... Open Model Agreement", required
    # when the release has a NOTICE file); the Open Model License also ends if its safety guardrails are bypassed
    # without a similar replacement. https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
    # and .../nvidia-open-model-agreement/ (read 2026-10-26).
    **dict.fromkeys(("LicenseRef-NVIDIA-Open-Model-License", "LicenseRef-NVIDIA-Open-Model-Agreement"), "permissive"),
    # CreativeML Open RAIL++-M (Stable Diffusion 2 and models fine-tuned from it, e.g. Marigold): commercial use
    # allowed; the use restrictions of its Attachment A pass on to every user and redistribution, which must include
    # the licence.
    # https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL (read 2026-10-27).
    "LicenseRef-CreativeML-OpenRAIL-PlusPlus-M": "permissive",
    # DINOv3 License (Meta, 2025-08-19; DINOv3 code and weights): royalty-free grant to use, reproduce, distribute and
    # modify, commercial use not excluded; redistribution must include the licence, use must follow its acceptable-use
    # and trade-control terms. https://ai.meta.com/resources/models-and-libraries/dinov3-license/ (read 2026-10-27).
    # "permissive" classifies these use terms only (a custom licence, not OSI); commercial use is not excluded, so the
    # variants are commercially "yes".
    "LicenseRef-DINOv3-License": "permissive",
    # Apple sample-code licence (e.g. apple/ml-depth-pro LICENSE): use, reproduction, modification and redistribution,
    # also commercial, keeping the notice; no patent licence.
    "LicenseRef-Apple-Sample-Code": "permissive",
    # SAM License (SAM 3 weights): royalty-free licence to use, reproduce, distribute and modify, commercial use not
    # excluded. Redistribution needs a copy of the agreement, publications an acknowledgement; trade controls apply and
    # military / weapons / nuclear / espionage uses (ITAR) are excluded; no reverse engineering of the materials.
    # https://huggingface.co/facebook/sam3/blob/main/LICENSE (read 2026-10-28). The weights are gated (the agreement is
    # accepted per user): they stay on Meta's host and are never mirrored (tests/test_policy.py pins it).
    "LicenseRef-SAM-License": "permissive",
    # Meta VGGT License v1 (2025-07-29): royalty-free use / modification / distribution, commercial use not excluded;
    # Acceptable Use Policy applies. https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt (read 2026-10-27).
    "LicenseRef-VGGT": "permissive",
    **dict.fromkeys(
        ("GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later", "AGPL-3.0-only", "LGPL-3.0-only"),
        "copyleft",
    ),
    # Share-alike: adapted weights must carry the same licence; commercial use allowed (detectron2 model zoo).
    # https://creativecommons.org/licenses/by-sa/3.0/
    "CC-BY-SA-3.0": "copyleft",
    **dict.fromkeys(
        (
            "CC-BY-NC-4.0",
            "CC-BY-NC-SA-3.0",
            "CC-BY-NC-SA-4.0",
            "CC-BY-NC-ND-4.0",
            "LicenseRef-NVIDIA-Source-Code-License-NC",  # NVIDIA research licence: non-commercial use only
            "LicenseRef-research-only",  # terms that allow research use only
            "LicenseRef-noncommercial",  # custom terms that forbid commercial use (e.g. InStereo2K)
            "LicenseRef-none",  # upstream publishes no licence at all: no grant, so no commercial use
            "LicenseRef-FAIR-Noncommercial-Research",  # Meta FAIR Noncommercial Research License v1 (VGGT-Omega)
            # Apple Machine Learning Research Model licence (the DFN CLIP weights): research purposes only, no
            # commercial or product use. https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14/blob/main/LICENSE (read
            # 2026-11-10).
            "LicenseRef-Apple-AMLR",
        ),
        "noncommercial",
    ),
}
NOT_STATED = "not stated"  # dataset terms only; weights without terms of their own are `ASSUMED`
# Weights released without a licence of their own are taken to be covered by the code licence; cards say so
# explicitly and listings show it.
ASSUMED = "assumed from code"
# A Hugging Face model card's `license: other`: terms stated elsewhere (the card's `evidence`), recorded as written.
OTHER = "other"
_RANK: dict[CodeClass, int] = {"permissive": 0, "copyleft": 1, "noncommercial": 2}
_COMMERCIAL: dict[CodeClass, Commercial] = {"permissive": "yes", "copyleft": "yes", "noncommercial": "no"}
_TOKEN = re.compile(r"\s+(AND|OR)\s+")


def licence_class(expression: str, where: str = "licence") -> CodeClass:
    """Class of an SPDX expression of ids joined by `AND` / `OR` (no parentheses).

    `AND` takes the most restrictive class (all terms apply), `OR` the least restrictive (the user may choose).

    Raises:
        CardError: Unknown id, parentheses, mixed `AND` / `OR`, or copyleft `AND` non-commercial (incompatible).
    """
    parts = _TOKEN.split(expression.strip())
    ids, ops = parts[::2], set(parts[1::2])
    if "(" in expression or len(ops) > 1:
        msg = f"{where}: {expression!r}: only `A AND B ...` or `A OR B ...` without parentheses is supported"
        raise CardError(msg)
    unknown = [i for i in ids if i not in CLASSES]
    if unknown:
        msg = f"{where}: unknown licence id(s) {unknown}; known: {sorted(CLASSES)} (add new ones to _core/licences.py)"
        raise CardError(msg)
    classes = {CLASSES[i] for i in ids}
    if ops == {"OR"}:
        return min(classes, key=_RANK.__getitem__)
    if {"copyleft", "noncommercial"} <= classes:
        msg = f"{where}: {expression!r} combines copyleft and non-commercial terms, which cannot both be met"
        raise CardError(msg)
    return max(classes, key=_RANK.__getitem__)


def weights_licence(weights: str, code: str) -> str:
    """The licence expression that holds for the weights: the code licence if the card says `ASSUMED`."""
    return code if weights == ASSUMED else weights


def describe_weights(weights: str, code: str) -> str:
    """Weights licence for listings, e.g. `"MIT (assumed from code)"`."""
    return f"{code} ({ASSUMED})" if weights == ASSUMED else weights


def weights_commercial(licence: str, where: str = "weights") -> Commercial:
    """Commercial status of a weights licence expression (resolve `ASSUMED` with `weights_licence` first).

    Raises:
        CardError: `"not stated"` (write `ASSUMED`) or an unknown expression.
    """
    if licence in {NOT_STATED, ASSUMED}:
        msg = f"{where}: {licence!r}: weights without a licence of their own are {ASSUMED!r} (the code licence)"
        raise CardError(msg)
    if licence == OTHER:
        return "unclear"
    return _COMMERCIAL[licence_class(licence, where)]


def code_commercial(cls: CodeClass) -> Commercial:
    """Commercial status of a code class (copyleft allows commercial use, with its obligations)."""
    return _COMMERCIAL[cls]


@dataclass(frozen=True)
class Dataset:
    """A training dataset (`_core/datasets.toml`).

    Attributes:
        id: Short id used in cards, e.g. `"sceneflow"`.
        name: Display name.
        licence: SPDX id, `LicenseRef-*`, or `"not stated"`.
        commercial: Whether the dataset terms allow commercial use.
        url: Page stating the terms.
    """

    id: str
    name: str
    licence: str
    commercial: Commercial
    url: str


UNSTATED = "unstated"  # training-data id for "not stated upstream"


@functools.cache
def datasets() -> dict[str, Dataset]:
    """All known training datasets by id."""
    data = tomllib.loads((importlib.resources.files("fastforks._core") / "datasets.toml").read_text())
    out = {}
    for key, t in data.items():
        commercial = t["commercial"]
        if commercial not in {"yes", "no", "unclear"}:
            msg = f"datasets.toml [{key}]: commercial {commercial!r} not in yes / no / unclear"
            raise CardError(msg)
        out[key] = Dataset(key, t["name"], t["licence"], commercial, t["url"])
    return out


def data_commercial(ids: tuple[str, ...]) -> Commercial:
    """Commercial status of the training data alone.

    Whether weights inherit the terms of their training data is legally unsettled, so it does not enter a variant's
    commercial status (user decision 2026-10-31): `ModelCard.status` lists restrictive, unclear or unstated data as
    a note only. This helper just classifies the data.
    """
    known = datasets()
    if any(i == UNSTATED or known[i].commercial != "yes" for i in ids):
        return "unclear"
    return "yes"


def worst(*statuses: Commercial) -> Commercial:
    """The most restrictive status: `no` < `unclear` < `yes`."""
    return "no" if "no" in statuses else "unclear" if "unclear" in statuses else "yes"
