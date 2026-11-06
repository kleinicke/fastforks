"""Stage our Hugging Face repos (kernel binaries, golden outputs, weight mirrors) in a local directory (no upload).

    uv run python tools/stage_hf.py kernels OUT [--from build/kernels] [--standin]
    uv run python tools/stage_hf.py goldens OUT --from $FASTFORKS_GOLDEN_PATH [--standin]
    FASTFORKS_WEIGHTS_PATH=... uv run python tools/stage_hf.py mirrors OUT [--standin]
    uv run python tools/stage_hf.py cards OUT
    uv run python tools/stage_hf.py cards --check-live

`kernels` copies the files the committed `manifest.toml`s list (hash- and size-checked) to `OUT/<lib>/<file>`, the
layout of the manifests' `url`, and writes the repo card. `goldens` copies every `<ref>.npz` of the `golden.toml`
files (hash-checked) to `OUT/<ref>.npz`, the layout `tools/fetch_goldens.py` reads, with a card listing the model
each output comes from. `mirrors` copies the weights whose card entry has `mirror = true` from the local weight
store (hash-checked) to `OUT/<family>/<file>`, with a card naming each file's licence and original URL. The tool
prints the upload command; it never uploads.

`cards` writes one model card (README.md only, no weights) per model family to `OUT/<repo name>/README.md`, the
repos `fastforks/<family>`, from the model cards, the golden tolerances, `docs/benchmarks.md` and `docs/data/`, and
prints the repo ids. Each card ends with a comment naming the fastforks commit and date it was staged from.
`cards --check-live` downloads the published card of every repo (anonymously, read-only), prints the repos whose
card differs from the freshly staged one (the commit comment is ignored) and exits 1 if there are any.

With `--standin`, OUT is the root of a local stand-in for huggingface.co and the repo is staged at
`OUT/<repo>/resolve/main/` (datasets under `OUT/datasets/`); `FASTFORKS_HF_ENDPOINT=file://OUT` then makes fastforks
download from it exactly as it will from the published repos.
"""

# ruff: noqa: ISC004, SLF001  # card text joins f-strings in lists; reuses the private renderers of gen_docs
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tomllib
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING

import gen_docs
import goldens as store
from fastforks._core import licences, registry
from fastforks._core.errors import WeightsError
from fastforks._core.files import MIRROR_URL, FileSpec, locate, sha256_of
from fastforks._core.registry import WEIGHTS_PATH_ENV
from fastforks._core.sizes import card_size_lines

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastforks._core.cards import ModelCard, Variant

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "src" / "fastforks" / "kernels"
GITHUB = "https://github.com/kleinicke/fastforks"
KERNELS_REPO = "https://huggingface.co/fastforks/kernels"
HF_URL = r"https://huggingface\.co/([^/]+/[^/]+)/resolve/"
RESERVED_REPOS = frozenset({"kernels", "goldens", "reference-outputs", "mirrors", "fastforks"})
REPOS = {
    "kernels": ("fastforks/kernels", "model"),
    "goldens": ("fastforks/reference-outputs", "dataset"),
    "mirrors": ("fastforks/mirrors", "model"),
    "cards": ("fastforks/<family>", "model"),
}


def _check(src: Path, sha256: str, size: int | None = None) -> None:
    if not src.is_file():
        sys.exit(f"missing: {src}")
    if size is not None and src.stat().st_size != size:
        sys.exit(f"{src}: size {src.stat().st_size}, manifest says {size}")
    if sha256_of(src) != sha256:
        sys.exit(f"{src}: SHA-256 differs from the recorded one")


def stage_kernels(src: Path, out: Path) -> list[str]:
    """Copy the manifest-listed binaries; return the card's table rows."""
    rows = []
    for manifest in sorted(KERNELS.glob("*/manifest*.toml")):  # + manifest.compat.toml (older-driver builds)
        m = tomllib.loads(manifest.read_text())
        lib = m["name"]
        for f in m["files"].values():
            _check(src / lib / f["name"], f["sha256"], f["size"])
            (out / lib).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / lib / f["name"], out / lib / f["name"])
        targets = ", ".join(m["files"])
        rows.append(
            f"| `{lib}` | {m['build_id']} | {m['toolkit']} | {targets} | {m['licence']} | {m['derived_from']} |"
        )
    for name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copyfile(ROOT / name, out / name)
    return rows


def stage_goldens(src: Path, out: Path) -> list[str]:
    """Copy the golden outputs; return the card's table rows."""
    cards = {v.ref: (card, v) for t in registry.TASKS for card, v in registry.cards(t).values()}
    rows = []
    out.mkdir(parents=True, exist_ok=True)
    for refs in (fam["refs"] for fam in store.load_all(registry.TASKS).values()):
        for ref, entry in sorted(refs.items()):
            if ref not in cards:
                sys.exit(f"{ref}: no card variant (a stale golden ref?)")
            _check(src / f"{ref}.npz", entry["sha256"])
            shutil.copyfile(src / f"{ref}.npz", out / f"{ref}.npz")
            card, v = cards[ref]
            st = card.status(v)
            rows.append(f"| `{ref}` | {card.family} | {card.licence.code} | {st.weights} | {entry['upstream']} |")
    shutil.copyfile(ROOT / "LICENSE", out / "LICENSE")
    return rows


def mirrored() -> dict[str, list[tuple[ModelCard, Variant, FileSpec]]]:
    """`<family>/<file>` -> [(card, variant, spec)] of every weight file that has a mirror."""
    out: dict[str, list[tuple[ModelCard, Variant, FileSpec]]] = {}
    for t in registry.TASKS:
        for card, v in registry.cards(t).values():
            for spec in v.weights:
                if spec.mirror is not None:
                    path = spec.mirror.removeprefix(MIRROR_URL.partition("{group}")[0])
                    out.setdefault(path, []).append((card, v, spec))
    return out


def stage_mirrors(out: Path) -> list[str]:
    """Copy the mirrored weights from the local weight store; return the card's table rows."""
    rows = []
    for path, uses in sorted(mirrored().items()):
        card, _, spec = uses[0]
        try:
            src = locate(spec, group=card.family, env_var=WEIGHTS_PATH_ENV, allow_download=False)
        except WeightsError as e:
            sys.exit(f"{path}: {e}")
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, out / path)
        lic = {licences.weights_licence(lic.weights, lic.code) for lic in (c.variant_licence(v) for c, v, _ in uses)}
        refs = ", ".join(f"`{v.ref}`" for _, v, _ in uses)
        rows.append(f"| `{path}` | {refs} | {' / '.join(sorted(lic))} | {spec.url} |")
    shutil.copyfile(ROOT / "LICENSE", out / "LICENSE")
    return rows


KERNELS_CARD = """\
---
license: other
license_name: per-library
tags: [cuda, kernels, fastforks]
---
# fastforks kernels

Prebuilt CUDA kernels of the [fastforks](https://github.com/kleinicke/fastforks) `mode="kernels"` backends, one file per
library, build and GPU architecture. fastforks downloads them on demand and checks each against the SHA-256 in the
library's `manifest.toml`; nothing here is loaded without that check.

Our kernel code is Apache-2.0 (`LICENSE`). Libraries built against CUTLASS headers also contain CUTLASS code
(BSD-3-Clause, see `THIRD_PARTY_NOTICES.md`). The toolkit and host compiler of every build are in its manifest.

| library | build | CUDA | targets | licence | derived from |
|---|---|---|---|---|---|
"""

GOLDENS_CARD = """\
---
license: other
license_name: per-model
tags: [stereo, test-data, fastforks]
---
# fastforks reference outputs

Reference outputs for the [fastforks](https://github.com/kleinicke/fastforks) parity tests: the disparity of each
model variant's original implementation (pinned commit, CPU float32) on the synthetic stereo pair of `tests/synth.py`.
The input pair and the files' layout are ours (Apache-2.0, `LICENSE`); each output is produced by the model and weights
named below, under their licences.

| file | family | code licence | weights | original |
|---|---|---|---|---|
"""


MIRRORS_CARD = """\
---
license: other
license_name: per-file
tags: [stereo, weights, fastforks]
---
# fastforks weight mirrors

Byte-identical copies of model weights for [fastforks](https://github.com/kleinicke/fastforks), used only when the
original URL below fails; fastforks checks each file against the SHA-256 in the model's card. We mirror only weights
whose licence allows redistribution; each file stays under its own licence (below, from the model card). We did not
train or change any of them.

| file | variants | weights licence | original URL |
|---|---|---|---|
"""


# ---- model cards (one HF model repo per family) ----

_HF_LICENCES = {
    "MIT": "mit",
    "Apache-2.0": "apache-2.0",
    "BSD-2-Clause": "bsd-2-clause",
    "BSD-3-Clause": "bsd-3-clause",
    "GPL-3.0-only": "gpl-3.0",
    "CC-BY-4.0": "cc-by-4.0",
    "CC-BY-NC-4.0": "cc-by-nc-4.0",
    "CC-BY-NC-SA-4.0": "cc-by-nc-sa-4.0",
}
_PIPELINE = {
    "stereo": "depth-estimation",
    "depth": "depth-estimation",
    "features": "image-feature-extraction",
    "classification": "image-classification",
    "segmentation": "image-segmentation",
    "promptable_segmentation": "mask-generation",
    "detection": "object-detection",
    "multiview": "image-to-3d",
}
_TASK_TAGS = {
    "stereo": ["stereo", "stereo-matching", "disparity"],
    "depth": ["mono-depth", "monocular-depth"],
    "features": ["backbone", "feature-extraction"],
    "classification": ["image-classification"],
    "segmentation": ["semantic-segmentation"],
    "promptable_segmentation": ["segment-anything", "promptable-segmentation"],
    "detection": ["object-detection", "instance-segmentation"],
    "multiview": ["multi-view-stereo", "3d-reconstruction", "camera-pose-estimation"],
}
_SNIPPET = {
    "stereo": 'result = model("left.png", "right.png")\nresult.{out}  # [H, W] float32 tensor, input resolution',
    "depth": (
        'result = model("room.jpg")\nresult.{out}  # [H, W] float32 tensor; result.kind says what the maps mean\n'
        '# models that take a camera: model("room.jpg", intrinsics=K), K [3, 3] in pixels'
    ),
    "features": 'result = model("photo.jpg", size=518)\nresult.{out}  # tokens [N, C], as Hugging Face\'s output',
    "classification": 'result = model("photo.jpg")\nresult.top(5)  # top-5 classes',
    "segmentation": 'result = model("photo.jpg")\nresult.labels  # [H, W] per-pixel class index',
    "promptable_segmentation": (
        'result = model("photo.jpg", input_points=[[x, y]], input_labels=[1])\nresult.masks  # [M, H, W] bool\n'
        '# SAM 3 (text prompts): model("photo.jpg", text="coin") -> boxes, scores, masks'
    ),
    "detection": 'result = model("photo.jpg")\nresult.boxes, result.scores, result.labels  # [N, 4], [N], [N]',
    "multiview": (
        'result = model(["view0.jpg", "view1.jpg", "view2.jpg"])  # the views of one scene, no cameras needed\n'
        "result.{out}  # [V, H, W] float32 tensor; result.kind says what the maps mean"
    ),
}
# Families whose call differs from their task's snippet (the family's runner decides, see tests/test_hf_cards.py).
_SNIPPET_FAMILY = {
    "sam3": (
        'result = model("photo.jpg", text="coin")  # a concept: every instance of it; or input_boxes= exemplar boxes\n'
        "result.boxes, result.scores, result.masks  # [N, 4], [N], [N, H, W] bool"
    ),
    "mvsanywhere": (
        'result = model(["ref.jpg", "src1.jpg", "src2.jpg"], intrinsics=K, cam_to_world=c2w)  # views[0] = reference\n'
        "result.{out}  # [1, H, W] float32 tensor; K [N, 3, 3] pixels, c2w [N, 4, 4] OpenCV camera-to-world"
    ),
    "patchmatchnet": (
        'result = model(["ref.jpg", "src1.jpg", "src2.jpg"], intrinsics=K, cam_to_world=c2w, depth_range=(0.5, 5.0))\n'
        "result.{out}  # [1, H, W] float32 tensor; views[0] = reference, K [N, 3, 3] pixels, c2w [N, 4, 4] OpenCV"
    ),
}
_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")


def repo_name(family: str) -> str:
    """HF repo name of a family: lowercase, `[a-z0-9-]` only (`_` becomes `-`)."""
    name = re.sub(r"-+", "-", re.sub(r"[^a-z0-9-]", "-", family.lower())).strip("-.")
    if not name or name in RESERVED_REPOS:
        sys.exit(f"family {family!r}: repo name {name!r} is empty or reserved ({sorted(RESERVED_REPOS)})")
    return name


def _families() -> dict[str, tuple[ModelCard, list[Variant]]]:
    fams: dict[str, tuple[ModelCard, list[Variant]]] = {}
    for task in registry.TASKS:
        for card, v in registry.cards(task).values():
            fams.setdefault(card.family, (card, []))[1].append(v)
    return fams


def _benchmark_tables() -> list[tuple[str, list[str], dict[str, list[list[str]]]]]:
    """(heading, header cells, {model id: [row cells]}) of every `model` table of docs/benchmarks.md."""
    lines = (ROOT / "docs" / "benchmarks.md").read_text().splitlines()
    out, heading, i = [], "", 0
    while i < len(lines):
        if lines[i].startswith("#"):
            heading = lines[i].lstrip("#").strip()
        elif lines[i].startswith("| model |"):
            cells = lambda ln: [c.strip() for c in _TABLE_ROW.match(ln).group(1).split("|")]  # noqa: E731
            head, rows = cells(lines[i]), {}
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                row = cells(lines[i])
                rows.setdefault(row[0].strip("`"), []).append(row[1:])
                i += 1
            out.append((heading, head, rows))
            continue
        i += 1
    return out


def _licence_front(card: ModelCard, variants: list[Variant]) -> dict[str, str]:
    """HF licence keys: the SPDX id of the weights if HF knows it, else `other` + name + link."""
    found = {card.variant_licence(v).weights_licence for v in variants}
    link = next((card.variant_licence(v).evidence for v in variants if card.variant_licence(v).evidence), None)
    link = link.split()[0].rstrip(";,.") if link and link.startswith("http") else card.upstream
    slug = lambda t: re.sub(r"[^a-z0-9.]+", "-", t.lower()).removeprefix("licenseref-").strip("-")  # noqa: E731
    if assumed_weights(card, variants):  # the authors state no weights licence: no firm id
        return {
            "license": "other",
            "license_name": "no-licence-stated"
            if slug(card.licence.code) == "none"
            else f"{slug(card.licence.code)}-code-weights-unstated",
            "license_link": link,
        }
    if len(found) == 1 and (only := next(iter(found))) in _HF_LICENCES:
        return {"license": _HF_LICENCES[only]}
    name = "mixed-see-variant-table" if len(found) > 1 else slug(next(iter(found)))
    if name == "none":
        name = "no-licence-stated"
    return {"license": "other", "license_name": name, "license_link": link}


def assumed_weights(card: ModelCard, variants: list[Variant]) -> bool:
    """True if every variant's weights licence is only `assumed from code`."""
    return all(card.variant_licence(v).weights == licences.ASSUMED for v in variants)


def _yaml(front: dict[str, str | list[str]]) -> str:
    def q(s: str) -> str:
        return json.dumps(s)

    lines = ["---"]
    for k, val in front.items():
        if isinstance(val, list):
            lines += [f"{k}:", *(f"- {q(x)}" for x in val)] if val else []
        else:
            lines.append(f"{k}: {q(val)}")
    return "\n".join([*lines, "---"])


def _bar(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def usage_example(card: ModelCard, first: Variant) -> list[str]:
    """The `load` and call lines of the card's quickstart: the family's real call contract, plus the opt-in note."""
    task = card.task
    nc = licences.weights_commercial(card.variant_licence(first).weights_licence) == "no"
    return [
        *([f"# non-commercial weights: run with {registry.NONCOMMERCIAL_ENV}=1 in the environment"] if nc else []),
        f'model = {task}.load("{first.id}")  # mode="auto": the preferred backend from recorded measurements',
        _SNIPPET_FAMILY.get(card.family, _SNIPPET[task]).format(out=card.outputs[0]),
    ]


def render_card(card: ModelCard, variants: list[Variant]) -> str:
    """The README.md (HF model card) of one family."""
    task, lic = card.task, card.licence
    base = sorted({m.group(1) for v in variants for w in v.weights if w.url and (m := re.match(HF_URL, w.url))})
    front = {
        "library_name": "fastforks",
        "pipeline_tag": _PIPELINE[task],
        **_licence_front(card, variants),
        "tags": list(dict.fromkeys([task, *_TASK_TAGS[task], "cuda", "pytorch", "fastforks"])),
    }
    first = variants[0]
    author_repos = ", ".join(f"<https://huggingface.co/{r}>" for r in base)
    lines = [
        _yaml(front),
        f"# {card.name} (fastforks)",
        "",
        card.summary,
        "",
        f"This repo is the model card only: it holds no weights. fastforks downloads the authors' weights from the "
        f"links below and checks them against the SHA-256 in the model card. Code: <{GITHUB}>. Kernels for the "
        f"`kernels` backend: <{KERNELS_REPO}>. Original code: <{card.upstream}> at "
        f"`{card.upstream_commit[:12]}`.",
        "",
        *([f"Author weight repos: {author_repos}.", ""] if base else []),
        f"Task: {task}. Tier: {card.tier}. Outputs: {', '.join(f'`{o}`' for o in card.outputs)}. Backends: "
        f"{', '.join(f'`{m}`' for m in card.modes)}.",
        "",
        gen_docs.TASK_IO[task],
        "",
        "## Variants",
        "",
        "Weights are the authors' files (not hosted here). SHA-256 shortened to 12 hex digits; the full hashes are in "
        f"[`card.toml`]({GITHUB}/blob/main/src/fastforks/{task}/models/{card.family}/card.toml).",
        "",
        "| variant | weights (author link) | SHA-256 | weights licence | commercial use |",
        "|---|---|---|---|---|",
    ]
    for v in variants:
        st = card.status(v)
        files = [f"[{w.name}]({w.url})" if w.url else f"{w.name} (not published)" for w in v.weights]
        sums = [f"`{w.sha256[:12]}`" for w in v.weights]
        lines.append(
            _bar([f"`{v.ref}`", "<br>".join(files), "<br>".join(sums), st.weights, gen_docs._COMMERCIAL[st.commercial]])
        )
    lines += [
        "",
        "## Licence",
        "",
        f"Original-code strings, recorded as they are (research-use package; check them before any other use). Code: "
        f"{lic.code}. Files: {', '.join(f'`{f}`' for f in lic.files)}.",
        "",
        *(
            [f"Code licence: {lic.code}; the authors do not state a separate weights licence.", ""]
            if assumed_weights(card, variants)
            else []
        ),
        f"Training data: {gen_docs.data_text(lic.training_data, lic.training_note)}.",
        *(["", *(f"- Licence note: {n}" for n in card.licence_notes)] if card.licence_notes else []),
    ]
    if lic.evidence:
        lines += ["", f"Licence evidence: {lic.evidence if ' ' in lic.evidence else f'<{lic.evidence}>'}"]
    lines += ["", "## Input size", "", *card_size_lines(card, variants, heading=False)]
    lines += ["", "## Speed and memory", ""]
    lines += _speed_section(card.task, variants)
    lines += ["", "## Accuracy", ""]
    lines += _accuracy_section(card)
    if card.limitations:
        lines += ["", "## Limitations", "", *(f"- {x}" for x in card.limitations)]
    lines += [
        "",
        "## Usage",
        "",
        "```python",
        f"from fastforks import {task}",
        "",
        *usage_example(card, first),
        "result.backend  # what ran, e.g. Backend(mode='portable', device='cuda:0', dtype='float16')",
        "```",
        "",
        f"Install with `pip install fastforks` (see <{GITHUB}>). Other variants: "
        + ", ".join(f"`{v.id}`" for v in variants)
        + ".",
        "",
        "## Kernels backend",
        "",
    ]
    if card.kernel_libs:
        kernels = ", ".join(f"`{k}`" for k in card.kernel_libs)
        lines += [
            f'`{task}.load("{first.id}", mode="kernels")` runs our CUDA kernels ({kernels}). '
            "It needs a prebuilt kernel for your GPU architecture (downloaded from "
            f"<{KERNELS_REPO}> on first use) and a recent NVIDIA driver; if there is none it raises "
            '`BackendUnavailableError` with the reason and never falls back silently. `mode="auto"` picks `kernels` '
            "when it can run and logs the choice."
        ]
    else:
        lines += ["No kernels backend" + (f": {card.no_kernels_reason}." if card.no_kernels_reason else ".")]
    return "\n".join(lines) + "\n"


def _ms(cell: str) -> float:
    """Milliseconds of a fixed-size cell (`113.0 ms / 1 / 2`), 0 if not a time."""
    m = re.match(r"([\d.]+) ms", cell)
    return float(m[1]) if m else 0.0


def _speed_section(task: str, variants: list[Variant]) -> list[str]:
    ids = {v.id for v in variants}
    lines: list[str] = []
    std = gen_docs._std_input(task, {})
    tables = _benchmark_tables()
    prefix = task.replace("_", " ")

    def sized(heading: str) -> str:
        m = re.search(r", (\d+x\d+)$", heading)
        return m[1] if m and heading.lower().startswith(prefix + ",") else ""

    by_id = {v.id: v for v in variants}
    shared: dict[str, str] = {}
    at_std = {i for h, _, rows in tables if sized(h) == std for i in rows}
    for heading, head, rows in tables:
        size = sized(heading)
        # one standard size per task; a model without a row there keeps its other tables (the size is in the heading)
        hit = [i for i in rows if i in ids and (not size or std in {gen_docs._OWN, size} or i not in at_std)]
        dups = gen_docs.same_as(hit, {i: rows[i] for i in hit})
        shared |= dups
        hit = [i for i in hit if i not in dups]
        if hit:
            lines += ["", f"### {heading}", "", _bar(head), _bar(["---"] * len(head))]
            lines += [_bar([gen_docs.labelled(variants, by_id[i]), *r]) for i in hit for r in rows[i]]
    if lines:
        lines = [*lines[1:], "", "Fixed-size cells: time (ms) / peak allocated / peak reserved memory (MiB)."]
        ratio = [
            i
            for _, head, rows in tables
            if head[-1].startswith("12 MP")
            for i in rows
            if i in ids and any(_ms(r[-1]) < 0.9 * _ms(r[-2]) for r in rows[i] if _ms(r[-1]) and _ms(r[-2]))
        ]
        lines += ["", *gen_docs._same_note(shared)] if shared else []
        if ratio:
            lines += ["", "Resizes to its working resolution: time depends on the aspect ratio, not the input pixels."]
        lines += [
            "",
            "Hardware, protocol and caveats: "
            f"[benchmarks]({GITHUB}/blob/main/docs/benchmarks.md) (RTX 5060 Ti 16 GB, sm_120). "
            "Original = the original code named above at the precision of portable (fp16, or `(fp32)` where fp16 is "
            "unsupported or not finite); original fp32 = its default fp32 run. Peak memory is the torch allocator's.",
        ]
        return lines
    return ["Not measured yet."]


def _accuracy_section(card: ModelCard) -> list[str]:
    """Accuracy on public datasets (original code, portable, kernels) and the golden-output parity check."""
    measured = gen_docs._measured(card.task, card, gen_docs._eval_rows(card.task))
    lines = [ln.replace("### Measured", "### On public datasets") for ln in measured[1:]]
    return [*lines, *([""] if lines else []), "### Parity with the original code", "", *_parity_section(card)]


def _parity_section(card: ModelCard) -> list[str]:
    path = ROOT / "src" / "fastforks" / card.task / "models" / card.family / "golden.toml"
    tol = tomllib.loads(path.read_text()).get("tolerance") if path.is_file() else None
    if not tol:
        return ["No golden tolerance recorded."]
    return [
        "Every variant is checked against reference outputs of the original code at its pinned commit. "
        + gen_docs.parity_text(card, tol)
    ]


SOURCE_RE = re.compile(r"\n*<!-- staged from fastforks [^>]*-->\s*$")


def source_line() -> str:
    """`<!-- staged from fastforks <short hash> (<commit date>) -->` of the current checkout (`unknown` outside git)."""
    try:
        h, d = subprocess.run(
            ["git", "-C", str(ROOT), "log", "-1", "--format=%h %cs"], capture_output=True, text=True, check=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError, ValueError):
        h, d = "unknown", "unknown"
    return f"<!-- staged from fastforks {h} ({d}) -->"


def strip_source(card: str) -> str:
    """The card without its source-commit comment and trailing whitespace (what `--check-live` compares)."""
    return SOURCE_RE.sub("", card.replace("\r\n", "\n")).rstrip()


def staged_cards() -> dict[str, str]:
    """`repo id -> card text` of every family (source-commit comment included)."""
    names: dict[str, str] = {}
    fams = _families()
    for family in fams:
        name = repo_name(family)
        if name in names:
            sys.exit(f"repo name {name!r} is used by families {names[name]!r} and {family!r}")
        names[name] = family
    src = source_line()
    return {
        f"fastforks/{n}": render_card(*fams[family]).rstrip() + f"\n\n{src}\n" for n, family in sorted(names.items())
    }


def fetch_live(repo_id: str) -> str | None:
    """The published README.md of a public repo (anonymous GET); None if the repo or the file does not exist yet."""
    try:
        with urllib.request.urlopen(f"https://huggingface.co/{repo_id}/raw/main/README.md", timeout=30) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        if e.code == HTTPStatus.NOT_FOUND:
            return None
        raise


def differing(cards: dict[str, str], fetch: Callable[[str], str | None] = fetch_live) -> list[str]:
    """Repo ids whose live card differs from the staged one (ignoring the source-commit comment) or is missing."""
    return [r for r, text in cards.items() if (live := fetch(r)) is None or strip_source(live) != strip_source(text)]


def stage_cards(out: Path) -> list[str]:
    """Write `OUT/<repo name>/README.md` per family; return the repo ids."""
    cards = staged_cards()
    for repo_id, text in cards.items():
        (out / repo_id.split("/", 1)[1]).mkdir(parents=True)
        (out / repo_id.split("/", 1)[1] / "README.md").write_text(text)
    return list(cards)


def check_live() -> None:
    """Print the repos whose published card differs from the staged one; exit 1 if any (read-only)."""
    cards = staged_cards()
    diff = differing(cards)
    for repo_id in diff:
        print(f"differs: {repo_id}")
    print(f"{len(diff)} of {len(cards)} live cards differ from the staged ones")
    sys.exit(1 if diff else 0)


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=sorted(REPOS))
    ap.add_argument(
        "out", type=Path, nargs="?", help="staging directory (must not exist; with --standin: the stand-in root)"
    )
    ap.add_argument("--from", dest="src", type=Path, help="source directory (default for kernels: build/kernels)")
    ap.add_argument("--check-live", action="store_true", help="cards: compare with the published cards, upload nothing")
    ap.add_argument("--standin", action="store_true", help="stage into a huggingface.co stand-in rooted at OUT")
    a = ap.parse_args()
    if a.check_live and a.what != "cards":
        ap.error("--check-live is for cards")
    if a.out is None and not a.check_live:
        ap.error("OUT is required")
    return a


def main() -> None:
    a = _args()
    if a.check_live:
        check_live()
    repo, kind = REPOS[a.what]
    out = a.out / ("datasets/" if kind == "dataset" else "") / repo / "resolve" / "main" if a.standin else a.out
    if out.exists():
        sys.exit(f"{out} exists; pick a new directory")
    if a.what == "cards":
        if a.standin:
            sys.exit("cards has no --standin")
        out.mkdir(parents=True)
        for repo_id in (ids := stage_cards(out)):
            print(repo_id)
        print(f"staged {len(ids)} model cards in {out}")
        return
    if a.what == "kernels":
        rows, card = stage_kernels(a.src or ROOT / "build" / "kernels", out), KERNELS_CARD
    elif a.what == "mirrors":
        rows, card = stage_mirrors(out), MIRRORS_CARD
    elif a.src is None:
        sys.exit("goldens need --from DIR")
    else:
        rows, card = stage_goldens(a.src, out), GOLDENS_CARD
    (out / "README.md").write_text(card + "\n".join(rows) + "\n")
    print(f"staged {len(rows)} entries in {out}")
    if a.standin:
        print(f"use it: FASTFORKS_HF_ENDPOINT=file://{a.out.resolve()}")
    else:
        print(f"upload (needs the maintainers' OK): hf upload --repo-type {kind} {repo} {out} .")


if __name__ == "__main__":
    main()
