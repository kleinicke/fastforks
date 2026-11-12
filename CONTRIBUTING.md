# Contributing

## Setup

```
uv sync                                           # dev tools + CUDA 13.0 torch (GPU machines)
uv sync --no-default-groups --group dev --group cpu   # CPU-only torch (what CI uses)
uv run pre-commit install
```

Checks (all must pass before a commit; CI runs all but the GPU tests):

```
uv run ruff format && uv run ruff check && uv run ty check
uv run pytest -m "not gpu"     # CPU: the fast tests (unit, policy, card, docs)
uv run pytest -m "not gpu" --slow   # also the slow ones (> 1 s: golden parity, exports, batch): before a release
uv run pytest -m gpu           # GPU parity of portable and kernels (needs a CUDA GPU)
uv run pytest -m "not gpu" --slow -o junit_family=legacy --junitxml=j.xml && uv run python tools/coverage_report.py j.xml   # which test kinds ran per family
```

Golden-parity tests need the golden outputs: `python tools/fetch_goldens.py --all DIR` and
`FASTFORKS_GOLDEN_PATH=DIR`. Local weights can be used without downloading via `FASTFORKS_WEIGHTS_PATH`, kernel
binaries via `FASTFORKS_KERNEL_PATH` (e.g. `build/kernels` after `tools/build_kernels.py`, which needs
`uv sync --group kernels`). Tests that need more than
~20 GB RAM on the CPU (Marigold's `torch.export`) are skipped unless `FASTFORKS_HEAVY_CPU_TESTS=1`; run them alone.
The ONNX export tests (ONNX Runtime against `module()`, ~30 min on the CPU) run only for a release, with
`FASTFORKS_ONNX_TESTS=1`.
`tools/stage_hf.py` stages the kernel binaries, the goldens and the weight mirrors for their Hugging Face repos
(`fastforks/kernels`, dataset `fastforks/reference-outputs`, `fastforks/mirrors`) and prints the upload command; uploads are a
maintainer step. With `--standin` it builds a local copy of those repos under OUT instead, and
`FASTFORKS_HF_ENDPOINT=file://OUT` points every download from them there (other URLs are unchanged), so new kernel
builds and goldens can be tested end to end before they are published.

## Code rules

- ruff runs with `select = ["ALL"]`. Every ignore, per-file ignore and `noqa` carries a reason. Vendored upstream
  network code (`models/*/net/`) has a relaxed per-file set so it stays diffable against upstream; correctness rules
  stay on there.
- ty runs on everything; a `ty: ignore[rule]` needs a reason after it.
- Google-style docstrings on public code.
- Runtime imports are limited to torch, numpy, pillow, platformdirs, safetensors, isal and the stdlib (`tests/test_policy.py`;
  laspy only in the point-cloud writer). Model code that needs another package vendors or rewrites the needed part (einops, cv2, ...), except the optional libraries timm, torchvision and transformers (extras in `pyproject.toml`): a family that builds its network with one imports it lazily and `load` fails with the pip command when it is missing.
- `mode="kernels"` never falls back: if a kernel cannot run, raise `KernelUnavailableError` with the reason (arch, driver,
  torch, missing binary). Every result carries the `Backend` that actually ran.

## Adding a model

A model is a directory `src/fastforks/<task>/models/<family>/`; S2M2 (`stereo/models/s2m2/`) is the reference.

```
<family>/
  card.toml      # data only (schema 2): licence, training data, io, params/options, modes, variants + weights
  LICENSE*       # the upstream licence file(s), verbatim; listed in card licence.files
  __init__.py    # one-line docstring
  model.py       # Params dataclass + one Runner per backend + RUNNERS
  net/           # vendored upstream network code, as close to upstream as possible
  kernels/       # optional: the custom-kernel forward path
```

`model.py` skeleton (the full contract is the docstring of `fastforks/stereo/_runner.py`):

```python
@dataclass(frozen=True)
class Params:  # = card [params] + variant params + [options]; checked at load
    iters: int


class Portable(Runner):
    dtypes = ("float32", "float16")  # what this code implements on CUDA; the card picks the validated one
    Params = Params

    def __init__(self, ctx: Context) -> None:
        super().__init__(ctx)
        self.net = Net().eval().to(ctx.device)
        self.net.load_state_dict(ctx.state_dict())  # hash-checked, cached, restricted unpickler

    def __call__(self, left, right):  # scaled + padded [B, 3, H, W] -> {"disparity": [B, H, W], ...}
        return {"disparity": self.net(left, right, iters=self.ctx.params.iters)[:, 0]}

    def module(self):  # plain-PyTorch nn.Module for torch.export / ONNX
        return Forward(self.net, lambda net, l, r: net(l, r)[:, 0])


RUNNERS = {"portable": Portable, "kernels": Kernels}
```

The core converts and checks images, scales and pads them as the card `[io]` says, crops the outputs, checks their
names and shapes, splits batches by `Runner.batch` and records the backend and non-default options in every result.

Architecture is card data, not code: constants such as `max_disp` or the encoder go in `[params]` (a variant overrides
them in its own `params`), and a variant that needs other padding sets `io = { pad = { multiple = 96 } }` (merged into
the family `[io]`; `ctx.io` is the effective value). `pad = "none"` means the core pads nothing (the net pads itself).
A policy test rejects module-level numbers in `model.py` unless they carry a `# fixed: <reason>` comment. Keep a
value a `[params]` entry, not an `[options]` one, unless the portable and the kernels path both run correctly at other
values.

Other tasks follow the same layout; their runner contract is the docstring of `fastforks/<task>/_runner.py`, and
`fastforks/_core/tasks/<task>.py` says what a card of the task may declare (a new task adds its module there
and is registered from it):

| task | outputs (others go to `extras`) | `[output]` | `[io]` keys | who applies `[io]` |
|---|---|---|---|---|
| stereo | `disparity` (required), `confidence`, any extra map | - | `input`, `pad`, `min_size`, `mean`, `std` | core (runner `io = "raw"`: the runner) |
| depth | one of `depth` / `inverse_depth` / `points`; `intrinsics`, `confidence`, `mask`, `normals`, extras | `kind` | `input`, `pad`, `min_size`, `mean`, `std` | the runner (`scale(image, ctx.io)`; it resizes itself) |
| classification | `logits` | `label_set` | `input`, `min_size`, `mean`, `std`, `resize`, `crop`, `interpolation` | core (timm eval transform) |
| segmentation | `logits` | `label_set` | `input`, `pad`, `min_size`, `mean`, `std` | the runner |
| promptable_segmentation | `pred_masks` or `masks` (required), `iou_scores`, `boxes`, `scores` | - | `input`, `min_size`, `mean`, `std` | the runner (it resizes / pads and maps prompts and masks back itself) |
| detection | `boxes` (required), `scores`, `labels`, `masks` | `label_set` | `input`, `min_size`, `mean`, `std` | the runner (it resizes / pads and maps boxes and masks back itself) |
| features | `last_hidden_state` (required), `pooler_output`, `hidden_states` | - | `input`, `pad`, `min_size`, `mean`, `std`, `interpolation` | core (resizes to a multiple of `pad.multiple`, the patch stride) |
| multiview | `depth` or `world_points` (required), `confidence`, `mask`, `intrinsics`, `cam_to_world`, extras | `kind` | `input`, `pad`, `min_size`, `mean`, `std` | the runner (`scale(images, ctx.io)`; it resizes itself) |

`mean` / `std` need `input = "mean_std"`. A variant may override `outputs` and `[output]` keys (`outputs = [...]`,
`output = { kind = "metric" }`), e.g. a metric checkpoint of a relative-depth family or another label set.

Classification ids follow timm, `<name>.<tag>` (`resnet50.tv2_in1k`). Mark the variant that the untagged name should
load with `default_tag = true` (the tag `timm.create_model("<name>")` picks): `classification.load("resnet50")` then
loads it, and results record the full id. At most one variant per name and task is the default.

`limitations = [...]` (optional, top level of the card, before the first table): short one-line sentences (<= 300
characters, unique) for what a user should know and the card does not otherwise say: failure modes, input-size
limits, missing kernels or variants, graph breaks / `torch.export` limits, nondeterminism, checkpoints upstream
releases but we do not ship. Only documented facts (card comments, xfail reasons, results files); leave it out when
unsure. The model pages show it; the card tests list the families that have none.

Checklist:

1. **Licence.** The upstream repository has a licence file (unlicensed repositories are not accepted). Copy it
   verbatim and list it in `licence.files`; set `licence.code` (SPDX expression, ids from `_core/licences.py`),
   `weights` (SPDX, `LicenseRef-*`, or `"assumed from code"` when upstream gives the weights no licence of their own:
   the code licence then covers them, and listings say so), `evidence` (URL of the upstream line stating the licence, at
   the pinned commit) and `training_data` (ids from `_core/datasets.toml`, or `"unstated"`, with `training_note`;
   per-variant overrides in the variant's `licence = {...}`). The commercial status is computed, never written: see
   `ModelCard.status`. The code class decides the distribution (`dists.toml`).
2. **Upstream pin.** `upstream.commit` is the commit the vendored code comes from. Keep per-file licence headers.
3. **Weights.** Only the authors' URLs (or a mirror if the licence clearly allows redistribution). Record `size` and
   `sha256`; `tools/check_urls.py` checks the URLs nightly.
4. **Dependencies.** Strip or vendor anything outside the allowlist. Code used by more than one family (e.g. a
   backbone) goes into `src/fastforks/shared/<module>/` with its LICENSE and a `meta.toml` (permissive only); list
   it in the card's `uses`. Families never import each other (`tests/test_policy.py`).
5. **Goldens.** Write `tools/golden/<family>.py`: it runs the reference implementation (upstream at a pinned
   commit, plus a minimal patch in `tools/golden/patches/` only if upstream cannot run as is; CPU float32) on
   `tests/synth.py`'s pair and records `<id>@<revision>` in `src/fastforks/<task>/models/<family>/golden.toml`
   (`[refs]`; not shipped in wheels). The family's `[tolerance]` there bounds our portable CPU path (`cpu_atol`) and
   GPU portable / kernels (`gpu_mean`, `gpu_p99`, px). `tests/test_export.py` checks that `Runner.module()` exports
   with `torch.export`.
6. **Kernels backend.** Kernels go into a shared library under `src/fastforks/kernels/<lib>/` (reuse an existing one if
   it fits) with a `meta.toml` (`licence`, `derived_from`; must be permissive); list the libraries in
   `modes.kernels.kernels`. Rebuild with `tools/build_kernels.py`, which regenerates the manifest, then run
   `pytest -m gpu --record` on a GPU: it writes the validated kernel builds per variant into `[validated]` of the
   family's golden.toml, and the CPU suite fails for any kernels variant whose current builds were not validated.
7. **Revision.** Bump a variant's `revision` whenever weights or numerics change, and regenerate its golden.
8. **Docs.** Measure the largest size that runs (`tools/bench_sizes.py run --out sizes.jsonl --modes portable --
   <id>@<revision>` on a GPU, then `tools/extract_sizes.py sizes.jsonl`, which merges it into
   `src/fastforks/_core/sizes.json`; `tests/test_sizes.py` fails for a ref without it), then `uv run python
   tools/gen_docs.py` (the docs test fails if the model pages are stale).
9. **Tier.** New models start as `experimental`; `stable` once goldens, GPU parity and a benchmark entry exist.

## Speed check

`docs/data/speed.jsonl` is the speed and memory floor. On a GPU, `python tools/speed_check.py run --out rows.jsonl
--check [--only REF ...]` measures the portable and kernels rows and fails when one is more than 5 % (or its noise band)
slower than its floor row, or uses more than 5 % more peak memory; with the rows of an earlier run, `--changed-since
COMMIT` re-measures only the models whose source files changed since then. `tests/test_speed_floor.py`
checks the floor itself without a GPU: per model and input, kernels must not be slower than portable, and neither may
be more than 3 % slower than the faster original (`upstream`, `upstream-default`). A row that cannot meet that needs an
entry in `docs/data/speed_allowlist.json` with its measured reason; remove the entry when the row is fixed.

## Packaging

The core wheel `fastforks-core` ships every model family with permissive code; `dists.toml` cuts the others into
separate distributions by code-licence class, so `pip install fastforks-core` never pulls non-commercial or GPL code and
`[nc]` adds it. The `fastforks` metapackage has no code: it is `fastforks-core[nc,libs]`. `tools/build_dists.py` builds
all wheels. Non-commercial weights download only with `FASTFORKS_NONCOMMERCIAL=1` (the test suite sets it). A model's
class is set by its card; a new non-permissive class needs an entry in `dists.toml`.
