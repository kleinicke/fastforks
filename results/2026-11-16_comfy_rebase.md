# ComfyUI pack on work/ports (2026-11-16)

Worktree `wt-comfy`, branch `work/comfyui-merge` (from `origin/work/comfyui`), command to test:
`review/cpu_tests.sh --light wt-comfy <unit> tests/test_comfyui.py` (capped unit, CPU), RTX 5060 Ti box, no GPU needed.

## What was done
- `origin/work/comfyui` (local `work/comfyui` is an ancestor of it) carried 48 commits that are cherry-picked copies of
  ports commits (CodeFormer, DAT, HAT, MVSplat, ...): a plain merge of `origin/work/ports` conflicted in 60 files
  (cards, docs tables, uv.lock). Merge commit instead = ports tree + the pack's own files (`integrations/comfyui`,
  `docs/comfyui.md`, `docs/plans/comfyui-windows-check.md`, `tests/test_comfyui.py`, the `ty` include / `comfy`
  stubs in `pyproject.toml`, the mkdocs nav line). The diff against `origin/work/ports` is only those 10 files.
  History keeps the branch as a parent, so the push to `origin/work/comfyui` is a fast-forward.
- Depth node: video models (family `video_depth_anything`, runner `video = True`) are no longer listed; they get the
  new node "Video depth (fastforks)": the frames run as one clip through `model.video(frames)`, scaled by one min / max
  for the whole clip (per-frame scaling would flicker). `convert.depth_image` also takes a `[T, H, W]` stack now (the
  `expand` assumed 2-D). The ids come from the card family (`_VIDEO_DEPTH`), not the name.
- Library API names the nodes use (checked by `tests/test_comfyui.py::test_every_node_matches_the_library` and by a
  stub-ComfyUI script): `fastforks.load / list_models(task, commercial) / model_card(...).{outputs, family, options,
  output_info, variants, variant_ref}`, `restoration.load(path, ...)`, `restoration._api._checkpoint_family` (private,
  no public replacement yet; `noqa` added), `RestorationModel`, `model.video`, result fields. All exist on ports.
  All 20 nodes: non-empty model lists, existing cards, `run` arguments equal the declared inputs.

## Results
| Check | Result |
|---|---|
| `tests/test_comfyui.py` (13 old + 2 new) | 15 passed |
| `tests/test_docs.py tests/test_policy.py` | 1859 passed, 62 slow skipped |
| ruff check / format on `integrations`, `tests/test_comfyui.py` | clean |
| Stub-ComfyUI end to end, CPU, `FASTFORKS_OFFLINE=1`, weights from `FASTFORKS_WEIGHTS_PATH` / `~/.cache/fastforks` (read in place) | Depth (`depth-anything-v2-s`, 2 x 128 x 160) and Video depth (`video-depth-anything-s`, 4 frames) run, output in [0, 1] |
| Cache setup | `FASTFORKS_CACHE` = `<models>/fastforks`, `FASTFORKS_READ_CACHES` = the normal fastforks cache (as docs/comfyui.md) |

## Windows check still needs (docs/plans/comfyui-windows-check.md, now 7 steps)
Install into ComfyUI's embedded Python without replacing torch; 20 nodes appear; Depth `auto` backend warning;
RemoveBackground / Segment / Interpolate; unload frees VRAM; long paths; fp16 / bf16 on RTX 20xx; new: Video depth
and the `.pth` upscale files. Not tested in a real ComfyUI since the merge (only the stub; the Linux real-ComfyUI run is
results/2026-11-13_comfyui_nodes.md, before the new nodes).
Not released or published.
