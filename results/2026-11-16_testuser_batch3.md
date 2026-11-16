# Test user batch 3 (2026-11-16)

Fresh venv (`uv venv --python 3.12` + `uv pip install <worktree>`), scratch dir, docs only; six personas. Fixes are on work/ports; commit hash of the last fix: bb1217d9.

## Fixed
| finding | fix |
|---|---|
| metric3d / da3metric without intrinsics return "metric" depth silently (wrong metres) | warns once per call: "its metres assume the canonical camera ... pass intrinsics=K"; runner attribute `assumed_camera` (metric3d both runners, DA3 when intrinsics are optional); tests + golden / odd-size filters |
| pose boxes passed as 0..1 fractions were accepted silently (no person found) | UserWarning when all boxes are <= 1 on an image >= 32 px |
| ReSplat (all multi-view) with K = identity or normalised K: NaN, in kernels a device-side assert (dead CUDA context) | InputError "principal point must lie inside the image", fast test |
| TinyViT portable on CUDA crashed (weights on CPU) | found by the family smoke; root fix in `_optimize` |
| no way to get a cutout PNG, keypoints on a picture, a normal map picture from the docs | three short recipes in docs/usage.md (matting alpha cannot be saved as .png by design; `DepthResult.save` writes depth only; MoGe normals need `normals=True`) |

## Batch-2 open items
- metric3d metres without intrinsics: fixed (above).
- Stereo with swapped left / right: no warning added. The disparity of a swapped pair is not reliably detectable (distant or flat scenes look the same; models clamp to >= 0). Already in docs/usage.md (Known limits): "pass the left camera's image first".
- ComfyUI Depth node lists video models: on branch work/comfyui (a236bd40) `Depth._ids` lists every depth model with a `depth` / `inverse_depth` output, including video models, which then run frame by frame (a flicker warning from the library). Fix there: exclude `variant` models with `video = true` from that list (or route them to a video node). Not applied: that branch is 333 commits behind work/ports and must be rebased onto it before release; not rebased here.

## Personas (no change needed, low severity)
- Pose: top-down models need boxes (clear error); xywh boxes rejected clearly; `PoseResult.save("x.png")` errors (hint mentions np.save of boxes, odd); the face6 model error says "person boxes" for face boxes.
- Matting: alpha .png refused (message says .tif).
- Multi-view: views must be exactly one size (clear error; a 426 vs 425 px pair from two crops is a typical trap); ReSplat needs intrinsics (clear); AnySplat, VGGT, PatchmatchNet OK incl. .ply save.
- Interpolation: `timestep` outside [0, 1] rejected clearly; result attribute is `.images`-style (guessing `.image` fails, docs show it); sizes must match.
- Normals: lotus-normal unit vectors OK; `result.save` for normals refuses with a clear message (recipe added).
- Intrinsics: geocalib OK (f ~ 409 px, vfov 0.96 rad on a 640 x 426 photo).
- OCR: ppocrv5 returns an empty list on a photo without text (fine); `OCRResult.save` missing: not added (users write `json.dumps(texts)` themselves; not clearly worth the code).
