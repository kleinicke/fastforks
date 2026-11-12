# Design: a Gaussian splatting / radiance field library

Status (2026-11-09): design only, nothing built. The working name in this document is `fastsplat`; the final name
is open (check PyPI, GitHub and trademarks before choosing). The roadmap of which methods it gets, in which order,
is in [roadmap.md](roadmap.md#the-splat-library); fastforks' side of the work is in
[fastforks-next.md](fastforks-next.md).

**Order of work.** There are no dates. fastforks is developed first, until we judge it good enough for now; then
development moves here. The base is not an idea but an existing research project of ours, whose rasteriser is about
3x faster than gsplat's for standard splatting (see [Starting point](#starting-point)).

## Why a separate library

fastforks runs a published network with fixed weights: an image goes in, a tensor comes out, the call is stateless.
Gaussian splatting and NeRF optimise a new representation for every scene: a training loop with datasets,
densification, schedules, evaluation and export. That is a different product with a different API, so it gets its own
package. fastforks becomes an optional dependency of it (camera poses, depth and normal priors, masks, features), not
the other way round.

NeRF and 3DGS go into one library because everything around the renderer is shared: camera models, dataset loaders,
metrics and evaluation protocols, the viewer, export. The library is **splatting-first**: rasterised primitives are
where practice and new papers are. Ray-marched fields (Instant-NGP, Zip-NeRF) come later, mainly as quality baselines,
and the newer methods that mix both (ray-traced Gaussians, 3DGUT, EVER, radiant foam) fit the same renderer
interface.

## Goals

1. **Fast**: the fastest permissively licensed trainer and renderer on NVIDIA GPUs, measured end to end (training
   wall-clock time to the paper's quality on Mip-NeRF 360 / Tanks and Temples / Deep Blending), not per kernel. The
   starting point is the rasteriser of our research project, about 3x faster than gsplat's for standard splatting
   (what that number covers is recorded at the start of M0, see [Starting point](#starting-point)).
2. **Faithful**: every method reproduces its paper's numbers on the paper's scenes, recorded in a reproduction card
   and checked in CI, the way fastforks checks goldens against the authors' code.
3. **Permissive**: Apache-2.0, no code from the Inria codebase or from its derivatives (see
   [Licence policy](#licence-policy)), so commercial products can use it.
4. **Easy, then hackable**: one call or one command trains a scene with a named method; below that, every layer is
   reachable and replaceable without losing speed where it is not changed (see [Layers](#layers)).
5. **Compatible**: gsplat's `rasterization()` signature, its strategies and the Inria PLY layout work, so switching
   from gsplat is a one-line import change, and nerfstudio can use us as a backend.
6. **Installs without a compiler**: no JIT, no torch-version wheel matrix (see [Kernels](#l0-kernels-and-distribution)).
7. **Grows into a product**: the same engine powers a desktop app (capture → poses → train → clean → export → view /
   share) that can compete with commercial apps.

Non-goals (for now): generative 3D (text- or image-to-3D with diffusion), a general SLAM system (GS-SLAM methods may
come later as methods), mesh-first reconstruction (NeuS-style SDF training). Meshes come from splats.

## The competition and what to take from each

| project | licence | strength to match | our angle |
|---|---|---|---|
| Inria 3DGS | non-commercial | the reference everyone builds on | permissive licence, faster, many methods in one place |
| gsplat | Apache-2.0 | hackable building blocks, readable trainer, nerfstudio integration, several methods | ~3x faster kernels, no compile step, faithful per-paper reproduction, regular releases (gsplat's last is v1.5.3 from 2025-07; newer work sits unreleased on main), API-compatible |
| LichtFeld Studio | GPL-3.0 | training speed, GUI with live view, frequent releases | a permissive Python library first; app later on the same engine |
| Brush | Apache-2.0 | Rust + wgpu: any GPU, browser, Android | CUDA speed first; Metal later; wgpu for the viewer |
| 3DGRUT (NVIDIA) | Apache-2.0 | ray-traced Gaussians, distorted / rolling-shutter cameras (3DGUT) | 3DGUT and 3DGRT as methods behind the same API |
| OpenSplat | AGPL-3.0 | CUDA, ROCm, Metal and CPU from one C++ code base | permissive; Python-first |
| Scaniverse, Postshot, Polycam, Luma, KIRI | commercial | one-flow capture → splat, polish, sharing; Scaniverse trains on-device | the open engine + the fastforks pose and prior models; the app is a later layer |

Details (licences, activity, which paper repos build on what) are collected in
[roadmap.md](roadmap.md#ecosystem-reference).

## Starting point

**What exists.** A research project of ours with a rasteriser about 3x faster than gsplat's for standard splatting,
plus the sparse-view geometry pipeline built on it. Two things are written down at the start of M0, because the whole speed
goal and the licence rest on them:

- **What the 3x measures**: forward only or forward and backward, resolution, number of Gaussians, GPU, gsplat
  version, and whether it is the kernel alone or a training step. A training step is also densification, the
  optimiser, SSIM and data loading, so a 3x kernel does not give 3x training.
- **Where every file came from**: written by us, taken from gsplat (fine, Apache-2.0), or derived from Inria's
  `diff-gaussian-rasterization` or a paper repo built on it (must be rewritten before it enters the library).

**Two routes.** Open decision.

| | A: start inside gsplat | B: a new base, heavily inspired by gsplat |
|---|---|---|
| how | fork or branch gsplat; put our kernels behind its `rasterization()`; keep its trainer, strategies and tests | new repository with the layers of this document; gsplat's API shape and its good Python code are adapted file by file |
| first trained scene | at once: the trainer, the strategies and every method gsplat ships already run | only after the reference tier, scene and trainer exist (M0) |
| correctness | gsplat's tests and its users' experience come with it | has to be earned again; the compat layer runs gsplat's tests |
| our architecture | constrained by gsplat's interfaces: its JIT build and wheel matrix, a flat package, kernel signatures that do not have tiers, output presets or a deterministic mode | free: prebuilt kernels, tiers, attribute schema, cards, protocols |
| upstream | merges work until the kernels diverge, which is at once; easy to offer PRs | no merges; improvements to gsplat go upstream as separate PRs |
| provenance | gsplat's tree as it is | a fresh tree, checked per file |
| risk | we end up maintaining a diverged fork with gsplat's structure | we rewrite things that already work, and it takes longer until research can use it |

**What to take from whom**, on either route:

- **gsplat**: the `rasterization()` signature, the strategies, camera models, parsers, the readable example trainer,
  its test cases.
- **Taming 3DGS**: the faster backward pass and the sparse optimiser update (ideas; the code is Inria-derived).
- **fused-ssim**: the loss kernel (MIT).
- **LichtFeld Studio**: what makes its training fast, and the live view (GPL-3.0: look, do not copy).
- **Brush**: the GPU sort without CUB, and how one code base serves several GPU vendors (Apache-2.0).
- **3DGRUT**: the renderer interface that fits rasterised and ray-traced primitives, and the distorted cameras.
- **nerfstudio**: the method registry and typed configs, and the lesson of what became too heavy to change.
- **fastforks**: prebuilt kernels downloaded by hash, cards, goldens, the no-silent-fallback rule.

**Recommendation: B, with a short step on route A first.** The first task of M0 is a spike that is thrown away
afterwards: plug the research rasteriser into gsplat's `simple_trainer` behind `rasterization()`. It costs little
and gives three things: the end-to-end number against gsplat on the reference GPU, a profile of a full training
step (which says what else must become a kernel), and the first test of the gsplat-compatible layer. The route is
decided after the spike:

- Stay on A if our kernels fit gsplat's interfaces without losing speed, and tiers, output presets and prebuilt
  distribution can be added without rewriting its core. Then the better move may be to contribute upstream.
- Go to B otherwise. The rest of this document describes B; the layers, cards and protocols also apply to A as far
  as gsplat's structure allows.

## Licence policy

- The library is **Apache-2.0**.
- **gsplat** is Apache-2.0: its code and kernels may be used as a starting point. Files taken from it keep their
  copyright header, our `NOTICE` names gsplat, and changed files say so (Apache-2.0 §4).
- **Inria code is never copied**, and neither are the modified Inria files inside paper repos, whatever licence the
  paper repo puts on them (the Inria licence's §4.2 keeps the research-only limitation on modified versions): a
  changed Inria function or a patched `diff-gaussian-rasterization` kernel remains a derivative of Inria's
  non-commercial code. Code a paper's authors wrote themselves, clearly separable (a new file, a new loss, a new
  strategy), under a permissive licence, may be used with attribution.
- **A permissive headline licence is not enough.** Many MIT / Apache paper repos still vendor `simple-knn` or a fork
  of `diff-gaussian-rasterization` (4DGS, Deformable 3DGS, Triangle Splatting, pixelSplat / MVSplat), and some append
  the Inria terms to an MIT file (WildGaussians, SpacetimeGaussians). Every file we adapt gets a per-file provenance
  check. The permissive sources checked so far: gsplat, 3dgrut, Brush (Apache-2.0), fused-ssim, fused-bilagrid, SPZ
  and splat-transform (MIT).
- **How a paper is ported** (detailed in [Porting a paper](#porting-a-paper)): diff the paper repo against the Inria
  commit it forked to learn *what* changed; implement that change in our architecture from the paper and the diff's
  behaviour; check against the paper's numbers and, where possible, against outputs of the authors' code (outputs,
  not source, are the reference). Because our structure differs from Inria's, the result is a rewrite by
  construction. Reading and writing are separated: one session reads the diff and writes a description in prose
  (behaviour, constants, schedules), and another implements from that description and the paper, without the source
  in its context. This matters most for agent-driven porting, since an agent that has just read a function tends to
  write it again. A similarity check against the Inria repositories and the paper repo runs in CI.
- **Every method records its provenance** in its card: paper, reference repo, that repo's licence, which files (if
  any) were taken and under which licence, and a patent note.
- **Patents**: methods can be patented independently of any code. Before a method is marked commercially usable, its
  card records a patent search (Google Patents, the authors' institution). Not legal advice; a lawyer checks before a
  commercial release.
- **Datasets** used for benchmarks keep their own licences; the library downloads them on request, never ships them.

## Layers

From the bottom up. Each layer only depends on the ones below it; users can enter at any layer.

```
L7  app (Rust): viewer, desktop app with live training, web viewer
L6  CLI / train(scene, method=...)            ← easy entry point
L5  methods: plugins (representation extension, strategy, losses, renderer, appearance, schedule)
L4  trainer: ~300-line readable loop + hooks
L3  scene: typed per-primitive attribute set, cameras, optimiser groups
L2  data: loaders, poses (COLMAP or fastforks), priors, masks, undistortion
L1  ops: differentiable operations, three tiers (fused / composable / reference)
L0  kernels: CUDA (later Metal), C ABI, prebuilt per GPU architecture
```

### L0: kernels and distribution

The fastforks approach, carried over:

- CUDA code built ahead of time into **cubins** (driver-API modules) per GPU architecture, plus PTX for newer GPUs,
  loaded through the CUDA driver API. A cubin does not depend on Python, torch or the OS, so one build serves every
  torch version and both Linux and Windows.
- Code that needs host-side library calls (CUB radix sort for the tile / depth sort, CUTLASS GEMMs, multi-kernel
  launches) goes into **shared libraries with a plain C ABI** taking raw device pointers and a stream: still
  independent of the torch version. Built per OS (`.so`, `.dll`).
- Binaries are not in the wheel. They are downloaded by hash from Hugging Face into the user cache on first use, with
  an offline path (`FASTSPLAT_KERNEL_PATH`) for Docker and air-gapped machines. The wheel is pure Python.
- Allocation stays in Python (torch tensors). Kernels that need a data-dependent output size (number of tile
  intersections) use the usual two-pass pattern: count, allocate in Python, fill. This keeps memory under torch's
  caching allocator and visible to users.
- Kernels are launched on torch's current stream; CUDA graphs are possible for fixed-size inner loops.
- **Source build, opt-in.** The kernel sources ship in the repository, and `fastsplat build-kernels` compiles the
  same cubins / shared libraries locally with the same build tool, into the same cache, recorded as a local build
  (not a validated one). It needs a CUDA toolkit and a host compiler, but not one that matches the torch build,
  because the binaries never link torch. Uses: GPUs or CUDA versions we do not publish for, air-gapped machines that
  will not download binaries, users who audit what they run, and **kernel hacking**: with
  `FASTSPLAT_KERNEL_SRC=<checkout>` the loader rebuilds changed kernels (ninja, incremental) before the first launch
  and logs that it did. This is gsplat's JIT advantage (edit a `.cu`, rerun) without making it the default. It lives in
  the shared kernel runtime, so fastforks gets the same option.
- Never compile during `pip install` (build isolation, no GPU on build machines, slow failing installs: gsplat's most
  common issue type) and never silently on first inference: a minutes-long first call with a compiler error is the
  failure we are avoiding. Building happens only when asked for by command or environment variable, and is logged.
- Shared with fastforks: the loader (manifest, target selection `sm_XY` → older same-major → PTX, hash check,
  driver-version check), the build tool and the Hugging Face publishing check. Extract them into a small shared
  package (`fastforks-kernels-runtime` or similar) rather than copying: both libraries then get Windows support and
  loader fixes at once.

### L1: ops

Public, differentiable operations with stable signatures, each a `torch.autograd.Function`:

- projection (pinhole, fisheye, and the unscented transform for 3DGUT cameras), covariance from scale / rotation,
  spherical harmonics evaluation, tile intersection and sorting, alpha compositing forward / backward, depth and
  normal rendering, per-pixel extra channels (features, semantics, any N-D attribute), 2D Gaussian (surfel)
  rasterisation, losses that are bottlenecks (fused SSIM, L1), and the optimiser step (fused Adam, updates only for
  visible Gaussians, optional bf16 state).

Every operation exists in up to three **tiers** with identical outputs (within a recorded tolerance):

| tier | what | when |
|---|---|---|
| fused | several steps in one kernel (e.g. SH + projection + rasterisation), fastest | when the configuration matches exactly what was fused |
| composable | one kernel per step, steps glued in Python | when a user replaces or adds a step; still fast |
| reference | pure PyTorch, slow, readable | verification, prototyping a new paper, CPU / MPS debugging |

**Extra outputs on the fast tier.** Research changes losses and pruning far more often than the rasteriser, so a
loss change must not force the slow tier. The fused and composable rasterisers return, as opt-in flags that cost
nothing when off:

- per pixel: expected depth, median depth, unbiased / planar depth (PGSR-style: distance to the local plane through
  the normal), rendered normal, accumulated alpha, final transmittance;
- per Gaussian and view: visible (yes / no), maximum blending weight, total contribution (sum of weights), and the
  2D mean gradient densification already uses.

These are designed into the M1 kernels, because adding outputs to a fused kernel later means rewriting it.

Kernels are prebuilt, so a flag is a compiled variant, not a branch in the inner loop. A short, fixed list of
**output presets** is built (for example `rgb`, `rgb+depth`, `rgb+depth+normal`, and `geometry` with every per-pixel
output; each with or without the per-Gaussian statistics). `mode="auto"` picks the smallest preset that covers the
request and logs it; a combination outside the list runs on the composable tier. The list is fixed in M1 and kept
short, because it multiplies with tiers, forward / backward and later Metal.

**Channel counts.** The rasteriser blends any per-Gaussian vector `[N, C]`, not only RGB: depth alone (1), RGB (3),
RGB plus depth (4), or feature vectors of tens to hundreds of channels (CLIP / DINO / SAM feature fields, learned
features decoded by an MLP). gsplat compiles one kernel copy per count from a fixed list (1, 2, 3, 4, 5, 8, 9, 16,
17, … 512, 513) and zero-pads everything else. Ours:

- **Specialised kernels for 1, 3 and 4 channels**, prebuilt like the output presets. 1 channel is for depth, masks
  and other single values, and skips the spherical-harmonics step, which is where most of its saving comes from.
- **One generic kernel for any other count**, with the count as a runtime argument: no padding, no rebuild.
- **Wide features in a second pass**: the first pass keeps which Gaussians contribute to each pixel and with what
  weight, and a second kernel blends an arbitrary feature tensor with those weights, so the RGB path stays fast.
- The user passes a tensor; `mode="auto"` picks the specialised kernel when one exists, the generic one otherwise,
  and logs which.

To measure in M1: the generic kernel against the specialised ones at 1, 3 and 4 channels. If the gap is small, the
specialised list stays this short; if it is large, 16 and 32 are the next candidates.

**Deterministic mode.** The backward pass accumulates gradients with float atomics, so two runs differ slightly
(measured in our research: up to 0.056 per scan on sparse views, about 0.1 dB PSNR from image order alone).
`deterministic=True` switches the backward pass to a fixed-order reduction (per-tile partial sums, then an ordered
sum), and pins the data order and seeds. Slower, meant for A/B comparisons, not for production. Bit-identical means on the same GPU model, driver and kernel
build, not across machines. The ordered sum needs one partial sum per Gaussian-tile intersection, so it also uses
more memory; the docs state how much. Designed in M1 for
the same reason as the extra outputs; the reference tier gets it through `torch.use_deterministic_algorithms`.

Selection follows the fastforks rule *no silent fallback*: `mode="auto"` chooses the fastest tier valid for the
configuration and **logs the choice** (and why a faster tier was not valid); a forced `mode="fused"` raises with the
reason when it cannot run. `report()` lists which tier each op ran in.

Typed signatures: `jaxtyping` annotations (`Float[Tensor, "N 3"]`) on every public op, checked with `beartype` in
tests and debug mode (off in production for speed). `ty` and `ruff` as in fastforks.

### L2: data

- Loaders: COLMAP (binary and text), nerfstudio `transforms.json`, Blender synthetic, LLFF, image folders and video
  files (frame sampling with blur / sharpness selection).
- Camera models: pinhole, OpenCV radial-tangential, fisheye (equidistant / Kannala-Brandt), rolling shutter;
  undistortion when a method needs pinhole.
- **Poses without COLMAP via fastforks**: VGGT, MapAnything, or fastforks retrieval pairs + permissive matchers
  (ALIKED / DISK + LightGlue, RoMa) fed into the `pycolmap` global mapper, with bundle adjustment by pycolmap
  (Caspar GPU BA where built). No own SfM, RANSAC or BA solver: PoseLib / pycolmap / OpenCV are optional
  dependencies (see [roadmap.md](roadmap.md#geometry-sfm-ransac-bundle-adjustment-calibration)). MASt3R / DUSt3R are non-commercial and not a default. COLMAP stays
  supported and is the default for benchmark reproductions (papers used it).
- **Priors via fastforks**: monocular depth (Depth Anything, MoGe), normals, masks (SAM 2 / SAM 3 for objects,
  sky, people), features (DINOv3 / CLIP / SigLIP for feature fields).
- **Stereo and multi-view depth priors**, the stronger source for geometry: in our sparse-view research, stereo depth
  with confidence plus TSDF fusion gave the best results, and monocular depth helped only as local hole fill.
  fastforks already ships `stereo` (FoundationStereo; needs rectified pairs) and `multiview` (MVSAnywhere,
  PatchMatchNet; any posed images). L2 provides depth with a confidence map per view, a multi-view consistency check
  (reproject into neighbours, keep depths that agree), and fusion into a point cloud or TSDF for initialisation and
  free-space pruning.
- Point cloud initialisation from SfM points, from fastforks multi-view depth, from fused stereo / MVS depth, from any
  mesh or point cloud file, or random.
- Caching of decoded, resized images and priors on disk; an async loader that keeps the GPU busy.

### L3: scene

A scene is a set of primitives with **named, typed attributes** registered in a schema:

```python
@attribute("opacity", shape=(1,), activation="sigmoid", init=constant(0.1), lr=0.05,
           on_split="copy", on_clone="copy", on_prune="drop")
```

- The base 3DGS schema: means, scales (log), rotations (quaternions), opacities (logit), SH coefficients.
- A method **extends** the schema (Scaffold-GS anchors, per-Gaussian features, beta kernels, time attributes for
  4D, appearance codes) instead of forking the scene class.
- Densification operations (split, clone, prune, relocate) are generic over the schema: each attribute declares how
  it behaves, so a new attribute works with existing strategies.
- Optimiser groups, learning-rate schedules and the optimiser state follow the schema (state is split / pruned
  with the primitives).
- Cameras are a batch structure (intrinsics, extrinsics, model, distortion, time, image size), optionally optimised
  (pose refinement).

### L4: trainer

- A short, readable training loop (target: ~300 lines) that users can copy and edit, like gsplat's
  `simple_trainer.py`, and that the library itself uses.
- Hooks at fixed points (before / after render, loss, backward, step, densify, eval, checkpoint), so methods and
  users add behaviour without editing the loop.
- Hooks on the primitives, not only on the loop: per-iteration penalties on attributes (e.g. an opacity penalty),
  custom pruning on a schedule (e.g. prune Gaussians in space that external depth says is empty, as our free-space
  pruning does), and initialisation from any source (mesh, point cloud, fused depth). Each new hook is checked
  against at least two methods (e.g. our sparse-view pipeline and MCMC), so one pipeline does not shape the API.
- Reproducible runs: `seed=` and a fixed image order by default; `deterministic=True` (see L1) for A/B comparisons.
  Without it, every comparison needs repeat runs; the card records which mode produced which number.
- Typed configs (dataclasses), one per method, composable; CLI generated from them (tyro or similar).
- Checkpoints: the scene schema + tensors + optimiser state + config + the library and kernel versions.
- Multi-GPU later (data-parallel over cameras, or Grendel-style primitive sharding for huge scenes).

### L5: methods

A method is a plugin bundling: schema extensions, a densification strategy, losses, a renderer choice, an appearance
model (e.g. bilateral grids, per-image embeddings), schedules and its reproduction card. Methods compose where the
paper allows it (MCMC densification + anti-aliasing + appearance model).

Every method has a **reproduction card** (TOML, like fastforks model cards): paper, reference repo and commit,
licence and provenance, the paper's reported metrics per scene, our measured metrics and training time per device,
the tolerance, and the status (`reproduced` / `close` / `not yet`). The tolerance comes from the spread between
repeat runs (ours, and the authors' code where its licence lets us run it), not from the paper's table alone: paper
numbers are often not reproducible to 0.1 dB even with the authors' code. Each measured number also names the evaluation
protocol, the precision (parameters, optimiser state such as bf16 Adam, kernels), whether the run was deterministic,
and the number of repeats, because these change results slightly and numbers are only comparable within one setting.

### Evaluation protocols

Protocols are named code, not documentation, because the same benchmark is measured differently by different papers:
DTU chamfer distance in the 2DGS protocol and in the PGSR protocol give different numbers for the same mesh (in our
research 0.846 vs 0.882 are only comparable within one protocol). Each protocol fixes the scene list, image split,
downscaling, masks, culling and metric code:

| protocol | measures | from |
|---|---|---|
| `mipnerf360` | PSNR / SSIM / LPIPS | M0 |
| `tnt-nvs`, `deepblending` | PSNR / SSIM / LPIPS | M1 |
| `dtu-2dgs`, `dtu-pgsr` | chamfer distance of the mesh | M2 (with meshing) |
| `tnt-geometry` | F-score of the mesh | M2 |
| sparse-view DTU (the split and view count our pipeline is measured on; named after the paper it comes from) | chamfer distance of the mesh | M2 |

`fastsplat eval --protocol <name>` and the cards refer to these names; a new protocol is added as code with a test
that reproduces a published number.

### Post-processing: `fastsplat.clean` and `fastsplat.mesh`

Library layers, not app features, because cleanup and meshing are where much of the quality comes from (in our
research, floater culling plus stereo hole fill improved the dense-view chamfer distance from 0.502 to 0.480). The app calls them.

- `fastsplat.clean`: floater culling (by opacity, size, visibility count, distance to fused depth), hole filling from
  stereo / MVS depth, crop boxes, mask-based deletion (SAM via fastforks).
- `fastsplat.mesh`: TSDF fusion of rendered depth with all parameters settable, including the minimum weight (Open3D's
  default of 3 is what produced empty meshes on sparse views), marching cubes, mesh cleanup. Mesh extraction
  specific to a method (GOF tetrahedra, RaDe-GS) lives in that method.
- Combining the meshes of two methods (per-face fusion, our best dense and sparse results) starts as a recipe in
  `methods/`, and becomes a library function only once it works beyond our pipeline.

### L6: easy entry

```python
import fastsplat

scene = fastsplat.load_scene("garden/", poses="colmap")  # or poses="vggt" via fastforks
result = fastsplat.train(scene, method="mcmc", steps=30_000)
result.metrics  # PSNR / SSIM / LPIPS on the held-out views, with the protocol named
result.save("garden.ply")  # Inria-compatible PLY; also .spz, .sog
```

```
fastsplat train garden/ --method mcmc
fastsplat eval runs/garden --protocol mipnerf360
fastsplat export runs/garden --format spz
fastsplat view runs/garden
fastsplat bench --method 3dgs --scenes mipnerf360   # end-to-end time and quality on this machine
```

### L7: app (Rust)

Rust where it pays off: the desktop app, file formats and data import.

- **Viewer: reuse the 3D Visualizer** (`ply-visualizer`, MIT, our own), not a new one. It already renders splats
  (3DGS PLY, SPZ, SPLAT, KSPLAT, SOG via Spark on WebGL), point clouds, meshes, depth and cameras, and runs as a
  VS Code extension, a web page, a Tauri desktop workspace and a Python / MCP package (`viz3d`) with live
  `viewer.update(...)`. What fastsplat needs on top: live splat updates during training (send changed Gaussians, or
  render server-side with our rasteriser and stream frames once scenes exceed what the browser holds), training
  cameras and their errors, and a pause / step / densification view. The Scientific Image Visualizer
  (`tiff-visualizer`, MIT) covers the 2D side: rendered vs ground-truth images, depth, normals, error maps, EXR.
- **Formats crate**: PLY / SPZ / SOG reading and writing, compression, exposed to Python through PyO3 (fast I/O in
  the library too).
- **Desktop app**: grows out of the visualizer's Tauri workspace (which already combines both engines). Drop in images or a video → poses (fastforks) → training with **live preview** → cleanup tools
  (crop box, floater removal, segmentation-based deletion via SAM) → export / share. Training runs in the Python
  engine as a separate process; the app streams the scene state over shared memory. A Rust trainer (for mobile,
  on-device) is a later decision, not a starting point.
- Windows first-class (that is where most desktop users are), then Linux, then macOS once Metal kernels exist.

## Compatibility with gsplat and nerfstudio

- `fastsplat.compat.gsplat.rasterization(...)` takes gsplat's arguments and returns gsplat's outputs, so
  `from fastsplat.compat import gsplat` replaces `import gsplat` in existing code. Tested by running gsplat's own
  test cases against it.
- `DefaultStrategy` and `MCMCStrategy` equivalents with the same parameters.
- PLY files are read and written in the Inria layout (what every viewer reads).
- A nerfstudio plugin (`ns-train fastsplat-...`) so nerfstudio users get our speed without changing tools.
- **Preferred: a new repository, not a fork of gsplat** (the decision is open, see
  [Starting point](#starting-point)). Three reasons: the CUDA backend and the build / distribution model
  are replaced completely (a fork would carry gsplat's JIT and wheel matrix and diverge at once, so upstream merges
  stop working anyway); the layering (L0–L7, tiers, method cards) differs from gsplat's flat package; and per-file
  provenance is easier to keep clean in a fresh tree. But **adapt rather than rewrite** where gsplat's Apache-2.0
  Python code is good: strategies, camera models, COLMAP / Blender parsers, the example trainer's structure. Copied
  files keep gsplat's copyright header, are listed in NOTICE, and pass the same provenance check as any permissive
  source (see [Licence policy](#licence-policy)). Rewriting what works costs time and makes compatibility harder.
- Friendly stance: credit gsplat in the README and NOTICE; where a fix or speed-up applies to their code, offer it
  upstream as a PR. Never use their name for ours.

## Porting a paper

1. **Card first**: paper, reference repo + commit, its licence, the base it forked (Inria commit, gsplat version),
   the paper's reported numbers per scene.
2. **Diff**: compare the paper repo with the base commit it forked; list what changed (rasteriser, densification,
   losses, representation, schedules). This is for understanding, not copying: the result of this step is a prose
   description, and step 3 is done from it without the source (see the licence policy).
3. **Reference tier**: implement the change as a plugin on the reference / composable tiers. Run the paper's scenes;
   compare against the paper's numbers and, if available, renders of the authors' code.
4. **Kernels**: once the method reproduces, move hot spots into composable or fused kernels; the parity tests check
   the new tier against the reference tier.
5. **Done** when the card says `reproduced` (within the tolerance on every reported scene) and the speed rows are
   recorded.

The steps are repetitive and checkable against numbers, so they suit agent-driven porting, as fastforks' golden
pipeline does for models.

## Testing

- **Parity**: every kernel tier against the reference tier (forward outputs and gradients), on random scenes with
  edge cases (degenerate covariances, Gaussians behind the camera, huge / tiny scales, empty tiles), like fastforks'
  kernel fuzz tests.
- **Gradient checks**: `torch.autograd.gradcheck` on tiny scenes in float64 on the reference tier.
- **Golden renders**: fixed scenes and cameras rendered by each tier, compared to stored outputs (hash-referenced,
  not committed).
- **Reproduction CI**: nightly / per-release on GPU: a short schedule on a few scenes against recorded curves; full
  reproductions before each release. The budget is one 16 GB card: nightly runs stay at a short schedule on two or
  three scenes; scenes that do not fit 16 GB at the paper's resolution are run on a rented GPU and marked so in the
  card.
- **Speed floor**: end-to-end training time per method on the reference GPU (RTX 5060 Ti), with regressions failing
  the release check, as in fastforks.

## Platforms

1. Linux x86-64, NVIDIA sm_80+ (same as fastforks' kernels).
2. Windows native, NVIDIA (same cubins; shared libraries built with MSVC). Needed for the desktop app.
3. Apple silicon via Metal kernels. fastforks already ships Metal kernels for 19 families, so the loader and build
   path exist. A 3DGS training step needs: projection, a GPU radix sort (no CUB on Metal: write one; Brush's wgpu
   sort is an Apache-2.0 reference), tile rasterisation forward and backward (float atomics: available on Apple
   GPUs), SH evaluation and fused Adam. Roughly the M1 kernel set again, so medium effort. Expect a few times
   slower training than a desktop RTX card, but unified memory fits large scenes. OpenSplat (AGPL-3.0) already
   trains on Metal: look, do not copy. The reference tier runs on MPS from day one for development.
4. WebGPU for the viewer only.
5. Not planned: NVIDIA Jetson (nobody trains scenes there; fastforks gets Jetson support for inference, see
   [fastforks-next.md](fastforks-next.md#jetson-and-arm64-linux)). ARM64 Linux servers with NVIDIA GPUs come with
   the shared kernel runtime once fastforks builds for them.

## Benchmarks and claims

- Headline numbers are **end-to-end**: wall-clock training time to the paper's PSNR on the standard scenes, and
  rendering FPS at a fixed resolution, same device, same image downscaling, same number of steps.
- Against gsplat, Inria, LichtFeld Studio, Brush and OpenSplat, and against the fast-training methods (Taming 3DGS,
  DashGaussian, Speedy-Splat), run by us on the same machine, with versions and commands published next to the
  numbers. A claim of the fastest end-to-end training must beat these too, not only gsplat. They are Inria-derived or
  non-commercial, so they are run from their own repositories for the comparison and ported as methods, never used
  as code.
- Kernel-level speed-ups (e.g. the existing rasteriser's 3x) are reported separately and never as the headline.

## Repository layout (proposed)

```
fastsplat/
  python/fastsplat/
    ops/            # L1 public ops + tier selection
    kernels/        # CUDA sources per library, build.toml, manifest.toml (as in fastforks)
    data/           # L2
    scene/          # L3 schema, attributes, cameras
    train/          # L4 loop, hooks, configs
    methods/<name>/ # L5 plugin + card.toml
    clean/  mesh/   # post-processing (floaters, hole fill, TSDF, meshes)
    eval/           # named protocols
    compat/         # gsplat API, nerfstudio plugin
    cli.py
  crates/
    formats/        # PLY / SPZ / SOG (Rust, PyO3 bindings)
    viewer/         # wgpu viewer (native + web)
    app/            # desktop app (later)
  tests/  tools/  docs/
```

## Milestones

| milestone | contents | done when |
|---|---|---|
| M0 foundations | record of what the research rasteriser's 3x measures and where each file came from; the spike in gsplat's trainer and the route decision (see [Starting point](#starting-point)); shared kernel runtime extracted from fastforks; reference-tier ops; scene schema; loaders (COLMAP, transforms.json); trainer loop with seeds and fixed image order; protocol registry with `mipnerf360` | vanilla 3DGS on the reference tier matches Inria's published numbers on 2-3 scenes |
| M1 fast 3DGS | our rasteriser as fused + composable tiers, with the extra outputs (depth variants, normals, alpha, transmittance, per-Gaussian stats) and deterministic mode designed in; fixed list of output presets; fused SSIM and fused Adam; gsplat-compatible API; PLY export; `bench`; a minimal live view of training through `viz3d`; `tnt-nvs` and `deepblending` protocols | vanilla 3DGS reproduced on all Mip-NeRF 360 scenes, end-to-end faster than gsplat on the reference GPU; deterministic mode gives bit-identical runs |
| M2 research core: geometry | MCMC, AbsGS / revised densification, Mip-Splatting, Taming 3DGS; 2DGS; PGSR-style planar depth with multi-view NCC and geometry losses (after the patent check); stereo / MVS priors with confidence from fastforks; hooks on primitives; `fastsplat.clean` and `fastsplat.mesh` (TSDF); `dtu-2dgs`, `dtu-pgsr`, `tnt-geometry` protocols. Then port our sparse-view pipeline (stereo prior, free-space pruning, cull / fill, mesh fusion) as the first own method | 2DGS and PGSR reproduce their DTU numbers in their own protocols; our pipeline reproduces its own numbers on fastsplat; first public release, with our pipeline as the showcase and the licence of every prior it uses stated |
| M3 breadth | the rest of gsplat parity (bilateral grids / PPISP, pose refinement, appearance embeddings, 3DGUT cameras); nerfstudio plugin; GOF and RaDe-GS meshes; compression and formats (SPZ, SOG); dynamic scenes; VGGT / MapAnything poses; DashGaussian and the other speed methods; Windows native | every method gsplat ships, reproduced; 10+ methods with cards; Windows wheels; docs site |
| M4 viewer and app | Rust formats crate; live training view in the 3D Visualizer (`viz3d`); desktop app on its Tauri workspace with cleanup | capture-to-share in one flow on Windows and Linux |
| M5 more renderers | ray-traced Gaussians (3DGRT), EVER / radiant foam / triangle or beta primitives, Instant-NGP and Zip-NeRF baselines, Metal kernels (could move earlier: the Mac is the developer machine) | renderer interface proven by at least two non-rasterised methods |

**Check after M1.** If vanilla 3DGS is not clearly faster end to end than gsplat on the reference GPU (the target
is set from the spike's numbers), work stops before M2 and we decide again: fix the remaining bottlenecks, or
contribute the kernels to gsplat instead of building a library around them.

**Priors in the showcase.** Our pipeline's strongest prior is stereo depth. Per the fastforks cards, FoundationStereo
and MVSAnywhere are non-commercial; `c_foundationstereo` (NVIDIA Open Model License) and PatchMatchNet (MIT) are the
candidates for a commercially usable path. The release either reproduces the result with those or says plainly that
the showcase numbers need non-commercial weights.

**Scope.** M0–M2 are run strictly as a library for our own research: everything in them is something our
geometry work needs, and M2 is done when our own pipeline runs on it. Until M4 the only live view is the minimal `viz3d` one from M1.
Public users get it at the end of M2, with our
result as the showcase. M4 (the app) is effectively a second product and starts only once the engine has proven
itself in our research. Licence condition for our own work: PGSR and the other Inria-derived bases are written from
the papers on fastsplat's layers, never copied from the code our earlier results ran in; our own additions are ours.

## Open questions

- gsplat or a new base: decided after the M0 spike (see [Starting point](#starting-point)).
- Final name, and whether the library lives under the fastforks brand (shared website, shared kernel runtime) or
  stands alone.
- Rust trainer for on-device / mobile training (Scaniverse-style): when, and whether it shares kernels (C ABI makes
  it possible) or uses WebGPU / Metal like Brush.
- Whether to offer part of the speed-up upstream to gsplat first, and which part.
- Which patent searches are needed before the first commercial-use claim. Applications already found touch
  RaDe-GS-style depth rasterisation, planar splatting, NeRF-to-3DGS initialisation and the Instant-NGP hash grid
  (list in [roadmap.md](roadmap.md#patents-found-titles-only-not-exhaustive-no-claims-read)).
- Hosting of benchmark datasets and results (Hugging Face datasets, as fastforks does for kernels).
