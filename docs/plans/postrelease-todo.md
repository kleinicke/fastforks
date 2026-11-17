# Post-release work

## Measurement and implementation coverage

Local generated site data on 2026-11-17 at 23:38 CET covers 578 variants.

- [ ] Portable FP32 quality is absent for 175 variants and original FP32 quality for 64. Fifty-four variants have
  no available quality score. COCO, DAVIS, DTU, MegaDepth, prompted-depth, volumetric, OCR, edge and pose evaluations
  are in progress.
- [ ] Compile has a measured row for 562 variants and a recorded failure or planned entry for 13. Three LoMa
  variants are queued.
- [ ] TensorRT has a measured row for 506 variants and a recorded failure or planned entry for 69. Three LoMa
  variants are queued. The two new LoMa engines built, and their data-dependent output allocation is under review.
- [ ] Kernel work is tracked separately in [the family and variant list](missing-kernel-modes-0.2.1.md).

Counts use the generated site's per-variant quality and speed entries. A variant can have additional gaps at other
resolutions or on other datasets. Preserve the existing comparison protocols when filling them.

## Website model examples

- [x] Inventory models missing example images on the website.
- [x] Generate and check representative examples for the 578 variants in the current inventory.
- [ ] Ensure the example metadata and image assets are available to the website; verify the resulting model pages.

The existing generator is `tools/examples.py`, with metadata in `docs/data/examples.jsonl`. Earlier generated assets
are in `site_examples`; they are not committed to this repository. Classification
examples can contain top-five text without a rendered output image, so count website coverage separately from image
files. Do not treat generated assets as published until they have actually been made available to the website.

Local inventory on 2026-11-17: the generated site data contains 573 variants. Fourteen have no example metadata;
101 classification variants have text-only examples; and 289 variants reference an image absent from the newer
local asset directory. The older batch's output directory was not found during the 2026-11-15 survey, and images
may already exist on the live site, so these figures are not a count of broken website pages. The per-variant local
inventory is `2026-11-17_missing_site_examples.tsv` in the experiment workspace.
The generator's `_done()` currently skips successful metadata rows even if the corresponding local file is missing;
restore those assets from their original location or explicitly regenerate them rather than relying on the default skip.

A public HTTP check on 2026-11-17 found 174 output WebPs and 17 input WebPs returning 404 among the generated site's
asset references. A retry confirmed that `crestereo-eth3d.webp` returns a WebP. The missing Marigold v2 depth example
was regenerated with the portable BF16 backend. The archive at
`2026-11-17_fastforks_missing_live_assets.zip` in the experiment workspace’s results directory now contains all 191 missing WebPs
and the new example assets, along with their SHA-256 hashes. Each WebP was decoded successfully before packaging.

The local metadata audit now covers all 578 variants, with rendered images for 477 and text-only examples for
101 classifiers. The asset package also includes the five newly added LoMa examples. Site upload and live-page
verification remain open.

On 2026-11-17 the missing `span-x2` and `span-x4` examples were generated and visually checked; their metadata is in
`docs/data/examples.jsonl`. The two rendered images and distinct 2x/4x input images are in the local asset directory
above and still need to be supplied to the website. Generation now gives those two degraded inputs different names.
