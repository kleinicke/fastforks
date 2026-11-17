# Post-release work

## Measurement and implementation coverage

Local generated site data on 2026-11-17 (573 variants; a snapshot, not the live site):

- [ ] Portable FP32 quality is absent for 251 variants; original FP32 quality is absent for 62. These counts include
  55 variants with no quality dataset entry, so they are coverage gaps rather than a promise that every evaluation
  is currently runnable or applicable. Classification and flow campaigns are still producing additional results.
- [ ] Compile has a measured row for 560 variants and a recorded failure or planned entry for 13.
- [ ] TensorRT has a measured row for 501 variants, a recorded failure or planned entry for 71, and no row for one
  (`clipseg`). A recorded time is not evidence that task accuracy was validated.
- [ ] 73 variants belong to families without a kernels mode. All have a stated family-level reason, but some reasons
  merely say the implementation has not been built; review and develop candidates, and measure speed, memory, and
  correctness before considering them complete. The user is handling kernel development in parallel; see
  [the family and variant list](missing-kernel-modes-0.2.1.md).

Counts use the generated site's per-variant quality and speed entries; multiple evaluation datasets or resolutions
per variant can have additional gaps. Preserve the existing comparison protocols when filling them.

## Website model examples

- [ ] Inventory models missing example images on the website (roughly 170 reported; confirm current coverage).
- [ ] Generate and check representative examples for missing models, using suitable inputs and recording attribution.
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
`results/2026-11-17_fastforks_missing_live_assets.zip` in the experiment workspace now contains all 191 missing WebPs
and the new example assets, along with their SHA-256 hashes. Each WebP was decoded successfully before packaging.

On 2026-11-17 the missing `span-x2` and `span-x4` examples were generated and visually checked; their metadata is in
`docs/data/examples.jsonl`. The two rendered images and distinct 2x/4x input images are in the local asset directory
above and still need to be supplied to the website. Generation now gives those two degraded inputs different names.
