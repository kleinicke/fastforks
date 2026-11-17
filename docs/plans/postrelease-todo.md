# Post-release work

## Website model examples

- [ ] Inventory models missing example images on the website (roughly 170 reported; confirm current coverage).
- [ ] Generate and check representative examples for missing models, using suitable inputs and recording attribution.
- [ ] Ensure the example metadata and image assets are available to the website; verify the resulting model pages.

The existing generator is `tools/examples.py`, with metadata in `docs/data/examples.jsonl`. Earlier generated assets
are in `/home/florian/projects/data/site_examples`; they are not committed to this repository. Classification
examples can contain top-five text without a rendered output image, so count website coverage separately from image
files. Do not treat generated assets as published until they have actually been made available to the website.

Local inventory on 2026-11-17: the generated site data contains 573 variants. Fourteen have no example metadata;
101 classification variants have text-only examples; and 289 variants reference an image absent from the newer
local asset directory. The older batch's output directory was not found during the 2026-11-15 survey, and images
may already exist on the live site, so these figures are not a count of broken website pages. The per-variant local
inventory is `results/2026-11-17_missing_site_examples.tsv` in the experiment workspace.
