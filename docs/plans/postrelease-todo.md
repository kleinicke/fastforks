# Post-release work

## Website model examples

- [ ] Inventory models missing example images on the website (roughly 170 reported; confirm current coverage).
- [ ] Generate and check representative examples for missing models, using suitable inputs and recording attribution.
- [ ] Ensure the example metadata and image assets are available to the website; verify the resulting model pages.

The existing generator is `tools/examples.py`, with metadata in `docs/data/examples.jsonl`. Earlier generated assets
are in `/home/florian/projects/data/site_examples`; they are not committed to this repository. Classification
examples can contain top-five text without a rendered output image, so count website coverage separately from image
files. Do not treat generated assets as published until they have actually been made available to the website.
