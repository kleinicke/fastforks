# Disk space, 2026-11-15

Both disks were full (`/` 28 GB free, `/mnt/hdd` 28 GB free), so `move-to-hdd` could free nothing meaningful; nothing was moved to the HDD.

Freed on `/`: 28 GB -> 107 GB (after the weights downloads started), by **hard-linking identical weight files**, nothing deleted:
- `~/.cache/huggingface/hub/blobs` vs `data/ff-weights`, `data/gated_models`, `~/.cache/fastforks/blobs`: 10 large files (sam, sam2, sam3, vggt-omega, dinov3 ...) byte-compared (`cmp`) and replaced by hard links: 14.9 GB.
- Same, all files >= 100 MB across `data/ff-weights`, `data/gated_models`, `~/.cache/fastforks/blobs`, `~/.cache/huggingface/hub/blobs`, `data/{multiview/weights,mono,matching,classification,segmentation,matting,restoration}` (same size, `filecmp` deep compare, then `os.link` + atomic rename): 70.9 GB. Script: `results/raw/2026-11-15_gaps/solve/dedup_links.py`.
- Every path still exists with identical content; nothing running was touched (no file was rewritten in place).

Not done: `rm -rf` of regenerable caches (`~/.cache/ff_export_j/engines` 6.6 GB, `ff_newuser/venv` 5.8 GB, `ff_launch/tmp` 1.6 GB) was denied by the permission classifier; they stay (candidates for the user).
Candidates for the user if more space is needed (HDD is also full, 28 GB free): `data/multiview/dtu_2dgs/official` (13 GB zips, extracted copy exists), `data/multiview/synth_v4` 79 GB / `synth_v5` 49 GB (synthetic sets), `~/.cache/uv` 12 GB.
