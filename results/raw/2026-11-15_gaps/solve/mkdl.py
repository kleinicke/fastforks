"""dry.txt -> dl_order.txt: refs smallest first that the library may download (no non-commercial gate, not HDD-cold)."""
import re, sys
from pathlib import Path
here = Path(__file__).parent
size, nc, info = {}, set(), {}
for l in (here / "dry.txt").read_text().splitlines():
    m = re.match(r"(\S+)@\d+: licence .*?, (\d+(?:\.\d+)?) (MB|GB)$|(\S+)@\d+: licence .*size unknown", l)
    if m and m.group(1):
        size[m.group(1)] = float(m.group(2)) * (1024 if m.group(3) == "GB" else 1)
    elif m and m.group(4):
        size[m.group(4)] = 0.0
    m = re.match(r"(\S+)@\d+: needs \S+ to download", l)
    if m:
        nc.add(m.group(1))
cold = {"marigold-v2-depth"}  # /mnt/hdd/projects/INDEX.md: data/marigold-v2 (41 G study weights)
refs = [r.strip() for r in (here / "uncached_refs.txt").read_text().split()]
ok = sorted((r for r in refs if r in size and r not in nc and r not in cold), key=lambda r: size[r])
(here / "dl_order.txt").write_text("".join(f"{r}\t{size[r]:.0f}\n" for r in ok))
(here / "dl_skipped.txt").write_text("".join(
    f"{r}\t{'noncommercial gate (FASTFORKS_NONCOMMERCIAL=1)' if r in nc else 'HDD-cold list' if r in cold else 'no size in dry run'}\n"
    for r in refs if r not in ok))
print(len(ok), "downloadable", sum(size[r] for r in ok) / 1024, "GB;", len(refs) - len(ok), "skipped")
