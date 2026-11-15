import json, sys
from fastforks._core import registry
from fastforks._core.files import present
rows = {}
for task in registry.TASKS:
    for card, v in registry.cards(task).values():
        for s in v.weights:
            if s.url is None: continue
            r = rows.setdefault(s.sha256, dict(sha=s.sha256, name=s.name, size=s.size, url=s.url, mirror=s.mirror, member=s.member, asize=s.archive_size, refs=[], cached=False, nc=False))
            r["refs"].append(v.ref)
            r["cached"] = r["cached"] or present(s, card.family, "FASTFORKS_WEIGHTS_PATH")
            r["nc"] = r["nc"] or registry.nc_weights_blocked(card, v)
json.dump(list(rows.values()), open(sys.argv[1], "w"))
print(len(rows))
