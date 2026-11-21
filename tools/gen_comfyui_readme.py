"""Write the model lists of the ComfyUI packs' READMEs from the registry (what each pack's dropdowns list).

The block between `README_START` and `README_END` of integrations/comfyui/README.md (the fastforks pack: every model,
non-commercial ones tagged) and integrations/comfyui-commercial/README.md (only models cleared for commercial use) is
`nodes.readme_models()`, from the installed cards; the rest of each README is hand-written.

    uv run python tools/gen_comfyui_readme.py          # write
    uv run python tools/gen_comfyui_readme.py --check  # exit 1 if a README is out of date

tests/test_comfyui.py runs the check (with every model family installed).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import types

ROOT = Path(__file__).resolve().parents[1]
READMES = {
    False: ROOT / "integrations" / "comfyui" / "README.md",
    True: ROOT / "integrations" / "comfyui-commercial" / "README.md",
}


def implementation() -> types.ModuleType:
    """The packs' `fastforks_comfyui.nodes` (imported without ComfyUI: nothing of it is needed for the lists)."""
    if "fastforks_comfyui" not in sys.modules:
        spec = importlib.machinery.ModuleSpec("fastforks_comfyui", None, is_package=True)
        spec.submodule_search_locations = [str(ROOT / "integrations" / "comfyui")]
        sys.modules["fastforks_comfyui"] = importlib.util.module_from_spec(spec)
    return importlib.import_module("fastforks_comfyui.nodes")


def render(text: str, *, commercial_only: bool) -> str:
    """`text` (a README) with its generated block replaced by the current lists."""
    nodes = implementation()
    start, end = text.index(nodes.README_START), text.index(nodes.README_END) + len(nodes.README_END)
    return text[:start] + nodes.readme_models(commercial_only=commercial_only) + text[end:]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if a README is out of date")
    args = ap.parse_args()
    stale = []
    for commercial_only, path in READMES.items():
        text = path.read_text(encoding="utf-8")
        new = render(text, commercial_only=commercial_only)
        if new != text:
            stale.append(path)
            if not args.check:
                path.write_bytes(new.encode("utf-8"))
    if args.check and stale:
        sys.exit(f"out of date (run tools/gen_comfyui_readme.py): {', '.join(str(p) for p in stale)}")


if __name__ == "__main__":
    main()
