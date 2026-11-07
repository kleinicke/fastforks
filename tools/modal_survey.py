"""Run tools/speed_survey.py on a Modal cloud GPU (https://modal.com); rows land in docs/data/speed/<slug>.jsonl.

One-time setup on the Mac (the login is a browser OAuth that writes ~/.modal.toml):

    uvx modal setup        # or: pip install modal && python3 -m modal setup

Then, cheapest first:

    uvx modal run tools/modal_survey.py::prefetch                 # CPU only: weights, goldens, scene
    uvx modal run tools/modal_survey.py::main --gpu L4 --subset    # cross-device subset (~10-20 min)
    uvx modal run tools/modal_survey.py::main --gpu L4             # every ref, same file (resumes)
    uvx modal run tools/modal_survey.py::main --gpu H100 --only "resnet50.a1_in1k vit_base_patch16_224.augreg_in21k"

`--gpu` takes Modal's names (T4, L4, A10, L40S, A100-40GB, A100-80GB, H100, H200, B200). The local entrypoint waits
for the run, then copies the rows and the device's devices.toml entry into this checkout; add `devices/<slug>.md` to
mkdocs.yml's nav and run `uv run python tools/gen_docs.py`. With `modal run --detach` the run outlives the terminal;
fetch its rows later with `modal run tools/modal_survey.py::fetch`.

Everything that is not a measurement stays off the GPU: `prefetch` fills the `fastforks-survey` volume on a CPU
container (about 110 GB of weights the first time, then nothing), so the GPU container only verifies and measures. The
rows are written to the volume as they are measured (committed every few minutes), so a preempted or killed run
continues where it stopped. The gated families (DINOv3, SAM 3) need a Hugging Face token: create a Modal secret
`huggingface` with HF_TOKEN and set FASTFORKS_MODAL_SECRETS=huggingface when running.
"""

from __future__ import annotations

import dataclasses
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import modal  # ty: ignore[unresolved-import] - not a project dependency: `uvx modal` brings it

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/repo")  # the checkout inside the container
VOL = Path("/vol")  # the persistent volume: caches, goldens, scene, rows
VENV = "/opt/venv"
PY = f"{VENV}/bin/python"
GROUPS = ("dev", "cu130", "vision")  # the lock's cu130 torch + timm / transformers for the originals
MIDDLEBURY = "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip"  # the floor's stereo scene
COMMIT_EVERY_S = 300

app = modal.App("fastforks-survey")
volume = modal.Volume.from_name("fastforks-survey", create_if_missing=True)
secrets = [modal.Secret.from_name(s) for s in os.environ.get("FASTFORKS_MODAL_SECRETS", "").split(",") if s]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "unzip", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .add_local_file(ROOT / "pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", f"{REPO}/uv.lock", copy=True)
    .run_commands(  # the dependency layer: rebuilt only when the lock changes, not on every source edit
        f"cd {REPO} && UV_PROJECT_ENVIRONMENT={VENV} uv sync --frozen --no-install-project --no-default-groups "
        + " ".join(f"--group {g}" for g in GROUPS)
    )
    .env(
        {
            "PYTHONPATH": f"{REPO}/src:{REPO}/tools",
            "FASTFORKS_CACHE": f"{VOL}/fastforks",
            "HF_HOME": f"{VOL}/hf",
            "TORCH_HOME": f"{VOL}/torch",
            "FASTFORKS_GOLDEN_PATH": f"{VOL}/golden",
            "FASTFORKS_DATA": f"{VOL}/data",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        ROOT,
        str(REPO),
        ignore=[".git", ".venv", "site", "runs", "o", "**/__pycache__", "**/.pytest_cache", "**/.ruff_cache", "tests"],
    )
)


def _sh(*cmd: str, cwd: Path = REPO) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


def _committer(stop: threading.Event) -> threading.Thread:
    """Commit the volume every few minutes, so a preempted container loses at most that much."""

    def loop() -> None:
        while not stop.wait(COMMIT_EVERY_S):
            volume.commit()

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def _scene() -> None:
    """The Middlebury Motorcycle pair of the floor's 994x1482 stereo / depth rows (KITTI needs a login: synthetic)."""
    left = VOL / "data/Middlebury/MiddEval3/trainingH/Motorcycle/im0.png"
    if left.is_file():
        return
    dest = VOL / "data/Middlebury"
    dest.mkdir(parents=True, exist_ok=True)
    zipf = dest / "MiddEval3-data-H.zip"
    if _sh("curl", "-fsSL", "-o", str(zipf), MIDDLEBURY) or _sh("unzip", "-qo", str(zipf), "-d", str(dest)):
        print("Middlebury scene not fetched: stereo / depth rows use synthetic inputs", file=sys.stderr)
    zipf.unlink(missing_ok=True)


def _goldens() -> None:
    if not any((VOL / "golden").glob("*")):
        _sh(PY, "tools/fetch_goldens.py", "--all", str(VOL / "golden"))


@app.function(image=image, volumes={str(VOL): volume}, secrets=secrets, cpu=4, memory=8192, timeout=6 * 3600)
def prefetch(only: str = "", subset: bool = False) -> None:  # noqa: FBT001, FBT002 - modal CLI flags
    """Weights + kernel binaries of every arch, goldens, scene into the volume, on a CPU container (no GPU billed)."""
    import _devices  # noqa: PLC0415 - container-only import (PYTHONPATH)
    import speed_survey  # noqa: PLC0415

    stop = threading.Event()
    _committer(stop)
    _goldens()
    _scene()
    any_dev = next(d for d in _devices.devices() if d.backend == "cuda")
    dev = dataclasses.replace(any_dev, reference=False)  # plan() reads only the backend
    refs = sorted({j["ref"] for j in speed_survey.plan(dev, only.split(), subset=subset) if not j["skip"]})
    env = {**os.environ, "FASTFORKS_NONCOMMERCIAL": "1"}
    cmd = [PY, "-m", "fastforks", "download", *refs, "--kernels", "--all-archs"]
    print("$", " ".join(cmd[:4]), f"... ({len(refs)} refs) --kernels --all-archs", flush=True)
    rc = subprocess.run(cmd, cwd=REPO, env=env, check=False).returncode
    stop.set()
    volume.commit()
    if rc:
        print(f"download exit {rc}: a file is missing (the GPU run's `prepare` says which)", file=sys.stderr)


def _entry() -> tuple[str, str]:
    """This GPU's devices.toml entry (`speed_survey.py detect`), and its slug."""
    out = subprocess.run(
        [PY, "tools/speed_survey.py", "detect"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout
    slug = re.match(r"\[([^\]]+)\]", out.strip())
    if not slug:
        msg = f"detect printed no entry: {out!r}"
        raise RuntimeError(msg)
    return out, slug.group(1)


@app.function(
    image=image,
    volumes={str(VOL): volume},
    secrets=secrets,
    cpu=8,
    memory=32768,  # weights load on the host first
    gpu="L4",  # overridden per call with .with_options(gpu=...)
    timeout=12 * 3600,
    retries=modal.Retries(max_retries=2, initial_delay=10.0),  # preemption: the survey resumes from the volume rows
)
def survey(commit: str, only: str = "", subset: bool = False, force: bool = False) -> str:  # noqa: FBT001, FBT002
    """Detect, prepare, run; returns the device slug (rows: /vol/rows/<slug>.jsonl, entry: /vol/devices/<slug>.toml)."""
    _sh("nvidia-smi", "-q", "-d", "CLOCK,POWER")
    entry, slug = _entry()
    registry = REPO / "docs/data/devices.toml"
    if slug not in tomllib.loads(registry.read_text()):
        registry.write_text(registry.read_text().rstrip() + "\n\n" + entry)
    (VOL / "devices").mkdir(parents=True, exist_ok=True)
    (VOL / "devices" / f"{slug}.toml").write_text(entry)
    rows, local = VOL / "rows" / f"{slug}.jsonl", REPO / "docs/data/speed" / f"{slug}.jsonl"
    rows.parent.mkdir(parents=True, exist_ok=True)
    if local.is_file() and not rows.is_file():  # rows this checkout already has: kept, not re-measured
        shutil.copy(local, rows)
    local.parent.mkdir(parents=True, exist_ok=True)
    local.unlink(missing_ok=True)
    local.symlink_to(rows)  # speed_survey writes through the link: every row is on the volume at once
    volume.commit()

    stop = threading.Event()
    _committer(stop)
    os.environ["FASTFORKS_COMMIT"] = commit
    sel = [*(["--only", *only.split()] if only else []), *(["--subset"] if subset else [])]
    t0 = time.monotonic()
    try:
        if _sh(PY, "tools/speed_survey.py", "prepare", "--device", slug, *sel):
            print("prepare: a file is missing; its rows will be `failed`", file=sys.stderr)
        print(f"prepare took {time.monotonic() - t0:.0f} s", flush=True)
        rc = _sh(PY, "tools/speed_survey.py", "run", "--device", slug, *sel, *(["--force"] if force else []))
    finally:
        stop.set()
        volume.commit()
    print(f"survey of {slug}: exit {rc}, {time.monotonic() - t0:.0f} s in all", flush=True)
    return slug


def _fetch(slugs: list[str] | None = None) -> list[str]:
    """Copy the volume's rows (and missing devices.toml entries) into this checkout."""
    registry = ROOT / "docs/data/devices.toml"
    known = tomllib.loads(registry.read_text())
    if slugs is None:
        try:
            slugs = [e.path.split("/")[-1].removesuffix(".toml") for e in volume.listdir("devices")]
        except FileNotFoundError:
            slugs = []
    got = []
    for slug in slugs:
        try:
            rows = b"".join(volume.read_file(f"rows/{slug}.jsonl"))
        except FileNotFoundError:
            continue
        (ROOT / "docs/data/speed" / f"{slug}.jsonl").write_bytes(rows)
        if slug not in known:
            entry = b"".join(volume.read_file(f"devices/{slug}.toml")).decode()
            registry.write_text(registry.read_text().rstrip() + "\n\n" + entry)
        got.append(slug)
        print(f"docs/data/speed/{slug}.jsonl: {len(rows.splitlines())} rows")
    return got


def _next_steps(slugs: list[str]) -> None:
    nav = (ROOT / "mkdocs.yml").read_text()
    for slug in slugs:
        if f"devices/{slug}.md" not in nav:
            print(f"add `devices/{slug}.md` to mkdocs.yml's nav (next to the Apple M1 Pro page)")
    print("then: uv run python tools/gen_docs.py && uv run --group docs mkdocs build --strict")


def _commit_here() -> str:
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
    )
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "src"], capture_output=True, text=True, check=False
    )
    return head.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


@app.local_entrypoint()
def main(gpu: str = "L4", only: str = "", subset: bool = False, force: bool = False) -> None:  # noqa: FBT001, FBT002
    """Survey one Modal GPU type and copy its rows here (`--only` takes space-separated refs)."""
    slug = survey.with_options(gpu=gpu).remote(_commit_here(), only, subset, force)
    _next_steps(_fetch([slug]))


@app.local_entrypoint()
def fetch() -> None:
    """Copy the rows of every surveyed GPU from the volume (after `modal run --detach`)."""
    _next_steps(_fetch())
