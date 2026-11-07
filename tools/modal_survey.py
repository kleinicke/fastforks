"""Run tools/speed_survey.py on a Modal cloud GPU (https://modal.com); rows land in docs/data/speed/<slug>.jsonl.

One-time setup on the Mac (the login is a browser OAuth that writes ~/.modal.toml):

    uvx modal setup        # or: pip install modal && python3 -m modal setup

Then, cheapest first:

    uvx modal run tools/modal_survey.py::upload --src /Volumes/Daten/fastforks-weights  # the gated weights, once
    uvx modal run tools/modal_survey.py::prefetch                 # CPU only: weights, goldens, scene
    uvx modal run tools/modal_survey.py::main --gpu L4 --subset    # cross-device subset (~10-20 min)
    uvx modal run tools/modal_survey.py::main --gpu L4             # every ref, same file (resumes)
    uvx modal run tools/modal_survey.py::main --gpu H100 --only "resnet50.a1_in1k vit_base_patch16_224.augreg_in21k"
    uvx modal run tools/modal_survey.py::profile --gpu L4 --only depth-pro --out trace.jsonl   # kernel groups
    uvx modal run tools/modal_survey.py::shootout --gpu L4 --out gemm.json                     # GEMM tiles vs cuBLAS
    uvx modal run tools/modal_survey.py::sweep --gpu L4 --out calls.json    # every kernels call of a forward
    uvx modal run tools/modal_survey.py::sections --gpu L4 --out sections.json  # wall time by section
    uvx modal run tools/modal_survey.py::lab --gpu L4 --spec lab.json --out lab-out.json  # kernel variants

`--gpu` takes Modal's names (T4, L4, A10, L40S, A100-40GB, A100-80GB, H100, H200, B200). The local entrypoint waits
for the run, then copies the rows and the device's devices.toml entry into this checkout; add `devices/<slug>.md` to
mkdocs.yml's nav and run `uv run python tools/gen_docs.py`. With `modal run --detach` the run outlives the terminal;
fetch its rows later with `modal run tools/modal_survey.py::fetch`.

Everything that is not a measurement stays off the GPU: `prefetch` fills the `fastforks-survey` volume on a CPU
container (about 110 GB of weights the first time, then nothing), so the GPU container only verifies and measures. The
rows are written to the volume as they are measured (committed every few minutes), so a preempted or killed run
continues where it stopped.

The gated families (DINOv3, SAM 3, VGGT-Omega) cannot be downloaded without a Hugging Face token, so `upload` copies
the weights this Mac already has (a FASTFORKS_WEIGHTS_PATH folder, about 8 GB) into the volume's weights/ once; the
containers search there first. Everything else downloads inside Modal, which is much faster than a home upload. A gated
file that is neither uploaded nor fetchable (no token) is only a `failed` row. To fetch the rest of them too, create a
Modal secret `huggingface` with HF_TOKEN and set FASTFORKS_MODAL_SECRETS=huggingface when running.
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
VOL = Path("/vol")  # the persistent volume: caches, goldens, scene, rows, uploaded weights
WEIGHTS = VOL / "weights"  # `upload`'s copy of local FASTFORKS_WEIGHTS_PATH folders
PY = "/usr/local/bin/python"  # the image's Python: the functions run in it too, so it holds every dependency
GROUPS = ("dev", "cu130", "vision")  # the lock's cu130 torch + timm / transformers for the originals
MIDDLEBURY = "https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-data-H.zip"  # the floor's stereo scene
COMMIT_EVERY_S = 300

app = modal.App("fastforks-survey")
volume = modal.Volume.from_name("fastforks-survey", create_if_missing=True)
secrets = [modal.Secret.from_name(s) for s in os.environ.get("FASTFORKS_MODAL_SECRETS", "").split(",") if s]

_deps = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "unzip", "libgl1", "libglib2.0-0")
    .pip_install("uv")
    .add_local_file(ROOT / "pyproject.toml", f"{REPO}/pyproject.toml", copy=True)
    .add_local_file(ROOT / "uv.lock", f"{REPO}/uv.lock", copy=True)
    .run_commands(  # the dependency layer, into the image's own Python: rebuilt only when the lock changes
        f"cd {REPO} && UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen --inexact --no-install-project "
        "--no-default-groups " + " ".join(f"--group {g}" for g in GROUPS)
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
)


def _with_source(img: modal.Image) -> modal.Image:
    """The checkout on top of `img`: the only layers a source edit rebuilds."""
    return (
        img.add_local_dir(
            ROOT,
            str(REPO),
            ignore=[
                ".git",
                ".venv",
                "site",
                "runs",
                "o",
                "**/__pycache__",
                "**/.pytest_cache",
                "**/.ruff_cache",
                "tests",
            ],
            copy=True,  # built in, so the project can be installed (its metadata: `fastforks.__version__`)
        ).run_commands(f"uv pip install --system --no-deps -e {REPO}")  # seconds; the only layer a source edit rebuilds
    )


image = _with_source(_deps)
CUTLASS = "/opt/cutlass"
# `kernel_lab.py` sources built in the container: gcc, the kernels group's pip nvcc, the CUTLASS headers
lab_image = _with_source(
    _deps.apt_install("build-essential")
    .run_commands(
        f"cd {REPO} && UV_PROJECT_ENVIRONMENT=/usr/local uv sync --frozen --inexact --no-install-project "
        "--no-default-groups " + " ".join(f"--group {g}" for g in (*GROUPS, "kernels")),
        f"git clone --depth 1 --branch v4.7.1 https://github.com/NVIDIA/cutlass {CUTLASS}",
    )
    .env({"FASTFORKS_LAB_CUTLASS": CUTLASS})
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


def _weights_path() -> None:
    """FASTFORKS_WEIGHTS_PATH: the uploaded weights/, plus each folder in it that holds only folders.

    `<dir>/<group>/<name>` already finds a family folder of files (sam3/model.safetensors). A folder of folders holds
    names with a sub-folder, possibly of another group (the ConvNeXt `dinov3-convnext-*/model.safetensors` under
    dinov3/), so it is searched itself. A folder of files is not: its generic names (model.safetensors) would be
    offered to every other family, which only the size check would turn down.
    """
    if WEIGHTS.is_dir():
        nested = [d for d in sorted(WEIGHTS.iterdir()) if d.is_dir() and all(p.is_dir() for p in d.iterdir())]
        os.environ["FASTFORKS_WEIGHTS_PATH"] = os.pathsep.join(map(str, [WEIGHTS, *nested]))


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
    _weights_path()
    _goldens()
    _scene()
    any_dev = next(d for d in _devices.devices() if d.backend == "cuda")
    dev = dataclasses.replace(any_dev, reference=False)  # plan() reads only the backend
    refs = sorted({j["ref"] for j in speed_survey.plan(dev, only.split(), subset=subset) if not j["skip"]})
    env = {**os.environ, "FASTFORKS_NONCOMMERCIAL": "1"}
    # one call: a file that cannot be fetched (gated, no token) is a `MISSING` line at the end, the rest still come
    rc = subprocess.run(
        [PY, "-m", "fastforks", "download", *refs, "--kernels", "--all-archs"], cwd=REPO, env=env, check=False
    ).returncode
    stop.set()
    volume.commit()
    if rc:
        print("some files were not fetched (the MISSING lines above); their rows will be `failed`", file=sys.stderr)


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
    # no gpu here: `main` sets it per call (.with_options), so the CPU-only `prefetch` runs without GPU billing set up
    timeout=12 * 3600,
    retries=modal.Retries(max_retries=2, initial_delay=10.0),  # preemption: the survey resumes from the volume rows
)
def survey(commit: str, only: str = "", subset: bool = False, force: bool = False) -> str:  # noqa: FBT001, FBT002
    """Detect, prepare, run; returns the device slug (rows: /vol/rows/<slug>.jsonl, entry: /vol/devices/<slug>.toml)."""
    _sh("nvidia-smi", "-q", "-d", "CLOCK,POWER")
    _weights_path()
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


@app.function(image=image, volumes={str(VOL): volume}, secrets=secrets, cpu=8, memory=32768, timeout=3600)
def trace(only: str, modes: str = "portable kernels") -> str:
    """`speed_check.py run --trace-diff` on this GPU: CUDA time per kernel family and launching op."""
    _sh("nvidia-smi", "--query-gpu=name,power.limit,clocks.max.sm", "--format=csv")
    _weights_path()
    out = VOL / "traces" / "last.jsonl"  # on the volume, not /tmp
    out.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["tools/speed_check.py", "run", "--trace-diff", "--sides", "lib", "--force", "--out", str(out)]
    _sh(PY, *cmd, "--only", *only.split(), "--modes", *modes.split())
    return out.read_text() if out.is_file() else ""


@app.function(image=image, volumes={str(VOL): volume}, cpu=4, memory=16384, timeout=1800)
def gemm(m: str = "20195 577") -> str:
    """`tools/gemm_shootout.py` on this GPU: `dpt_fused` tile configurations vs cuBLAS / cuBLASLt, as JSON."""
    _sh("nvidia-smi", "--query-gpu=name,power.limit,clocks.max.sm", "--format=csv")
    out = VOL / "traces" / "gemm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _sh(PY, "tools/gemm_shootout.py", "--m", *m.split(), "--out", str(out))
    return out.read_text() if out.is_file() else ""


@app.function(image=image, volumes={str(VOL): volume}, cpu=4, memory=32768, timeout=1800)
def calls(model: str = "depth-pro") -> str:
    """`tools/call_sweep.py` on this GPU: every `dpt_fused` / attention call of one forward vs cuBLAS / cuDNN / SDPA."""
    _sh("nvidia-smi", "--query-gpu=name,power.limit,clocks.max.sm", "--format=csv")
    _weights_path()
    out = VOL / "traces" / "calls.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _sh(PY, "tools/call_sweep.py", "--model", model, "--out", str(out))
    return out.read_text() if out.is_file() else ""


@app.function(image=image, volumes={str(VOL): volume}, cpu=4, memory=32768, timeout=1800)
def section_times(model: str = "depth-pro") -> str:
    """`tools/section_time.py` on this GPU: wall time by section, streams on / off, clocks, kernel families."""
    _sh("nvidia-smi", "--query-gpu=name,power.limit,clocks.max.sm", "--format=csv")
    _weights_path()
    out = VOL / "traces" / "sections.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _sh(PY, "tools/section_time.py", "--model", model, "--out", str(out))
    return out.read_text() if out.is_file() else ""


@app.function(image=lab_image, volumes={str(VOL): volume}, cpu=8, memory=32768, timeout=3600)
def lab_run(spec: str) -> str:
    """`tools/kernel_lab.py` on this GPU: `spec` is its JSON with each source's text in place of its path."""
    import json  # noqa: PLC0415

    _sh("nvidia-smi", "--query-gpu=name,power.limit,clocks.max.sm", "--format=csv")
    _weights_path()
    _scene()
    d = Path("/tmp/lab")  # noqa: S108 - container scratch
    d.mkdir(parents=True, exist_ok=True)
    sp = json.loads(spec)
    for name, text in sp.get("sources", {}).items():
        (d / f"{name}.cu").write_text(text)
        sp["sources"][name] = f"{name}.cu"
    (d / "spec.json").write_text(json.dumps(sp))
    import torch  # noqa: PLC0415 - the image's torch

    gpu = re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name().lower()).strip("-")
    out = VOL / "traces" / f"lab-{gpu}-{int(time.time())}.json"  # concurrent runs on other GPUs do not collide
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    img = VOL / "data/Middlebury/MiddEval3/trainingH/Motorcycle/im0.png"
    _sh(PY, "tools/kernel_lab.py", "--spec", str(d / "spec.json"), "--image", str(img), "--out", str(out))
    return out.read_text() if out.is_file() else ""


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
def upload(src: str = "") -> None:
    """Copy local weight folders (`--src`, else FASTFORKS_WEIGHTS_PATH) into the volume; skips files already there."""
    dirs = [Path(p).expanduser() for p in (src or os.environ.get("FASTFORKS_WEIGHTS_PATH", "")).split(os.pathsep) if p]
    if not dirs:
        msg = "pass --src DIR (or set FASTFORKS_WEIGHTS_PATH)"
        raise SystemExit(msg)
    try:
        have = {e.path.lstrip("/").removeprefix("weights/"): e.size for e in volume.listdir("weights", recursive=True)}
    except FileNotFoundError:
        have = {}
    todo: dict[str, Path] = {}
    for d in dirs:
        for f in sorted(d.rglob("*")):
            rel = f.relative_to(d).as_posix()
            if f.is_file() and not f.name.startswith(".") and have.get(rel) != f.stat().st_size:
                todo.setdefault(rel, f)
    size = sum(f.stat().st_size for f in todo.values())
    print(f"uploading {len(todo)} files, {size / 1e9:.1f} GB ({len(have)} already in the volume)", flush=True)
    with volume.batch_upload(force=True) as batch:
        for rel, f in todo.items():
            batch.put_file(f, f"/weights/{rel}")


@app.local_entrypoint()
def fetch() -> None:
    """Copy the rows of every surveyed GPU from the volume (after `modal run --detach`)."""
    _next_steps(_fetch())


@app.local_entrypoint()
def profile(out: str, gpu: str = "L4", only: str = "depth-pro", modes: str = "portable kernels") -> None:
    """Kernel-group trace of `--only` refs on one GPU, rows to `--out` (read them with a short script or jq)."""
    rows = trace.with_options(gpu=gpu).remote(only, modes)
    Path(out).write_text(rows)
    print(f"{out}: {len(rows.splitlines())} rows")


@app.local_entrypoint()
def shootout(out: str, gpu: str = "L4", m: str = "20195 577") -> None:
    """GEMM shoot-out (`tools/gemm_shootout.py`) on one GPU, JSON to `--out`."""
    Path(out).write_text(gemm.with_options(gpu=gpu).remote(m))
    print(f"{out} written")


@app.local_entrypoint()
def sweep(out: str, gpu: str = "L4", model: str = "depth-pro") -> None:
    """Per-call sweep (`tools/call_sweep.py`) of one model's kernels forward on one GPU, JSON to `--out`."""
    Path(out).write_text(calls.with_options(gpu=gpu).remote(model))
    print(f"{out} written")


@app.local_entrypoint()
def sections(out: str, gpu: str = "L4", model: str = "depth-pro") -> None:
    """Section timing (`tools/section_time.py`) of one model's kernels forward on one GPU, JSON to `--out`."""
    Path(out).write_text(section_times.with_options(gpu=gpu).remote(model))
    print(f"{out} written")


@app.local_entrypoint()
def lab(spec: str, out: str, gpu: str = "L4") -> None:
    """Kernel lab (`tools/kernel_lab.py`): the variants of `--spec` on one GPU, JSON to `--out`."""
    import json  # noqa: PLC0415

    sp = json.loads(Path(spec).read_text())
    base = Path(spec).resolve().parent
    sp["sources"] = {n: (base / p).read_text() for n, p in sp.get("sources", {}).items()}
    Path(out).write_text(lab_run.with_options(gpu=gpu).remote(json.dumps(sp)))
    print(f"{out} written")
