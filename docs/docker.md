# Docker images that never download at run time

Fetch weights and kernel binaries while the image is built, then forbid every download at run time with
`FASTFORKS_OFFLINE=1` (`HF_HUB_OFFLINE=1` is honoured too). A file that is not in the image then raises an error naming
the file and the `fastforks download` command that fetches it, instead of reaching for the network.

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim AS build
ENV FASTFORKS_CACHE=/opt/fastforks-cache VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python -m venv $VIRTUAL_ENV
# torch is a dependency of fastforks; for a specific CUDA build add e.g. --index-url https://download.pytorch.org/whl/cu130
# (it must match the kernels, see `fastforks env`). For a lean or commercial image install fastforks-core with the extras
# the models you ship need, e.g. "fastforks-core[timm]" (see the README).
RUN --mount=type=cache,target=/root/.cache/pip pip install fastforks
# explicit model list; --arch = the GPUs this image runs on (no GPU needed to build). Drop --kernels for CPU-only.
# HF_TOKEN (gated or private repos) only as a BuildKit secret: never ENV / ARG, it would stay in a layer.
RUN --mount=type=secret,id=hf_token \
    HF_TOKEN="$(cat /run/secrets/hf_token 2>/dev/null)" \
    fastforks download hitnet-eth3d igev-stereo-sceneflow --kernels --arch sm_86,sm_89 --commercial-only \
 && fastforks download hitnet-eth3d igev-stereo-sceneflow --kernels --arch sm_86,sm_89 --check \
 && chmod -R a+rX /opt/fastforks-cache

FROM python:3.12-slim
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/fastforks-cache /opt/fastforks-cache
ENV PATH=/opt/venv/bin:$PATH FASTFORKS_CACHE=/opt/fastforks-cache FASTFORKS_OFFLINE=1 \
    TRITON_CACHE_DIR=/tmp/triton TORCHINDUCTOR_CACHE_DIR=/tmp/inductor
USER 1000
```

Gated weights land in the Hugging Face cache, so copy `HF_HOME` (or `HF_HUB_CACHE`) too, or put the file on
`FASTFORKS_WEIGHTS_PATH`.

Build with `docker build --secret id=hf_token,env=HF_TOKEN .` (or leave the secret out); run with
`docker run --gpus all --read-only --tmpfs /tmp IMAGE ...`. Without `--gpus` the image runs the portable CPU path.

- `download ... --check` as a last `RUN` fails the build (exit 1, `MISSING ...` on stderr) if a file cannot be resolved
  without network, so a broken image is found at build time, not at the first request. It never touches the network and
  prints where each file resolves from. "Resolvable" includes `FASTFORKS_WEIGHTS_PATH` / `FASTFORKS_KERNEL_PATH` and the
  Hugging Face / torch hub caches; in a clean build stage those are empty, so every hit is in `FASTFORKS_CACHE`. `--dry-run` lists
  the size and licence class per model first and fetches nothing.
- GPU -> `--arch`: RTX 30xx `sm_86`, A100 `sm_80`, RTX 40xx / L4 `sm_89`, H100 `sm_90`, B200 `sm_100`, RTX 50xx `sm_120` (all in [Other GPUs](OTHER_GPUS.md#what-runs-where)); several: `--arch sm_89,sm_120`.
- `--arch` takes `sm_89`, `8.9` or `native`, validated against the kernel manifest and resolved as at run time (an
  `sm_86` GPU gets the binary the runtime would pick). `--arch` needs `--kernels` and excludes `--all-archs`; targets like `sm_90a` are not offered (use `sm_90`); an all-archs set is
  needlessly large for an image.
- `--kernels-only` fetches the kernel binaries without weights (all libraries, or those of the named models); `--arch` / `--all-archs` / `--check` work as with `--kernels`.
- The cache is a plain directory; treat it as opaque, copy it whole, and use the same fastforks major version to read it.
  fastforks creates cache directories private (0700), so the build stage ends with `chmod -R a+rX` (above) or the
  non-root user cannot read them. A read-only cache then works: files are only read, a failed write of the verification
  stamp is ignored. Inside a container `download` refuses to run while `FASTFORKS_OFFLINE` is set (unset it for that step).
- Instead of baking, mount a pre-filled cache as a read-only volume (`-v cache:/opt/fastforks-cache:ro`), or point
  `FASTFORKS_WEIGHTS_PATH` / `FASTFORKS_KERNEL_PATH` at read-only directories of the original files.
- JIT caches are written at run time (Triton, `torch.compile`): point `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` at
  a writable directory (`--tmpfs`), or the first call is slow and a read-only filesystem fails. `torch.compile` also needs
  a C++ compiler in the image.
- A `nvidia/cuda:*-runtime` base needs a host driver that is new enough for the kernels' CUDA version; `fastforks env`
  prints what the kernels need.

## Licences

Baking weights into an image that you push to a registry is redistribution. `fastforks download` without a model list
fetches every installed model, including non-commercial and GPL families, and warns about it. Build the image from your
own explicit model list, use `--commercial-only` to skip models not cleared for commercial use (with a model list it fails if you name one that is not cleared), and **do not push
images holding non-commercial weights to public registries**. `fastforks info MODEL` shows the terms.

## Offline installs (wheelhouse)

On a machine with network: `pip download fastforks torch -d wheelhouse` (add `--index-url https://download.pytorch.org/whl/cu130`
for a CUDA build of torch that matches the kernels), then copy `wheelhouse/` and the cache (`fastforks download ...`).
On the air-gapped machine: `pip install --no-index --find-links wheelhouse fastforks`.
