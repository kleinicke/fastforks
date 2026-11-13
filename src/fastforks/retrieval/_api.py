"""Retrieval API: `list_models`, `model_card`, `load`, the model / result types, and the `top_k` / `pairs` search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from fastforks._core import model as core
from fastforks._core.api import task_api
from fastforks._core.bench import BenchInputs, BenchSize, synthetic_image
from fastforks._core.errors import InputError
from fastforks._core.image import ImageLike, is_batch, to_rgb_batch
from fastforks._core.model import Backend
from fastforks._core.outputs import NumpyOutputs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastforks.retrieval._runner import Runner

__all__ = ["Backend", "RetrievalModel", "RetrievalResult", "list_models", "load", "model_card", "pairs", "top_k"]

TASK = "retrieval"

_CHUNK = 1 << 26  # query x database scores per matmul chunk (256 MB of float32)


@dataclass(frozen=True)
class RetrievalResult(NumpyOutputs):
    """Output of a retrieval model, on the model's device.

    Attributes:
        descriptor: Global descriptor `[dim]` float32 of unit L2 length (`[B, dim]` for a batch); the dot product of
            two descriptors is their cosine similarity.
        model: Versioned model id.
        backend: The backend that produced the result.
        options: Load options that differ from the validated defaults.
    """

    descriptor: torch.Tensor
    model: str
    backend: Backend
    options: Mapping[str, Any] = field(default_factory=dict)


class RetrievalModel(core.Model):
    """A loaded retrieval model. Call it with an image (or a batch).

    Besides the attributes of `fastforks.Model`:

    Attributes:
        dim: Descriptor size.
    """

    _runner: Runner

    @property
    def dim(self) -> int:
        """Descriptor size (the card's `params.dim`, a variant's own `params.dim` first)."""
        return int({**self.card.params, **self.variant.params}["dim"])

    def __call__(self, image: ImageLike) -> RetrievalResult:
        """Global descriptor of an image (resized to the network size as the card's `[params]` say).

        Args:
            image: Image (path, PIL, numpy HWC / BHWC, or torch CHW / BCHW; see `fastforks._core.image`).

        Returns:
            The result; a batch gives `[B, dim]` descriptors.

        Raises:
            InputError: Unsupported or too small image.
        """
        x = to_rgb_batch(image, self.device)
        self._check_size(x)

        def run(chunk: torch.Tensor) -> dict[str, torch.Tensor]:
            return self._check_outputs(self._runner(chunk), {"descriptor": (chunk.shape[0], self.dim)})

        d = self._batched(run, x)["descriptor"]
        return RetrievalResult(d if is_batch(image) else d[0], self.variant.ref, self.backend, self.options)


def _matrix(x: torch.Tensor | RetrievalResult, name: str) -> torch.Tensor:
    t = x.descriptor if isinstance(x, RetrievalResult) else x
    if not isinstance(t, torch.Tensor) or t.ndim not in {1, 2} or not t.is_floating_point() or t.numel() == 0:
        msg = f"{name} must be a non-empty float tensor [dim] or [N, dim] (or a RetrievalResult), got {t!r:.80}"
        raise InputError(msg)
    return t.reshape(-1, t.shape[-1])


def top_k(
    query: torch.Tensor | RetrievalResult, database: torch.Tensor | RetrievalResult | None = None, k: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """The `k` most similar database descriptors of each query (exact search: matmul + top-k on their device).

    Args:
        query: Descriptors `[Q, dim]` (or one `[dim]`), e.g. `RetrievalResult.descriptor`.
        database: Descriptors `[N, dim]` on the same device; `None`: the queries themselves, each without itself.
        k: Neighbours per query (at most `N`, `N - 1` without a database).

    Returns:
        `(scores, index)`, both `[Q, k]`, best first: the dot products (cosine similarities of unit descriptors) as
        float32 and the database rows as int64.

    Raises:
        InputError: Not float tensors `[dim]` / `[N, dim]`, different sizes or devices, or `k` out of range.
    """
    q = _matrix(query, "query")
    db = q if database is None else _matrix(database, "database")
    if db.shape[1] != q.shape[1] or db.device != q.device:
        msg = f"query [{q.shape[1]}] on {q.device} and database [{db.shape[1]}] on {db.device} do not match"
        raise InputError(msg)
    n = db.shape[0] - (database is None)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n:
        msg = f"k must be an integer in [1, {n}], got {k!r}"
        raise InputError(msg)
    scores, index = [], []
    rows = max(1, _CHUNK // db.shape[0])
    for i in range(0, q.shape[0], rows):
        s = q[i : i + rows].float() @ db.float().T
        if database is None:  # never the query itself
            s[torch.arange(s.shape[0], device=s.device), torch.arange(i, i + s.shape[0], device=s.device)] = -torch.inf
        v, j = s.topk(k, dim=1)
        scores.append(v)
        index.append(j)
    return torch.cat(scores), torch.cat(index)


def pairs(descriptors: torch.Tensor | RetrievalResult, k: int = 20) -> torch.Tensor:
    """Image pairs to match for structure from motion: each image with its `k` most similar others.

    As hloc's `pairs_from_retrieval`, but each unordered pair once: `[i, j]` with `i < j`, sorted.

    Args:
        descriptors: `[N, dim]` descriptors of all images.
        k: Neighbours per image (at most `N - 1`).

    Returns:
        `[P, 2]` int64 image indices on the descriptors' device.

    Raises:
        InputError: As `top_k`.
    """
    _, j = top_k(descriptors, None, k)
    i = torch.arange(j.shape[0], device=j.device)[:, None].expand_as(j)
    p = torch.stack((torch.minimum(i, j), torch.maximum(i, j)), -1).reshape(-1, 2)
    return torch.unique(p, dim=0)


def bench_inputs(model: RetrievalModel, size: BenchSize) -> BenchInputs:
    """`fastforks bench` call: one synthetic image `[1, 3, H, W]` (resized by the model as usual)."""
    return (synthetic_image(size.height, size.width, model.device),), {}


list_models, model_card, load = task_api(TASK, RetrievalModel)
