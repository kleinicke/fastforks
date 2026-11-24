"""Small linear-algebra operators for ONNX export graphs (ONNX has no SVD or eigendecomposition)."""

from __future__ import annotations

import torch


def nearest_rotation(b: torch.Tensor, *, squarings: int = 12) -> torch.Tensor:
    """The rotation closest to each `b` `[N, 3, 3]` (`argmax_R tr(R^T b)`, the SVD's `U diag(1, 1, det) V^T`).

    Horn's 4 x 4 quaternion matrix of `b` has that rotation's quaternion as its top eigenvector. Its eigenvalues lie
    within `+-sqrt(3) |b|_F`, so after scaling and a shift of 2 they are non-negative; repeated squaring (normalised
    each step) then leaves only the top eigenvector in every column, which is read from the strongest column.
    Matmuls, norms and a gather only, so ONNX and TensorRT can take it. Matches the SVD to ~1e-5 on random matrices.
    """
    b00, b01, b02 = b[:, 0, 0], b[:, 0, 1], b[:, 0, 2]
    b10, b11, b12 = b[:, 1, 0], b[:, 1, 1], b[:, 1, 2]
    b20, b21, b22 = b[:, 2, 0], b[:, 2, 1], b[:, 2, 2]
    k = torch.stack(
        [
            torch.stack([b00 + b11 + b22, b21 - b12, b02 - b20, b10 - b01], -1),
            torch.stack([b21 - b12, b00 - b11 - b22, b01 + b10, b02 + b20], -1),
            torch.stack([b02 - b20, b01 + b10, b11 - b00 - b22, b12 + b21], -1),
            torch.stack([b10 - b01, b02 + b20, b12 + b21, b22 - b00 - b11], -1),
        ],
        -2,
    )
    scale = b.flatten(1).norm(dim=-1).clamp_min(1e-12)[:, None, None]
    p = k / scale + 2 * torch.eye(4, dtype=b.dtype, device=b.device)
    for _ in range(squarings):
        p = p @ p
        p = p / p.flatten(1).norm(dim=-1)[:, None, None]
    i = p.norm(dim=-2).argmax(-1)
    q = torch.gather(p, 2, i[:, None, None].expand(-1, 4, 1))[..., 0]
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        -2,
    )
