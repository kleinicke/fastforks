"""Deterministic synthetic stereo pair shared by the golden tools and the tests (numpy only, stable across versions)."""

from __future__ import annotations

from typing import Any

import numpy as np

SHIFT = 8  # at 4 px RAFT-Stereo's Middlebury checkpoint (trained on large disparities) diverges


def pair(h: int = 100, w: int = 150, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return uint8 `[H, W, 3]` left / right images of a random texture at constant disparity `SHIFT` px.

    `right[y, x] == left[y, x + SHIFT]`, i.e. `x_left - x_right = SHIFT`.
    """
    rng = np.random.default_rng(seed)
    coarse = rng.random((h // 5 + 1, (w + SHIFT) // 5 + 1, 3))
    texture = np.kron(coarse, np.ones((5, 5, 1)))[:h, : w + SHIFT] * 0.8 + rng.random((h, w + SHIFT, 3)) * 0.2
    img = (texture * 255).round().astype(np.uint8)
    return np.ascontiguousarray(img[:, :w]), np.ascontiguousarray(img[:, SHIFT:])


def image(h: int = 100, w: int = 150, seed: int = 0) -> np.ndarray:
    """Return the uint8 `[H, W, 3]` left image of `pair` (single-image tasks)."""
    return pair(h, w, seed)[0]


def size(min_size: int = 1, h: int = 100, w: int = 150) -> tuple[int, int]:
    """Test image size `h x w`, raised to a model's `io.min_size` (e.g. PSMNet's 64 px pooling at 1/4 needs 241)."""
    return max(h, min_size), max(w, min_size)


def prompts() -> dict[str, dict[str, object]]:
    """Fixed prompts on `image()` for promptable segmentation, as call arguments, by golden key prefix.

    `""`: the centre point (foreground, multimask output); `"box_"`: a box + the centre point (single mask).
    """
    point = {"input_points": [[75, 50]], "input_labels": [1]}
    return {
        "": {**point, "multimask_output": True},
        "box_": {**point, "input_boxes": [20, 15, 120, 85], "multimask_output": False},
    }


def concept() -> str:
    """Text prompt of the concept-segmentation goldens (SAM 3 on scikit-image's `coins.png`, stored as `input`)."""
    return "coin"


def concepts() -> dict[str, dict[str, Any]]:
    """Concept prompts of the SAM 3 goldens by key prefix (the model's keyword arguments; boxes in pixels of coins.png).

    The text alone; one positive exemplar box alone (upstream text "visual"); the text with a positive and a
    negative exemplar box (fractional coordinates: ROI align between pixels).
    """
    return {
        "": {"text": concept()},
        "box_": {"input_boxes": [[305.0, 16.0, 365.0, 72.0]], "input_boxes_labels": [1]},
        "mixed_": {
            "text": concept(),
            "input_boxes": [[143.5, 236.25, 201.0, 287.75], [22.0, 34.0, 66.5, 74.5]],
            "input_boxes_labels": [1, 0],
        },
    }


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera-to-world `[4, 4]` of a camera at `eye` looking at `target` (OpenCV axes, world y down)."""
    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    c2w[:3, 3] = eye
    return c2w


def _texture(s: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """RGB in [0, 1] at plane coordinates `(s, t)` in [0, 1]: bilinear random grids at two scales."""
    out = np.zeros((*s.shape, 3))
    for cells, weight in ((18, 0.75), (90, 0.25)):
        grid = rng.random((cells + 2, cells + 2, 3))
        gs, gt = np.clip(s, 0, 1) * cells, np.clip(t, 0, 1) * cells
        i, j = np.floor(gs).astype(int), np.floor(gt).astype(int)
        fs, ft = (gs - i)[..., None], (gt - j)[..., None]
        out += weight * (
            grid[i, j] * (1 - fs) * (1 - ft)
            + grid[i + 1, j] * fs * (1 - ft)
            + grid[i, j + 1] * (1 - fs) * ft
            + grid[i + 1, j + 1] * fs * ft
        )
    return out


def scene(n: int = 4, h: int = 120, w: int = 160, seed: int = 0) -> dict[str, np.ndarray]:
    """A textured slanted plane rendered from `n` cameras by exact ray-plane intersection (consistent views).

    The world frame is camera 0's (OpenCV axes); the plane is ~3 units away. Returns `images` uint8 `[N, H, W, 3]`,
    `intrinsics` `[N, 3, 3]`, `cam_to_world` `[N, 4, 4]`, `depth` (ground truth z) `[N, H, W]` float32 and
    `depth_range` `[2]` (near, far) float32.
    """
    target = np.array([0.0, 0.0, 3.0])
    eyes = [np.zeros(3)] + [
        np.array([0.35 * np.cos(a), 0.2 * np.sin(a), 0.1 * np.sin(2 * a)])
        for a in np.linspace(0, 2 * np.pi, n - 1, endpoint=False)
    ]
    c2ws = [_look_at(e, target) for e in eyes]
    c0 = np.linalg.inv(c2ws[0])  # camera 0 is the world frame
    c2ws = [c0 @ c for c in c2ws]
    k = np.array([[0.9 * w, 0, (w - 1) / 2], [0, 0.9 * w, (h - 1) / 2], [0, 0, 1]])
    normal = np.array([0.25, -0.15, -1.0]) / np.linalg.norm([0.25, -0.15, -1.0])
    e1 = np.cross(normal, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)
    origin = target - 3.0 * e1 - 3.0 * e2  # the textured square spans 6 x 6 units around the target
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    rays_cam = np.stack([xs, ys, np.ones_like(xs)], -1) @ np.linalg.inv(k).T  # z = 1
    images, depths = [], []
    for c2w in c2ws:
        rays = rays_cam @ c2w[:3, :3].T
        lam = ((target - c2w[:3, 3]) @ normal) / (rays @ normal)  # ray parameter; = z-depth since rays_cam z = 1
        pts = c2w[:3, 3] + rays * lam[..., None]
        s, t = ((pts - origin) @ e1) / 6, ((pts - origin) @ e2) / 6
        images.append((_texture(s, t, np.random.default_rng(seed)) * 255).round().astype(np.uint8))
        depths.append(lam)
    depth = np.stack(depths).astype(np.float32)
    near, far = float(np.min(depth)) * 0.8, float(np.max(depth)) * 1.25
    return {
        "images": np.stack(images),
        "intrinsics": np.repeat(k[None], n, 0).astype(np.float32),
        "cam_to_world": np.stack(c2ws).astype(np.float32),
        "depth": depth,
        "depth_range": np.array([near, far], np.float32),
    }
