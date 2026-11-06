# Stereo geometry

`fastforks.stereo.geometry` turns a calibrated camera pair into the rectified image pair a stereo model expects, and
its disparity back into depth and point clouds. It needs numpy and torch only; OpenCV is used only as a test oracle.

```python
from fastforks.stereo import geometry as g

rig = g.read_calib("calib_cam_to_cam.txt", "kitti-unrectified")  # a StereoRig
rect = g.rectify(rig)  # rotations, rectified intrinsics, Q; no maps yet
left_r, right_r = rect.remap(left, right)  # builds the maps once (per device), then reuses them
disparity = ...  # any fastforks stereo model on (left_r, right_r)
depth = rect.depth(disparity)  # metres, rectified left view
cloud = rect.point_cloud(disparity, colors=left_r, frame="camera")  # cloud.points, cloud.colors
depth_left = rect.unrectify(depth)  # depth in the original left image
```

Already rectified data (KITTI `P_rect`, Middlebury, ETH3D) needs no remapping:

```python
rig = g.read_calib("calib.txt")  # Middlebury: a RectifiedRig(f, baseline, cx, cy, doffs)
cloud = rig.point_cloud(disparity, colors=left)
rig = g.RectifiedRig(f=721.5, cx=609.6, cy=172.9, baseline=0.54)  # or by hand
```

With `doffs > 0` (Middlebury), points farther than `f * baseline / doffs` need a negative disparity, which the networks do
not predict: their depth is plausible but wrong. Rectify a `StereoRig` instead (`rectify` gives both images one
principal point, `doffs = 0`).

## End-to-end example

A raw (unrectified) KITTI pair and its `calib_cam_to_cam.txt` to a coloured point cloud in the original left camera
frame and a depth map at the raw left image's pixels. The images stay on the model's device from the remap to the
cloud; `tests/test_geometry_example.py` runs this block on rendered images with a stand-in model. `calib_cam_to_cam.txt`
is KITTI raw's calibration file; the images must have its `S_xx` size (else `InputError` names both sizes).

```python
import numpy as np
import torch
from PIL import Image

from fastforks import stereo
from fastforks.stereo import geometry as g

model = stereo.load("igev-stereo-sceneflow")  # on the GPU if there is one
rig = g.read_calib("calib_cam_to_cam.txt", "kitti-unrectified")  # both cameras with their distortion, R, T
rect = g.rectify(rig)  # rotations, rectified intrinsics, Q; the maps are built on first use


def raw(path):  # [3, H, W] uint8 on the model's device
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1).to(model.device)


left_r, right_r = rect.remap(raw("left.png"), raw("right.png"))  # rectified on that device, maps cached there
disparity = model(left_r, right_r).disparity  # [H, W] pixels, rectified left view
cloud = rect.point_cloud(disparity, colors=left_r.permute(1, 2, 0), frame="camera")  # original left camera frame
points, colors = cloud.points[cloud.valid].cpu().numpy(), cloud.colors[cloud.valid].cpu().numpy()  # [N, 3] each
depth_left = rect.unrectify(rect.depth(disparity))  # [h, w] metres at the raw left pixels, NaN where unknown
```

## Camera models

One frozen dataclass per model; all satisfy the `Camera` protocol (`project(points) -> (pixels, valid)`,
`unproject(pixels) -> (rays, valid)`, numpy or torch in, the same type out, float64). `g.camera(model_id, w, h, fx,
fy, cx, cy, *coefficients)` builds one from a model id and its coefficients in file order.

| class | id | coefficients | checked against |
|---|---|---|---|
| `Pinhole` | `pinhole` | none | closed form |
| `PinholeOpenCV` | `pinhole-opencv` | 4, 5, 8, 12 or 14 (OpenCV order) | `cv2.projectPoints` / `undistortPoints`, 1e-9 px |
| `KB4` | `kb4` (`fisheye-opencv`, `kb3` / `fisheye-kb3`, `fisheye-equidistant`) | `k1..k4` (kb3: `k0..k3` or 3 terms; equidistant: none) | `cv2.fisheye`, 1e-9 px; Rust |
| `Fisheye624` | `fisheye624` | 12 (Aria / Project Aria) | Rust `camera-models`, 1e-8 px |
| `ThinPrismFisheye` | `thin-prism-fisheye` | 8, COLMAP `THIN_PRISM_FISHEYE` (ETH3D DSLR) | COLMAP formula |
| `DoubleSphere` | `double-sphere` (`ds`) | `xi alpha` | paper formula, Rust |
| `UCM` | `ucm` (`omni`) | `xi` | paper formula, Rust |
| `EUCM` | `eucm` | `alpha beta` | paper formula, Rust |
| `FOV` | `fov` | `w` | paper formula, Rust |
| `E57Pinhole` | `e57-pinhole` | none | Rust |
| `E57Spherical` | `e57-spherical` | none | Rust |
| `E57Cylindrical` | `e57-cylindrical` | none | Rust |

The E57 models are the image representations of ASTM E2807 (`.e57` laser scans): pinhole images looking along -z
with y down, and spherical (azimuth / elevation) and cylindrical panoramas around +z. They project and unproject like
the others (e.g. to colour a scan or turn a panorama into rays), but `rectify` rejects them. Reading `.e57` files
themselves (a binary point cloud container) is out of scope.

The tests compare with reference values from a Rust implementation (`tools/geometry_goldens/ff_goldens.rs`, regeneration
steps in its header), stored in `tests/data/geometry_goldens.json`, so the tests need no Rust toolchain.
`pinhole-opencv` marks rays past the distortion fold (where the radial polynomial turns back, e.g. 44 degrees off
axis for k1 = -0.35) invalid in both directions, so they cannot alias pixels nearer the centre; OpenCV and the Rust
reference return them.

## Rectification

`rectify(rig, *, size=None, focal=None, alpha=None, fov_max=120.0)` works the same way for every camera model:

- `R1`, `R2` follow Bouguet's method (as `cv2.stereoRectify`): each camera is rotated half way, then both onto the
  baseline.
- `K_new` is one focal length and principal point for both views. Default: the left camera's `fx` at the rectified
  size and the image centre, the same as `model(left, right, rig=rig)`; `focal=` sets the focal length. `alpha=`
  instead picks `K_new` as OpenCV does (0: only valid pixels, 1: all source pixels) from unprojected samples of the
  image borders, as are the valid-pixel rectangles `roi_left` / `roi_right`.
- Each rectified pixel `p` maps to the source pixel `cam.project(R_i^T K_new^-1 p)`.
- Wide-angle and fisheye images are rectified onto a plane. Border rays beyond `fov_max` (full angle, degrees, < 180)
  are clamped to it, so a 200 degree lens gives a finite image of the central `fov_max` degrees.
- Vertical rigs and swapped cameras raise `InputError`.

`Rectification` holds `R1 R2 P1 P2 Q roi_left roi_right` (OpenCV conventions) and builds the maps lazily per device.
`maps(device)` returns them as `[2, H, W, 2]` source pixel coordinates (-1 where invalid) plus the valid mask.
`remap(left, right, *, device=None, return_valid=False)` warps both images in one `grid_sample` call. It accepts HW,
HWC or BHWC numpy and HW, CHW or BCHW torch, keeps the dtype (integers rounded) and fills invalid pixels with 0.
`depth` and `point_cloud` mark rectified pixels without a source pixel (the black border with `alpha > 0`) invalid,
so disparities a network predicts there do not end up in the cloud; `unrectify` ignores them too.

### Agreement with OpenCV

| what | tolerance |
|---|---|
| `R1`, `R2` vs `cv2.stereoRectify` / `cv2.fisheye.stereoRectify` | 1e-12 |
| `P1`, `P2`, `Q`, mild distortion (k1 = -0.04) | 1e-6 px |
| `P1`, `P2`, `Q`, strong distortion (k1 = -0.28), alpha 0 / 0.5 / 1 | 0.05 / 0.25 / 0.4 px |
| maps vs `cv2.initUndistortRectifyMap` with the same `R`, `P` | 1e-3 px |
| `remap` uint8 / uint16 vs `cv2.remap` `INTER_LINEAR` | 1 LSB |

The gap for strong distortion comes from OpenCV: `stereoRectify` undistorts the border points with only 5
fixed-point iterations, which have not converged at k1 = -0.28. fastforks iterates to convergence; the rectified
images differ by at most a fraction of a pixel in scale. `cv2.fisheye.stereoRectify` picks `K_new` with its own
`balance` rule, so only the rotations and the maps (given our `P`) are compared there.

## Speed and memory

Both images of a pair, RGB uint8, strong distortion, alpha 0; RTX 5060 Ti, Ryzen 7 5800X with 4 threads, median of 10
(CPU) / 20 (GPU) runs (`tools/bench_geometry.py`):

| size | OpenCV CPU remap | torch CPU remap | torch GPU remap (tensors on GPU / numpy in and out) | GPU peak allocated |
|---|---|---|---|---|
| 720x1280 | 1.9 ms | 29 ms | 0.39 / 1.6 ms | 71 MB |
| 1080x1920 | 6.3 ms | 64 ms | 1.1 / 6.3 ms | 156 MB |
| 3000x4000 | 36 ms | 428 ms | 8.1 / 45 ms | 894 MB |

The first `remap` on a device also builds the maps (float32, every camera model): 86 / 165 / 819 ms on the CPU, against
3 / 7 / 50 ms for `cv2.initUndistortRectifyMap`, and 0.24-0.32 s on the GPU, of which all but 9 / 14 / 65 ms is CUDA
loading its kernels on first use (a second rectification on a warm GPU builds in that time). The first `unrectify` builds
its inverse map with the iterative undistortion: 0.2-0.4 s on the GPU, 0.6 / 1.5 / 7.4 s on the CPU. Without a GPU and
with OpenCV installed, `cv2.remap` with the maps from `rect.maps()` is about 10x faster than the torch CPU path.

## Calibration files

`read_calib(path, format=None, cameras=None, size=None, unit=None, right=None, names=None)` detects the format from the content:

| format | files | result |
|---|---|---|
| `kitti` | `calib_cam_to_cam.txt` (raw, stereo 2012/2015), `P_rect_xx` | `RectifiedRig` |
| `kitti-unrectified` | the same file, `K_xx D_xx R_xx T_xx` | `StereoRig` |
| `kitti-odometry` | odometry / object `calib.txt`, `P0..P3` | `RectifiedRig` |
| `middlebury` | Middlebury 2014/2021 and ETH3D `calib.txt` (baseline in mm, `doffs`) | `RectifiedRig` |
| `opencv` | `cv2.FileStorage` YAML / XML of `stereoCalibrate` results (pinhole or fisheye) | `StereoRig` |
| `kalibr` | Kalibr `camchain.yaml` (pinhole with radtan / equidistant / fov, omni, ds, eucm; omni + radtan later) | `StereoRig` |
| `tum` | TUM VI `camchain.yaml` (the Kalibr format; same reader as `kalibr`) | `StereoRig` |
| `oak` | OAK-D / DepthAI `eepromToJson` calibration (`cameraData`, cm; sockets 1 and 2, `cameras=` picks others) | `StereoRig` |
| `realsense` | RealSense profile JSON: two streams with `rs2_intrinsics`, one `rs2_extrinsics` (column-major `R`); `names=` picks streams | `StereoRig` |
| `ros` | ROS `camera_info` YAML of the left camera, the right one as `right=`; `R = R_r^T R_l`, `T = R_r^T (P_r[0,3] / f, 0, 0)` | `StereoRig` |
| `colmap` | sparse model folder or `cameras.txt` / `.bin` with `images`; two images (or `names=`); `X_r = R_r R_l^T (X_l - t_l) + t_r`; `cx, cy` read as `c - 0.5`; every COLMAP camera model | `StereoRig` |
| `euroc` | `cam0/sensor.yaml` (`cam1` next to it or `right=`); `T_BS` inverted to the left-to-right pose | `StereoRig` |
| `zed` | Stereolabs ZED factory calibration `SN<serial>.conf`; `size=` picks the resolution section | `StereoRig` |
| `aria` | Project Aria device calibration JSON or MPS `online_calibration.jsonl` (first line): the SLAM pair by default (Gen1 `camera-slam-*`, Gen2 `slam-front-*`), `names=` picks other labels; `FisheyeRadTanThinPrism` as `Fisheye624`, `T_Device_Camera` poses; image sizes as `projectaria_tools` gives them (`size=` overrides; VRS RGB / ET streams are smaller than the calibration) | `StereoRig` |
| `nerfstudio` | nerfstudio / instant-ngp `transforms.json`: two frames by `names=` (`file_path` or file name), `PINHOLE` / `OPENCV` / `OPENCV_FISHEYE`, shared or per-frame intrinsics, OpenGL camera-to-world poses; scale arbitrary (`unit=` a number, the factor to metres); intrinsics are for the stored `w x h` (`rig.resized` for `images_4`) | `StereoRig` |

Results are in metres. KITTI baselines use the exact `t_x = (P[0, 3] - cx P[2, 3]) / f` of each projection (the
`P[0, 3]` difference alone is 0.2 % short on odometry sequence 00). Kalibr `cameras=(1, 0)` reads cam1 as the left
camera. KITTI files without camera 2 (KITTI-360 `perspective.txt`) default to `cameras=(0, 1)`. ZED files hold
every resolution (2K, FHD, FHD1200, HD, SVGA, VGA); the rotation is `Rodrigues(RX, CV, RZ)` of the chosen resolution
and the translation `(-Baseline, TY, TZ)` in mm, as in Stereolabs' OpenCV samples. Rotations must be orthonormal to 1e-4 (files round to about 1e-6); they are snapped to the nearest exact
rotation. YAML is read with a small built-in parser (no PyYAML); anchors, aliases and flow mappings
raise `InputError`. DSO `camera.txt` (one camera, no pair) is not a stereo format. The OAK and RealSense readers are not verified on a device: the vendors' extrinsics direction ("from" and "to") differs between SDK versions, so a file that puts the right camera on the left of the left one (`T_x > 0`, swapped cameras) raises `InputError` with the fix; a baseline outside 5 mm to 1 m warns about a wrong `unit=`. COLMAP translations are in the model's arbitrary units; `unit=` takes a number, the factor to metres. RealSense `inverse_brown_conrady` is only accepted with zero coefficients.

The Gen1 Aria SLAM cameras sit sideways and face away from each other (`rectify` rejects that pair as vertical); Gen2's front pair is a usable horizontal one.

### Any camera profile with intrinsics and poses

For a dataset without a `read_calib` format, build the two cameras and pass their camera-to-world poses to
`StereoRig.from_poses` (keyword-only poses: world frame and unit are written at the call). `axes="opencv"` (x right,
y down, z forward) is the default, `"opengl"` is nerfstudio / Blender; `unit` is the unit of the translations
(`"m"`, `"cm"`, `"mm"` or a number, the factor to metres); non-rigid matrices (scale, shear, a projection) raise `InputError`. SHOW3D (poses in mm;
`headset0` / `headset1` are a 64 mm stereo pair, frame 0 of a sample scene):

```python
import json, numpy as np
from PIL import Image
from fastforks import stereo
from fastforks.stereo import geometry as g


def show3d(name, frame=0):  # camera + 4x4 T_WorldFromCamera of camera_calibration/<name>.json
    c = json.load(open(f"camera_calibration/{name}.json"))
    t = c["T_WorldFromCamera_by_index"][str(frame)] if "T_WorldFromCamera_by_index" in c else c
    return g.camera("pinhole", c["ImageSizeX"], c["ImageSizeY"], c["fx"], c["fy"], c["cx"], c["cy"]), t[
        "T_WorldFromCamera"
    ]


(left, pose_l), (right, pose_r) = show3d("headset0"), show3d("headset1")
rig = g.StereoRig.from_poses(
    left, right, left_to_world=pose_l, right_to_world=pose_r, unit="mm"
)  # rig.baseline 0.064 m
rect = g.rectify(rig, focal=450)  # the rectified focal length is yours to choose
left_r, right_r = rect.remap(np.array(Image.open("headset0_0.png")), np.array(Image.open("headset1_0.png")))
depth = rect.depth(stereo.load("las2-s")(left_r, right_r).disparity)  # metres, rectified left view
```

The left camera is the one whose right neighbour sits on its +x side; the other order raises in `rectify` ("swap the
two cameras"). Pose lists are not read for you: pass the matrices of the frame you want.

Points from `depth` / `points` are in the left camera's frame; to world coordinates use `p @ R.T + t` with the
rotation `R` and translation `t` of `left_to_world`.

### Rigs with many cameras

A calibration file with many cameras (COLMAP, Kalibr, Aria, nerfstudio) gives one pair per call: pick it with
`read_calib(path, names=("cam_a", "cam_b"))` (or `cameras=(i, j)` by index). Load the model once and loop:

```python
model = stereo.load("las2-s")
for a, b in [("cam0", "cam1"), ("cam1", "cam2")]:
    rig = g.read_calib("rig.yaml", names=(a, b))
    result = model(left_image[a], right_image[b], rig=rig)
```

Keep the unit in mind: COLMAP and nerfstudio poses are in scene units, so depth is too, until you pass `unit=` (a
number, the factor to metres). Pinhole pairs turned more than about 20 degrees against each other, or a size far above
the model's training size, can give disparities beyond the model's range (about 176 px for las2-s) and wrong depth;
`rectify` warns ("poor rectification") when such a turn is large (fisheye and 360 inputs are not limited by the turn)
or when less than 10 % of the rectified image is valid.

Out of scope: calibrating a rig from checkerboard images, and rectification without calibration.

## Rectified size and focal length

Sizes are `(height, width)` everywhere, the order of tensor shapes (calibration files keep their own order and are
converted on read). A size that swaps portrait and landscape against the images or the calibration raises
`InputError`. `rectify(rig, size=..., focal=...)`: the rectified size defaults to the image size and the focal length
to the left camera's `fx` (`focal=` and `alpha=` exclude each other); a `RectifiedRig` without `cx, cy` takes the
image centre `((W - 1) / 2, (H - 1) / 2)`. `rig.resized((h, w))` rescales a calibration to other images with
`c' = (c + 0.5) s - 0.5`. A calibrated call does this for images of another size only if the aspect ratio stays
(one pixel of rounding): a change of aspect ratio (crops or pads) raises `InputError` (shift `cx, cy` for crops; pass
`rig.resized(...)` for an anisotropic resize on purpose). A size change that keeps the aspect ratio is assumed to be a
resize of the whole view, and logs one warning per (calibration, size); a same-aspect crop cannot be told from it by
the image alone, so give such images a calibration of the cropped size (`cx, cy` shifted by the crop offset), and a `RectifiedRig` with a `size` is rescaled the same way (other aspect ratios raise `InputError`). `rect.depth(..., view="right")`, `rect.point_cloud(..., view="right")` and
`rect.unrectify(..., view="right")` use the right camera's rotation and maps; holes are NaN.

Calibrated calls keep the rectification maps of the last 8 (calibration, image size, `size`, `focal`) combinations per model (FIFO; an older one is rebuilt when used again). Each holds GPU maps of the rectified size, so many different rigs in one process cycle through that bound; the GPU memory that the allocator reserved for it stays reserved until `torch.cuda.empty_cache()`.
