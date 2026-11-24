"""Generate the example ComfyUI workflows (UI format) next to this file.

Usage: python make_workflows.py OBJECT_INFO_JSON [OUT_DIR ...]
OBJECT_INFO_JSON is the output of `curl localhost:8188/object_info`; it supplies
widget order and slot names so the files match the installed nodes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from types import FunctionType

PRIMS = {"STRING", "INT", "FLOAT", "BOOLEAN", "COLORCODE", "COMBO"}
OI: dict[str, Any] = {}  # ComfyUI's /object_info, read in `main`
Node = dict[str, Any]
Src = tuple[Node, int]  # (source node, output slot)


class WF:
    """One workflow in ComfyUI's UI format: nodes and links."""

    def __init__(self) -> None:
        """Start an empty workflow."""
        self.nodes: list[Node] = []
        self.links: list[list[Any]] = []
        self.nid = self.lid = 0

    def add(
        self, typ: str, x: int, y: int, *, title: str | None = None, size: tuple[int, int] = (300, 200), **vals: object
    ) -> Node:
        """Add a node. vals: widget overrides by name, link inputs as (node, slot)."""
        self.nid += 1
        spec = OI.get(typ)
        node = {
            "id": self.nid,
            "type": typ,
            "pos": [x, y],
            "size": list(size),
            "flags": {},
            "order": self.nid,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {},
        }
        if title:
            node["title"] = title
        widgets = []
        if spec:
            ins = {**spec["input"]["required"], **spec["input"].get("optional", {})}
            for name, t in ins.items():
                kind = t[0]
                forced = len(t) > 1 and t[1].get("forceInput")
                if not forced and (isinstance(kind, list) or kind in PRIMS):
                    default = t[1].get("default") if len(t) > 1 else None
                    if isinstance(kind, list) and default is None:
                        default = kind[0]
                    if kind == "COMBO" and default is None:
                        default = (t[1].get("options") or [None])[0]
                    widgets.append(vals.pop(name, default))
                else:
                    node["inputs"].append({"name": name, "type": kind, "link": None})
            for i, (n, ty) in enumerate(zip(spec["output_name"], spec["output"], strict=True)):
                node["outputs"].append({"name": n, "type": ty, "links": [], "slot_index": i})
        node["widgets_values"] = vals.pop("_widgets", widgets)
        for name, src in vals.items():  # remaining = link inputs
            self.link(cast("Src", src), node, name)
        self.nodes.append(node)
        return node

    def link(self, src: Src, node: Node, name: str) -> None:
        """Connect output `src` to the input `name` of `node`."""
        s_node, s_slot = src
        self.lid += 1
        slot = [i["name"] for i in node["inputs"]].index(name)
        node["inputs"][slot]["link"] = self.lid
        s_node["outputs"][s_slot]["links"].append(self.lid)
        self.links.append([self.lid, s_node["id"], s_slot, node["id"], slot, node["inputs"][slot]["type"]])

    def save(self, path: Path) -> None:
        """Write the workflow JSON."""
        data = {
            "last_node_id": self.nid,
            "last_link_id": self.lid,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }
        Path(path).write_text(json.dumps(data, indent=1), newline="\n")  # LF on Windows too


def note(g: WF, text: str, x: int = 0, y: int = -260) -> None:
    """A note node above the graph."""
    g.add("Note", x, y, size=(520, 200), _widgets=[text])


def load(g: WF, img: str, x: int, y: int) -> Node:
    """A LoadImage node for `img` (in ComfyUI's input folder)."""
    return g.add("LoadImage", x, y, size=(300, 340), _widgets=[img, "image"])


def preview(  # noqa: PLR0913 - position, output slot and kind of the previewed output
    g: WF, src: Node, x: int, y: int, *, slot: int = 0, mask: bool = False
) -> Node:
    """Preview output `slot` of `src` (a MASK through MaskToImage)."""
    if mask:
        m = g.add("MaskToImage", x, y, size=(200, 30), mask=(src, slot))
        src, slot = m, 0
        x += 230
    return g.add("PreviewImage", x, y, size=(300, 300), images=(src, slot))


def shootout(  # noqa: PLR0913 - layout knobs of the one helper all shoot-outs share
    g: WF,
    node_type: str,
    img_input: str,
    branches: list[dict[str, Any]],
    *,
    img: Node,
    mask_out: bool = False,
    img_slot: int = 0,
    out_slot: int = 0,
    step: int = 420,
    **common: object,
) -> None:
    """One loader feeding N Fastforks nodes (different model/backend), each with a preview."""
    for k, br in enumerate(branches):
        y = k * step
        vals = {**common, **br, img_input: (img, img_slot)}
        n = g.add(node_type, 380, y, size=(330, 230), title=f"{br['model']} ({br.get('backend', 'auto')})", **vals)
        preview(g, n, 740, y, slot=out_slot, mask=mask_out)


WORKFLOWS: dict[str, FunctionType] = {}


def other(  # noqa: PLR0913, PLR0917 - position, title, input wiring and widget values of one foreign node
    g: WF,
    typ: str,
    x: int,
    y: int,
    title: str,
    src: Src,
    *,
    link: str = "image",
    out: int = 0,
    mask: bool = False,
    extra: dict[str, Src] | None = None,
    **vals: object,
) -> Node:
    """A node of another ComfyUI pack (installed separately) as one more competing branch, with its preview."""
    n = g.add(typ, x, y, size=(330, 260), title=title, **{link: src}, **vals)
    for name, s in (extra or {}).items():
        g.link(s, n, name)
    return preview(g, n, x + 360, y, slot=out, mask=mask)


def wf(fn: FunctionType) -> FunctionType:
    """Register a workflow builder under its function name."""
    WORKFLOWS[fn.__name__] = fn
    return fn


@wf
def w01_depth_shootout() -> WF:
    """Example 01: depth shootout."""
    g = WF()
    note(
        g,
        "Depth: four fastforks models on one image, plus two from the comfyui_controlnet_aux pack (right column). "
        "Change `model` in any node (32 depth models) or "
        "right-click > Bypass a branch. Mute branches you do not want to run.",
    )
    img = load(g, "example.png", 0, 0)
    shootout(
        g,
        "FastforksDepth",
        "image",
        [
            {"model": "depth-anything-v2-s"},
            {"model": "distill-any-depth-s"},
            {"model": "da3mono-large"},
            {"model": "zoedepth-nk"},
        ],
        img=img,
    )
    # the same task in other packs (comfyui_controlnet_aux)
    other(
        g,
        "DepthAnythingV2Preprocessor",
        1160,
        0,
        "controlnet_aux: Depth Anything V2 vits",
        (img, 0),
        ckpt_name="depth_anything_v2_vits.pth",
        resolution=518,
    )
    other(g, "MiDaS-DepthMapPreprocessor", 1160, 420, "controlnet_aux: MiDaS", (img, 0), resolution=512)
    return g


@wf
def w02_backend_compare() -> WF:
    """Example 02: backend compare."""
    g = WF()
    note(
        g,
        "Same model, five backends: auto (what fastforks picks on this Mac), portable "
        "(plain PyTorch on MPS), kernels (Metal kernels where there are any), mlx and mlx portable (Apple MLX). "
        "Results should match; time and memory differ. "
        "Run the queue twice: the first run includes loading and compilation.",
    )
    img = load(g, "example.png", 0, 0)
    shootout(
        g,
        "FastforksDepth",
        "image",
        [
            {"model": "depth-anything-v2-l [non-commercial]", "backend": b}
            for b in ("auto", "portable", "kernels", "mlx", "mlx portable")
        ],
        img=img,
        dtype="float16",
    )
    return g


@wf
def w03_background_removal() -> WF:
    """Example 03: background removal."""
    g = WF()
    note(g, "Background removal: five matting/saliency models. The IMAGE output is the cut-out (RGBA).")
    img = load(g, "example.png", 0, 0)
    shootout(
        g,
        "FastforksRemoveBackground",
        "image",
        [{"model": m} for m in ("birefnet-lite", "isnet-general-use", "inspyrenet-fast", "u2net", "ben2")],
        img=img,
        step=380,
    )
    # the same task in the ComfyUI-RMBG pack
    other(g, "RMBG", 1160, 0, "ComfyUI-RMBG: INSPYRENET", (img, 0), model="INSPYRENET")
    other(g, "RMBG", 1160, 380, "ComfyUI-RMBG: BEN2", (img, 0), model="BEN2")
    return g


@wf
def w04_segment_prompted() -> WF:
    """Example 04: segment prompted."""
    g = WF()
    note(
        g,
        "Promptable segmentation: a click at the image centre (positive_points = 384,384). "
        "Edit the point in pixels, add a box 'x0,y0,x1,y1' or negative points. "
        "The click lands on the dress only because the image centre happens to be there: "
        "move it to pick another object. "
        "Only the two bottom nodes (SAM 3, CLIPSeg) take a text prompt: edit `text` there ('pink dress', 'grass'); "
        "the point-based SAM nodes error on text, leave it empty. "
        "Compare SAM 2.1, MobileSAM and EfficientViT-SAM.",
    )
    img = load(g, "example.png", 0, 0)
    shootout(
        g,
        "FastforksSegment",
        "image",
        [{"model": m} for m in ("sam2.1-hiera-small", "mobile-sam", "efficientvit-sam-l0", "sam-vit-b")],
        img=img,
        mask_out=True,
        positive_points="384,384",
    )
    # text prompt: no points, the `text` field says what to segment (SAM 3 and CLIPSeg understand it)
    for k, m in enumerate(("sam3", "clipseg")):
        y = 1700 + k * 420
        n = g.add(
            "FastforksSegment",
            380,
            y,
            size=(330, 230),
            title=f"{m}: text prompt",
            model=m,
            image=(img, 0),
            text="pink dress",
        )
        preview(g, n, 740, y, mask=True)
    # SAM 2.1 small in ComfyUI-segment-anything-2 (kijai)
    sam = g.add(
        "DownloadAndLoadSAM2Model",
        1160,
        0,
        size=(300, 160),
        model="sam2.1_hiera_small.safetensors",
        segmentor="single_image",
        device="mps",
        precision="fp32",
    )
    pt = g.add("PrimitiveString", 1160, 700, size=(300, 80), title="click point (JSON)", value='[{"x": 384, "y": 384}]')
    other(
        g,
        "Sam2Segmentation",
        1160,
        220,
        "segment-anything-2: SAM 2.1 small",
        (img, 0),
        mask=True,
        extra={"sam2_model": (sam, 0), "coordinates_positive": (pt, 0)},
        keep_model_loaded=False,
    )
    return g


@wf
def w05_detection() -> WF:
    """Example 05: detection."""
    g = WF()
    note(
        g,
        "Object detection: DETR-family, YOLOX and RF-DETR. Output is a mask of the boxes and a "
        "text list. Set `classes` (e.g. 'person, car') to filter, raise score_threshold to cut noise. "
        "For open-vocabulary detection swap in grounding-dino-tiny and type what to find in `classes`.",
    )
    img = load(g, "scene.png", 0, 0)
    for k, m in enumerate(["rf-detr-s-coco", "yolox-s", "rtdetrv2-s-coco", "dfine-s-coco"]):
        y = k * 420
        n = g.add("FastforksDetect", 380, y, size=(330, 260), title=m, model=m, image=(img, 0))
        preview(g, n, 740, y, slot=0, mask=True)
        g.add("PreviewAny", 1260, y, size=(300, 200), source=(n, 1))
    return g


@wf
def w06_pose() -> WF:
    """Example 06: pose."""
    g = WF()
    note(
        g,
        "Pose estimation (skeleton overlay). Top-down models (rtmpose/vitpose/dwpose) need person "
        "boxes, the one-stage rtmo-m finds people itself (the detector node on the left feeds the others). "
        "Switch `model` to rtmw-x/dwpose-l for whole body.",
    )
    img = load(g, "scene.png", 0, 0)
    det = g.add(
        "FastforksDetect",
        0,
        400,
        size=(330, 260),
        title="person detector (feeds top-down models)",
        model="rtmdet-m-person",
        image=(img, 0),
    )
    for k, m in enumerate(["rtmo-m", "rtmpose-m", "vitpose-plus-small", "dwpose-m"]):
        y = k * 420
        vals = {"model": m, "image": (img, 0)}
        if m != "rtmo-m":
            vals["detections"] = (det, 1)
        n = g.add("FastforksPose", 380, y, size=(330, 260), title=m, **vals)
        preview(g, n, 740, y)
    other(g, "DWPreprocessor", 1160, 0, "controlnet_aux: DWPose", (img, 0))
    other(g, "OpenposePreprocessor", 1160, 420, "controlnet_aux: OpenPose", (img, 0))
    return g


@wf
def w07_upscale() -> WF:
    """Example 07: upscale."""
    g = WF()
    note(
        g,
        "4x upscaling of a small image: GAN (Real-ESRGAN), light CNN (SPAN), transformer (SwinIR). "
        "tile limits memory; 0 runs one pass.",
    )
    img = load(g, "small.png", 0, 0)
    shootout(
        g,
        "FastforksUpscale",
        "image",
        [{"model": m} for m in ("realesrgan-x4plus", "realesr-general-x4v3", "span-x4", "swinir-m-x4-real")],
        img=img,
        step=480,
    )
    # ComfyUI's own upscale-model nodes with the same Real-ESRGAN checkpoint (models/upscale_models)
    up = g.add("UpscaleModelLoader", 1160, 0, size=(300, 60), model_name="RealESRGAN_x4plus.pth")
    other(
        g,
        "ImageUpscaleWithModel",
        1160,
        120,
        "ComfyUI built-in: RealESRGAN_x4plus",
        (img, 0),
        extra={"upscale_model": (up, 0)},
    )
    return g


@wf
def w08_interpolation() -> WF:
    """Example 08: interpolation."""
    g = WF()
    note(
        g,
        "Frame interpolation: four frames are batched, each model doubles the frame count "
        "(multiplier 2; raise it for slow motion). The preview shows the frames; use Save Image/Video nodes "
        "to export.",
    )
    ys = [0, 360, 720, 1080]
    imgs = [load(g, f"clip{i}.png", 0, y) for i, y in enumerate(ys)]
    b = g.add("ImageBatch", 330, 0, size=(200, 50), image1=(imgs[0], 0), image2=(imgs[1], 0))
    b = g.add("ImageBatch", 330, 120, size=(200, 50), image1=(b, 0), image2=(imgs[2], 0))
    b = g.add("ImageBatch", 330, 240, size=(200, 50), image1=(b, 0), image2=(imgs[3], 0))
    for k, m in enumerate(["rife-4.26", "film", "ema-vfi-small", "rife-4.25-lite"]):
        y = k * 520
        n = g.add("FastforksInterpolate", 600, y, size=(330, 260), title=m, model=m, frames=(b, 0))
        preview(g, n, 960, y)
    other(g, "RIFE VFI", 1330, 0, "Frame-Interpolation: RIFE 4.9", (b, 0), link="frames", ckpt_name="rife49.pth")
    other(g, "FILM VFI", 1330, 520, "Frame-Interpolation: FILM", (b, 0), link="frames", ckpt_name="film_net_fp32.pt")
    return g


@wf
def w09_stereo() -> WF:
    """Example 09: stereo."""
    g = WF()
    note(
        g,
        "Stereo depth from a left/right pair: classic cost-volume (IGEV, ACVNet) vs recurrent (RAFT-Stereo) "
        "vs transformer (S2M2). Output is a disparity visualisation.",
    )
    left = load(g, "stereo_left.png", 0, 0)
    right = load(g, "stereo_right.png", 0, 400)
    for k, m in enumerate(
        ["raft-stereo-middlebury", "igev-stereo-middlebury", "fast-acvnet-plus-generalization", "waft-stereo-s"]
    ):
        y = k * 420
        n = g.add("FastforksStereo", 380, y, size=(330, 260), title=m, model=m, left=(left, 0), right=(right, 0))
        preview(g, n, 740, y)
    return g


@wf
def w10_optical_flow() -> WF:
    """Example 10: optical flow."""
    g = WF()
    note(
        g,
        "Optical flow between neighbouring frames (colour-coded) and, bottom, a warp: frame 1 is pushed "
        "along the flow of the chosen model, so a good flow gives a copy of frame 2. "
        "Compare RAFT, SEA-RAFT, GMFlow and WAFT.",
    )
    imgs = [load(g, f"clip{i}.png", 0, i * 360) for i in range(3)]
    b = g.add("ImageBatch", 330, 0, size=(200, 50), image1=(imgs[0], 0), image2=(imgs[1], 0))
    b = g.add("ImageBatch", 330, 120, size=(200, 50), image1=(b, 0), image2=(imgs[2], 0))
    for k, m in enumerate(["raft-small", "sea-raft-s", "gmflow-scale1-mixdata", "waft-a1"]):
        y = k * 360
        n = g.add("FastforksFlow", 600, y, size=(330, 260), title=m, model=m, frames=(b, 0))
        preview(g, n, 960, y)
        if k == 0:
            w = g.add("FastforksWarp", 1300, y, size=(200, 50), image=(b, 0), flow=(n, 1))
            preview(g, w, 1540, y)
    other(
        g,
        "Unimatch_OptFlowPreprocessor",
        1330,
        720,
        "controlnet_aux: GMFlow",
        (b, 0),
        out=1,
        ckpt_name="gmflow-scale1-mixdata.pth",
    )
    return g


@wf
def w11_edges() -> WF:
    """Example 11: line art and edges."""
    g = WF()
    note(
        g,
        "Line art and edge maps (white on black), the hints of lineart, soft-edge and M-LSD ControlNets: "
        "lineart, anime line art, AnyLine, HED, TEED (soft edges) and M-LSD (straight line segments).",
    )
    img = load(g, "example.png", 0, 0)
    shootout(
        g,
        "FastforksEdges",
        "image",
        [{"model": m} for m in ("lineart", "lineart-anime", "anyline", "hed", "teed", "mlsd")],
        img=img,
        step=380,
    )
    return g


@wf
def w12_inpaint() -> WF:
    """Example 12: inpainting and object removal."""
    g = WF()
    note(
        g,
        "Inpainting with LaMa. Top: the object under a click (Segment, SAM 2.1) is removed; the mask is grown a "
        "little so no outline is left. Bottom: a fixed 200 x 200 square at the centre is filled. To paint your own "
        "mask, open the image in the mask editor (right-click > Open in MaskEditor) and connect the MASK output of "
        "Load Image to `mask` instead.",
    )
    img = load(g, "example.png", 0, 0)
    seg = g.add(
        "FastforksSegment",
        380,
        0,
        size=(330, 230),
        title="object under the click",
        image=(img, 0),
        positive_points="384,384",
    )
    grow = g.add("GrowMask", 740, 0, size=(240, 80), mask=(seg, 0), expand=12)
    n = g.add("FastforksInpaint", 1010, 0, size=(330, 230), title="remove the object", image=(img, 0), mask=(grow, 0))
    preview(g, grow, 1010, 300, mask=True)
    preview(g, n, 1370, 0)
    base = g.add("SolidMask", 380, 640, size=(240, 100), value=0.0, width=768, height=768)
    box = g.add("SolidMask", 380, 780, size=(240, 100), value=1.0, width=200, height=200)
    sq = g.add(
        "MaskComposite",
        740,
        640,
        size=(240, 130),
        destination=(base, 0),
        source=(box, 0),
        _widgets=[284, 284, "add"],  # x, y (they would clash with the node position), operation
    )
    n = g.add("FastforksInpaint", 1010, 640, size=(330, 230), title="fill a square", image=(img, 0), mask=(sq, 0))
    preview(g, n, 1370, 640)
    return g


@wf
def w13_segment_like() -> WF:
    """Example 13: segment like a reference."""
    g = WF()
    note(
        g,
        "Segment like a reference (INSID3): one example image with a mask of a concept (here the object under a "
        "click on example.png, from Segment) finds the same concept in other images, no prompt per image. "
        "Batch more reference images and masks (one mask each) for a broader concept.",
    )
    ref = load(g, "example.png", 0, 0)
    seg = g.add(
        "FastforksSegment",
        380,
        0,
        size=(330, 230),
        title="reference mask (click)",
        image=(ref, 0),
        positive_points="384,384",
    )
    preview(g, seg, 740, 0, mask=True)
    imgs = [load(g, f"clip{i}.png", 0, 420 + i * 360) for i in range(3)]
    b = g.add("ImageBatch", 380, 420, size=(200, 50), image1=(imgs[0], 0), image2=(imgs[1], 0))
    b = g.add("ImageBatch", 380, 540, size=(200, 50), image1=(b, 0), image2=(imgs[2], 0))
    n = g.add(
        "FastforksSegmentLike",
        620,
        420,
        size=(330, 230),
        image=(b, 0),
        reference_image=(ref, 0),
        reference_mask=(seg, 0),
    )
    preview(g, n, 980, 420, mask=True)
    return g


@wf
def w14_video_matte() -> WF:
    """Example 14: video matting."""
    g = WF()
    note(
        g,
        "Video matting (MatAnyone, non-commercial weights): a mask of the first frame (here the object under a "
        "click, from Segment; move the click to pick another object) is followed through the clip, giving an alpha "
        "per frame. Only the first mask of a batch is used. Frames are read one at a time, so long clips fit.",
    )
    imgs = [load(g, f"clip{i}.png", 0, i * 360) for i in range(4)]
    b = g.add("ImageBatch", 330, 0, size=(200, 50), image1=(imgs[0], 0), image2=(imgs[1], 0))
    b = g.add("ImageBatch", 330, 120, size=(200, 50), image1=(b, 0), image2=(imgs[2], 0))
    b = g.add("ImageBatch", 330, 240, size=(200, 50), image1=(b, 0), image2=(imgs[3], 0))
    first = g.add(
        "FastforksSegment",
        600,
        0,
        size=(330, 230),
        title="first-frame mask (click)",
        image=(imgs[0], 0),
        positive_points="384,384",
    )
    n = g.add("FastforksVideoMatte", 960, 0, size=(330, 230), frames=(b, 0), mask=(first, 0))
    preview(g, n, 1320, 0, slot=1, mask=True)
    return g


def main() -> None:
    """Read the object info, write every workflow to each output folder."""
    OI.update(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    outs = [Path(p) for p in sys.argv[2:]] or [Path(__file__).parent]
    for name, fn in WORKFLOWS.items():
        for o in outs:
            o.mkdir(parents=True, exist_ok=True)
            fn().save(o / f"fastforks-{name}.json")
    print(len(WORKFLOWS), "workflows ->", *outs)


if __name__ == "__main__":
    main()
