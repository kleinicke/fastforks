"""Generate the example ComfyUI workflows (UI format) next to this file.

Usage: python make_workflows.py OBJECT_INFO_JSON [OUT_DIR ...]
OBJECT_INFO_JSON is the output of `curl localhost:8188/object_info`; it supplies
widget order and slot names so the files match the installed nodes.
"""

import json
import sys
from pathlib import Path

PRIMS = {"STRING", "INT", "FLOAT", "BOOLEAN"}
OI = json.load(open(sys.argv[1]))


class WF:
    def __init__(self):
        self.nodes, self.links, self.nid, self.lid = [], [], 0, 0

    def add(self, typ, x, y, w=None, title=None, size=(300, 200), **vals):
        """Add a node. vals: widget overrides by name, link inputs as (node, slot)."""
        self.nid += 1
        spec = OI.get(typ)
        node = {"id": self.nid, "type": typ, "pos": [x, y], "size": list(size), "flags": {},
                "order": self.nid, "mode": 0, "inputs": [], "outputs": [], "properties": {}}
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
                    widgets.append(vals.pop(name, default))
                else:
                    node["inputs"].append({"name": name, "type": kind, "link": None})
            for i, (n, ty) in enumerate(zip(spec["output_name"], spec["output"])):
                node["outputs"].append({"name": n, "type": ty, "links": [], "slot_index": i})
        node["widgets_values"] = vals.pop("_widgets", widgets)
        for name, src in vals.items():  # remaining = link inputs
            self.link(src, node, name)
        self.nodes.append(node)
        return node

    def link(self, src, node, name):
        s_node, s_slot = src
        self.lid += 1
        for i in node["inputs"]:
            if i["name"] == name:
                i["link"] = self.lid
                typ = i["type"]
        s_node["outputs"][s_slot]["links"].append(self.lid)
        self.links.append([self.lid, s_node["id"], s_slot, node["id"],
                           [i["name"] for i in node["inputs"]].index(name), typ])

    def save(self, path):
        data = {"last_node_id": self.nid, "last_link_id": self.lid, "nodes": self.nodes,
                "links": self.links, "groups": [], "config": {}, "extra": {}, "version": 0.4}
        Path(path).write_text(json.dumps(data, indent=1))


def note(g, text, x=0, y=-260):
    g.add("Note", x, y, size=(520, 200), _widgets=[text])


def load(g, img, x, y):
    return g.add("LoadImage", x, y, size=(300, 340), _widgets=[img, "image"])


def preview(g, src, x, y, slot=0, mask=False):
    if mask:
        m = g.add("MaskToImage", x, y, size=(200, 30), mask=(src, slot))
        src, slot = m, 0
        x += 230
    return g.add("PreviewImage", x, y, size=(300, 300), images=(src, slot))


def shootout(g, node_type, img_input, branches, extra=None, img=None, mask_out=None,
             img_slot=0, out_slot=0, step=420, **common):
    """One loader feeding N Fastforks nodes (different model/backend), each with a preview."""
    for k, br in enumerate(branches):
        y = k * step
        vals = {**common, **br, img_input: (img, img_slot)}
        n = g.add(node_type, 380, y, size=(330, 230), title=f"{br['model']} ({br.get('backend', 'auto')})", **vals)
        preview(g, n, 740, y, slot=out_slot, mask=mask_out)
    return


WORKFLOWS = {}


def wf(fn):
    WORKFLOWS[fn.__name__] = fn
    return fn


@wf
def w01_depth_shootout():
    g = WF()
    note(g, "Depth: four models on one image. Change `model` in any node (32 depth models) or "
            "right-click > Bypass a branch. Mute branches you do not want to run.")
    img = load(g, "example.png", 0, 0)
    shootout(g, "FastforksDepth", "image", [
        {"model": "depth-anything-v2-s"}, {"model": "distill-any-depth-s"},
        {"model": "da3mono-large"}, {"model": "zoedepth-nk"}], img=img)
    return g


@wf
def w02_backend_compare():
    g = WF()
    note(g, "Same model, five backends: auto (what fastforks picks on this Mac), portable "
            "(plain PyTorch on MPS), kernels (Metal kernels where there are any), mlx and mlx portable (Apple MLX). Results should match; time and memory differ. "
            "Run the queue twice: the first run includes loading and compilation.")
    img = load(g, "example.png", 0, 0)
    shootout(g, "FastforksDepth", "image", [
        {"model": "depth-anything-v2-l [non-commercial]", "backend": b} for b in ("auto", "portable", "kernels", "mlx", "mlx portable")],
        img=img, dtype="float16")
    return g


@wf
def w03_background_removal():
    g = WF()
    note(g, "Background removal: five matting/saliency models. The IMAGE output is the cut-out (RGBA).")
    img = load(g, "example.png", 0, 0)
    shootout(g, "FastforksRemoveBackground", "image", [
        {"model": m} for m in ("birefnet-lite", "isnet-general-use", "inspyrenet-fast", "u2net", "ben2")],
        img=img, step=380)
    return g


@wf
def w04_segment_prompted():
    g = WF()
    note(g, "Promptable segmentation: a click at the image centre (positive_points = 384,384). "
            "Edit the point in pixels, add a box 'x0,y0,x1,y1' or negative points. "
            "Compare SAM 2.1, MobileSAM and EfficientViT-SAM.")
    img = load(g, "example.png", 0, 0)
    shootout(g, "FastforksSegment", "image", [
        {"model": m} for m in ("sam2.1-hiera-small", "mobile-sam", "efficientvit-sam-l0", "sam-vit-b")],
        img=img, mask_out=True, positive_points="384,384")
    return g


@wf
def w05_detection():
    g = WF()
    note(g, "Object detection: DETR-family, YOLOX and RF-DETR. Output is a mask of the boxes and a "
            "text list. Set `classes` (e.g. 'person, car') to filter, raise score_threshold to cut noise. "
            "For open-vocabulary detection swap in grounding-dino-tiny and type what to find in `classes`.")
    img = load(g, "scene.png", 0, 0)
    for k, m in enumerate(["rf-detr-s-coco", "yolox-s", "rtdetrv2-s-coco", "dfine-s-coco"]):
        y = k * 420
        n = g.add("FastforksDetect", 380, y, size=(330, 260), title=m, model=m, image=(img, 0))
        preview(g, n, 740, y, slot=0, mask=True)
        g.add("PreviewAny", 1260, y, size=(300, 200), source=(n, 1))
    return g


@wf
def w06_pose():
    g = WF()
    note(g, "Pose estimation (skeleton overlay). Top-down models (rtmpose/vitpose/dwpose) need person "
            "boxes, the one-stage rtmo-m finds people itself (the detector node on the left feeds the others). Switch `model` to rtmw-x/dwpose-l for whole body.")
    img = load(g, "scene.png", 0, 0)
    det = g.add("FastforksDetect", 0, 400, size=(330, 260), title="person detector (feeds top-down models)",
                model="rtmdet-m-person", image=(img, 0))
    for k, m in enumerate(["rtmo-m", "rtmpose-m", "vitpose-plus-small", "dwpose-m"]):
        y = k * 420
        vals = {"model": m, "image": (img, 0)}
        if m != "rtmo-m":
            vals["detections"] = (det, 1)
        n = g.add("FastforksPose", 380, y, size=(330, 260), title=m, **vals)
        preview(g, n, 740, y)
    return g


@wf
def w07_upscale():
    g = WF()
    note(g, "4x upscaling of a small image: GAN (Real-ESRGAN), light CNN (SPAN), transformer (SwinIR). "
            "tile limits memory; 0 runs one pass.")
    img = load(g, "small.png", 0, 0)
    shootout(g, "FastforksUpscale", "image", [
        {"model": m} for m in ("realesrgan-x4plus", "realesr-general-x4v3", "span-x4", "swinir-m-x4-real")],
        img=img, step=480)
    return g


@wf
def w08_interpolation():
    g = WF()
    note(g, "Frame interpolation: four frames are batched, each model doubles the frame count "
            "(multiplier 2; raise it for slow motion). The preview shows the frames; use Save Image/Video nodes "
            "to export.")
    ys = [0, 360, 720, 1080]
    imgs = [load(g, f"clip{i}.png", 0, y) for i, y in enumerate(ys)]
    b = g.add("ImageBatch", 330, 0, size=(200, 50), image1=(imgs[0], 0), image2=(imgs[1], 0))
    b = g.add("ImageBatch", 330, 120, size=(200, 50), image1=(b, 0), image2=(imgs[2], 0))
    b = g.add("ImageBatch", 330, 240, size=(200, 50), image1=(b, 0), image2=(imgs[3], 0))
    for k, m in enumerate(["rife-4.26", "film", "ema-vfi-small", "rife-4.25-lite"]):
        y = k * 520
        n = g.add("FastforksInterpolate", 600, y, size=(330, 260), title=m, model=m, frames=(b, 0))
        preview(g, n, 960, y)
    return g


@wf
def w09_stereo():
    g = WF()
    note(g, "Stereo depth from a left/right pair: classic cost-volume (IGEV, ACVNet) vs recurrent (RAFT-Stereo) "
            "vs transformer (S2M2). Output is a disparity visualisation.")
    left = load(g, "scene.png", 0, 0)
    right = load(g, "scene_b.png", 0, 400)
    for k, m in enumerate(["raft-stereo-middlebury", "igev-stereo-middlebury",
                           "fast-acvnet-plus-generalization", "waft-stereo-s"]):
        y = k * 420
        n = g.add("FastforksStereo", 380, y, size=(330, 260), title=m, model=m, left=(left, 0), right=(right, 0))
        preview(g, n, 740, y)
    return g


@wf
def w10_optical_flow():
    g = WF()
    note(g, "Optical flow between neighbouring frames (colour-coded) and, bottom, a warp: frame 1 is pushed "
            "along the flow of the chosen model, so a good flow gives a copy of frame 2. "
            "Compare RAFT, SEA-RAFT, GMFlow and WAFT.")
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
    return g


if __name__ == "__main__":
    outs = [Path(p) for p in sys.argv[2:]] or [Path(__file__).parent]
    for name, fn in WORKFLOWS.items():
        for o in outs:
            o.mkdir(parents=True, exist_ok=True)
            fn().save(o / f"fastforks-{name}.json")
    print(len(WORKFLOWS), "workflows ->", *outs)
