"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

# fastforks: torchvision / einops / huggingface_hub dropped, relative imports, only the "last" stacking function (the
# released config's); the training parts of UniDepthV2 (losses, forward_train / forward_test, get_params, infer: the
# runners in ../model.py do its pre- and post-processing) are not vendored
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import Decoder
from .dinov2 import dinov2_vitl14
from .utils import last_stack

STACKING_FNS = {
    "last": last_stack,
}


def get_paddings(original_shape, aspect_ratio_range):
    # Original dimensions
    H_ori, W_ori = original_shape
    orig_aspect_ratio = W_ori / H_ori

    # Determine the closest aspect ratio within the range
    min_ratio, max_ratio = aspect_ratio_range
    target_aspect_ratio = min(max_ratio, max(min_ratio, orig_aspect_ratio))

    # fastforks: no padding inside the range (upstream's int(H_ori * W_ori / H_ori) can give W_ori - 1, e.g. 352 x 480:
    # a -1 px pad and a broken output)
    if orig_aspect_ratio == target_aspect_ratio:
        return (0, 0, 0, 0), (H_ori, W_ori)
    if orig_aspect_ratio > target_aspect_ratio:  # Too wide
        W_new = W_ori
        H_new = int(W_ori / target_aspect_ratio)
        pad_top = (H_new - H_ori) // 2
        pad_bottom = H_new - H_ori - pad_top
        pad_left, pad_right = 0, 0
    else:  # Too tall
        H_new = H_ori
        W_new = int(H_ori * target_aspect_ratio)
        pad_left = (W_new - W_ori) // 2
        pad_right = W_new - W_ori - pad_left
        pad_top, pad_bottom = 0, 0

    return (pad_left, pad_right, pad_top, pad_bottom), (H_new, W_new)


def get_resize_factor(original_shape, pixels_range, shape_multiplier=14):
    # Original dimensions
    H_ori, W_ori = original_shape
    n_pixels_ori = W_ori * H_ori

    # Determine the closest number of pixels within the range
    min_pixels, max_pixels = pixels_range
    target_pixels = min(max_pixels, max(min_pixels, n_pixels_ori))

    # Calculate the resize factor
    resize_factor = (target_pixels / n_pixels_ori) ** 0.5
    new_width = int(W_ori * resize_factor)
    new_height = int(H_ori * resize_factor)
    new_height = ceil(new_height / shape_multiplier) * shape_multiplier
    new_width = ceil(new_width / shape_multiplier) * shape_multiplier

    return resize_factor, (new_height, new_width)


def _postprocess(tensor, shapes, paddings, interpolation_mode="bilinear"):
    # interpolate to original size
    tensor = F.interpolate(
        tensor, size=shapes, mode=interpolation_mode, align_corners=False
    )

    # remove paddings
    pad1_l, pad1_r, pad1_t, pad1_b = paddings
    tensor = tensor[..., pad1_t : shapes[0] - pad1_b, pad1_l : shapes[1] - pad1_r]
    return tensor


def _postprocess_intrinsics(K, resize_factors, paddings):
    batch_size = K.shape[0]
    K_new = K.clone()

    for i in range(batch_size):
        scale = resize_factors[i]
        pad_l, _, pad_t, _ = paddings[i]

        K_new[i, 0, 0] /= scale  # fx
        K_new[i, 1, 1] /= scale  # fy
        K_new[i, 0, 2] /= scale  # cx
        K_new[i, 1, 2] /= scale  # cy

        K_new[i, 0, 2] -= pad_l  # cx
        K_new[i, 1, 2] -= pad_t  # cy

    return K_new


class UniDepthV2(nn.Module):  # fastforks: no PyTorchModelHubMixin
    def __init__(
        self,
        config,
        eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        self.eps = eps
        self.build(config)

    def encode_decode(self, inputs, image_metas=[]):
        B, _, H, W = inputs["image"].shape

        # shortcut eval should avoid errors
        if len(image_metas) and "paddings" in image_metas[0]:
            inputs["paddings"] = torch.tensor(
                [image_meta["paddings"] for image_meta in image_metas],
                device=self.device,
            )[
                ..., [0, 2, 1, 3]
            ]  # lrtb
            inputs["depth_paddings"] = torch.tensor(
                [image_meta["depth_paddings"] for image_meta in image_metas],
                device=self.device,
            )
            if (
                self.training
            ):  # at inference we do not have image paddings on top of depth ones (we have not "crop" on gt in ContextCrop)
                inputs["depth_paddings"] = inputs["depth_paddings"] + inputs["paddings"]

        if inputs.get("camera", None) is not None:
            inputs["rays"] = inputs["camera"].get_rays(shapes=(B, H, W))

        features, tokens = self.pixel_encoder(inputs["image"])
        inputs["features"] = [
            self.stacking_fn(features[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]
        inputs["tokens"] = [
            self.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ]

        outputs = self.pixel_decoder(inputs, image_metas)
        outputs["rays"] = outputs["rays"].unflatten(1, (H, W)).permute(0, 3, 1, 2)  # fastforks: einops 'b (h w) c -> b c h w'
        pts_3d = outputs["rays"] * outputs["radius"]
        outputs.update({"points": pts_3d, "depth": pts_3d[:, -1:]})

        return inputs, outputs

    @property
    def device(self):
        return next(self.parameters()).device

    def build(self, config):
        assert config["model"]["pixel_encoder"]["name"] == "dinov2_vitl14"  # fastforks: the one vendored factory
        pixel_encoder_factory = dinov2_vitl14
        pixel_encoder_config = {
            **config["training"],
            **config["model"]["pixel_encoder"],
            **config["data"],
        }
        pixel_encoder = pixel_encoder_factory(pixel_encoder_config)

        config["model"]["pixel_encoder"]["patch_size"] = (
            14 if "dino" in config["model"]["pixel_encoder"]["name"] else 16
        )
        pixel_encoder_embed_dims = (
            pixel_encoder.embed_dims
            if hasattr(pixel_encoder, "embed_dims")
            else [getattr(pixel_encoder, "embed_dim") * 2**i for i in range(4)]  # noqa: B009 - upstream
        )
        config["model"]["pixel_encoder"]["embed_dim"] = getattr(  # noqa: B009 - upstream
            pixel_encoder, "embed_dim"
        )
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims
        config["model"]["pixel_encoder"]["depths"] = pixel_encoder.depths
        config["model"]["pixel_encoder"]["cls_token_embed_dims"] = getattr(
            pixel_encoder, "cls_token_embed_dims", pixel_encoder_embed_dims
        )

        pixel_decoder = Decoder(config)

        self.pixel_encoder = pixel_encoder
        self.pixel_decoder = pixel_decoder

        self.slices_encoder_range = list(
            zip([0, *self.pixel_encoder.depths[:-1]], self.pixel_encoder.depths)
        )

        stacking_fn = config["model"]["pixel_encoder"]["stacking_fn"]
        assert (
            stacking_fn in STACKING_FNS
        ), f"Stacking function {stacking_fn} not found in {STACKING_FNS.keys()}"
        self.stacking_fn = STACKING_FNS[stacking_fn]
        self.shape_constraints = config["data"]["augmentations"]["shape_constraints"]
        self.interpolation_mode = "bilinear"
