"""DeDoDe (Parskatt/DeDoDe, vendored in `net/`): encoders, refiners, detector, descriptor and inference helpers.

The VGG-BN encoders, the convolutional refiners, the detector and descriptor networks, keypoint sampling, descriptor
sampling and dual-softmax matching. Used by the `dedode` (DeDoDe v2) and `loma` (DaD detector + DeDoDe descriptor)
matching families.
"""

from .net.dedode import (
    DESCRIPTOR_B,
    DESCRIPTOR_G,
    DETECTOR_L,
    DETECTOR_S,
    VGG,
    VGG11_BN,
    VGG19_BN,
    ConvRefiner,
    Decoder,
    Descriptor,
    Detector,
    VGGDINOv2,
    autocast,
    describe_keypoints,
    dual_softmax_match,
    get_grid,
    sample_keypoints,
    vgg_bn_features,
)

__all__ = [
    "DESCRIPTOR_B",
    "DESCRIPTOR_G",
    "DETECTOR_L",
    "DETECTOR_S",
    "VGG",
    "VGG11_BN",
    "VGG19_BN",
    "ConvRefiner",
    "Decoder",
    "Descriptor",
    "Detector",
    "VGGDINOv2",
    "autocast",
    "describe_keypoints",
    "dual_softmax_match",
    "get_grid",
    "sample_keypoints",
    "vgg_bn_features",
]
