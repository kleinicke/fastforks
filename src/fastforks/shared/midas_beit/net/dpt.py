# fastforks: vendored from MiDaS (isl-org/MiDaS@4545977, MIT, see LICENSE) midas/dpt_depth.py (DPT, DPTDepthModel,
# _make_fusion_block) and midas/backbones/beit.py (_make_beit_backbone, _make_pretrained_beitl16_{512,384},
# forward_beit), BEiT backbones only. `DPT.forward(x, features=True)` additionally returns the decoder features ZoeDepth
# reads through forward hooks (`MidasCore.layer_names`); everything else is upstream.
import torch
import torch.nn as nn

from .beit import Beit
from .blocks import FeatureFusionBlock_custom, Interpolate, _make_scratch, forward_adapted_unflatten, make_backbone_default


def forward_beit(pretrained, x):
    # fastforks: = forward_adapted_unflatten(pretrained, x, "forward_features") with the hooks returned directly
    b, c, h, w = x.shape
    return forward_adapted_unflatten(pretrained, pretrained.model.forward_features(x, pretrained.hooks), h, w)


def _make_beit_backbone(
        model,
        features=[96, 192, 384, 768],
        size=[384, 384],
        hooks=[0, 4, 8, 11],
        vit_features=768,
        use_readout="ignore",
        start_index=1,
        start_index_readout=1,
):
    # fastforks: the MethodType patches live in net/beit.py's classes
    return make_backbone_default(model, features, size, hooks, vit_features, use_readout, start_index,
                                 start_index_readout)


def _make_pretrained_beitl16_512(use_readout="ignore", hooks=None):
    model = Beit(img_size=512)  # fastforks: = timm.create_model("beit_large_patch16_512", pretrained=False)

    hooks = [5, 11, 17, 23] if hooks is None else hooks

    features = [256, 512, 1024, 1024]

    return _make_beit_backbone(
        model,
        features=features,
        size=[512, 512],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
    )


def _make_pretrained_beitl16_384(use_readout="ignore", hooks=None):
    model = Beit(img_size=384)  # fastforks: = timm.create_model("beit_large_patch16_384", pretrained=False)

    hooks = [5, 11, 17, 23] if hooks is None else hooks
    return _make_beit_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
        use_readout=use_readout,
    )


def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPT(nn.Module):
    def __init__(
        self,
        head,
        features=256,
        backbone="beitl16_512",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "beitl16_512": [5, 11, 17, 23],
            "beitl16_384": [5, 11, 17, 23],
        }[backbone]

        # Instantiate backbone and reassemble blocks
        # fastforks: = midas.blocks._make_encoder for the BEiT-L backbones
        self.pretrained = {
            "beitl16_512": _make_pretrained_beitl16_512,
            "beitl16_384": _make_pretrained_beitl16_384,
        }[backbone](use_readout=readout, hooks=hooks)
        self.scratch = _make_scratch([256, 512, 1024, 1024], features, groups=1, expand=False)

        self.number_layers = len(hooks) if hooks is not None else 4
        size_refinenet3 = None
        self.scratch.stem_transpose = None

        self.forward_transformer = forward_beit

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        if self.number_layers >= 4:
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head


    def forward(self, x, features=False):
        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        layer_1, layer_2, layer_3, layer_4 = layers

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        if not features:
            return self.scratch.output_conv(path_1)
        # fastforks: ZoeDepth MidasCore hooks: out_conv = output_conv[3] (the ReLU before the last conv)
        out_conv = self.scratch.output_conv[0:4](path_1)
        out = self.scratch.output_conv[4:](out_conv)
        return out, [out_conv, layer_4_rn, path_4, path_3, path_2, path_1]


class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs["features"] if "features" in kwargs else 256
        head_features_1 = kwargs["head_features_1"] if "head_features_1" in kwargs else features
        head_features_2 = kwargs["head_features_2"] if "head_features_2" in kwargs else 32
        kwargs.pop("head_features_1", None)
        kwargs.pop("head_features_2", None)

        head = nn.Sequential(
            nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

    def forward(self, x, features=False):
        if not features:
            return super().forward(x).squeeze(dim=1)
        out, feats = super().forward(x, features=True)
        return out.squeeze(dim=1), feats
