# fastforks: upstream bridgedepth/monocular/depth_anything.py, reduced to what inference uses. The DINOv2 ViT and the
# DPT head come from fastforks.shared.depth_anything_v2 (upstream's copies are Depth Anything V2's, same modules and
# state-dict keys); BridgeDepth's head stops after `output_conv1` (the `forward` below). No hub download
# (`from_pretrained`): the BridgeDepth checkpoint holds these weights. `infer` runs in the caller's precision (upstream
# wraps it in torch.inference_mode + float16 autocast whenever the weights are float32, also on the CPU).
import torch.nn as nn

from fastforks.shared.depth_anything_v2 import DINOv2, DPTHead


class BridgeDPTHead(DPTHead):
    def forward(self, out_features, patch_h, patch_w):
        scratch = self.scratch
        reductions = (scratch.layer1_rn, scratch.layer2_rn, scratch.layer3_rn, scratch.layer4_rn)
        layers = []
        for i, features in enumerate(out_features):
            x = features[0]  # use_clstoken is False
            x = x.permute(0, 2, 1).reshape(x.shape[0], x.shape[-1], patch_h, patch_w)
            x = self.projects[i](x)
            layers.append(reductions[i](self.resize_layers[i](x)))

        layer1, layer2, layer3, layer4 = layers
        del layers, x
        path = scratch.refinenet4(layer4, size=layer3.shape[2:])
        del layer4
        path = scratch.refinenet3(path, layer3, size=layer2.shape[2:])
        del layer3
        path = scratch.refinenet2(path, layer2, size=layer1.shape[2:])
        del layer2
        path = scratch.refinenet1(path, layer1)
        return scratch.output_conv1(path)


class DepthAnything(nn.Module):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    use_bn = False
    use_clstoken = False

    def __init__(self, encoder="vitl"):
        super(DepthAnything, self).__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11],
            'vitl': [4, 11, 17, 23],
            'vitg': [9, 19, 29, 39]
        }

        self.encoder = encoder

        self.pretrained = DINOv2(encoder)  # fastforks: upstream dinov2.hub.backbones.dinov2_<encoder>14(pretrained=False)

        self.depth_head = BridgeDPTHead(self.pretrained.embed_dim, self.model_configs[encoder]["features"], self.use_bn,
                                        out_channels=self.model_configs[encoder]["out_channels"],
                                        use_clstoken=self.use_clstoken)

    @property
    def feature_dim(self) -> int:
        return self.model_configs[self.encoder]["features"] // 2

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, image):
        patch_h, patch_w = image.shape[-2] // 14, image.shape[-1] // 14

        features = self.pretrained.get_intermediate_layers(image, self.intermediate_layer_idx[self.encoder], return_class_token=True)

        out_feature = self.depth_head(features, patch_h, patch_w)

        return out_feature

    def infer(self, image):
        return self.forward(image).float()
