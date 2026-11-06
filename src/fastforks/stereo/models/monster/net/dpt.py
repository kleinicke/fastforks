import torch
import torch.nn as nn
import torch.nn.functional as F

# fastforks: MonSter's Depth-Anything-V2-list3/depth_anything_v2/dpt.py, only the class it adds (DPTHead_decoder: the
# DPT fusion path returning the 1/4..1/32 features). The ViT, the unchanged DPTHead and the blocks are the shared,
# unmodified Depth Anything V2 code (fastforks.shared.depth_anything_v2; MonSter's DPTHead only adds an unused
# `only_feat` argument). cv2 / torchvision imports and the DepthAnythingV2 / DepthAnythingV2_decoder wrappers dropped.
from fastforks.shared.depth_anything_v2 import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPTHead_decoder(nn.Module):
    def __init__(
        self, 
        in_channels, 
        features=256, 
        use_bn=False, 
        out_channels=[256, 512, 1024, 1024], 
        use_clstoken=False
    ):
        super(DPTHead_decoder, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )
        
        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        
        # head_features_1 = features
        # head_features_2 = 32
        
        # self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        # self.scratch.output_conv2 = nn.Sequential(
        #     nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True),
        #     nn.Identity(),
        # )
    
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out

        # print('layer_1', layer_1.shape)
        # print('layer_2', layer_2.shape)
        # print('layer_3', layer_3.shape)
        # print('layer_4', layer_4.shape)
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        # path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])        
        # path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        # path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        # path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_4_rn.shape[2:])
        up_path_4 = F.interpolate(path_4, size=layer_3_rn.shape[2:], mode='bilinear', align_corners=True)  
        path_3 = self.scratch.refinenet3(up_path_4, layer_3_rn, size=layer_3_rn.shape[2:])
        up_path_3 = F.interpolate(path_3, size=layer_2_rn.shape[2:], mode='bilinear', align_corners=True)
        path_2 = self.scratch.refinenet2(up_path_3, layer_2_rn, size=layer_2_rn.shape[2:])
        up_path_2 = F.interpolate(path_2, size=layer_1_rn.shape[2:], mode='bilinear', align_corners=True)
        path_1 = self.scratch.refinenet1(up_path_2, layer_1_rn, size=layer_1_rn.shape[2:])

        return path_1, path_2, path_3, path_4

        # feat_4 = self.scratch.refinenet4(layer_4_rn, size=layer_4_rn.shape[2:])
        # up_feat_4 = F.interpolate(feat_4, size=layer_3_rn.shape[2:], mode='bilinear', align_corners=True)  
        # feat_3 = self.scratch.refinenet3(up_feat_4, layer_3_rn, size=layer_3_rn.shape[2:])
        # up_feat_3 = F.interpolate(feat_3, size=layer_2_rn.shape[2:], mode='bilinear', align_corners=True)
        # feat_2 = self.scratch.refinenet2(up_feat_3, layer_2_rn, size=layer_2_rn.shape[2:])
        # up_feat_2 = F.interpolate(feat_2, size=layer_1_rn.shape[2:], mode='bilinear', align_corners=True)
        # feat_1 = self.scratch.refinenet1(up_feat_2, layer_1_rn, size=layer_1_rn.shape[2:])

        # if only_feat:
        #     return [layer_4, layer_3, layer_2, layer_1]
        # else:
        
        #     out = self.scratch.output_conv1(path_1)
        #     out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        #     # features_decoder = out.clone()
        #     out = self.scratch.output_conv2(out)
            
        #     return out, [layer_4, layer_3, layer_2, layer_1]
