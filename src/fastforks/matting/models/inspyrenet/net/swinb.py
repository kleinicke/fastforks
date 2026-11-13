"""InSPyReNet's Swin-B backbone (`backbones/SwinTransformer.py` `SwinB`) on `fastforks.shared.swin_v1`.

Upstream's backbone is the Swin detection code BiRefNet vendors too; its `forward` returns the patch embedding in front
of the four normalised stages (shared swin_v1 drops it). Shared swin_v1 masks shifted windows with -inf (upstream
-100, `exp(-100)` ~ 4e-44 in the softmax) and runs SDPA.
"""

from fastforks.shared.swin_v1 import SwinTransformer


class _SwinB(SwinTransformer):
    def forward(self, x):
        x = self.patch_embed(x)
        wh, ww = x.size(2), x.size(3)
        outs = [x.contiguous()]
        x = x.flatten(2).transpose(1, 2)
        for i, layer in enumerate(self.layers):
            x_out, h, w, x, wh, ww = layer(x, wh, ww)
            x_out = getattr(self, f"norm{i}")(x_out)
            outs.append(x_out.view(-1, h, w, self.num_features[i]).permute(0, 3, 1, 2).contiguous())
        return tuple(outs)


def SwinB(pretrained=False):  # noqa: N802 - upstream's name
    if pretrained:
        msg = "ImageNet backbone weights are not loaded here; the InSPyReNet checkpoint holds the backbone"
        raise ValueError(msg)
    return _SwinB(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12)
