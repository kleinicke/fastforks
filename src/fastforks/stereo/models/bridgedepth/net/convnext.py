# fastforks: the parts of torchvision's ConvNeXt (torchvision/models/convnext.py, torchvision/ops/misc.py, v0.29.1,
# BSD-3-Clause, see LICENSE-torchvision) that BridgeDepth's ConvNextDPT uses: `features` only (no classifier), module
# layout and state-dict keys as upstream. Inference only: stochastic depth (identity in eval mode) is an `nn.Identity`,
# no weight initialisation (the checkpoint sets every parameter), no API-usage logging.
from functools import partial
from typing import Callable, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class Permute(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return torch.permute(x, self.dims)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = nn.Identity()  # fastforks: StochasticDepth(stochastic_depth_prob, "row") in eval mode

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class CNBlockConfig:
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers


class ConvNeXt(nn.Module):
    def __init__(
        self,
        block_setting: list[CNBlockConfig],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
    ) -> None:
        super().__init__()
        block = CNBlock
        norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: list[nn.Module] = []

        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            nn.Sequential(  # fastforks: Conv2dNormActivation(3, C, 4, stride 4, padding 0, LayerNorm2d, no activation, bias)
                nn.Conv2d(3, firstconv_output_channels, kernel_size=4, stride=4, padding=0, bias=True),
                norm_layer(firstconv_output_channels),
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            stage: list[nn.Module] = []
            for _ in range(cnf.num_layers):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
