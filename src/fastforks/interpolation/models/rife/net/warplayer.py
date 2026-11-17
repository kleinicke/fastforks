# Ported from hzwer/Practical-RIFE model/warplayer.py (MIT, see ../LICENSE).
# fastforks patches: the base grid is built per call (no global cache keyed by device and size) and the sampling runs
# in float32 (upstream's fp16 mode builds the grid in fp16: up to ~0.5 px off at the right edge of 1920 px).
import torch


def warp(tenInput, tenFlow):
    tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device).view(
        1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
    tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device).view(
        1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
    tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

    tenFlow = tenFlow.float()
    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    g = (tenGrid + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput.float(), grid=g, mode='bilinear', padding_mode='border', align_corners=True).to(tenInput.dtype)
