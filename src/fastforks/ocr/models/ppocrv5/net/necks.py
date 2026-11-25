"""Detection necks (PaddleOCR `db_fpn.py`, `intracl.py`): RSEFPN (mobile) and LKPAN with IntraCL blocks (server)."""

from __future__ import annotations

import torch
from torch import nn

from .backbones import _SE
from .layers import up


class _RSELayer(nn.Module):
    def __init__(self, cin: int, cout: int, k: int) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
        self.se_block = _SE(cout, slope=0.2)  # det_mobilenet_v3 SEModule: hardsigmoid(slope 0.2, offset 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        return x + self.se_block(x)


class RSEFPN(nn.Module):
    """RSEFPN with shortcuts: `[c2, c3, c4, c5]` -> `[B, out, H/4, W/4]`."""

    def __init__(self, cins: list[int], cout: int) -> None:
        super().__init__()
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        for c in cins:  # created interleaved, as upstream
            self.ins_conv.append(_RSELayer(c, cout, 1))
            self.inp_conv.append(_RSELayer(cout, cout // 4, 3))

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        in2, in3, in4, in5 = (conv(c) for conv, c in zip(self.ins_conv, x, strict=True))
        out4 = in4 + up(in5, 2)
        out3 = in3 + up(out4, 2)
        out2 = in2 + up(out3, 2)
        p2, p3, p4, p5 = (conv(c) for conv, c in zip(self.inp_conv, (out2, out3, out4, in5), strict=True))
        return torch.cat([up(p5, 8), up(p4, 4), up(p3, 2), p2], 1)


class _IntraCL(nn.Module):
    def __init__(self, c: int, rf: int = 2) -> None:
        super().__init__()
        r = c // rf
        self.reduce = nn.Conv2d(c, r, 1)
        self.ret = nn.Conv2d(r, c, 1)
        self.v7, self.v5, self.v3 = (nn.Conv2d(r, r, (k, 1), padding=(k // 2, 0)) for k in (7, 5, 3))
        self.q7, self.q5, self.q3 = (nn.Conv2d(r, r, (1, k), padding=(0, k // 2)) for k in (7, 5, 3))
        self.c7, self.c5, self.c3 = (nn.Conv2d(r, r, k, padding=k // 2) for k in (7, 5, 3))
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reduce(x)
        y = self.c7(y) + self.v7(y) + self.q7(y)
        y = self.c5(y) + self.v5(y) + self.q5(y)
        y = self.c3(y) + self.v3(y) + self.q3(y)
        return x + torch.relu(self.bn(self.ret(y)))


class LKPAN(nn.Module):
    """LKPAN (mode large) with IntraCL blocks: `[c2, c3, c4, c5]` -> `[B, out, H/4, W/4]`."""

    def __init__(self, cins: list[int], cout: int) -> None:
        super().__init__()
        q = cout // 4
        self.ins_conv = nn.ModuleList()
        self.inp_conv = nn.ModuleList()
        self.pan_head_conv = nn.ModuleList()
        self.pan_lat_conv = nn.ModuleList()
        for i, c in enumerate(cins):  # created interleaved, as upstream
            self.ins_conv.append(nn.Conv2d(c, cout, 1, bias=False))
            self.inp_conv.append(nn.Conv2d(cout, q, 9, padding=4, bias=False))
            if i > 0:
                self.pan_head_conv.append(nn.Conv2d(q, q, 3, 2, 1, bias=False))
            self.pan_lat_conv.append(nn.Conv2d(q, q, 9, padding=4, bias=False))
        self.incl = nn.ModuleList(_IntraCL(q) for _ in range(4))  # incl1 (p2) .. incl4 (p5)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        in2, in3, in4, in5 = (conv(c) for conv, c in zip(self.ins_conv, x, strict=True))
        # Release each projection as its replacement is computed, keeping the same operations and order.
        in4 = in4 + up(in5, 2)
        in3 = in3 + up(in4, 2)
        in2 = in2 + up(in3, 2)
        p2 = self.inp_conv[0](in2)
        del in2
        p3 = self.inp_conv[1](in3)
        del in3
        p4 = self.inp_conv[2](in4)
        del in4
        p5 = self.inp_conv[3](in5)
        del in5
        p3 = p3 + self.pan_head_conv[0](p2)
        p4 = p4 + self.pan_head_conv[1](p3)
        p5 = p5 + self.pan_head_conv[2](p4)
        p2 = self.pan_lat_conv[0](p2)
        p3 = self.pan_lat_conv[1](p3)
        p4 = self.pan_lat_conv[2](p4)
        p5 = self.pan_lat_conv[3](p5)
        p2 = self.incl[0](p2)
        p3 = self.incl[1](p3)
        p4 = self.incl[2](p4)
        p5 = self.incl[3](p5)
        return torch.cat([up(p5, 8), up(p4, 4), up(p3, 2), p2], 1)
