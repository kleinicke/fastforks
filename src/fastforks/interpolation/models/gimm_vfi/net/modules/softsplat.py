# Ported from GSeanCDAT/GIMM-VFI src/models/generalizable_INR/modules/softsplat.py (S-Lab License 1.0, see ../../LICENSE).
# fastforks patches: the cupy CUDA kernel `softsplat_out` (summation splatting) is `_splat` in plain torch (scatter_add,
# same bilinear weights, finite and bounds checks, float32 as upstream's custom_fwd(cast_inputs=torch.float32)), so it
# runs on any device; no NaN checks (host syncs); forward only.
import torch


def _splat(tenIn, tenFlow):
    # fastforks: softsplat_out - every source pixel adds its value times the bilinear weight of the four integer
    # neighbours of (x + flow_x, y + flow_y) that lie inside the image; non-finite targets are skipped.
    b, c, h, w = tenIn.shape
    gy, gx = torch.meshgrid(
        torch.arange(h, device=tenFlow.device, dtype=tenFlow.dtype),
        torch.arange(w, device=tenFlow.device, dtype=tenFlow.dtype),
        indexing="ij",
    )
    fx = gx + tenFlow[:, 0]
    fy = gy + tenFlow[:, 1]
    finite = torch.isfinite(fx) & torch.isfinite(fy)
    x0 = torch.where(finite, torch.floor(fx), torch.zeros_like(fx))
    y0 = torch.where(finite, torch.floor(fy), torch.zeros_like(fy))
    x1, y1 = x0 + 1.0, y0 + 1.0
    corners = (
        (x0, y0, (x1 - fx) * (y1 - fy)),
        (x1, y0, (fx - x0) * (y1 - fy)),
        (x0, y1, (x1 - fx) * (fy - y0)),
        (x1, y1, (fx - x0) * (fy - y0)),
    )
    out = tenIn.new_zeros([b, c, h * w + 1])  # the last slot collects the skipped samples
    src = tenIn.reshape(b, c, h * w)
    for cx, cy, wgt in corners:
        valid = finite & (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h)
        idx = torch.where(valid, cy * w + cx, torch.full_like(cx, h * w)).long()
        out.scatter_add_(2, idx.view(b, 1, h * w).expand(b, c, h * w), src * wgt.view(b, 1, h * w).to(src.dtype))
    return out[:, :, : h * w].view(b, c, h, w)


def softsplat(tenIn, tenFlow, tenMetric, strMode, return_norm=False):
    assert strMode.split("-")[0] in ["sum", "avg", "linear", "softmax"]

    if strMode == "sum":
        assert tenMetric is None
    if strMode == "avg":
        assert tenMetric is None
    if strMode.split("-")[0] == "linear":
        assert tenMetric is not None
    if strMode.split("-")[0] == "softmax":
        assert tenMetric is not None

    if strMode == "avg":
        tenIn = torch.cat(
            [
                tenIn,
                tenIn.new_ones([tenIn.shape[0], 1, tenIn.shape[2], tenIn.shape[3]]),
            ],
            1,
        )

    elif strMode.split("-")[0] == "linear":
        tenIn = torch.cat([tenIn * tenMetric, tenMetric], 1)

    elif strMode.split("-")[0] == "softmax":
        tenIn = torch.cat([tenIn * tenMetric.exp(), tenMetric.exp()], 1)

    # end
    tenOut = _splat(tenIn.float(), tenFlow.float())  # fastforks: was softsplat_func.apply (cupy), NaN checks removed

    if strMode.split("-")[0] in ["avg", "linear", "softmax"]:
        tenNormalize = tenOut[:, -1:, :, :]

        if len(strMode.split("-")) == 1:
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split("-")[1] == "addeps":
            tenNormalize = tenNormalize + 0.0000001

        elif strMode.split("-")[1] == "zeroeps":
            tenNormalize = torch.where(tenNormalize == 0.0, torch.ones_like(tenNormalize), tenNormalize)  # fastforks: out of place

        elif strMode.split("-")[1] == "clipeps":
            tenNormalize = tenNormalize.clip(0.0000001, None)

        # end

        if return_norm:
            return tenOut[:, :-1, :, :], tenNormalize

        tenOut = tenOut[:, :-1, :, :] / tenNormalize

    return tenOut
