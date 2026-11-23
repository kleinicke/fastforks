"""`torch.einsum` for the LM fit's per-pixel products of 1-3 element vectors and matrices, as broadcast products.

`torch.einsum` lowers `...Nij,...Njk->...Nik` with N = every pixel to one `bmm` of N tiny matrices, which cuBLAS
runs at a few hundred microseconds a call; a broadcast multiply and a sum over the (short) contracted axes is two
memory-bound kernels. The products are summed in a different order than cuBLAS does (one rounding apart).
"""

import torch


def einsum(spec: str, *operands: torch.Tensor) -> torch.Tensor:
    """`torch.einsum(spec, *operands)` by broadcasting: the output's axes, then the contracted ones, summed away.

    Every operand's `...` (or its unnamed leading axes) broadcasts against the others'; the output starts with `...`
    when any input has one. Contracted axes should be short: the product of all operands is materialised.
    """
    inputs, out = spec.replace(" ", "").split("->")
    terms = inputs.split(",")
    if not operands or len(terms) != len(operands):
        msg = f"{spec!r} names {len(terms)} operands, got {len(operands)}"
        raise ValueError(msg)
    if "..." in out and not out.startswith("..."):
        msg = f"{spec!r}: the output's '...' must come first"
        raise ValueError(msg)
    out = out.removeprefix("...")
    named = [t.replace("...", "") for t in terms]
    contracted = sorted({c for t in named for c in t} - set(out))
    order = list(out) + contracted
    ellipsis = max(op.dim() - len(t) for op, t in zip(operands, named, strict=True))
    views = []
    for op, term, letters in zip(operands, terms, named, strict=True):
        if "..." not in term and op.dim() != len(letters):
            msg = f"{spec!r}: operand {term!r} has {op.dim()} axes"
            raise ValueError(msg)
        lead = op.dim() - len(letters)
        x = op.permute(*range(lead), *(lead + letters.index(c) for c in order if c in letters))
        shape = [1] * (ellipsis - lead) + list(x.shape[:lead])
        sizes = iter(x.shape[lead:])
        shape += [next(sizes) if c in letters else 1 for c in order]
        views.append(x.reshape(shape))
    product = views[0]
    for x in views[1:]:
        product = product * x
    if contracted:
        product = product.sum(tuple(range(-len(contracted), 0)))
    return product
