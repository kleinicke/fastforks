"""Torch-only reshape/transpose patterns used by upstream FlowFormer."""
import re
from torch import nn

def _product(values):
    result = 1
    for value in values:
        result = result * value
    return result

def rearrange(tensor, pattern, **axes):
    left, right = pattern.split("->")
    source = [token.strip("()").split() for token in re.findall(r"\([^)]*\)|[A-Za-z_][A-Za-z_0-9]*", left)]
    target = [token.strip("()").split() for token in re.findall(r"\([^)]*\)|[A-Za-z_][A-Za-z_0-9]*", right)]
    shape = dict(axes)
    for group, size in zip(source, tensor.shape, strict=True):
        missing = [name for name in group if name not in shape]
        known = _product(shape[name] for name in group if name in shape)
        if len(missing) > 1:
            raise ValueError(f"Underdetermined FlowFormer reshape: {pattern}")
        if missing:
            shape[missing[0]] = size // known
    flat_source = [name for group in source for name in group]
    flat_target = [name for group in target for name in group]
    result = tensor.reshape(*(shape[name] for name in flat_source))
    result = result.permute(*(flat_source.index(name) for name in flat_target))
    return result.reshape(*(_product(shape[name] for name in group) for group in target))

class Rearrange(nn.Module):
    def __init__(self, pattern, **axes):
        super().__init__()
        self.pattern, self.axes = pattern, axes
    def forward(self, tensor):
        return rearrange(tensor, self.pattern, **self.axes)
