"""Direct inference weight placement for the large multi-view transformer checkpoints."""

import torch


def assign_inference_weights(
    net: torch.nn.Module, state: dict[str, torch.Tensor], device: torch.device, half_prefixes: tuple[str, ...]
) -> None:
    """Fill a meta network without building an intermediate float32 copy of its half layers."""
    expected = net.state_dict()
    for name, source in state.items():
        target = expected.get(name)
        if target is None or source.shape != target.shape:
            continue  # let the strict loader report invalid keys or shapes
        weight, dtype = source, target.dtype
        cast_half = name.startswith(half_prefixes) and target.is_floating_point()
        if cast_half:
            # Keep the constructor's intermediate cast for custom dtypes that can round twice.
            if dtype != torch.float32 or weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
                weight = weight.to(dtype=dtype)
            dtype = torch.float16
        # Host-staged half weights can be packed in place later. Own the storage even if they were already half.
        make_copy = cast_half and device.type == "cpu" and weight.dtype == dtype
        state[name] = weight.to(device=device, dtype=dtype, copy=make_copy).contiguous()
    net.load_state_dict(state, strict=True, assign=True)
