import torch

from fastforks._core.upload import device_tensor  # fastforks


def imagenet(img: torch.Tensor) -> torch.Tensor:
    imagenet_mean = device_tensor([0.485, 0.456, 0.406], device=img.device)  # fastforks: no host sync (a pageable upload syncs the host)
    imagenet_std = device_tensor([0.229, 0.224, 0.225], device=img.device)  # fastforks: no host sync (a pageable upload syncs the host)
    return (img - imagenet_mean[None, :, None, None]) / imagenet_std[None, :, None, None]
