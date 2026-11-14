# Vendored from huggingface/controlnet_aux b1aac318b15c4e65a21bc516dc236ad4bf593d46, src/controlnet_aux/teed/Fsmish.py (from
# xavysp/TEED, MIT). Unchanged except the removed `torch.jit.script` decorator.

"""
Script based on:
Wang, Xueliang, Honge Ren, and Achuan Wang.
 "Smish: A Novel Activation Function for Deep Learning Methods.
 " Electronics 11.4 (2022): 540.
"""

# import pytorch
import torch


# @torch.jit.script  # fastforks: removed (deprecated, warns on import in torch 2.14); a plain function
def smish(input):
    """
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(sigmoid(x))))
    See additional documentation for mish class.
    """
    return input * torch.tanh(torch.log(1 + torch.sigmoid(input)))
