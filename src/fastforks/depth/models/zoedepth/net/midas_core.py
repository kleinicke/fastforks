# fastforks: vendored from ZoeDepth (isl-org/ZoeDepth@d87f17b, MIT, see LICENSE) zoedepth/models/base_models/midas.py
# (Resize, PrepForMidas, MidasCore), inference parts only. Changes, all marked "fastforks:": the MiDaS network comes
# from fastforks.shared.midas_beit (vendored MiDaS) instead of torch.hub, its features are returned by the network
# instead of forward hooks, torchvision's Normalize is written out, Resize's parameter prints are dropped, a very thin
# side in Resize's "minimal" mode is one multiple instead of 0, Resize rounds with Python ints instead of NumPy.
import math

import torch
import torch.nn as nn


class Resize(object):
    """Resize sample to given size (width, height).
    """

    def __init__(
        self,
        width,
        height,
        resize_target=True,
        keep_aspect_ratio=False,
        ensure_multiple_of=1,
        resize_method="lower_bound",
    ):
        """Init.
        Args:
            width (int): desired output width
            height (int): desired output height
            resize_target (bool, optional):
                True: Resize the full sample (image, mask, target).
                False: Resize image only.
                Defaults to True.
            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to False.
            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        # fastforks: parameter prints dropped
        self.__width = width
        self.__height = height

        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        # fastforks: Python ints (round half to even as np.round), not NumPy: dynamo traces NumPy into tensors
        y = int(round(x / self.__multiple_of)) * self.__multiple_of

        if max_val is not None and y > max_val:
            y = math.floor(x / self.__multiple_of) * self.__multiple_of

        if y < min_val:
            y = math.ceil(x / self.__multiple_of) * self.__multiple_of

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            # fastforks: min_val (was 0): a very thin side (aspect ratio above ~32) rounded to 0 is one multiple
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__multiple_of)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__multiple_of)
        else:
            raise ValueError(
                f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, x):
        width, height = self.get_size(*x.shape[-2:][::-1])
        return nn.functional.interpolate(x, (height, width), mode='bilinear', align_corners=True)


class PrepForMidas(object):
    def __init__(self, resize_mode="minimal", keep_aspect_ratio=True, img_size=384, do_resize=True):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        net_h, net_w = img_size
        # fastforks: = torchvision Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.normalization = lambda x: (x - 0.5) / 0.5
        self.resizer = Resize(net_w, net_h, keep_aspect_ratio=keep_aspect_ratio, ensure_multiple_of=32, resize_method=resize_mode) \
            if do_resize else nn.Identity()

    def __call__(self, x):
        return self.normalization(self.resizer(x))


class MidasCore(nn.Module):
    def __init__(self, midas, keep_aspect_ratio=True, img_size=384):
        # fastforks: inference only (frozen, features always fetched: layer_names out_conv, l4_rn, r4, r3, r2, r1)
        super().__init__()
        self.core = midas
        self.output_channels = [256] * 5  # fastforks: = MIDAS_SETTINGS["DPT_BEiT_L_384"]
        self.prep = PrepForMidas(keep_aspect_ratio=keep_aspect_ratio, img_size=img_size)

    def forward(self, x, denorm=False, return_rel_depth=False):
        # fastforks: denorm (ImageNet de-normalisation of the input) is not supported
        if denorm:
            raise ValueError("denorm is not supported")
        x = self.prep(x)
        rel_depth, out = self.core(x, features=True)  # fastforks: = the forward hooks' outputs
        if return_rel_depth:
            return rel_depth, out
        return out
