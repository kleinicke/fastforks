# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FFS checkpoint key remapping.

The FFS commercial ckpt is a research ``*_serialize.pth`` that pickles an
entire ``nn.Module`` instance. (fastforks: TAO's meta-path stub finder and
its unrestricted ``torch.load`` are removed; the family's ``model.py`` reads
the pickle with the restricted loader and hands the flat state dict here.)

Beyond the pickle stub, three load-time bridges are needed against the
TAO ``FastFoundationStereo`` wrapper. These cover only the slots that
cannot be expressed via the configurable knobs on the wrapper alone:

  - Prefix rename ``cost_agg.atts.4.* -> cost_agg.atts.cost_vol_disp.*``:
    research FFS keys the cost-volume attention by the integer ``'4'``;
    TAO ``HourGlass`` keys the same module by the string ``'cost_vol_disp'``.
    The ``HourGlass`` knobs control divisor and forward order but not
    the dict key name, so we remap at load.
  - Optional missing keys (``dx``): the wrapper registers ``dx`` as an
    int8 buffer (so device moves work via ``.to``); research FFS stores
    it as a plain attribute and the ckpt has no entry. ``strict=False``
    + reporting under ``optional_missing`` keeps the loader honest.
  - ``.bn`` / ``.IN`` -> ``.norm`` rewiring at every TAO ``Conv``: handled
    automatically by the ``_remap_norm_keys`` pre-hook registered on
    each ``Conv`` instance — this loader does not need to touch it.
"""

from typing import Dict, Tuple

import torch


# Research FFS keys cost_agg.atts by the integer '4'; TAO HourGlass keys
# the same module 'cost_vol_disp'. Prefix-based so all four state slots
# (``in_proj_*``, ``out_proj.*``, ``norm*.*``, ``self_attn.*``) remap.
#
# corr_feature_att / classifier in the ckpt are wrapped under ``.layers``;
# the wrapper uses inline ``nn.Sequential`` so the wrapper key prefix is
# ``corr_feature_att.X.*`` / ``classifier.X.*`` (no ``.layers.`` segment).
# (fastforks: the timm ``edgenext_small`` backbone keys load unchanged into ``fastforks.shared.edgenext``.)
_PREFIX_REMAP_RULES: Dict[str, str] = {
    'cost_agg.atts.4.': 'cost_agg.atts.cost_vol_disp.',
    'corr_feature_att.layers.': 'corr_feature_att.',
    'classifier.layers.': 'classifier.',
}

# Buffers/parameters that ``FastFoundationStereo`` registers but the
# research FFS ckpt does not contain. ``strict=False`` would silently let
# any missing key through; this list is the explicit whitelist.
_OPTIONAL_MISSING_KEYS: Tuple[str, ...] = (
    'dx',
)


def _strip_prefix(sd: Dict[str, torch.Tensor],
                  prefix: str = 'module.') -> Dict[str, torch.Tensor]:
    """Drop a leading prefix on every key (e.g. DDP's ``module.``)."""
    if not prefix:
        return sd
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        if k.startswith(prefix):  # noqa: FURB188 - upstream code
            k = k[len(prefix):]
        out[k] = v
    return out


def _apply_prefix_remap(sd: Dict[str, torch.Tensor],
                        rules: Dict[str, str]) -> Dict[str, torch.Tensor]:
    """Rename keys whose prefix matches an entry in ``rules`` (most-specific wins).

    Rules are tried in descending prefix-length order, so a longer (more
    specific) prefix wins over a shorter one even if both match. Raises
    ValueError on collision — i.e. the remap produces a key that the output
    already holds.
    """
    if not rules:
        return sd
    sorted_rules = sorted(rules.items(), key=lambda kv: -len(kv[0]))
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        for old, new in sorted_rules:
            if k.startswith(old):
                k = new + k[len(old):]
                break
        if k in out:
            raise ValueError(
                f"Prefix remap collision on key {k!r}; check _PREFIX_REMAP_RULES.")
        out[k] = v
    return out


def remap_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Research FFS state dict -> TAO ``FastFoundationStereo`` names (the key steps of TAO's
    ``load_ffs_pretrained``; the unpickling is done by the caller)."""
    if any(k.startswith('model.') for k in sd):
        sd = _strip_prefix(sd, prefix='model.')
    sd = _strip_prefix(sd, prefix='module.')
    return _apply_prefix_remap(sd, _PREFIX_REMAP_RULES)


_TAO_EDGENEXT_RULES: Tuple[Tuple[str, str], ...] = (
    ('.dwconv.', '.conv_dw.'),
    ('.pwconv1.', '.mlp.fc1.'),
    ('.pwconv2.', '.mlp.fc2.'),
)


def tao_edgenext_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """TAO ``EdgeNeXt`` backbone keys (``feature.downsample_layers.*``, ``feature.stages.S.N.*``, ``dwconv`` /
    ``pwconv1`` / ``pwconv2``) -> the timm layout of ``fastforks.shared.edgenext`` (``feature.stem.*``,
    ``feature.stages.S.{downsample,blocks.N}.*``, ``conv_dw`` / ``mlp.fc1`` / ``mlp.fc2``); other keys unchanged."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in sd.items():
        parts = k.split('.')
        if parts[0] == 'feature' and parts[1] == 'downsample_layers':
            i = int(parts[2])
            k = '.'.join(['feature', 'stem'] + parts[3:] if i == 0 else ['feature', 'stages', str(i), 'downsample'] + parts[3:])
        elif parts[0] == 'feature' and parts[1] == 'stages':
            k = '.'.join(parts[:3] + ['blocks'] + parts[3:])
            for old, new in _TAO_EDGENEXT_RULES:
                k = k.replace(old, new)
        out[k] = v
    return out
