# Ported from GSeanCDAT/GIMM-VFI src/models/generalizable_INR/raft/__init__.py (S-Lab License 1.0, see ../../LICENSE).
# fastforks patches: the RAFT weights come with the GIMM-VFI checkpoint (no raft-things.pth load), argparse.Namespace for the args.
from .raft import RAFT
import argparse
import torch
from .extractor import BasicEncoder


def initialize_RAFT(model_path="pretrained_ckpt/raft-things.pth", device="cuda"):
    """Initializes the RAFT model."""
    args = argparse.Namespace()  # fastforks: was ArgumentParser() used as a namespace
    args.raft_model = model_path
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    model = RAFT(args)
    # fastforks: the weights are part of the GIMM-VFI checkpoint (`flow_estimator.*`), not loaded here

    return model
