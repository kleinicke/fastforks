"""FlexGlue nets: the DINOv3 ConvNeXt-B descriptor (`descriptor.py`) and the sparse matcher (`flexglue.py`).

Ported from cvg/FlexGlue 9992029 (MIT). fastforks changes: the sparse mode only (no refiner, dense or sampled
matching, early stop or `torch.compile` padding), batch 1, only the last layer's head; see each module's docstring.
"""
