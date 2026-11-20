"""The accuracy adapter uses the same detector geometry for both sparse runners."""

from eval_matching import SPARSE_OPTIONS, sparse_options
from fastforks._core import registry


def test_sparse_accuracy_detector_geometry() -> None:
    """LoMa and FlexGlue's original adapters use the 1024 px detector settings in their cards."""
    for ref in ("loma-g", "aliked-flexglue"):
        card, _ = registry.resolve("matching", ref)
        assert card.options["long_side"] == 1024
        assert card.options["max_keypoints"] == 2048
        assert sparse_options(ref) == {}
    for ref in ("aliked-lightglue", "xfeat"):
        assert sparse_options(ref) == SPARSE_OPTIONS
