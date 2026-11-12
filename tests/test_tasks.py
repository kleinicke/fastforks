"""The depth, classification, segmentation, features, promptable segmentation and multiview front ends, on fake families
(CPU)."""

from __future__ import annotations

import importlib
import importlib.util
import types
from typing import TYPE_CHECKING, Any, ClassVar, assert_type, get_args

import numpy as np
import pytest
import torch

from fastforks import (
    BackendUnavailableError,
    CardError,
    InputError,
    ModelNotFoundError,
    classification,
    depth,
    features,
    multiview,
    promptable_segmentation,
    segmentation,
)
from fastforks._core import registry, tasks
from fastforks._core.bench import BenchSize, bench_inputs
from fastforks._core.cards import Io, parse_card
from fastforks._core.instances import Instances
from fastforks._core.labels import label_set
from fastforks._core.model import Backend
from fastforks._core.preprocess import resize_crop
from fastforks.classification._runner import Runner as ClsRunner
from fastforks.depth._runner import Runner as DepthRunner
from fastforks.detection._runner import Runner as DetRunner
from fastforks.features._runner import Forward
from fastforks.features._runner import Runner as FeatRunner
from fastforks.flow._runner import Runner as FlowRunner
from fastforks.matching._runner import Runner as MatchingRunner
from fastforks.matting._runner import Runner as MattingRunner
from fastforks.multiview._runner import Runner as MultiViewRunner
from fastforks.pose._runner import Runner as PoseRunner
from fastforks.promptable_segmentation._api import MAX_EXEMPLARS, ConceptModelId, _model_class
from fastforks.promptable_segmentation._runner import Runner as PromptRunner
from fastforks.segmentation._runner import Runner as SegRunner
from fastforks.tracking._runner import Runner as TrackingRunner

W0 = {"name": "w.pth", "size": 1, "sha256": "0" * 64, "url": "https://example.com/w.pth"}


pytestmark = pytest.mark.usefixtures("card_min_size")  # tiny images through fake runners


def card_data(outputs, output, io=None):
    return {
        "schema": 2,
        "name": "Fake",
        "summary": "Fake family for tests.",
        "tier": "experimental",
        "outputs": outputs,
        "output": output,
        "upstream": {"repo": "https://example.com/fake", "commit": "0" * 40},
        "licence": {"code": "MIT", "weights": "MIT", "files": [], "training_data": ["unstated"]},
        "io": io or {"input": "0-1"},
        "modes": {"portable": {"cuda_dtype": "float32"}},
        "variants": [{"id": "fake", "revision": 1, "weights": [W0]}],
    }


@pytest.fixture
def fake(monkeypatch):
    """`fake(task, runner, outputs, output, io)`: make `<task>.load("fake")` build `runner`."""

    def install(task, runner, outputs, output, io=None):
        card = parse_card(card_data(outputs, output, io), family="fake", task=task, where="fake card")
        monkeypatch.setattr(registry, "resolve", lambda t, _m: (card, card.variants[0]) if t == task else None)
        monkeypatch.setattr(registry, "import_model", lambda _c: types.SimpleNamespace(RUNNERS={"portable": runner}))

    return install


class Depth(DepthRunner):
    batch = None
    intrinsics = "optional"

    def __call__(self, image, intrinsics):
        d = 1 + image[:, 0]
        out = {"depth": d, "mask": d > 1.5}
        if intrinsics is None:
            out["intrinsics"] = torch.eye(3).expand(image.shape[0], 3, 3)
        return out


def test_depth(fake):
    fake("depth", Depth, ["depth", "mask", "intrinsics"], {"kind": "metric"})
    m = depth.load("fake", device="cpu")
    img = np.full((6, 7, 3), 255, np.uint8)
    r = m(img)
    assert r.depth is not None
    assert r.mask is not None
    assert r.kind == "metric"
    assert r.depth.shape == (6, 7)
    assert r.depth.dtype == torch.float32
    assert r.mask.dtype == torch.bool
    assert r.inverse_depth is None
    torch.testing.assert_close(r.intrinsics, torch.eye(3))
    k = torch.tensor([[100.0, 0, 3], [0, 100, 3], [0, 0, 1]])
    rb = m(np.stack([img, img]), intrinsics=k)
    assert rb.depth is not None
    assert rb.depth.shape == (2, 6, 7)
    torch.testing.assert_close(rb.intrinsics, k.expand(2, 3, 3))  # given intrinsics are passed through
    with pytest.raises(InputError, match="intrinsics"):
        m(img, intrinsics=torch.eye(4))
    with pytest.raises(InputError, match="1 or 2 matrices"):
        m(np.stack([img, img]), intrinsics=torch.eye(3).expand(3, 3, 3))


def test_depth_intrinsics_policy(fake):
    class Unused(Depth):
        intrinsics = "unused"

    class Required(Depth):
        intrinsics = "required"

    fake("depth", Unused, ["depth", "mask", "intrinsics"], {"kind": "scale_invariant"})
    with pytest.raises(InputError, match="does not use camera intrinsics"):
        depth.load("fake", device="cpu")(np.zeros((4, 4, 3), np.uint8), intrinsics=torch.eye(3))
    fake("depth", Required, ["depth", "mask", "intrinsics"], {"kind": "scale_invariant"})
    with pytest.raises(InputError, match="needs the camera intrinsics"):
        depth.load("fake", device="cpu")(np.zeros((4, 4, 3), np.uint8))
    with pytest.raises(TypeError, match="positional"):
        depth.load("fake", device="cpu")(np.zeros((4, 4, 3), np.uint8), torch.eye(3))  # ty: ignore[too-many-positional-arguments]


def test_depth_card_rules():
    with pytest.raises(CardError, match="kind"):
        parse_card(card_data(["depth"], {"kind": "absolute"}), family="f", task="depth", where="c")
    with pytest.raises(CardError, match="must include"):
        parse_card(card_data(["confidence"], {"kind": "metric"}), family="f", task="depth", where="c")
    card = parse_card(card_data(["depth", "rays"], {"kind": "metric"}), family="f", task="depth", where="c")
    assert card.outputs == ("depth", "rays")  # non-standard maps are allowed; they go to DepthResult.extras


class Cls(ClsRunner):
    batch = 2
    seen: list[tuple[int, ...]] = []  # noqa: RUF012 - test log

    def __call__(self, image):
        Cls.seen.append(tuple(image.shape))
        logits = torch.zeros(image.shape[0], 1000)
        logits[:, 3] = image.mean((1, 2, 3)) * 10
        return {"logits": logits}


def test_classification(fake):
    Cls.seen = []
    io = {"input": "mean_std", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "resize": 8, "crop": 6}
    fake("classification", Cls, ["logits"], {"label_set": "imagenet-1k"}, io)
    m = classification.load("fake", device="cpu")
    assert m.class_names[0] == "tench"
    r = m(np.full((10, 20, 3), 255, np.uint8))
    assert Cls.seen == [(1, 3, 6, 6)]  # shorter side 8, centre crop 6
    assert r.logits.shape == (1000,)
    torch.testing.assert_close(r.logits[3], torch.tensor(10.0))  # (1 - 0.5) / 0.5 = 1
    top = r.top(2)
    assert top[0][0] == "tiger shark"
    assert top[0][1] == pytest.approx(r.probs[3].item())
    rb = m(np.zeros((5, 10, 10, 3), np.uint8))  # batch 5 in chunks of 2
    assert rb.logits.shape == (5, 1000)
    assert [s[0] for s in Cls.seen[1:]] == [2, 2, 1]
    assert len(rb.top(1)) == 5
    assert m.num_classes == 1000
    assert m.id2label[3] == "tiger shark"
    assert len(m.id2label) == m.num_classes


def test_classification_untagged_timm_name():
    if "resnet50.tv2_in1k" not in registry.cards("classification"):
        pytest.skip("no resnet50 card")
    assert registry.resolve("classification", "resnet50")[1].id == "resnet50.tv2_in1k"
    assert registry.resolve("classification", "resnet50@1")[1].ref == "resnet50.tv2_in1k@1"
    assert (
        classification.model_card("resnet50").family
        == registry.resolve("classification", "resnet50.tv2_in1k")[0].family
    )


def test_classification_bad_logits(fake):
    class Wrong(Cls):
        def __call__(self, image):
            return {"logits": torch.zeros(image.shape[0], 10)}

    fake("classification", Wrong, ["logits"], {"label_set": "imagenet-1k"})
    with pytest.raises(CardError, match="expected"):
        classification.load("fake", device="cpu")(np.zeros((8, 8, 3), np.uint8))


def test_unknown_label_set():
    with pytest.raises(CardError, match="unknown label set"):
        parse_card(card_data(["logits"], {"label_set": "nope"}), family="f", task="classification", where="c")


class Seg(SegRunner):
    def __call__(self, image):
        logits = torch.zeros(image.shape[0], 150, *image.shape[-2:], dtype=torch.float16)
        logits[:, 2] = image[:, 0]
        return {"logits": logits}


def test_segmentation(fake):
    fake("segmentation", Seg, ["logits"], {"label_set": "ade20k-150"})
    m = segmentation.load("fake", device="cpu")
    img = np.zeros((5, 6, 3), np.uint8)
    img[:, 3:, 0] = 255
    r = m(img)
    assert r.logits is None
    assert r.labels.dtype == torch.int64
    assert r.labels.shape == (5, 6)
    assert (r.labels[:, 3:] == 2).all()
    assert m.class_names[2] == "sky"
    rb = m(np.stack([img, img]), logits=True)
    assert rb.logits is not None
    assert rb.labels.shape == (2, 5, 6)
    assert rb.logits.shape == (2, 150, 5, 6)
    assert rb.logits.dtype == torch.float32


def test_segmentation_probabilities_become_log_space(fake):
    class Probs(SegRunner):
        probabilities = True  # mask classification: per-class probabilities, not logits

        def __call__(self, image):
            p = torch.zeros(image.shape[0], 150, *image.shape[-2:])
            p[:, 2] = 0.5 * image[:, 0]
            p[:, 4] = 0.25
            return {"logits": p}

    fake("segmentation", Probs, ["logits"], {"label_set": "ade20k-150"})
    img = np.zeros((5, 6, 3), np.uint8)
    img[:, 3:, 0] = 255
    r = segmentation.load("fake", device="cpu")(img, logits=True)
    assert r.logits is not None
    torch.testing.assert_close(r.logits[2, 0, 3], torch.tensor(0.5).log())
    torch.testing.assert_close(r.logits[4, 0, 0], torch.tensor(0.25).log())
    assert torch.isfinite(r.logits).all()  # zero probabilities are floored
    assert (r.logits.argmax(0) == r.labels).all()
    assert (r.labels[:, 3:] == 2).all()
    assert (r.labels[:, :3] == 4).all()


def test_module_of_a_runner_without_one(fake):
    fake("segmentation", Seg, ["logits"], {"label_set": "ade20k-150"})
    with pytest.raises(BackendUnavailableError, match="no exportable module"):
        segmentation.load("fake", device="cpu").module()


class Prompt(PromptRunner):
    """Low-res logits: +1 left of the first point's x (or the box's x1), -1 right of it."""

    batch = 1
    prompts = frozenset({"points", "boxes"})
    mask_size = (4, 4)

    def embed(self, image):
        return {"mean": image.mean()[None]}

    def predict(self, features, size, *, points, labels, boxes, masks, multimask):  # noqa: ARG002, PLR0913 - interface
        x = points[:, 0, 0] if points is not None else boxes[:, 2]
        cols = (torch.arange(4) + 0.5) * size[1] / 4
        m = 3 if multimask else 1
        low = torch.where(cols < x[:, None], 1.0, -1.0)[:, None, None, :].expand(-1, m, 4, 4) + 0 * features["mean"]
        return {"pred_masks": low, "iou_scores": torch.linspace(0, 1, m).expand(low.shape[0], m)}

    def upsample(self, pred_masks, size):
        return torch.nn.functional.interpolate(pred_masks, size, mode="nearest")


def test_promptable_segmentation(fake):
    fake("promptable_segmentation", Prompt, ["pred_masks", "iou_scores"], {})
    m = promptable_segmentation.load("fake", device="cpu")
    assert m.prompts == {"points", "boxes"}
    img = np.zeros((8, 12, 3), np.uint8)
    r = m(img, input_points=[[6.0, 2.0]], input_labels=[1])  # no P: dropped from the result
    assert r.masks.shape == (3, 8, 12)
    assert r.masks.dtype == torch.bool
    assert r.masks[:, :, :6].all()
    assert not r.masks[:, :, 6:].any()
    assert r.iou_scores.shape == (3,)
    assert r.pred_masks.shape == (3, 4, 4)
    emb = m.get_image_embeddings(img)
    assert isinstance(emb, promptable_segmentation.ImageEmbeddings)
    rb = m(emb, input_boxes=[[0, 0, 3, 8], [0, 0, 12, 8]], multimask_output=False)  # P = 2, one mask each
    assert rb.masks.shape == (2, 1, 8, 12)
    assert rb.masks[0, 0, :, :3].all()
    assert not rb.masks[0, 0, :, 3:].any()
    assert rb.masks[1].all()
    embs = m.get_image_embeddings(np.stack([img, img]))  # one object for the batch, embs[i] per image
    assert isinstance(embs, promptable_segmentation.ImageEmbeddings)
    assert len(embs) == 2
    assert embs[1].image_size == (8, 12)
    assert len(embs[-1]) == 1
    assert m(embs[1], input_boxes=[0, 0, 3, 8]).masks.shape == (3, 8, 12)
    with pytest.raises(InputError, match="pass emb"):
        m(embs, input_boxes=[0, 0, 3, 8])
    with pytest.raises(IndexError):
        embs[2]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "no prompt given"),
        ({"input_points": [[1.0, 2.0]]}, "go together"),
        ({"input_masks": np.zeros((1, 4, 4))}, "takes no masks"),
        ({"input_points": [[1.0, 2.0, 3.0]], "input_labels": [1]}, "must end in"),
        ({"input_points": [[1.0, 2.0]], "input_labels": [1, 1]}, "without the 2"),
        ({"input_points": [[[1.0, 2.0]]], "input_labels": [[1]], "input_boxes": [[0, 0, 1, 1]] * 2}, "disagree"),
        ({"input_boxes": [0, 0, float("nan"), 1]}, "non-finite"),
        ({"input_boxes": "abc"}, "numeric"),
    ],
)
def test_promptable_segmentation_bad_prompts(fake, kwargs, match):
    fake("promptable_segmentation", Prompt, ["pred_masks", "iou_scores"], {})
    with pytest.raises(InputError, match=match):
        promptable_segmentation.load("fake", device="cpu")(np.zeros((8, 12, 3), np.uint8), **kwargs)


def test_promptable_segmentation_rejects_foreign_embeddings_and_batches(fake):
    fake("promptable_segmentation", Prompt, ["pred_masks", "iou_scores"], {})
    m = promptable_segmentation.load("fake", device="cpu")
    img = np.zeros((8, 12, 3), np.uint8)
    emb = m.get_image_embeddings(img)
    assert isinstance(emb, promptable_segmentation.ImageEmbeddings)
    other = type(emb)(emb.features, emb.image_size, "other@1", emb.backend)
    with pytest.raises(InputError, match="embeddings of other@1"):
        m(other, input_boxes=[0, 0, 1, 1])
    other_options = type(emb)(emb.features, emb.image_size, emb.model, emb.backend, {"size": 512})
    with pytest.raises(InputError, match="options"):
        m(other_options, input_boxes=[0, 0, 1, 1])
    with pytest.raises(InputError, match="one image"):
        m(np.stack([img, img]), input_boxes=[0, 0, 1, 1])


class Concept(Prompt):
    """Text prompts: 3 queries scored 0.9 / 0.2 / 0.7, query i's mask is the left (i + 1) quarter of the image."""

    prompts = frozenset({"text"})

    def segment_concept(self, features, text, boxes=None, labels=None):  # noqa: ARG002 - interface
        scores = torch.tensor([0.9, 0.2, 0.7]) if text == "coin" else torch.zeros(3)
        boxes = torch.tensor([[0.0, 0.0, 0.25, 1.0], [0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 0.75, 1.0]])
        low = torch.where(torch.arange(4) < torch.arange(1, 4)[:, None], 1.0, -1.0)[:, None, :].expand(3, 4, 4)
        return {"scores": scores, "boxes": boxes, "pred_masks": low + 0 * features["mean"]}


def test_concept_segmentation(fake):
    fake("promptable_segmentation", Concept, ["boxes", "scores", "masks"], {})
    m = promptable_segmentation.load("fake", device="cpu")
    img = np.zeros((8, 12, 3), np.uint8)
    r = m(img, text="coin")  # queries above 0.5, best first
    assert isinstance(r, promptable_segmentation.ConceptSegmentationResult)
    torch.testing.assert_close(r.scores, torch.tensor([0.9, 0.7]))
    torch.testing.assert_close(r.boxes, torch.tensor([[0.0, 0.0, 3.0, 8.0], [0.0, 0.0, 9.0, 8.0]]))
    assert r.masks is not None
    assert r.masks.shape == (2, 8, 12)
    assert r.masks[0, :, :3].all()
    assert not r.masks[0, :, 3:].any()
    assert r.masks[1, :, :9].all()
    assert len(m(m.get_image_embeddings(img), text="coin", threshold=0.1)) == 3
    none = m(img, text="giraffe")  # nothing found: empty, well-shaped
    assert len(none) == 0
    assert none.masks is not None
    assert none.masks.shape == (0, 8, 12)
    assert none.masks.dtype == torch.bool
    assert none.boxes.shape == (0, 4)
    bad: list[tuple[dict[str, Any], str]] = [
        ({"text": ""}, "non-empty"),
        ({"text": "coin", "input_boxes": [0, 0, 1, 1]}, "takes no boxes"),
        ({"input_boxes": [0, 0, 1, 1]}, "takes no boxes"),
        ({"text": "coin", "input_points": [[1.0, 2.0]], "input_labels": [1]}, "takes text and / or"),
        ({"text": "coin", "input_boxes_labels": [1]}, "needs input_boxes"),
        ({}, "no prompt given"),
    ]
    for kwargs, match in bad:
        with pytest.raises(InputError, match=match):
            m(img, **kwargs)


class Exemplar(Concept):
    """Text and exemplar boxes: records the runner's arguments."""

    prompts = frozenset({"text", "boxes"})
    calls: ClassVar[list[tuple[Any, ...]]] = []

    def segment_concept(self, features, text, boxes=None, labels=None):
        type(self).calls.append((text, boxes, labels))
        return super().segment_concept(features, "coin", boxes, labels)


def test_exemplar_boxes(fake):
    fake("promptable_segmentation", Exemplar, ["boxes", "scores", "masks"], {})
    m = promptable_segmentation.load("fake", device="cpu")
    img = np.zeros((8, 12, 3), np.uint8)
    Exemplar.calls.clear()
    alone = m(img, input_boxes=[0, 0, 6, 4])  # text None (the runner says "visual")
    assert isinstance(alone, promptable_segmentation.ConceptSegmentationResult)
    assert len(alone) == 2
    assert len(m(img, text="coin", input_boxes=[[0, 0, 6, 4], [3, 2, 12, 8]], input_boxes_labels=[1, 0])) == 2
    (t0, b0, l0), (t1, b1, l1) = Exemplar.calls
    assert t0 is None
    assert l0 is None
    torch.testing.assert_close(b0, torch.tensor([[0.0, 0.0, 0.5, 0.5]]))  # normalised x / W, y / H, on the CPU
    assert t1 == "coin"
    torch.testing.assert_close(b1, torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 1.0, 1.0]]))
    assert l1.tolist() == [1, 0]
    assert l1.dtype == torch.int64
    bad: list[tuple[dict[str, Any], str]] = [
        ({"input_boxes": [[0, 0, 1]]}, r"\[N, 4\]"),
        ({"input_boxes": [[[0, 0, 1, 1]]]}, r"\[N, 4\]"),
        ({"input_boxes": [[0, 0, 1, 1]], "input_boxes_labels": [1, 1]}, "2 labels for 1 boxes"),
        ({"input_boxes": [[0, 0, 1, 1]], "input_boxes_labels": [-1]}, "positive"),
        ({"input_boxes": [[0, 0, float("inf"), 1]]}, "non-finite"),
        ({"input_masks": np.zeros((1, 4, 4))}, "takes text and / or"),
    ]
    for kwargs, match in bad:
        with pytest.raises(InputError, match=match):
            m(img, **kwargs)


def test_exemplar_box_checks(fake, monkeypatch):
    fake("promptable_segmentation", Exemplar, ["boxes", "scores", "masks"], {})
    m = promptable_segmentation.load("fake", device="cpu")
    assert isinstance(m, promptable_segmentation.ConceptSegmentationModel)  # "fake" is no `ConceptModelId`
    img = np.zeros((8, 12, 3), np.uint8)
    Exemplar.calls.clear()
    assert len(m(img, text="coin", input_boxes=[])) == 2  # empty exemplar set + text: text alone
    assert len(m(img, text="coin", input_boxes=np.zeros((0, 4)))) == 2
    assert len(m(img, input_boxes=[[0, 0, 6, 4], [3, 2, 12, 8]], input_boxes_labels=[0, 0])) == 2  # negatives only
    assert len(m(img, input_boxes=[[-12, -8, 24, 16]])) == 2  # up to one image size outside: legal
    (t0, b0, l0), (_, b1, _), (_, _, l2), (_, b3, _) = Exemplar.calls
    assert (t0, b0, l0, b1) == ("coin", None, None, None)
    assert l2.tolist() == [0, 0]
    torch.testing.assert_close(b3, torch.tensor([[-1.0, -1.0, 2.0, 2.0]]))
    many = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), (MAX_EXEMPLARS, 1))
    assert len(m(img, input_boxes=many)) == 2
    bad: list[tuple[dict[str, Any], str]] = [
        ({"input_boxes": []}, "empty and no text"),
        ({"input_boxes": np.zeros((0, 4))}, "empty and no text"),
        ({"input_boxes": [[0, 0, 6, 4], [6, 0, 6, 4]]}, r"x1 > x0 and y1 > y0; box 1"),  # zero width
        ({"input_boxes": [[0, 4, 6, 2]]}, r"x1 > x0 and y1 > y0; box 0"),  # y1 < y0
        ({"input_boxes": [[0, 0, 1e9, 4]]}, "within one image size"),  # would make huge ROI grids
        ({"input_boxes": [[-13, 0, 6, 4]]}, "within one image size"),
        ({"input_boxes": np.concatenate([many, many[:1]])}, f"at most {MAX_EXEMPLARS}"),
        ({"input_boxes": [0, 0, 6, 4], "input_boxes_labels": 2}, "1 \\(positive\\) or 0"),
        ({"input_boxes": [0, 0, 6, 4], "input_boxes_labels": -10}, "no padding"),  # HF pad label
        ({"text": "coin", "input_points": [[1.0, 2.0]]}, "takes text and / or"),
    ]
    embedded = []
    with monkeypatch.context() as mp:
        mp.setattr(type(m), "_embed_one", lambda *_: embedded.append(1))  # any image encoder call would show
        mp.setattr(type(m), "_embed_tensor", lambda *_: embedded.append(1))
        for kwargs, match in bad:  # all caught before the image encoder runs (the size checks read the image size)
            with pytest.raises(InputError, match=match):
                m(img, **kwargs)
    assert not embedded


def test_concept_model_typing(fake):
    fake("promptable_segmentation", Exemplar, ["boxes", "scores", "masks"], {})
    m = promptable_segmentation.load("fake", device="cpu")
    assert type(m) is promptable_segmentation.ConceptSegmentationModel
    img = np.zeros((8, 12, 3), np.uint8)
    assert isinstance(m, promptable_segmentation.ConceptSegmentationModel)
    # `assert_type` is a no-op at run time (only `ty` checks it); the isinstance asserts hold under pytest as well.
    for out in (m(img, input_boxes=[0, 0, 6, 4]), m(img, text="coin", input_boxes=[0, 0, 6, 4]), m(img, text="coin")):
        assert isinstance(out, promptable_segmentation.ConceptSegmentationResult)
    assert_type(
        m(img, input_boxes=[0, 0, 6, 4]), promptable_segmentation.ConceptSegmentationResult
    )  # boxes alone, no labels: a concept call
    assert_type(m(img, text="coin", input_boxes=[0, 0, 6, 4]), promptable_segmentation.ConceptSegmentationResult)
    assert_type(m(img, text="coin"), promptable_segmentation.ConceptSegmentationResult)
    if TYPE_CHECKING:  # load picks the class from the id (ty checks these lines; never run)
        sam3 = promptable_segmentation.load("sam3")
        assert_type(sam3, promptable_segmentation.ConceptSegmentationModel)
        assert_type(sam3(img, input_boxes=[0, 0, 6, 4]), promptable_segmentation.ConceptSegmentationResult)
        sam = promptable_segmentation.load("sam-vit-b")
        assert_type(sam(img, input_boxes=[0, 0, 6, 4]), promptable_segmentation.PromptableSegmentationResult)


def test_concept_model_ids():
    """`ConceptModelId` (typing of `load`) lists exactly the installed concept models, also pinned (`@1`)."""
    ids = promptable_segmentation.list_models()
    concept = {i for i in ids if _model_class(i) is promptable_segmentation.ConceptSegmentationModel}
    assert concept
    assert {i.split("@")[0] for i in get_args(ConceptModelId)} == concept
    for i in get_args(ConceptModelId):
        assert _model_class(i) is promptable_segmentation.ConceptSegmentationModel
    assert _model_class("no-such-model") is promptable_segmentation.PromptableSegmentationModel


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"text": "coin"}, "takes no text"),
        ({"input_boxes": [0, 0, 1, 1], "threshold": 0.3}, "not a concept model"),
        ({"input_boxes": [0, 0, 1, 1], "input_boxes_labels": [1]}, "not a concept model"),
    ],
)
def test_concept_arguments_need_a_concept_model(fake, kwargs, match):
    fake("promptable_segmentation", Prompt, ["pred_masks", "iou_scores"], {})
    with pytest.raises(InputError, match=match):
        promptable_segmentation.load("fake", device="cpu")(np.zeros((8, 12, 3), np.uint8), **kwargs)


@pytest.mark.parametrize(
    ("hw", "resize", "crop", "want"),
    [
        ((10, 20), 8, 6, (6, 6)),
        ((20, 10), 8, 0, (16, 8)),
        ((4, 4), 0, 6, (6, 6)),
        ((7, 9), 7, 0, (7, 9)),
        ((427, 640), 256, 0, (256, 383)),  # torchvision truncates the long side
        ((640, 427), 256, 0, (383, 256)),
        ((375, 500), 256, 224, (224, 224)),
    ],
)
def test_resize_crop(hw, resize, crop, want):
    io = Io(resize=resize, crop=crop)
    assert tuple(resize_crop(torch.ones(1, 3, *hw), io).shape[-2:]) == want


def test_resize_crop_bicubic_stays_in_range():
    step = torch.zeros(1, 3, 32, 32)
    step[..., 16:] = 1
    out = resize_crop(step, Io(resize=13, interpolation="bicubic"))
    assert out.min() >= 0
    assert out.max() <= 1


@pytest.mark.parametrize(
    "task",
    [
        "stereo",
        "depth",
        "classification",
        "segmentation",
        "features",
        "promptable_segmentation",
        "multiview",
        "pose",
        "flow",
        "matching",
        "matting",
        "tracking",
    ],
)
def test_runner_without_call_cannot_be_built(task):
    base = {"stereo": __import__("fastforks.stereo._runner", fromlist=["Runner"]).Runner, "depth": DepthRunner}
    base |= {"classification": ClsRunner, "segmentation": SegRunner, "features": FeatRunner}
    base["promptable_segmentation"] = PromptRunner
    base["multiview"] = MultiViewRunner
    base["pose"] = PoseRunner
    base["flow"] = FlowRunner
    base["matching"] = MatchingRunner
    base["matting"] = MattingRunner
    base["tracking"] = TrackingRunner

    class NoCall(base[task]):
        pass

    with pytest.raises(TypeError, match="abstract"):
        NoCall(None)


def test_unknown_task_has_no_spec():
    assert set(tasks.TASK_SPECS) == set(registry.TASKS)
    with pytest.raises(KeyError):
        tasks.spec("nope")


def test_tasks_are_discovered_from_their_spec_modules():
    assert registry.TASKS[:4] == ("stereo", "depth", "classification", "segmentation")  # TaskSpec.order
    orders = [tasks.spec(t).order for t in registry.TASKS]
    assert orders == sorted(orders)
    for task in registry.TASKS:  # a spec module names a task package with model cards
        assert importlib.util.find_spec(f"fastforks.{task}.models") is not None, task


class Extras(DepthRunner):
    batch = 1
    intrinsics = "optional"

    def __call__(self, image, intrinsics):
        b = image.shape[0]
        f = intrinsics[:, 0, 0] if intrinsics is not None else torch.ones(b)
        return {"depth": f[:, None, None].expand(b, *image.shape[-2:]), "fov": f * 2}


def test_depth_extras_and_chunked_intrinsics(fake):
    fake("depth", Extras, ["depth", "fov"], {"kind": "metric"})
    m = depth.load("fake", device="cpu")
    k = np.stack([np.diag([f, f, 1.0]) for f in (10.0, 20.0, 30.0)])  # numpy intrinsics, one per image
    r = m(np.zeros((3, 4, 5, 3), np.uint8), intrinsics=k)  # runner batch 1: k is split with the images
    assert r.depth is not None
    torch.testing.assert_close(r.depth[:, 0, 0], torch.tensor([10.0, 20.0, 30.0]))
    torch.testing.assert_close(r.extras["fov"], torch.tensor([20.0, 40.0, 60.0]))
    r1 = m(np.zeros((4, 5, 3), np.uint8), intrinsics=[[5, 0, 2], [0, 5, 2], [0, 0, 1]])  # nested lists
    assert r1.extras["fov"].shape == ()
    with pytest.raises(InputError, match="intrinsics"):
        m(np.zeros((4, 5, 3), np.uint8), intrinsics="K")  # the error path


@pytest.mark.parametrize(
    ("maps", "match"),
    [
        (lambda x: {"depth": x[:, 0, :2]}, "shape"),
        (lambda x: {"confidence": x[:, 0]}, "at least one of"),
        (lambda x: {"depth": x[:, 0], "normals": x}, "card outputs"),
        (lambda x: {"depth": x[:, 0], "mask": x[:, 0]}, "torch.bool"),
        (lambda x: {"depth": x[:, 0], "confidence": x[:, 0] > 0}, "must be numbers"),
    ],
)
def test_depth_bad_runner_outputs(fake, maps, match):
    class Bad(DepthRunner):
        def __call__(self, image, intrinsics):  # noqa: ARG002 - unused
            return maps(image)

    fake("depth", Bad, ["depth", "mask", "confidence"], {"kind": "metric"})
    with pytest.raises(CardError, match=match):
        depth.load("fake", device="cpu")(np.zeros((4, 5, 3), np.uint8))


def test_half_precision_runner_outputs_become_float32(fake):
    class Half(DepthRunner):
        def __call__(self, image, intrinsics):  # noqa: ARG002 - unused
            d = image[:, 0].half()
            return {"depth": d, "mask": d > 0.5, "confidence": d}

    fake("depth", Half, ["depth", "mask", "confidence"], {"kind": "metric"})
    r = depth.load("fake", device="cpu")(np.zeros((4, 5, 3), np.uint8))
    dtypes = [None if t is None else t.dtype for t in (r.depth, r.mask, r.confidence)]
    assert dtypes == [torch.float32, torch.bool, torch.float32]


def test_runner_without_logits(fake):
    class NoLogits(Cls):
        def __call__(self, image):  # noqa: ARG002 - unused
            return {}

    fake("classification", NoLogits, ["logits"], {"label_set": "imagenet-1k"})
    with pytest.raises(CardError, match="at least one of"):
        classification.load("fake", device="cpu")(np.zeros((8, 8, 3), np.uint8))


def test_segmentation_logits_across_chunks(fake):
    class Seg1(Seg):
        batch = 1

    fake("segmentation", Seg1, ["logits"], {"label_set": "ade20k-150"})
    r = segmentation.load("fake", device="cpu")(np.zeros((3, 4, 5, 3), np.uint8), logits=True)
    assert r.logits is not None
    assert r.logits.shape == (3, 150, 4, 5)
    assert r.labels.shape == (3, 4, 5)


def test_data_config(fake):
    io = {"input": "imagenet", "resize": 256, "crop": 224, "interpolation": "bicubic"}
    fake("classification", Cls, ["logits"], {"label_set": "imagenet-1k"}, io)
    cfg = classification.load("fake", device="cpu").data_config
    assert cfg["input_size"] == (3, 224, 224)
    assert cfg["crop_pct"] == pytest.approx(0.875)
    assert cfg["interpolation"] == "bicubic"
    assert cfg["mean"] == (0.485, 0.456, 0.406)


@pytest.mark.parametrize(
    ("task", "outputs", "output", "io", "match"),
    [
        ("stereo", ["disparity"], {"kind": "metric"}, None, "unknown"),
        ("classification", ["logits", "features"], {"label_set": "imagenet-1k"}, None, "must be among"),
        ("depth", ["depth"], {"kind": "metric"}, {"input": "0-1", "resize": 518}, "unknown"),
        ("stereo", ["disparity"], {}, {"input": "imagenet", "mean": [0.1, 0.1, 0.1]}, "mean_std"),
        ("depth", ["depth", "depth"], {"kind": "metric"}, None, "unique"),
        ("classification", ["logits"], {"labels": "imagenet-1k"}, None, "unknown"),
    ],
)
def test_card_task_rules(task, outputs, output, io, match):
    with pytest.raises(CardError, match=match):
        parse_card(card_data(outputs, output, io), family="f", task=task, where="c")


def test_variant_output_overrides():
    data = card_data(["inverse_depth"], {"kind": "affine_invariant_inverse"})
    data["variants"].append(
        {"id": "fake-metric", "revision": 1, "weights": [W0], "outputs": ["depth"], "output": {"kind": "metric"}}
    )
    card = parse_card(data, family="f", task="depth", where="c")
    rel, metric = card.variants
    assert (rel.outputs, rel.output_info["kind"]) == (("inverse_depth",), "affine_invariant_inverse")
    assert (metric.outputs, metric.output_info["kind"]) == (("depth",), "metric")
    data["variants"][1]["output"] = {"kind": "absolute"}
    with pytest.raises(CardError, match="kind"):
        parse_card(data, family="f", task="depth", where="c")


@pytest.mark.parametrize(("name", "count"), [("imagenet-1k", 1000), ("ade20k-150", 150), ("cityscapes-19", 19)])
def test_label_sets(name, count):
    names = label_set(name)
    assert len(names) == count
    assert all(n.strip() == n and n for n in names)


def test_model_ids_unique_across_tasks():
    seen: dict[str, str] = {}
    for task in registry.TASKS:
        for mid in {*registry.cards(task), *registry.default_tags(task)}:  # untagged timm names too
            assert mid not in seen, f"{mid} is in {seen.get(mid)} and {task}"
            seen[mid] = task


class FeedForward(MultiViewRunner):
    min_views = 2
    first = torch.tensor(0.0)  # pixel value of the view the test expects first

    def __call__(self, images, intrinsics, cam_to_world, depth_range):  # noqa: ARG002 - the Runner signature
        n = images.shape[0]
        c2w = torch.eye(4).repeat(n, 1, 1)
        c2w[:, 0, 3] = images[:, 0, 0, 0]  # identifies the view; the reference (images[0]) must come first
        assert images[0, 0, 0, 0] == self.first
        return {"depth": 1 + images[:, 0], "cam_to_world": c2w, "conf3d": images[:, :1]}


class Calibrated(MultiViewRunner):
    intrinsics = "required"
    cam_to_world = "required"
    depth_range = "optional"
    per_view = False
    min_views = 2
    max_views = 4

    def __call__(self, images, intrinsics, cam_to_world, depth_range):  # noqa: ARG002 - the Runner signature
        far = depth_range[1] if depth_range is not None else torch.tensor(9.0)
        d = (1 + images[:1, 0]) * cam_to_world[0, 0, 3]  # reference view first
        return {"depth": d, "mask": d < far, "range": torch.stack([far, far])[None]}


def _views(n=3, h=4, w=5):
    return [np.full((h, w, 3), 10 * (i + 1), np.uint8) for i in range(n)]


def test_multiview_feed_forward(fake):
    fake("multiview", FeedForward, ["depth", "cam_to_world", "conf3d"], {"kind": "scale_invariant"})
    m = multiview.load("fake", device="cpu")
    FeedForward.first = torch.tensor(30 / 255)
    r = m(_views(), reference=2)  # view 2 is run first; outputs come back in input order
    assert r.views == (0, 1, 2)
    assert r.depth is not None
    assert r.depth.shape == (3, 4, 5)
    assert r.cam_to_world is not None
    torch.testing.assert_close(r.cam_to_world[:, 0, 3], torch.tensor([10, 20, 30]) / 255)
    assert r.extrinsics is not None
    torch.testing.assert_close(r.extrinsics[:, 0, 3], -torch.tensor([10, 20, 30]) / 255)
    assert r.extras["conf3d"].shape == (3, 1, 4, 5)
    assert r.kind == "scale_invariant"
    assert r.intrinsics is None
    FeedForward.first = torch.tensor(10 / 255)
    rb = m(np.stack(_views()))  # one [N, H, W, 3] batch
    assert rb.depth is not None
    torch.testing.assert_close(rb.depth, r.depth)
    cases: list[tuple[dict[str, Any], str]] = [
        ({"intrinsics": torch.eye(3)}, "does not use intrinsics"),
        ({"reference": [0, 1]}, "one int"),
        ({"reference": 3}, "indices"),
        ({"depth_range": (1, 2)}, "depth range"),
    ]
    for bad, match in cases:
        with pytest.raises(InputError, match=match):
            m(_views(), **bad)
    with pytest.raises(InputError, match="at least 2"):
        m(_views(1))
    with pytest.raises(InputError, match="one size"):
        m([*_views(2), np.zeros((5, 5, 3), np.uint8)])
    with pytest.raises(InputError, match="sequence"):
        m("a.png")


def test_multiview_calibrated(fake):
    fake("multiview", Calibrated, ["depth", "mask", "range"], {"kind": "metric"})
    m = multiview.load("fake", device="cpu")
    k = np.diag([5.0, 5.0, 1.0])
    c2w = np.stack([np.eye(4)] * 3)
    c2w[:, 0, 3] = [1.0, 2.0, 3.0]
    r = m(_views(), intrinsics=k, cam_to_world=c2w, reference=[2, 0])
    assert r.views == (2, 0)
    assert r.depth is not None
    torch.testing.assert_close(r.depth[:, 0, 0], torch.tensor([(1 + 30 / 255) * 3, (1 + 10 / 255) * 1]))
    assert r.intrinsics is not None
    assert r.intrinsics.shape == (2, 3, 3)  # the given cameras of the predicted views
    assert r.cam_to_world is not None
    torch.testing.assert_close(r.cam_to_world[:, 0, 3], torch.tensor([3.0, 1.0], dtype=torch.float64))  # as given
    assert r.mask is not None
    assert r.mask.dtype == torch.bool
    assert r.extras["range"].shape == (2, 2)
    r1 = m(_views(), intrinsics=k, cam_to_world=c2w, depth_range=(0.5, 2.5))
    assert r1.mask is not None
    assert r1.views == (0,)
    assert r1.mask.all()
    cases: list[tuple[dict[str, Any], str]] = [
        ({}, "needs intrinsics"),
        ({"intrinsics": k}, "needs cam_to_world"),
        ({"intrinsics": k, "cam_to_world": c2w[:2]}, "expected 3"),
        ({"intrinsics": k, "cam_to_world": c2w[0]}, r"\[N, 4, 4\]"),
        ({"intrinsics": k, "cam_to_world": c2w, "depth_range": (2, 1)}, "near < far"),
        ({"intrinsics": k, "cam_to_world": c2w, "reference": [0, 0]}, "distinct"),
    ]
    for bad, match in cases:
        with pytest.raises(InputError, match=match):
            m(_views(), **bad)
    with pytest.raises(InputError, match="at most 4"):
        m(_views(5), intrinsics=k, cam_to_world=np.stack([np.eye(4)] * 5))


def test_unproject_round_trip():
    import synth  # noqa: PLC0415

    s = synth.scene()
    depth_, k, c2w = (torch.from_numpy(s[n]) for n in ("depth", "intrinsics", "cam_to_world"))
    pts = multiview.unproject(depth_, k, c2w)
    torch.testing.assert_close(multiview.unproject(depth_, k)[..., 2], depth_)
    # every view sees the same plane: view 1's points projected into view 0 land at view 0's depth
    w2c = torch.linalg.inv(c2w[0].double())
    cam = pts[1].double().reshape(-1, 3) @ w2c[:3, :3].T + w2c[:3, 3]
    uv = cam @ k[0].double().T
    u, v = (uv[:, 0] / uv[:, 2]).round().long(), (uv[:, 1] / uv[:, 2]).round().long()
    inside = (u >= 0) & (u < 160) & (v >= 0) & (v < 120)
    torch.testing.assert_close(cam[inside, 2].float(), depth_[0][v[inside], u[inside]], atol=0.02, rtol=0)


def test_every_task_has_a_bench_hook():
    for task in registry.TASKS:
        assert callable(getattr(importlib.import_module(f"fastforks.{task}._api"), "bench_inputs", None)), task


@pytest.mark.parametrize(
    ("task", "runner", "outputs", "output"),
    [
        ("depth", "required", ["depth", "mask", "intrinsics"], {"kind": "metric"}),
        ("depth", "unused", ["depth", "mask", "intrinsics"], {"kind": "metric"}),
        ("classification", None, ["logits"], {"label_set": "imagenet-1k"}),
        ("segmentation", None, ["logits"], {"label_set": "ade20k-150"}),
    ],
)
def test_bench_inputs_make_a_valid_call(fake, task, runner, outputs, output):
    if task == "depth":
        runner = type("R", (Depth,), {"intrinsics": runner})
    else:
        runner = {"classification": Cls, "segmentation": Seg}[task]
    fake(task, runner, outputs, output)
    m = importlib.import_module(f"fastforks.{task}").load("fake", device="cpu")
    args, kwargs = bench_inputs(m, BenchSize(9, 13))
    assert args[0].shape == (1, 3, 9, 13)
    assert ("intrinsics" in kwargs) == (getattr(runner, "intrinsics", None) == "required")
    m(*args, **kwargs)


class Det(DetRunner):
    """One box (and its mask) over the left half of the image."""

    def __call__(self, image, score_threshold):  # noqa: ARG002 - test stub
        h, w = image.shape[-2:]
        masks = torch.zeros(1, h, w, dtype=torch.bool)
        masks[:, :, : w // 2] = True
        return {
            "boxes": torch.tensor([[0.0, 0.0, w / 2, h]]),
            "scores": torch.ones(1),
            "labels": torch.zeros(1, dtype=torch.int64),
            "masks": masks,
        }


@pytest.mark.parametrize(
    ("task", "runner", "outputs", "output"),
    [
        ("promptable_segmentation", Prompt, ["pred_masks", "iou_scores"], {}),
        ("detection", Det, ["boxes", "scores", "labels", "masks"], {"label_set": "coco-80"}),
    ],
)
def test_bench_inputs_instance_tasks(fake, task, runner, outputs, output):
    fake(task, runner, outputs, output)
    m = importlib.import_module(f"fastforks.{task}").load("fake", device="cpu")
    args, kwargs = bench_inputs(m, BenchSize(9, 13))
    assert args[0].shape == (3, 9, 13)
    r = m(*args, **kwargs)
    if task == "promptable_segmentation":  # the centre point: the fake's mask is true left of x = 6
        assert r.masks.shape == (3, 9, 13)
        assert r.masks[0, :, :6].all()
        assert not r.masks[0, :, 7:].any()
    else:
        assert r.boxes.tolist() == [[0.0, 0.0, 6.5, 9.0]]
        assert r.masks.shape == (1, 9, 13)
        assert r.masks[0, :, :6].all()


class Pose(PoseRunner):
    def __call__(self, image, boxes):  # noqa: ARG002 - a stub: keypoints from the boxes alone
        p = boxes.shape[0]
        centre = torch.stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2), -1)
        return {
            "keypoints": centre[:, None].expand(p, 17, 2).contiguous(),
            "keypoint_scores": torch.ones(p, 17),
            "scores": torch.ones(p),
        }


def test_bench_inputs_pose(fake):
    fake("pose", Pose, ["keypoints", "keypoint_scores", "boxes", "scores"], {"label_set": "coco-keypoints-17"})
    m = importlib.import_module("fastforks.pose").load("fake", device="cpu")
    args, kwargs = bench_inputs(m, BenchSize(256, 384))
    assert args[0].shape == (3, 256, 384)
    assert kwargs["boxes"].shape == (3, 4)  # one person box per 128 px of width
    r = m(*args, **kwargs)
    assert r.boxes.shape == (3, 4)
    assert r.keypoints.shape[0] == r.keypoint_scores.shape[0] == len(r) == 3


def _tagged_card(*variants):
    data = card_data(["logits"], {"label_set": "imagenet-1k"})
    data["variants"] = [{"id": i, "revision": 1, "weights": [W0], **kw} for i, kw in variants]
    return parse_card(data, family="fake", task="classification", where="c")


@pytest.fixture
def tagged(monkeypatch):
    """`tagged(card)`: make `card` the only installed classifier."""

    def install(card):
        entries = {v.id: (card, v) for v in card.variants}
        monkeypatch.setattr(registry, "cards", lambda t: entries if t == "classification" else {})
        monkeypatch.setattr(registry, "default_tags", registry.default_tags.__wrapped__)

    return install


def test_untagged_timm_name_loads_the_default_tag(tagged):
    tagged(
        _tagged_card(("resnet50.a1_in1k", {"default_tag": True}), ("resnet50.tv_in1k", {}), ("fakenet18.tv_in1k", {}))
    )
    assert registry.resolve("classification", "resnet50")[1].id == "resnet50.a1_in1k"
    assert registry.resolve("classification", "ResNet50@1")[1].ref == "resnet50.a1_in1k@1"
    assert registry.resolve("classification", "resnet50.tv_in1k")[1].id == "resnet50.tv_in1k"
    with pytest.raises(ModelNotFoundError, match="no revision '2'"):
        registry.resolve("classification", "resnet50@2")
    with pytest.raises(
        ModelNotFoundError, match=r"'fakenet18' has no default pretrained tag here; pass one of: fakenet18"
    ):
        registry.resolve("classification", "fakenet18")


def test_default_tag_rules(tagged):
    with pytest.raises(CardError, match="tagged id"):
        _tagged_card(("resnet50", {"default_tag": True}))
    with pytest.raises(CardError, match="default_tag"):
        _tagged_card(("resnet50.a1_in1k", {"default_tag": "yes"}))
    tagged(_tagged_card(("resnet50.a1_in1k", {"default_tag": True}), ("resnet50.tv_in1k", {"default_tag": True})))
    with pytest.raises(CardError, match=r"already means 'resnet50\.a1_in1k'"):
        registry.resolve("classification", "resnet50")


class Feat(FeatRunner):
    """Patch 2: cls + 1 register in front of the patches; token value = mean pixel of its patch + block index."""

    batch = None

    def __init__(self, ctx):
        super().__init__(ctx)
        self.hidden_size, self.num_layers, self.num_prefix_tokens, self.num_register_tokens = 3, 2, 2, 1

    def __call__(self, image, layers):
        p = torch.nn.functional.avg_pool2d(image, 2).flatten(2).transpose(1, 2)  # [B, h * w, 3]
        t = torch.cat([p.mean(1, keepdim=True), torch.zeros_like(p[:, :1]), p], 1)
        out = {"last_hidden_state": t + 10, "pooler_output": t[:, 0] + 10}
        if layers:
            out["hidden_states"] = torch.stack([t + i for i in layers], 1)
        return out

    def module(self):
        return Forward(torch.nn.Identity(), lambda _net, x: self(x, ())["last_hidden_state"])


FEAT_IO = {"input": "0-1", "pad": {"multiple": 2, "mode": "none"}, "interpolation": "bilinear"}


def test_features(fake):
    fake("features", Feat, ["last_hidden_state", "pooler_output", "hidden_states"], {}, FEAT_IO)
    m = features.load("fake", device="cpu")
    assert (m.stride, m.hidden_size, m.num_layers) == (2, 3, 2)
    img = np.zeros((6, 10, 3), np.uint8)
    img[:, :, 1] = 255
    r = m(img)  # size None: kept (multiples of 2)
    assert r.grid == (3, 5)
    assert r.last_hidden_state.shape == (2 + 15, 3)
    assert r.hidden_states is None
    assert r.layers == ()
    assert r.pooler_output is not None
    assert r.cls_token.shape == r.pooler_output.shape == (3,)
    assert r.register_tokens.shape == (1, 3)
    assert r.patch_tokens.shape == (15, 3)
    fm = r.feature_map()
    assert fm.shape == (3, 3, 5)
    assert torch.allclose(fm[1], torch.full((3, 5), 11.0))
    rb = m(np.stack([img, img]), size=4, layers=(0, -1))  # shorter side 6 -> 4: 4 x 6
    assert rb.grid == (2, 3)
    assert rb.layers == (0, 2)
    assert rb.hidden_states is not None
    assert len(rb.hidden_states) == 3
    h0, h1, h2 = rb.hidden_states
    assert h1 is None
    assert h0 is not None
    assert h2 is not None
    assert h2.shape == (2, 2 + 6, 3)
    assert torch.equal(h2 - h0, torch.full((2, 8, 3), 2.0))
    assert rb.feature_map(2).shape == (2, 3, 2, 3)
    assert m(img, layers="all").layers == (0, 1, 2)
    with pytest.raises(InputError):
        rb.feature_map(1)
    with pytest.raises(InputError):
        m(img, layers=(3,))
    with pytest.raises(InputError):
        m(img, size=(5, 6))
    assert m.module()(torch.rand(1, 3, 7, 10)).shape == (1, 2 + 20, 3)  # 7 x 10 -> 8 x 10


def test_instances_indexing_to_list_and_cpu():
    boxes = torch.tensor([[0.0, 0.0, 2.0, 3.0], [1.0, 1.0, 4.0, 4.0]])
    masks = torch.zeros(2, 5, 6, dtype=torch.bool)
    masks[1, 1:4, 1:4] = True
    backend = Backend(mode="portable", device="cpu", dtype="float32", kernels=())
    r = Instances(boxes, torch.tensor([0.9, 0.4]), torch.tensor([1, 0]), masks, ("a", "b"), "m@1", backend)
    assert len(r) == 2
    assert len(r[r.scores > 0.5]) == 1
    assert r[1].to_list()[0]["class_name"] == "a"
    assert r[1].to_list()[0]["mask"].sum() == 9
    host = r.cpu()
    assert host.masks is not None
    assert torch.equal(host.masks, masks)


@pytest.mark.parametrize(
    ("hw", "rule", "want"),
    [
        ((512, 683), ("mmseg", 2048, 512), (512, 683)),
        ((300, 401), ("mmseg", 2048, 512), (512, 684)),  # int(x + 0.5)
        ((100, 3000), ("short", 640, 2560), (85, 2560)),  # the long-side limit wins
        ((96, 145), ("short", 512, 100000), (512, 773)),  # 773.33
        ((96, 147), ("eomt", 512), (512, 784)),
        ((256, 385), ("eomt", 512), (512, 770)),
        ((64, 33), ("eomt", 512), (993, 512)),  # 992.97
    ],
)
def test_seginfer_target_size(hw, rule, want):
    from fastforks.shared.seginfer import target_size  # noqa: PLC0415

    assert target_size(*hw, rule) == want


def test_seginfer_target_size_rounding():
    from fastforks.shared.seginfer import target_size  # noqa: PLC0415

    assert target_size(2, 5, ("short", 4, 100)) == (4, 10)
    assert target_size(4, 9, ("short", 2, 100)) == (2, 5)  # 4.5 -> 5 (detectron2 int(x + 0.5))
    assert target_size(4, 9, ("eomt", 2)) == (2, 4)  # 4.5 -> 4 (Python round, as EoMT)
    with pytest.raises(ValueError, match="unknown resize rule"):
        target_size(4, 4, ("nearest", 2))


def test_seginfer_resize_pil_filters_downscales_only():
    from fastforks.shared.seginfer import resize  # noqa: PLC0415

    x = torch.rand(2, 3, 9, 13)
    assert resize(x, (9, 13), pil=True) is x
    torch.testing.assert_close(resize(x, (20, 29), pil=True), resize(x, (20, 29)))
    image = pytest.importorskip("PIL.Image")
    src = torch.rand(37, 53)
    pil = image.fromarray(src.numpy(), mode="F").resize((20, 14), image.Resampling.BILINEAR)
    got = resize(src[None, None], (14, 20), pil=True)[0, 0]
    torch.testing.assert_close(got, torch.from_numpy(np.array(pil)), atol=1e-4, rtol=0)


@pytest.mark.parametrize(
    ("length", "crop", "want"),
    [(512, 512, [0]), (300, 512, [0]), (1024, 512, [0, 512]), (700, 512, [0, 188]), (1100, 512, [0, 294, 588])],
)
def test_seginfer_eomt_windows(length, crop, want):
    from fastforks.shared.seginfer import eomt_windows  # noqa: PLC0415

    starts = eomt_windows(length, crop)
    assert starts == want
    assert starts[-1] + min(crop, length) == length


@pytest.mark.parametrize(
    ("hw", "crop", "stride"),
    [((37, 53), (16, 16), (12, 12)), ((10, 9), (16, 16), (8, 8)), ((33, 17), (16, 8), (10, 5))],
)
def test_seginfer_slide(hw, crop, stride):
    from fastforks.shared.seginfer import slide  # noqa: PLC0415

    x = torch.rand(2, 3, *hw)
    calls = []

    def fn(c):
        calls.append(c.shape[0])
        return c[:, :2] * 2

    torch.testing.assert_close(slide(fn, x, crop, stride), x[:, :2] * 2)  # every pixel covered, overlaps averaged
    assert all(n % 2 == 0 for n in calls)  # batch 2 stays together


def test_seginfer_mask_semantic():
    from fastforks.shared.seginfer import mask_semantic  # noqa: PLC0415

    masks, classes = torch.randn(2, 5, 7, 9), torch.randn(2, 5, 4)
    want = torch.einsum("bqc,bqhw->bchw", classes.softmax(-1)[..., :-1], masks.sigmoid())
    torch.testing.assert_close(mask_semantic(masks, classes), want)


def test_concept_threshold_validated_and_recorded(fake):
    fake("promptable_segmentation", Concept, ["boxes", "scores", "masks"], {})
    m = promptable_segmentation.load("fake", device="cpu")
    img = np.zeros((8, 12, 3), np.uint8)
    for bad in (float("nan"), float("inf"), -0.1, 1.5, True, "0.5"):  # R2-15: scores are in [0, 1]
        with pytest.raises(InputError, match="threshold"):
            m(img, text="coin", threshold=bad)
    assert m(img, text="coin").options["threshold"] == 0.5  # R2-15: the default is recorded too
    assert len(m(img, text="coin", threshold=1)) == 0  # the bounds are allowed
    assert m(img, text="coin", threshold=0.1).options["threshold"] == 0.1
