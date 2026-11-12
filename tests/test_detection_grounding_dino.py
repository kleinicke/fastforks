"""Grounding DINO: caption building, phrase masks, the processor's input path and the BERT tokenizer vs transformers."""

import numpy as np
import pytest
import torch

from fastforks._core import registry
from fastforks._core.errors import InputError
from fastforks.detection.models.grounding_dino.model import (
    MEAN,
    SEPARATORS,
    STD,
    Portable,
    phrase_masks,
    target_size,
)
from fastforks.shared.bert_tokenizer import BertTokenizer, basic

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", ".", "?", "cat", "dog", "hot", "##dog", "remote", "control"]


def _runner(tmp_path, max_len: int = 256) -> Portable:
    (tmp_path / "vocab.txt").write_text("\n".join(VOCAB) + "\n")
    runner = object.__new__(Portable)
    runner.tokenizer = BertTokenizer(tmp_path / "vocab.txt")
    runner.separators = {runner.tokenizer.vocab[s] for s in SEPARATORS}
    runner.max_len = max_len
    runner.mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    runner.std = torch.tensor(STD).view(1, 3, 1, 1)
    return runner


def test_caption_tokens_and_class_maps(tmp_path) -> None:
    r = _runner(tmp_path)
    ids, classes = r.caption(("Cat", "hotdog", "remote control"))
    assert ids == [2, 6, 4, 8, 9, 4, 10, 11, 4, 3]  # [CLS] cat . hot ##dog . remote control . [SEP]
    want = torch.zeros(3, 10)
    want[0, 1], want[1, 3:5], want[2, 6:8] = 1, 0.5, 0.5
    torch.testing.assert_close(classes, want)
    att, pos = phrase_masks(ids, r.separators)
    assert pos.tolist() == [0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
    assert att[3].tolist() == [i in (3, 4, 5) for i in range(10)]  # "hot" sees its phrase and the closing "."
    assert att[0].tolist() == [i == 0 for i in range(10)]  # [CLS] only itself
    with pytest.raises(InputError, match="tokens"):
        _runner(tmp_path, max_len=9).caption(("Cat", "hotdog", "remote control"))
    with pytest.raises(InputError, match="no tokens"):
        r.caption(("cat", "."))


def test_target_size_is_the_processors() -> None:
    assert target_size(480, 640) == (800, 1066)
    assert target_size(640, 480) == (1066, 800)
    assert target_size(800, 1344) == (793, 1333)
    assert target_size(800, 800) == (800, 800)


def test_basic_tokenizer_splits_punctuation_accents_and_cjk() -> None:
    assert basic("Ölgemälde,  hot-dog\t猫x") == ["olgemalde", ",", "hot", "-", "dog", "猫", "x"]


@pytest.mark.slow  # imports transformers
@pytest.mark.parametrize("hw", [(150, 130), (900, 1700), (800, 640)])  # upscale, downscale (long side), short side
def test_pixels_match_the_transformers_processor(tmp_path, hw: tuple[int, int]) -> None:
    transformers = pytest.importorskip("transformers")
    img = np.random.default_rng(0).integers(0, 256, (*hw, 3), dtype=np.uint8)
    want = transformers.GroundingDinoImageProcessor()(images=img, return_tensors="pt")["pixel_values"]
    got = _runner(tmp_path).pixels(torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255)
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, atol=1e-5, rtol=0)  # CPU: the same uint8 resize kernel


@pytest.mark.slow  # imports transformers
def test_phrase_masks_match_transformers() -> None:
    pytest.importorskip("transformers")
    from transformers.models.grounding_dino import modeling_grounding_dino as m  # noqa: PLC0415

    ids = [101, 2711, 1012, 3902, 2482, 1012, 2482, 1029, 102]
    att, pos = phrase_masks(ids, {101, 102, 1012, 1029})
    want_att, want_pos = m.generate_masks_with_special_tokens_and_transfer_map(torch.tensor([ids]))
    assert torch.equal(att, want_att[0])
    assert torch.equal(pos, want_pos[0])


@pytest.mark.slow
@pytest.mark.model
def test_bert_tokenizer_matches_transformers() -> None:
    transformers = pytest.importorskip("transformers")
    card, variant = registry.resolve("detection", "grounding-dino-tiny")
    vocab = registry.weights(card, variant, allow_download=False)[2]
    ours = BertTokenizer(vocab)
    theirs = transformers.BertTokenizer(str(vocab))
    texts = ["dining table . teddy bear .", "a photo of a  Hot-Dog!", "ein Ölgemälde 猫", "x" * 120, "naïve café's"]
    for text in texts:
        assert ours(text) == theirs(text)["input_ids"], text
