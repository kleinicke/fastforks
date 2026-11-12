from __future__ import annotations

import re

import pytest

from fastforks import CardError
from fastforks._core import licences


@pytest.mark.parametrize(
    ("expr", "cls"),
    [
        ("MIT", "permissive"),
        ("MIT AND Apache-2.0", "permissive"),
        ("Apache-2.0 AND BSD-3-Clause AND WTFPL", "permissive"),
        ("WTFPL AND GPL-3.0-only", "copyleft"),
        ("MIT AND GPL-3.0-only", "copyleft"),
        ("MIT AND CC-BY-NC-4.0", "noncommercial"),
        ("GPL-3.0-only OR MIT", "permissive"),
        ("CC-BY-NC-4.0 OR GPL-2.0-only", "copyleft"),
    ],
)
def test_class(expr, cls):
    assert licences.licence_class(expr) == cls


@pytest.mark.parametrize(
    ("expr", "match"),
    [
        ("Foo", "unknown licence id"),
        ("MIT AND (BSD-3-Clause OR GPL-3.0-only)", "without parentheses"),
        ("MIT AND BSD-3-Clause OR GPL-3.0-only", "without parentheses"),
        ("GPL-3.0-only AND CC-BY-NC-4.0", "cannot both be met"),
    ],
)
def test_class_rejects(expr, match):
    with pytest.raises(CardError, match=match):
        licences.licence_class(expr)


def test_weights_and_code():
    for unresolved in (licences.NOT_STATED, licences.ASSUMED):
        with pytest.raises(CardError, match="assumed from code"):
            licences.weights_commercial(unresolved)
    assert licences.weights_licence(licences.ASSUMED, "MIT") == "MIT"
    assert licences.weights_licence("CC-BY-NC-4.0", "MIT") == "CC-BY-NC-4.0"
    assert licences.describe_weights(licences.ASSUMED, "MIT") == "MIT (assumed from code)"
    assert licences.weights_commercial("Apache-2.0") == "yes"
    assert licences.weights_commercial("CC-BY-NC-SA-4.0") == "no"
    assert licences.code_commercial("copyleft") == "yes"


def test_data_is_never_no():
    assert licences.data_commercial(()) == "yes"
    assert licences.data_commercial(("sceneflow",)) == "unclear"
    assert licences.data_commercial(("unstated",)) == "unclear"
    assert licences.data_commercial(("middlebury", "kitti")) == "unclear"


def test_datasets_have_terms():
    for d in licences.datasets().values():
        assert d.url.startswith("https://")
        assert d.licence == licences.NOT_STATED or d.licence in licences.CLASSES


def test_repo_links_are_pinned():
    """GitHub / Hugging Face terms pages point at a commit, so the cited text cannot change under the link."""
    for d in licences.datasets().values():
        if re.match(r"https://(github\.com|huggingface\.co)/", d.url):
            assert re.search(r"/blob/[0-9a-f]{40}/", d.url), d.url


def test_worst():
    assert licences.worst("yes", "unclear") == "unclear"
    assert licences.worst("unclear", "no", "yes") == "no"
    assert licences.worst("yes") == "yes"
