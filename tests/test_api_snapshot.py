"""The public API matches tests/api_snapshot.txt; nothing the release has disappears unannounced."""

import api_snapshot


def test_api_matches_snapshot():
    current = api_snapshot.snapshot()
    committed = api_snapshot.SNAPSHOT.read_text()
    if current != committed:
        old, new = set(committed.splitlines()), set(current.splitlines())
        diff = [f"- {x}" for x in sorted(old - new)] + [f"+ {x}" for x in sorted(new - old)]
        raise AssertionError(
            "public API changed; check the diff, then run `uv run python tools/api_snapshot.py --update`:\n"
            + "\n".join(diff[:60])
        )


def test_removed_release_names_are_in_the_changelog():
    assert not api_snapshot.unlisted_removals(api_snapshot.snapshot()), (
        "removed / renamed release API: name each in CHANGELOG.md's Unreleased section"
    )


def test_removed_items():
    old = (
        "def fastforks.x.f(a, *, b=1)\noption depth.da.res aliases r,s\nname fastforks.x.Gone\ncli fastforks r: --fp16"
    )
    new = "def fastforks.x.f(a)\noption depth.da.res aliases s\ncli fastforks r: --half"
    assert api_snapshot.removed_items(old, new) == ["--fp16", "Gone", "b", "r"]
