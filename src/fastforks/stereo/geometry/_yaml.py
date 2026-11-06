"""A small YAML subset reader for calibration files (no PyYAML at runtime).

Covers what Kalibr camchains and OpenCV `FileStorage` YAML use: block mappings and sequences, flow sequences
(also spanning lines and nested), comments, `%YAML` directives, `---` markers, tags such as `!!opencv-matrix`, and
plain or quoted scalars. Anchors, aliases, flow mappings and multi-line strings are rejected with a clear error.
"""

from __future__ import annotations

import re

from fastforks._core.errors import InputError

Node = dict[str, "Node"] | list["Node"] | str | int | float | bool | None

_NUMBER = re.compile(r"[-+]?((\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?|\.(inf|Inf|INF|nan|NaN|NAN))")
_INT = re.compile(r"[-+]?\d+")
_TAG = re.compile(r"^!\S*\s*")
_CONSTANTS: dict[str, Node] = {"": None, "~": None, "null": None, "true": True, "false": False}
_TOKEN = re.compile(r"""\s*(\[|\]|,|'[^']*'|"[^"]*"|[^\[\],'"]+)""")  # one token at a time: linear, no backtracking
_MAX_DEPTH = 64


def _strip_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i].rstrip()
    return line.rstrip()


def _logical_lines(text: str) -> list[tuple[int, str, int]]:
    """`(indent, content, line number)` per logical line; flow sequences spanning lines are joined."""
    out: list[tuple[int, str, int]] = []
    pending: tuple[int, str, int] | None = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw.replace("\t", "    "))
        if pending is None:
            if not line.strip() or line.startswith(("%", "---", "...")):
                continue
            pending = (len(line) - len(line.lstrip()), line.strip(), number)
        else:
            pending = (pending[0], f"{pending[1]} {line.strip()}", pending[2])
        if pending[1].count("[") <= pending[1].count("]"):
            out.append(pending)
            pending = None
    if pending is not None:
        msg = f"line {pending[2]}: unclosed '['"
        raise InputError(msg)
    return out


def _scalar(text: str, number: int) -> Node:
    text = _TAG.sub("", text).strip()
    if text[:1] in "&*{|>":
        msg = f"line {number}: YAML anchors, aliases, flow mappings and block strings are not supported"
        raise InputError(msg)
    if text.startswith("["):
        return _flow(text, number)
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":  # noqa: PLR2004
        return text[1:-1]
    if text.lower() in _CONSTANTS:
        return _CONSTANTS[text.lower()]
    if _INT.fullmatch(text):
        return int(text)
    if _NUMBER.fullmatch(text):
        return float(re.sub(r"\.(inf|nan)", r"\1", text, flags=re.IGNORECASE))
    return text


def _flow_scalar(part: str, number: int) -> Node:
    if part[:1] not in "'\"" and len(part.split()) > 1:  # `[1 2, 3]`: two values without a comma
        msg = f"line {number}: malformed flow sequence (missing ',' in {part!r})"
        raise InputError(msg)
    return _scalar(part, number)


def _flow(text: str, number: int) -> list[Node]:
    """Parse a flow sequence `[a, [b, c], 'd']`."""
    parts, pos, end = [], 0, len(text.rstrip())
    while pos < end:
        match = _TOKEN.match(text, pos)
        if match is None:
            msg = f"line {number}: malformed flow sequence near column {pos + 1}"
            raise InputError(msg)
        parts.append(match.group(1).strip())
        pos = match.end()
    stack: list[list[Node]] = [[]]
    after_item = False  # the last token was a value or ']' : a ',' or ']' must follow
    for part in parts:
        if after_item and part not in {",", "]"}:
            msg = f"line {number}: malformed flow sequence (missing ',' before {part!r})"
            raise InputError(msg)
        after_item = part not in {"[", ","}
        if part == "[":
            stack.append([])
        elif part == "]" and len(stack) > 1:
            done = stack.pop()
            stack[-1].append(done)
        elif part == "]":
            msg = f"line {number}: unbalanced ']'"
            raise InputError(msg)
        elif part != ",":
            stack[-1].append(_flow_scalar(part, number))
    if len(stack) != 1 or len(stack[0]) != 1:
        msg = f"line {number}: malformed flow sequence"
        raise InputError(msg)
    return stack[0][0]  # ty: ignore[invalid-return-type] - the only item is the outer sequence


def _split_key(content: str) -> tuple[str, str] | None:
    """`(key, rest)` if the line is a mapping entry `key: rest`."""
    if content[:1] in "'\"":
        end = content.find(content[0], 1)
        if end > 0 and content[end + 1 : end + 2] == ":":
            return content[1:end], content[end + 2 :].strip()
        return None
    match = re.match(r"([^\[\]{},:]+?):(\s|$)", content)
    if match is None:
        return None
    return match.group(1).strip(), content[match.end() :].strip()


def _sequence(lines: list[tuple[int, str, int]], i: int, indent: int, depth: int) -> tuple[list[Node], int]:
    """Parse the block sequence (`- item` lines) starting at `lines[i]`."""
    seq: list[Node] = []
    while i < len(lines) and lines[i][0] == indent and (lines[i][1].startswith("- ") or lines[i][1] == "-"):
        rest, number = lines[i][1][1:].strip(), lines[i][2]
        i += 1
        if not rest:
            if i < len(lines) and lines[i][0] > indent:
                value, i = _block(lines, i, lines[i][0], depth + 1)
            else:
                value = None
        elif _split_key(rest) is not None:
            msg = f"line {number}: mappings inside block sequences are not supported"
            raise InputError(msg)
        else:
            value = _scalar(rest, number)
        seq.append(value)
    return seq, i


def _block(lines: list[tuple[int, str, int]], i: int, indent: int, depth: int = 0) -> tuple[Node, int]:
    """Parse the block starting at `lines[i]` with the given indentation; returns `(node, next index)`."""
    if depth > _MAX_DEPTH:
        msg = f"line {lines[i][2]}: nested deeper than {_MAX_DEPTH} levels"
        raise InputError(msg)
    if lines[i][1].startswith("- ") or lines[i][1] == "-":
        return _sequence(lines, i, indent, depth)
    mapping: dict[str, Node] = {}
    while i < len(lines) and lines[i][0] == indent:
        _, content, number = lines[i]
        entry = _split_key(content)
        if entry is None:
            msg = f"line {number}: expected 'key: value', got {content[:40]!r}"
            raise InputError(msg)
        key, rest = entry
        rest = _TAG.sub("", rest) if rest.startswith("!") else rest
        i += 1
        if rest:
            mapping[key] = _scalar(rest, number)
        elif i < len(lines) and (lines[i][0] > indent or (lines[i][0] == indent and lines[i][1].startswith("-"))):
            mapping[key], i = _block(lines, i, lines[i][0], depth + 1)
        else:
            mapping[key] = None
    return mapping, i


def loads(text: str) -> Node:
    """Parse a YAML document of the supported subset.

    Args:
        text: The document.

    Returns:
        Nested dicts / lists / scalars (None for an empty document).

    Raises:
        InputError: The document uses YAML features outside the subset or is malformed.
    """
    lines = _logical_lines(text)
    if not lines:
        return None
    node, i = _block(lines, 0, lines[0][0])
    if i != len(lines):
        msg = f"line {lines[i][2]}: unexpected indentation"
        raise InputError(msg)
    return node
