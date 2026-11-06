"""Per-symbol hashes of a kernel library: which exported functions a rebuild actually changed.

A library's `build_id` changes with any edit, so a family validated against `igev@a` would count as unvalidated
against `igev@b` even if only an unrelated kernel was added. `symbol_hashes` hashes every `extern "C"` function of a
library's sources together with what it can depend on (the functions whose names it mentions, transitively, and every
item that is not a function: macros, includes, structs, constants), the build settings and the toolchain. Comments and
whitespace do not count. Two builds with the same hash for a symbol compile the same code for it, so a family whose
validated symbols have unchanged hashes stays validated (`tests/test_golden.py`, `tools/build_kernels.py`).

The scan is conservative: a dependency it cannot tell apart is included, never dropped.
"""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from pathlib import Path

_IDENT = re.compile(r"[A-Za-z_]\w*")
_CALLEE = re.compile(r"([A-Za-z_]\w*)\s*\(")
_LAUNCH_BOUNDS = re.compile(r"__launch_bounds__\s*\([^)]*\)")  # looks like a call; not the function's name
_NAMESPACE = re.compile(r"^(?:extern\s+\"C\"\s+)?namespace\b")
_NON_FUNCTION = re.compile(
    r"^(?:extern\s+\"C\"\s+)?(?:typedef|using|struct|class|union|enum|namespace|template\s*<[^>]*>\s*(?:struct|class))\b"
)


def strip_comments(text: str) -> str:
    """`text` without `//` and `/* */` comments (string and character literals are kept as they are)."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":  # literal: copy to the closing quote
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i : j + 1])
            i = j + 1
        elif text.startswith("//", i):
            i = text.find("\n", i)
            i = n if i < 0 else i
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
            out.append(" ")
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _blank_literals(text: str) -> str:
    """`text` with the contents of string / character literals replaced by spaces (so braces in them do not count)."""
    return re.sub(
        r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        lambda m: m.group(0)[0] + " " * (len(m.group(0)) - 2) + m.group(0)[-1],
        text,
    )


def _line_end(blank: str, i: int) -> int:
    """End of the preprocessor line starting at `i` (backslash continuations included)."""
    while i < len(blank) and (blank[i] != "\n" or blank[i - 1] == "\\"):
        i += 1
    return i


def items(source: str) -> list[str]:
    """Top-level items of a CUDA source: preprocessor lines (with continuations), declarations and definitions."""
    text = strip_comments(source)
    blank = _blank_literals(text)
    result: list[str] = []
    start, depth, i, n = 0, 0, 0, len(blank)
    while i < n:
        if depth == 0 and blank[i] == "#" and not blank[start:i].strip():  # preprocessor line
            j = _line_end(blank, i)
            result.append(text[i:j])
            start = i = j
            continue
        c = blank[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                head = blank[start:i]
                k = i + 1
                while k < n and blank[k] in " \t\n":
                    k += 1
                is_type = not _NAMESPACE.match(
                    head.strip()
                ) and (  # a namespace needs no `;`: the next statement is new
                    bool(_NON_FUNCTION.match(head.strip())) or "=" in head.split("{")[0] or head.rstrip().endswith("]")
                )
                if not (is_type or (k < n and blank[k] == ";")):  # a function body ends the item
                    result.append(text[start : i + 1])
                    start = i + 1
        elif c == ";" and depth == 0:
            result.append(text[start : i + 1])
            start = i + 1
        i += 1
    if text[start:].strip():
        result.append(text[start:])
    return [" ".join(x.split()) for x in result if x.strip()]


def _is_function(item: str) -> bool:
    """A function definition: has a `{ ... }` body, a `(` before it, and is not a type / initialised variable."""
    head = item.split("{", 1)[0]
    return item.endswith("}") and "(" in head and "=" not in head and not _NON_FUNCTION.match(item)


_DEFINE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+(\w+)\(([^)]*)\)((?:\\\n|[^\n])*)", re.MULTILINE)


def expand_export_macros(text: str) -> str:
    """The `extern "C"` functions that function-like macros generate, as plain source (one per invocation).

    A macro such as `#define LN_K(V) extern "C" ... ln_v##V(...) { ... }` followed by `LN_K(1)` on a line of its own
    defines the export `ln_v1`. Each such invocation is expanded (parameters substituted, `##` pasted); everything
    else is left out. Returns "" when the source has no such macro.
    """
    text = strip_comments(text)
    out: list[str] = []
    for m in _DEFINE.finditer(text):
        name, params, body = m.group(1), [p.strip() for p in m.group(2).split(",")], m.group(3)
        if 'extern "C"' not in body:
            continue
        body = " ".join(body.replace("\\\n", " ").split())
        for call in re.finditer(rf"^[ \t]*{name}\(([^)\n]*)\)[ \t]*$", text, re.MULTILINE):
            args = [a.strip() for a in call.group(1).split(",")]
            expanded = body
            for param, arg in zip(params, args, strict=True):
                expanded = re.sub(rf"\s*##\s*{re.escape(param)}\b|\b{re.escape(param)}\b\s*##\s*", arg, expanded)
                expanded = re.sub(rf"(?<![\w#]){re.escape(param)}(?!\w)", arg, expanded)
            out.append(expanded)
    return "\n".join(out)


def exports(source_texts: Iterable[str]) -> list[str]:
    """Names of the `extern "C"` functions (the library's ctypes / driver API), macro-generated ones included."""
    out: set[str] = set()
    for text in source_texts:
        for t in (strip_comments(text), expand_export_macros(text)):
            out.update(re.findall(r'extern "C"\s+[\w\s*]*?\b(\w+)\s*\(', _LAUNCH_BOUNDS.sub("", t)))
    return sorted(out)


def symbol_hashes(source_texts: Iterable[str], names: Iterable[str], environment: str) -> dict[str, str]:
    """Hash of every exported function in `names`, 16 hex digits.

    Args:
        source_texts: The library's sources, in build order.
        names: The exported symbols (`exports`).
        environment: Anything else that changes the compiled code (build.toml, toolchain); part of every hash.

    Raises:
        ValueError: A name has no definition in the sources.
    """
    texts = list(source_texts)
    all_items = [it for text in texts for it in items(text)]
    all_items += [it for text in texts for it in items(expand_export_macros(text))]  # the macro-generated exports
    functions = [it for it in all_items if _is_function(it)]
    other = [it for it in all_items if not _is_function(it)]
    defined: dict[str, list[int]] = {}
    for idx, fn in enumerate(functions):
        for name in _CALLEE.findall(_LAUNCH_BOUNDS.sub("", fn.split("{", 1)[0])):
            defined.setdefault(name, []).append(idx)
    preamble = "\n".join(other)
    result: dict[str, str] = {}
    for name in names:
        if name not in defined:
            msg = f"exported function {name!r} has no definition in the sources"
            raise ValueError(msg)
        seen: set[int] = set()
        todo = list(defined[name])
        scanned = [preamble]
        while todo:
            idx = todo.pop()
            if idx in seen:
                continue
            seen.add(idx)
            scanned.append(functions[idx])
            for word in set(_IDENT.findall(functions[idx])) | set(_IDENT.findall(preamble)):
                todo.extend(defined.get(word, ()))
        h = hashlib.sha256(environment.encode())
        h.update(preamble.encode())
        for idx in sorted(seen):
            h.update(functions[idx].encode())
        result[name] = h.hexdigest()[:16]
    return result


def library_symbols(lib_dir: Path, build_toml: str, toolchain: Mapping[str, str]) -> dict[str, str]:
    """`symbol_hashes` of the library in `lib_dir` (its `build.toml` text and the manifest's `[toolchain]`)."""
    import tomllib  # noqa: PLC0415 - only the build tools and tests need it

    cfg = tomllib.loads(build_toml)
    texts = [(lib_dir / s).read_text() for s in cfg["sources"]]
    environment = build_toml + repr(sorted(toolchain.items()))
    return symbol_hashes(texts, exports(texts), environment)
