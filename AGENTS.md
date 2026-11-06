# Agent notes

Read CONTRIBUTING.md first: setup, code rules and the add-a-model checklist apply to agents too.

- Keep every change passing `ruff format`, `ruff check`, `ty check` and `pytest -m "not gpu"`.
- Vendored `net/` code stays byte-close to upstream: fix bugs there only with a comment, never reformat it.
- Never add a silent fallback. A backend that cannot run raises; `mode="auto"` is the only place that chooses, and
  it logs the choice.
- Never commit weights, golden outputs or kernel binaries (they are hash-referenced and downloaded).
- After a kernel rebuild, commit the `[validated]` entries a `pytest -m gpu --record` run wrote; never edit them by hand.
- Do not change a golden or bump a revision to make a test pass; explain the numerical change in the commit message.
- Commit messages: imperative subject, body says what and why; no tool or AI attribution lines.
