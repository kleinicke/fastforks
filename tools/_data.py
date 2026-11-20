"""Where the evaluation datasets live: `$FASTFORKS_DATA` (default `~/.cache/fastforks/data`)."""

from __future__ import annotations

import os
from pathlib import Path

DATA = Path(os.environ.get("FASTFORKS_DATA", "~/.cache/fastforks/data")).expanduser()
