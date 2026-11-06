"""`python -m fastforks ...`: the `fastforks` command line (for a venv whose `bin/` is not on PATH)."""

import sys

from fastforks._core.cli import main

if __name__ == "__main__":
    sys.exit(main())
