import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "spider_cortex_sim requires Python 3.10+ for the simulator CLI. "
        "Use `python3.10` or newer."
    )

from .cli import main


if __name__ == "__main__":
    main()
