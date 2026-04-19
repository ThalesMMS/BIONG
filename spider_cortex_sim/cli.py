from __future__ import annotations

from .cli import build_parser, run_cli


def main() -> None:
    """Parse CLI arguments and dispatch the requested workflow."""
    parser = build_parser()
    args = parser.parse_args()
    args._parser = parser
    run_cli(args)


if __name__ == "__main__":
    main()
