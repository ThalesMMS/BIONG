from __future__ import annotations

import argparse
import sys


def _argument_error(args: argparse.Namespace, message: str) -> None:
    """
    Handle a command-line argument error by delegating to an argparse parser (if present), printing the error to stderr, and exiting with status 2.
    
    Parameters:
        args (argparse.Namespace): Namespace that may contain a `_parser` attribute; if present, its `error` method will be invoked with `message`.
        message (str): The error message to report.
    
    Raises:
        SystemExit: Exits the process with status code 2.
    """
    parser = getattr(args, "_parser", None)
    if parser is not None:
        parser.error(message)
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)
