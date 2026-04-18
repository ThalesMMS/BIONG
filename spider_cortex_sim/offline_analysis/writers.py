from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path

from .utils import _coerce_optional_float


def _write_svg(path: Path, svg: str) -> None:
    """
    Write SVG content to the given filesystem path.

    Writes the provided SVG text to `path` using UTF-8 encoding, overwriting any existing file.
    """
    path.write_text(svg, encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """
    Write rows of mapping objects to a CSV file and include a header row.
    
    Opens `path` for writing with UTF-8 encoding and `newline=""`, overwriting any existing file. Writes a header using `fieldnames` (order preserved). For each mapping in `rows`, writes a row containing exactly the given `fieldnames`; missing keys are written as empty strings.
    
    Parameters:
        path (Path): Destination file path to write the CSV.
        rows (Sequence[Mapping[str, object]]): Iterable of mappings representing rows.
        fieldnames (Sequence[str]): Column names and order to include in the CSV.
    """
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _escape_markdown_table_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _markdown_table(rows: Sequence[Sequence[object]], headers: Sequence[str]) -> str:
    """
    Format tabular data as a Markdown table.
    
    If `rows` is empty, returns the literal "_No data available._".
    
    Parameters:
        rows (Sequence[Sequence[object]]): Sequence of row sequences; each inner sequence represents a row's cell values.
        headers (Sequence[str]): Sequence of column header strings.
    
    Returns:
        str: A Markdown-formatted table with a header line, a separator line, and one body line per row. Cell values are converted to strings.
    """
    if not rows:
        return "_No data available._"
    header_line = "| " + " | ".join(
        _escape_markdown_table_cell(header) for header in headers
    ) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(_escape_markdown_table_cell(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header_line, separator, *body])


def _markdown_value(value: object, *, digits: int = 2) -> str:
    """
    Format a Python value for human-readable Markdown/ASCII output.
    
    Parameters:
        value (object): The value to format.
        digits (int): Number of decimal places to use when formatting numeric values (default 2).
    
    Returns:
        str: A formatted string where:
            - `None` becomes "—".
            - booleans become "yes" or "no".
            - numeric-like values are formatted to `digits` decimal places.
            - non-empty strings and other objects use `str(value)`.
            - empty strings become "—".
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    numeric_value = _coerce_optional_float(value)
    if numeric_value is not None and not isinstance(value, str):
        return f"{numeric_value:.{digits}f}"
    text = str(value)
    return text if text else "—"


def _table(columns: Sequence[str], rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """
    Builds a simple table representation with stringified column names and dictionary rows.
    
    Parameters:
        columns (Sequence[str]): Column identifiers; each element is converted to `str`.
        rows (Sequence[Mapping[str, object]]): Row mappings; each mapping is shallow-copied into a `dict`.
    
    Returns:
        dict[str, object]: A mapping with keys:
            - "columns": list of column names as strings.
            - "rows": list of row dictionaries.
    """
    return {
        "columns": [str(column) for column in columns],
        "rows": [dict(row) for row in rows],
    }


def _table_rows(table: Mapping[str, object]) -> list[Mapping[str, object]]:
    """
    Return mapping rows stored under the "rows" key of a table mapping.
    
    Parameters:
        table (Mapping[str, object]): Mapping representing a table; may contain a "rows" key.
    
    Returns:
        list[Mapping[str, object]]: Mapping rows from `table["rows"]`, or an empty list if rows is invalid.
    """
    rows = table.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, Mapping)]
