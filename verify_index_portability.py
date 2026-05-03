from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path


_LINK_RE = re.compile(r"\]\(([^)]+)\)")


def _extract_links(markdown: str) -> list[str]:
    return [m.group(1).strip() for m in _LINK_RE.finditer(markdown)]


def _is_external(href: str) -> bool:
    return href.startswith("http://") or href.startswith("https://") or href.startswith("mailto:")


def _is_anchor_only(href: str) -> bool:
    return href.startswith("#")


def _strip_anchor(href: str) -> str:
    return href.split("#", 1)[0]


def validate_index(report_dir: Path) -> None:
    index_path = report_dir / "INDEX.md"
    if not index_path.exists():
        raise SystemExit(f"Missing INDEX.md in {report_dir}")

    markdown = index_path.read_text(encoding="utf-8")
    links = _extract_links(markdown)

    errors: list[str] = []
    for href in links:
        if not href:
            continue
        if _is_external(href) or _is_anchor_only(href):
            continue

        if href.startswith("/") or re.match(r"^[A-Za-z]:\\", href):
            errors.append(f"absolute link not portable: {href}")
            continue

        target = _strip_anchor(href)
        if not target:
            continue

        if target.startswith(".."):
            errors.append(f"link escapes report dir: {href}")
            continue

        portable_target = target[2:] if target.startswith("./") else target
        target_path = (report_dir / portable_target).resolve()
        try:
            target_path.relative_to(report_dir.resolve())
        except ValueError:
            errors.append(f"link escapes report dir: {href}")
            continue

        if not target_path.exists():
            errors.append(f"broken link target missing: {href}")

    if errors:
        raise SystemExit("\n".join(["Portability validation failed:"] + errors))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate INDEX.md links are portable and resolve after copying report dir.\n"
            "Note: this assumes INDEX.md links only point at existing artifacts (missing artifacts are rendered as plain text)."
        )
    )
    parser.add_argument("report_dir", type=Path, help="Path to report directory containing INDEX.md")
    args = parser.parse_args()

    report_dir = args.report_dir.resolve()
    if not report_dir.exists():
        raise SystemExit(f"Report dir not found: {report_dir}")

    with tempfile.TemporaryDirectory(prefix="report-portability-") as tmp:
        copied = Path(tmp) / report_dir.name
        shutil.copytree(report_dir, copied)
        validate_index(copied)

    print("OK")


if __name__ == "__main__":
    main()
