# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""
Add or update the mdapy copyright header in Python and C/C++ source files.

Usage:
    python add_copyright.py           # preview changes (dry run)
    python add_copyright.py --apply   # actually write changes

To update the year next year, change END_YEAR below.
"""

import argparse
import re
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

START_YEAR = 2022
END_YEAR   = 2026          # <── change this each year (e.g. 2027 next year)

# Directories to scan (relative to this script's location)
SCAN_DIRS = ["src", "tests"]

# File extensions and their comment style
EXTENSIONS = {
    ".py":  "#",
    ".cpp": "//",
    ".h":   "//",
    ".c":   "//",
    ".cu":  "//",
    ".cuh": "//",
}

# ── Derived constants ─────────────────────────────────────────────────────────

def _make_header(comment: str) -> str:
    return (
        f"{comment} Copyright (c) {START_YEAR}-{END_YEAR},"
        f" Yongchao Wu in Aalto University\n"
        f"{comment} This file is from the mdapy project,"
        f" released under the BSD 3-Clause License.\n"
    )

# Pattern that matches the FIRST copyright line regardless of the end year
_COPYRIGHT_PATTERN = re.compile(
    r"^(?:#|//) Copyright \(c\) \d{4}-\d{4}, Yongchao Wu in Aalto University"
)

# Pattern to capture just the end-year so we can update it
_YEAR_RANGE_PATTERN = re.compile(
    r"(\d{4}-)\d{4}(, Yongchao Wu in Aalto University)"
)

# ── Core logic ────────────────────────────────────────────────────────────────

def process_file(path: Path, comment: str, dry_run: bool) -> str:
    """Return a status string describing what was done (or would be done)."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return "skipped (non-UTF-8 encoding)"
    lines = text.splitlines(keepends=True)

    # Look for an existing mdapy copyright in the first few lines
    header_line_idx = None
    for i, line in enumerate(lines[:5]):
        if _COPYRIGHT_PATTERN.match(line):
            header_line_idx = i
            break

    desired_header = _make_header(comment)

    if header_line_idx is not None:
        # Copyright exists – check whether the end year is already correct
        existing_line = lines[header_line_idx]
        updated_line = _YEAR_RANGE_PATTERN.sub(
            rf"\g<1>{END_YEAR}\g<2>", existing_line
        )
        if updated_line == existing_line:
            return "ok (already up to date)"

        # Need to update the year
        new_lines = lines[:]
        new_lines[header_line_idx] = updated_line
        new_text = "".join(new_lines)
        if not dry_run:
            path.write_text(new_text, encoding="utf-8")
        return f"updated year → {END_YEAR}"

    else:
        # No copyright found – prepend the header
        # Preserve shebang line if present
        if lines and lines[0].startswith("#!"):
            new_text = lines[0] + desired_header + "".join(lines[1:])
        else:
            new_text = desired_header + text
        if not dry_run:
            path.write_text(new_text, encoding="utf-8")
        return "added header"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Write changes to disk (default is dry-run / preview only)"
    )
    args = parser.parse_args()
    dry_run = not args.apply

    root = Path(__file__).parent

    if dry_run:
        print("DRY RUN – no files will be modified. Pass --apply to write changes.\n")

    counts = {"ok (already up to date)": 0, "updated year": 0, "added header": 0, "skipped": 0}

    for dir_name in SCAN_DIRS:
        scan_dir = root / dir_name
        if not scan_dir.exists():
            print(f"  [skip] directory not found: {scan_dir}")
            continue

        for ext, comment in EXTENSIONS.items():
            for path in sorted(scan_dir.rglob(f"*{ext}")):
                # Skip compiled / cache artefacts
                if any(part.startswith("__pycache__") for part in path.parts):
                    continue
                if path.suffix == ".pyc":
                    continue

                status = process_file(path, comment, dry_run)

                # Bucket the status for the summary
                for key in counts:
                    if status.startswith(key):
                        counts[key] += 1
                        break
                else:
                    counts["skipped"] += 1

                rel = path.relative_to(root)
                if status not in ("ok (already up to date)",) and not status.startswith("skipped"):
                    print(f"  [{status}]  {rel}")

    print()
    print("Summary:")
    print(f"  already up to date : {counts['ok (already up to date)']}")
    print(f"  year updated       : {counts['updated year']}")
    print(f"  header added       : {counts['added header']}")
    print(f"  skipped            : {counts['skipped']}")
    if dry_run:
        print("\nRun with --apply to write the changes above.")


if __name__ == "__main__":
    main()
