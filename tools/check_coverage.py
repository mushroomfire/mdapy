# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.
"""Fail if any source file is below the per-file coverage threshold.

coverage.py only enforces a total threshold (``fail_under`` in
pyproject.toml). This script adds the per-file rule: every file must reach
``--min-file`` percent unless it is listed in EXEMPT with a reason.

Usage: python tools/check_coverage.py coverage.json [--min-file 95]
"""

import argparse
import json
import sys

# path suffix -> reason. Keep this short; every entry needs a justification.
EXEMPT: dict[str, str] = {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="JSON report from `coverage json`")
    parser.add_argument("--min-file", type=float, default=95.0)
    args = parser.parse_args(argv)

    with open(args.report, encoding="utf-8") as f:
        files = json.load(f)["files"]

    failures = []
    for path, data in sorted(files.items()):
        percent = data["summary"]["percent_covered"]
        exempt = next((r for s, r in EXEMPT.items() if path.endswith(s)), None)
        status = "exempt" if exempt else ("ok" if percent >= args.min_file else "LOW")
        print(f"{percent:6.1f}%  {status:6s}  {path}")
        if status == "LOW":
            failures.append(path)

    if failures:
        print(f"\n{len(failures)} file(s) below {args.min_file}%:", file=sys.stderr)
        for path in failures:
            print(f"  {path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
