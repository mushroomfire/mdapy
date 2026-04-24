#!/usr/bin/env bash
# Regenerate the auto-API source files from mdapy's Python docstrings.
#
# Run this after editing any .py file whose docstring should appear in the
# rendered docs. It rewrites doc/source/*.rst only — it does NOT run a
# sphinx build. Esbonio's live preview will pick up the new .rst files
# automatically and re-render; a manual "make html" is only needed for the
# final publishable output.
#
# Usage (from anywhere inside the repo):
#   bash doc/regen_api.sh
#
# Or from inside doc/:
#   ./regen_api.sh

set -eu

cd "$(dirname "$0")"

rm -rf source
sphinx-apidoc -o source ../src/mdapy/

rm -f source/modules.rst

# Rewrite source/mdapy.rst:
#   - drop the "Module contents" section and everything after it
#   - drop the original 5-line header
#   - replace it with "Index\n======\n"
python - <<'PY'
from pathlib import Path
p = Path("source/mdapy.rst")
lines = p.read_text().splitlines()

# Drop "Module contents" onwards (match the start of the section).
for i, line in enumerate(lines):
    if line.startswith("Module contents"):
        lines = lines[:i]
        break

# Drop the original 5-line header (module title + underline + automodule hint).
lines = lines[5:]

# Prepend a fresh "Index" heading.
lines = ["Index", "======", ""] + lines

p.write_text("\n".join(lines) + "\n")
PY

echo "✅ Regenerated doc/source/*.rst from current Python sources."
