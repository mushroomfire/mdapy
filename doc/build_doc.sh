#!/usr/bin/env bash
# Full documentation build (used by CI / ReadTheDocs / before a release).
#
# For local iteration, prefer ./regen_api.sh + esbonio's live preview
# — that avoids wiping _build on every Python-source change.
#
# Usage (from anywhere):
#   bash doc/build_doc.sh

set -eu

cd "$(dirname "$0")"

# 1. Refresh source/*.rst from the current Python sources.
./regen_api.sh

# 2. Clean rebuild of doc/_build/html/.
make clean && make html
