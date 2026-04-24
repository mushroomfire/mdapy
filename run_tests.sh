#!/usr/bin/env bash
# Run the mdapy test suite with one subprocess per test file.
#
# Why per-file isolation:
#   On macOS, ovito and freud both bundle voro++ + TBB; initializing one
#   corrupts the other's TBB state, so any process that imports both and
#   calls a Voronoi routine will segfault (the order of import doesn't
#   matter — both directions crash). This is an upstream incompatibility,
#   not an mdapy issue. Running each test file in its own process keeps
#   them apart.
#
# Usage:
#   ./run_tests.sh                 # run every tests/test_*.py
#   ./run_tests.sh test_csp.py     # run a subset (names resolved under tests/)
#
# Exit code: 0 on success, 1 on first failing file.

set -u

repo_root=$(cd "$(dirname "$0")" && pwd)
cd "$repo_root/tests" || exit 1

if [ $# -gt 0 ]; then
    files=("$@")
else
    files=(test_*.py)
fi

pass=0
fail=0
failed_files=()

for f in "${files[@]}"; do
    printf '\n=== %s ===\n' "$f"
    if python -m pytest "$f" -q --tb=short; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        failed_files+=("$f")
    fi
done

printf '\n==============================\n'
printf 'Passed files: %d\n' "$pass"
printf 'Failed files: %d\n' "$fail"
if [ $fail -gt 0 ]; then
    printf 'Failures:\n'
    for f in "${failed_files[@]}"; do
        printf '  - %s\n' "$f"
    done
    exit 1
fi
