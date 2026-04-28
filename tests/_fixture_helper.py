# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Fixture loader for the structure-analysis tests.

There is one .npz file per configuration in tests/fixtures/structure_analysis/.
Each file holds the wrapped positions, box matrix, boundary, AND every
algorithm's reference output (csp, cna, aja, ptm, ids, cnp, q4, q6, ...)
along with the parameters that produced them.

Algorithms that don't apply to a given configuration just leave their
keys out of the file. The `iter_fixtures(key)` helper yields only the
fixtures that contain the requested reference key, and pytest skips
the rest cleanly.
"""

from pathlib import Path
import numpy as np

import mdapy as mp
from mdapy.box import Box

FIXTURE_FILE_DIR = (
    Path(__file__).parent / "fixtures" / "structure_analysis"
)
MISC_DIR = Path(__file__).parent / "fixtures" / "misc"
ADVANCED_DIR = Path(__file__).parent / "fixtures" / "advanced"
INPUT_DIR = Path(__file__).parent / "input_files"


def load_misc(name):
    """Load tests/fixtures/misc/<name>.npz as a numpy NpzFile object."""
    return np.load(MISC_DIR / f"{name}.npz")


def load_advanced(name):
    """Load tests/fixtures/advanced/<name>.npz as a numpy NpzFile object."""
    return np.load(ADVANCED_DIR / f"{name}.npz", allow_pickle=False)


def input_path(filename):
    """Resolve a test sample input filename to a string absolute path."""
    return str(INPUT_DIR / filename)


def all_fixture_paths():
    return sorted(FIXTURE_FILE_DIR.glob("*.npz"))


def fixtures_with(key):
    """Return paths whose .npz contains the given reference key.

    A second `key` is supported for callers that want either of two
    options (e.g. CSP needs `csp` AND `csp_num_neighbors`)."""
    out = []
    for p in all_fixture_paths():
        with np.load(p) as data:
            if key in data.files:
                out.append(p)
    return out


def fixture_ids(paths):
    return [p.stem for p in paths]


def system_from_fixture(data):
    return mp.System(pos=data["pos"],
                     box=Box(data["box"], boundary=list(data["boundary"])))
