# Copyright (c) 2022-2026, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
import sys
from pathlib import Path

# Make sibling helpers like `_fixture_helper.py` importable without an
# explicit package path.
sys.path.insert(0, str(Path(__file__).parent))
