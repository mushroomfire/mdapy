# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A stand-in for VASP in the tests: run in a directory of VASP inputs, it
computes the Zhou EAM of the POSCAR and writes an OUTCAR as VASP would.
INCAR tags steer it: FAKE_FAIL exits with an error, FAKE_UNCONVERGED leaves
the electronic loop unconverged, FAKE_SILENT writes no OUTCAR."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from outcar_files import outcar_text  # noqa: E402

import mdapy  # noqa: E402
from mdapy.calculators import EAM  # noqa: E402

tags = {}
for line in Path("INCAR").read_text().splitlines():
    name, value = (part.strip() for part in line.split("=", 1))
    tags[name] = value
if "FAKE_FAIL" in tags:
    print("VERY BAD NEWS! internal error in subroutine fake")
    sys.exit(3)
system = mdapy.read("POSCAR", format="poscar")
elements = system.table["element"].cast(str).to_list()
species = list(dict.fromkeys(elements))
counts = [elements.count(e) for e in species]
result = EAM.zhou(species).compute(system, ("energy", "forces", "virial"))
steps = [
    (
        system.box.matrix,
        system["positions"],
        result.forces,
        result.energy,
        result.virial,
    )
]
if "FAKE_SILENT" in tags:
    sys.exit(0)
text = outcar_text(steps, species, counts, converged=["FAKE_UNCONVERGED" not in tags])
Path("OUTCAR").write_text(text)
print(" reached required accuracy - stopping structural energy minimisation")
