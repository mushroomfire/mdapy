# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The elastic constants of fcc aluminium (four atoms) from MDAPY's DFT against
the mesh of k-points, at the cutoff and smearing of the labels."""

import json
import os
import sys
import time

import numpy as np

from mdapy.build import build_crystal
from mdapy.calculators import DFT
from mdapy.properties.elastic import elastic_constants

out = {}
for n in [int(x) for x in sys.argv[1:]]:
    t = time.time()
    smearing, sigma = (
        os.environ.get("SMEARING", "gaussian"),
        float(os.environ.get("SIGMA", "0.1")),
    )
    dft = DFT(
        600.0,
        pseudopotentials={"Al": "Al.upf"},
        kpoints=(n, n, n),
        smearing=smearing,
        sigma=sigma,
    )
    crystal = build_crystal("fcc", 4.041, elements="Al")
    c = np.asarray(elastic_constants(crystal, dft, max_force=1e-3).constants)
    out[n] = {
        "C11": c[0, 0],
        "C12": c[0, 1],
        "C44": c[3, 3],
        "seconds": time.time() - t,
    }
    print(n, out[n], flush=True)
    tag = os.environ.get("SMEARING", "gaussian") + os.environ.get("SIGMA", "0.1")
    json.dump(out, open(f"kconv-{tag}-{'-'.join(sys.argv[1:])}.json", "w"))
