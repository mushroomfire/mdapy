# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Threshold displacement energies at 300 K, as Liu et al. (J. Nucl. Mater. 2025,
arXiv:2411.02834) compute them, with their NEP-ZBL potential: a 12 x 13 x 14 bcc
crystal of 4368 atoms of one metal or of the random equiatomic MoNbTaVW alloy,
relaxed at 300 K; for each random direction a recoil of an atom near the middle
(in the alloy, of the element given), runs of 6 ps with the border of one
lattice constant kept at 300 K, no electronic stopping, until a Frenkel pair
stays; E_d is 1 eV below the first energy that leaves one. The energy climbs
from 20 eV in steps of 10 eV and the last step is searched again in steps of 2
eV, which finds the same first energy as steps of 2 eV unless a window narrower
than 10 eV leaves a pair below it.

    python tde.py W W 200 0 out.json        (metal, recoil element, directions, seed)
    python tde.py HEA Nb 200 0 out.json
"""

import json
import os
import sys
import time
import urllib.request

import numpy as np

from mdapy import units
from mdapy.analysis import WignerSeitz
from mdapy.build import build_crystal
from mdapy.calculators import NEP
from mdapy.elements import atomic_masses
from mdapy.simulation import MolecularDynamics
from mdapy.simulation.controls import AdaptiveTimestep, Langevin

URL = ("https://raw.githubusercontent.com/Jonsnow-willow/GPUMD-Wizard/main/tutorials/"
       "Calculatiing_Material_Properties/MoNbTaVW/potentials/nep.txt")  # fmt: skip
material, element, count, seed, out = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
    sys.argv[5],
)
start = time.time()
if not os.path.exists("nep_monbtavw.txt"):
    urllib.request.urlretrieve(URL, "nep_monbtavw.txt")
gpu = NEP("nep_monbtavw.txt", device="cuda")
guess = {"V": 3.00, "Nb": 3.31, "Mo": 3.16, "Ta": 3.31, "W": 3.17, "HEA": 3.18}[
    material
]
crystal = build_crystal("bcc", guess, elements="W", repeat=(12, 13, 14))
rng = np.random.default_rng(seed)
if material == "HEA":
    kinds = np.resize(np.array(["Mo", "Nb", "Ta", "V", "W"]), len(crystal))
    rng.shuffle(kinds)
else:
    kinds = np.full(len(crystal), material)
crystal = crystal.with_columns(element=kinds)
with MolecularDynamics(crystal, gpu, timestep=0.001) as md:
    md.velocities(600, seed=seed + 1)
    md.run(10000, ensemble="npt", temperature=300, pressure=0.0)  # 10 ps
    warm = md.system
a = (warm.box.volume / (12 * 13 * 14)) ** (1 / 3)
reference = build_crystal("bcc", a, elements="W", repeat=(12, 13, 14)).with_columns(
    element=kinds
)
sites = WignerSeitz(reference)
names = np.asarray(warm["element"])
L = warm.box.lengths
inside = warm.box.to_fractional(warm["positions"]) * L
border = ~((inside > a) & (inside < L - a)).all(axis=1)
center = warm.box.origin + warm.box.matrix.sum(axis=0) / 2
middle = np.argsort(np.linalg.norm(warm["positions"] - center, axis=1))[:400]
middle = middle[names[middle] == element]
mass = atomic_masses([element])[0]


def stays(energy, atom, direction, trial):
    """Whether a recoil of energy eV leaves a Frenkel pair after 6 ps."""
    v = warm["velocities"].copy()
    v[atom] = np.sqrt(2 * energy / (mass * units.AMU_A2_PER_PS2_TO_EV)) * direction
    controls = [
        AdaptiveTimestep(0.015),
        Langevin(border, 300.0, damping=0.1, seed=trial),
    ]
    with MolecularDynamics(warm.with_columns(velocities=v), gpu, timestep=0.001) as md:
        for frame in md.iterate(
            10_000_000, ensemble="nve", save_every=500, controls=controls
        ):
            if frame.info["time"] >= 6.0:
                break
        last = md.system
    return sites.compute(last).vacancy_count > 0


directions, energies, trials = [], [], 0
for i in range(count):
    d = rng.normal(size=3)
    d /= np.linalg.norm(d)
    atom = int(rng.choice(middle))
    low, energy, found = None, 20.0, None
    while energy <= 400.0:
        trials += 1
        if stays(energy, atom, d, trials):
            found = energy
            break
        low, energy = energy, energy + 10.0
    if found is not None and low is not None:
        for e in (
            low + 2.0,
            low + 4.0,
            low + 6.0,
            low + 8.0,
        ):  # the first in steps of 2 eV
            trials += 1
            if stays(e, atom, d, trials):
                found = e
                break
    directions.append(d.tolist())
    energies.append(None if found is None else found - 1.0)
    print(
        f"{material} {element} {i}: {energies[-1]} eV, {trials} runs, {time.time() - start:.0f} s",
        flush=True,
    )
    with open(out, "w") as f:
        json.dump({"material": material, "element": element, "lattice": a, "directions": directions,
                   "tde": energies, "runs": trials, "seconds": time.time() - start}, f)  # fmt: skip
