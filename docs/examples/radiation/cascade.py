# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""One cascade in W or in the random equiatomic MoNbTaVW alloy at 300 K, as Liu et
al. (J. Nucl. Mater. 2025, arXiv:2411.02834) run them, with their NEP-ZBL
potential (AGPL-3.0, downloaded from GPUMD-Wizard): the crystal relaxed at 300 K
and zero pressure, a PKA of the energy given along a <135> direction (its signs
and order from the run index; in the alloy its element too), the border of 3
lattice constants at 300 K, an adaptive time step of at most 0.015 Å and 1 fs,
the electronic stopping of Lindhard and Scharff by Bragg's rule above 10 eV; the
Frenkel pairs by Wigner-Seitz every picosecond, the clusters of vacancies and
interstitials at the end.

    python cascade.py W 5000 0 out.json [SIZE] [PS]

With an energy of 0 it only relaxes the crystal and keeps it in
warm-MATERIAL-SIZE.mdapy, which the runs of the alloy then share (one random
arrangement, every run its own PKA); the runs of W relax their own.
"""

import itertools
import json
import os
import sys
import time
import urllib.request

import numpy as np

from mdapy import units
from mdapy.analysis import ClusterAnalysis, WignerSeitz
from mdapy.build import build_crystal
from mdapy.calculators import NEP
from mdapy.elements import atomic_masses, atomic_numbers
from mdapy.simulation import MolecularDynamics
from mdapy.simulation.controls import AdaptiveTimestep, ElectronicStopping, Langevin

URL = ("https://raw.githubusercontent.com/Jonsnow-willow/GPUMD-Wizard/main/tutorials/"
       "Calculatiing_Material_Properties/MoNbTaVW/potentials/nep.txt")  # fmt: skip
material, energy, run, out = (
    sys.argv[1],
    float(sys.argv[2]),
    int(sys.argv[3]),
    sys.argv[4],
)
size = int(sys.argv[5]) if len(sys.argv) > 5 else 50
total = float(sys.argv[6]) if len(sys.argv) > 6 else 50.0  # ps of the cascade
start = time.time()
if not os.path.exists("nep_monbtavw.txt"):
    urllib.request.urlretrieve(URL, "nep_monbtavw.txt")
gpu = NEP("nep_monbtavw.txt", device="cuda")
elements = ["W"] if material == "W" else ["Mo", "Nb", "Ta", "V", "W"]
a0 = 3.17 if material == "W" else 3.18
crystal = build_crystal("bcc", a0, elements="W", repeat=(size, size, size))
rng = np.random.default_rng(1000 + run)
if material != "W":  # equiatomic and random
    kinds = np.resize(np.array(elements), len(crystal))
    rng.shuffle(kinds)
    crystal = crystal.with_columns(element=kinds)
cache = f"warm-{material}-{size}.mdapy"
if material != "W" and energy > 0 and os.path.exists(cache):
    import mdapy.io

    warm = mdapy.io.read(cache)
else:
    with MolecularDynamics(crystal, gpu, timestep=0.001) as md:
        md.velocities(600, seed=run + 1)
        md.run(
            10000, ensemble="npt", temperature=300, pressure=0.0, thermo=100
        )  # 10 ps
        warm = md.system
    if energy == 0:
        import mdapy.io

        mdapy.io.write(cache + ".part", warm, format="mdapy")
        os.replace(cache + ".part", cache)
        sys.exit(0)
a = (warm.box.volume / size**3) ** (1 / 3)  # the mean lattice constant
reference = build_crystal("bcc", a, elements="W", repeat=(size, size, size))
reference = reference.with_columns(element=np.asarray(warm["element"]))
# the PKA: an atom near the middle, a <135> direction of the run
names = np.asarray(warm["element"])
center = warm.box.origin + warm.box.matrix.sum(axis=0) / 2
near = np.argsort(np.linalg.norm(warm["positions"] - center, axis=1))[:200]
if material != "W":
    near = near[names[near] == elements[run % len(elements)]]
pka = int(near[0])
orders = list(itertools.permutations((1, 3, 5)))
signs = list(itertools.product((1, -1), repeat=3))
direction = np.array(orders[run % 6], dtype=float) * np.array(signs[(run // 6) % 8])
direction /= np.linalg.norm(direction)
mass = (
    warm["mass"][pka]
    if "mass" in warm.table.columns
    else atomic_masses([names[pka]])[0]
)
v = warm["velocities"].copy()
v[pka] = np.sqrt(2 * energy / (mass * units.AMU_A2_PER_PS2_TO_EV)) * direction
shot = warm.with_columns(velocities=v)
# Lindhard and Scharff of each element in the target, by Bragg's rule
e2, bohr_radius, bohr_speed = 14.399645, 0.529177, 21876.9
density = len(shot) / shot.box.volume
fractions = {el: float(np.mean(names == el)) for el in elements}
grid = np.geomspace(1.0, 2.0 * energy, 200)
table = {}
for el in elements:
    z1, m1 = atomic_numbers([el])[0], atomic_masses([el])[0]
    k = sum(c * 8 * np.pi * e2 * bohr_radius * z1 ** (7 / 6) * atomic_numbers([t])[0]
            / (z1 ** (2 / 3) + atomic_numbers([t])[0] ** (2 / 3)) ** 1.5 for t, c in fractions.items())  # fmt: skip
    speeds = np.sqrt(2 * grid / (m1 * units.AMU_A2_PER_PS2_TO_EV))
    table[el] = (grid, density * k * speeds / bohr_speed)
L = shot.box.lengths
inside = shot.box.to_fractional(shot["positions"]) * L
core = ((inside > 3 * a) & (inside < L - 3 * a)).all(axis=1)
controls = [
    AdaptiveTimestep(0.015),
    Langevin(~core, 300.0, damping=0.1, seed=run + 1),
    ElectronicStopping(table, min_energy=10.0),
]
sites = WignerSeitz(reference)
pairs, times = [], []
with MolecularDynamics(shot, gpu, timestep=0.001) as md:
    for frame in md.iterate(
        10_000_000, ensemble="nve", thermo=200, save_every=1000, controls=controls
    ):
        pairs.append(int(sites.compute(frame).vacancy_count))
        times.append(float(frame.info["time"]))
        if frame.info["time"] >= total:
            break
    thermo = md.thermo
    last = md.system
loss = float(thermo["electronic_loss"].to_numpy()[-1])
ws = sites.compute(last)
occupancy = np.asarray(ws.occupancy)
vacancies = reference["positions"][occupancy == 0]
extra = np.repeat(
    np.nonzero(occupancy > 1)[0], (occupancy[occupancy > 1] - 1).astype(int)
)
interstitials = reference["positions"][extra]


def cluster_sizes(points, cutoff):
    if len(points) == 0:
        return []
    import mdapy

    system = mdapy.System(points, last.box, elements="W")
    return [int(s) for s in ClusterAnalysis(cutoff=cutoff).compute(system).sizes]


result = {
    "material": material, "energy": energy, "run": run, "atoms": len(crystal), "lattice": a,
    "pka_element": str(names[pka]), "direction": direction.tolist(), "electronic_loss": loss,
    "damage_energy": energy - loss, "pairs": pairs, "times": times, "final": int(ws.vacancy_count),
    "peak": max(pairs), "vacancy_clusters": cluster_sizes(vacancies, 1.2 * a),
    "interstitial_clusters": cluster_sizes(interstitials, 1.55 * a), "seconds": time.time() - start,
}  # fmt: skip
print(
    {
        k: result[k]
        for k in (
            "material",
            "energy",
            "run",
            "pka_element",
            "damage_energy",
            "peak",
            "final",
            "seconds",
        )
    },
    flush=True,
)
with open(out, "w") as f:
    json.dump(result, f)
