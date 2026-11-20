# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Between two builds of its neighbor list, MDAPY's dynamics gives MDAPY's
potentials the list with its skin as it is, and they pass over the pairs
beyond their cutoffs: the runs are those of a list of the pairs within the
cutoff at each step, to the last bit, the pairs being summed in the same
order. Other calculators are given those pairs."""

import numpy as np
import pytest

from mdapy.calculators import EAM, LennardJones
from mdapy.calculators._base import Calculator
from mdapy.simulation import MolecularDynamics
from simulation.test_domain import copper, potentials


def run(system, potential, steps=60):
    with MolecularDynamics(system, potential, timestep=0.002) as md:
        md.velocities(900, seed=1)
        md.run(steps, ensemble="nvt", temperature=900, thermo=20)
        return md.system, md.thermo


@pytest.mark.parametrize(
    "name", ["EAM", "Lennard-Jones", "ADP", "MEAM", "NEP", "Tersoff",
             "Stillinger-Weber", "Morse", "Uhlenbeck-Ford"]
)  # fmt: skip
def test_the_runs_of_the_list_with_its_skin_are_those_of_the_pairs(
    monkeypatch, tmp_path, name
):
    if name == "EAM":
        potential, system = EAM.zhou(["Cu"]), copper(5)
    elif name == "Lennard-Jones":
        potential, system = LennardJones({"Cu": (0.4, 2.3)}, 5.0), copper(5)
    else:
        potential, system = potentials(tmp_path)[name]
    assert type(potential)._takes_wider_lists
    a, ta = run(system, potential)
    monkeypatch.setattr(type(potential), "_takes_wider_lists", False)
    b, tb = run(system, potential)
    np.testing.assert_array_equal(a["positions"], b["positions"])
    np.testing.assert_array_equal(a["velocities"], b["velocities"])
    assert ta.equals(tb)


def test_other_calculators_get_the_pairs_within_their_cutoff():
    """A calculator that would count the pairs beyond its cutoff."""
    eam = EAM.zhou(["Cu"])
    seen = []

    class Counting(Calculator):
        cutoff = eam.cutoff
        implemented = eam.implemented

        def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
            seen.append(float(neighbors.distances.max()))
            return eam.compute(system, properties, neighbors=neighbors)

    with MolecularDynamics(copper(5), Counting(), timestep=0.002) as md:
        md.velocities(900, seed=1)
        md.run(30)
    assert len(seen) == 31 and max(seen) < eam.cutoff
