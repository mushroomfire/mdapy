# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""OVITO as a reference: a system through a pipeline of OVITO modifiers.
Imported by test modules after pytest.importorskip("ovito")."""

import numpy as np
from ovito.data import DataCollection, ParticleType
from ovito.pipeline import Pipeline, StaticSource

from mdapy.elements import SYMBOLS


def ovito_compute(system, *modifiers):
    """The output data collection of OVITO for the system after modifiers;
    with an element column, the elements are the particle types, numbered
    from 1 by atomic number."""
    data = DataCollection()
    cell = np.zeros((3, 4))
    cell[:, :3] = system.box.matrix.T  # OVITO keeps the cell vectors as columns
    cell[:, 3] = system.box.origin
    data.create_cell(cell, pbc=tuple(bool(p) for p in system.box.pbc))
    particles = data.create_particles(count=len(system))
    particles.create_property("Position", data=system.positions)
    if "element" in system.table.columns:
        elements = system.table["element"].to_list()
        names = sorted(set(elements), key=SYMBOLS.index)
        types = particles.create_property(
            "Particle Type", data=[names.index(e) + 1 for e in elements]
        )
        for k, name in enumerate(names):
            types.types.append(ParticleType(id=k + 1, name=name))
    pipeline = Pipeline(source=StaticSource(data=data))
    for modifier in modifiers:
        pipeline.modifiers.append(modifier)
    return pipeline.compute()
