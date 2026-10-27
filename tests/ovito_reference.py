# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""OVITO as a reference: a system through a pipeline of OVITO modifiers.
Imported by test modules after pytest.importorskip("ovito")."""

import numpy as np
from ovito.data import DataCollection, ParticleType
from ovito.pipeline import Pipeline, StaticSource

from mdapy.elements import SYMBOLS


def ovito_source(system):
    """The system as the source of an OVITO pipeline; with an element column,
    the elements are the particle types, numbered from 1 by atomic number,
    a mass column becomes the masses and an id column the identifiers."""
    data = DataCollection()
    cell = np.zeros((3, 4))
    cell[:, :3] = system.box.matrix.T  # OVITO keeps the cell vectors as columns
    cell[:, 3] = system.box.origin
    data.create_cell(cell, pbc=tuple(bool(p) for p in system.box.pbc))
    particles = data.create_particles(count=len(system))
    particles.create_property("Position", data=system["positions"])
    if "mass" in system.table.columns:
        particles.create_property("Mass", data=system.table["mass"].to_numpy())
    if "id" in system.table.columns:
        particles.create_property(
            "Particle Identifier", data=system.table["id"].to_numpy()
        )
    if "element" in system.table.columns:
        elements = system.table["element"].to_list()
        names = sorted(set(elements), key=SYMBOLS.index)
        types = particles.create_property(
            "Particle Type", data=[names.index(e) + 1 for e in elements]
        )
        for k, name in enumerate(names):
            types.types.append(ParticleType(id=k + 1, name=name))
    return StaticSource(data=data)


def ovito_compute(system, *modifiers):
    """The output data collection of OVITO for the system after modifiers."""
    pipeline = Pipeline(source=ovito_source(system))
    for modifier in modifiers:
        pipeline.modifiers.append(modifier)
    return pipeline.compute()
