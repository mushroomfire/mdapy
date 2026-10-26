# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import numpy as np
import polars as pl
import pytest

import mdapy
from mdapy.build import build_crystal
from mdapy.identify import CommonNeighborAnalysis
from mdapy.interop import from_ovito, to_ovito

pytest.importorskip("ovito")
from ovito.data import DataCollection  # noqa: E402
from ovito.io import export_file, import_file  # noqa: E402
from ovito.modifiers import (  # noqa: E402
    CommonNeighborAnalysisModifier,
    ComputePropertyModifier,
)
from ovito.pipeline import Pipeline, StaticSource  # noqa: E402


def water(seed=0, box=None, pbc=True):
    """Two water molecules near the faces of a triclinic box, with every
    column OVITO has a property for."""
    rng = np.random.default_rng(seed)
    if box is None:
        box = mdapy.Box([[6.0, 0, 0], [1.0, 6.0, 0], [0.5, 0.3, 7.0]], [1, 2, 3], pbc)
    positions = (
        np.array(
            [
                [0.2, 3.0, 3.0],
                [-0.6, 3.5, 3.0],
                [0.9, 3.6, 3.2],
                [3.0, 0.1, 6.9],
                [3.0, 0.9, 6.5],
                [3.0, -0.5, 6.4],
            ]
        )
        + box.origin
    )
    n = len(positions)
    system = mdapy.System(positions, box, elements=["O", "H", "H"] * 2)
    system = system.with_columns(
        mass=[15.999, 1.008, 1.008] * 2,
        velocities=rng.normal(size=(n, 3)),
        forces=rng.normal(size=(n, 3)),
        charge=[-0.8, 0.4, 0.4] * 2,
        id=np.arange(1, n + 1),
        molecule=[1, 1, 1, 2, 2, 2],
        image=rng.integers(-2, 3, (n, 3)).astype(np.int32),
        movable=rng.random((n, 3)) > 0.5,
        atomic_energies=rng.normal(size=n),
        extra=rng.normal(size=n),
        tensor=rng.normal(size=(n, 9)),
    )
    bonds = pl.DataFrame(
        {
            "atoms": [[0, 1], [0, 2], [3, 4], [3, 5]],
            "type": [1, 1, 1, 1],
            "type_label": ["OH"] * 4,
        }
    )
    topology = mdapy.Topology(bonds=bonds, angles=[[1, 0, 2], [4, 3, 5]])
    return system.with_topology(topology).with_info(
        energy=-3.25, step=10, note="water", virial=np.eye(3)
    )


def assert_same_columns(back, system, names):
    for name in names:
        expected, found = system.table[name].to_numpy(), back.table[name].to_numpy()
        assert found.dtype == expected.dtype, name
        np.testing.assert_array_equal(found, expected, err_msg=name)


def test_round_trip():
    system = water(pbc=[True, True, False])
    back = from_ovito(to_ovito(system))
    assert back.box == system.box
    np.testing.assert_array_equal(back["positions"], system["positions"])
    assert back.table["element"].to_list() == system.table["element"].to_list()
    assert back.table["type"].to_list() == [1, 2, 2, 1, 2, 2]  # by first appearance
    others = [c for c in system.table.columns if c not in ("positions", "element")]
    assert_same_columns(back, system, others)
    assert back.topology.bonds.equals(system.topology.bonds)
    assert back.topology.angles.equals(system.topology.angles)
    info = dict(back.info)
    np.testing.assert_array_equal(info.pop("virial"), np.eye(3))
    assert info == {"energy": -3.25, "step": 10, "note": "water"}


def test_ovito_properties_and_types():
    data = to_ovito(water())
    particles = data.particles
    assert {"Velocity", "Force", "Mass", "Charge", "Particle Identifier"} <= set(
        particles.keys()
    )
    assert {"Molecule Identifier", "Periodic Image", "Potential Energy"} <= set(
        particles.keys()
    )
    types = {t.id: (t.name, t.radius) for t in particles.particle_types.types}
    assert types[1][0] == "O" and types[2][0] == "H"
    assert types[1][1] > types[2][1] > 0  # the radii of OVITO's elements
    assert [(t.id, t.name) for t in particles.bonds.bond_types.types] == [(1, "OH")]
    np.testing.assert_array_equal(np.asarray(data.cell)[:, 3], [1, 2, 3])


def test_bonds_through_periodic_boundaries_keep_their_length():
    system = water()
    pipeline = Pipeline(source=StaticSource(data=to_ovito(system)))
    pipeline.modifiers.append(
        ComputePropertyModifier(
            operate_on="bonds", output_property="Length", expressions=["BondLength"]
        )
    )
    lengths = np.asarray(pipeline.compute().particles.bonds["Length"])
    atoms = system.topology.bonds["atoms"].to_numpy()
    vectors = system.box.minimum_image(
        system["positions"][atoms[:, 1]] - system["positions"][atoms[:, 0]]
    )
    np.testing.assert_allclose(lengths, np.linalg.norm(vectors, axis=1), atol=1e-12)
    assert lengths.max() < 1.2  # the molecules cross the faces of the box


def test_ovito_writes_what_mdapy_reads(tmp_path):
    lammps_box = mdapy.Box([[6.0, 0, 0], [1.0, 6.0, 0], [0.5, 0.3, 7.0]], [1, 2, 3])
    system = water(box=lammps_box)
    export_file(
        to_ovito(system), tmp_path / "water.data", "lammps/data", atom_style="full"
    )
    read = mdapy.read(tmp_path / "water.data")
    assert read.box == system.box
    np.testing.assert_allclose(read["positions"], system["positions"], atol=1e-10)
    assert read.table["type"].to_list() == [1, 2, 2, 1, 2, 2]
    assert read.table["element"].to_list() == system.table["element"].to_list()
    assert_same_columns(read, system, ["molecule", "id"])
    np.testing.assert_allclose(
        read.table["charge"].to_numpy(), system.table["charge"].to_numpy()
    )
    np.testing.assert_array_equal(
        read.topology.bonds["atoms"].to_numpy(),
        system.topology.bonds["atoms"].to_numpy(),
    )


def test_mdapy_writes_what_ovito_reads(tmp_path):
    lammps_box = mdapy.Box([[6.0, 0, 0], [1.0, 6.0, 0], [0.5, 0.3, 7.0]], [1, 2, 3])
    system = water(box=lammps_box)
    mdapy.write(tmp_path / "water.data", system, atom_style="full")
    back = from_ovito(import_file(tmp_path / "water.data", atom_style="full").compute())
    assert back.box == system.box
    np.testing.assert_allclose(back["positions"], system["positions"], atol=1e-10)
    assert back.table["element"].to_list() == system.table["element"].to_list()
    assert_same_columns(back, system, ["id", "molecule"])
    np.testing.assert_allclose(
        back.table["charge"].to_numpy(), system.table["charge"].to_numpy()
    )
    np.testing.assert_array_equal(
        back.topology.bonds["atoms"].to_numpy(),
        system.topology.bonds["atoms"].to_numpy(),
    )


def test_results_of_an_ovito_pipeline_come_back():
    crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(5, 5, 5))
    crystal = crystal.gather(np.arange(1, len(crystal)))  # a vacancy
    pipeline = Pipeline(source=StaticSource(data=to_ovito(crystal)))
    pipeline.modifiers.append(CommonNeighborAnalysisModifier())
    analyzed = from_ovito(pipeline.compute())
    ours = CommonNeighborAnalysis().compute(crystal)
    np.testing.assert_array_equal(
        analyzed.table["structure"].to_numpy(), ours.structure
    )
    assert analyzed.info["CommonNeighborAnalysis.counts.FCC"] == len(crystal) - 12


def test_types_without_elements():
    plain = mdapy.System([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]], mdapy.Box(5.0))
    back = from_ovito(to_ovito(plain))  # no types at all
    assert back.table.columns == ["positions"]
    labelled = plain.with_columns(type=[1, 2, 1], type_label=["CT", "HC", "CT"])
    back = from_ovito(to_ovito(labelled))
    assert back.table["type"].to_list() == [1, 2, 1]
    assert back.table["type_label"].to_list() == ["CT", "HC", "CT"]
    numbered = from_ovito(to_ovito(plain.with_columns(type=[2, 1, 2])))
    assert numbered.table["type"].to_list() == [2, 1, 2]
    assert "element" not in numbered.table.columns


def test_terms_of_every_kind():
    chain = mdapy.System(np.arange(15.0).reshape(5, 3), mdapy.Box(20.0), elements="C")
    topology = mdapy.Topology(
        bonds=[[0, 1], [1, 2], [2, 3], [3, 4]],
        angles=[[0, 1, 2], [1, 2, 3]],
        dihedrals=pl.DataFrame(
            {"atoms": [[0, 1, 2, 3]], "type": [2], "type_label": ["CCCC"]}
        ),
        impropers=[[1, 0, 2, 3]],
    )
    back = from_ovito(to_ovito(chain.with_topology(topology)))
    for kind in ("bonds", "angles", "dihedrals", "impropers"):
        assert getattr(back.topology, kind).equals(getattr(topology, kind)), kind


def test_data_without_a_cell():
    data = DataCollection()
    particles = data.create_particles(count=2)
    particles.create_property("Position", data=[[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]])
    particles.create_property("Selection", data=[1, 0])
    particles.create_property("Name", data=["a", "b"])  # strings are left out
    system = from_ovito(data)
    assert not system.box.pbc.any()
    assert system.table.columns == ["positions", "selection"]
    assert system.topology is None
