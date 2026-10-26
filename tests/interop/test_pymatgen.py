# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

import numpy as np
import pytest

import mdapy
from mdapy.errors import InvalidParameter, MissingProperty
from mdapy.interop import from_pymatgen, to_pymatgen

pytest.importorskip("pymatgen.core")
from pymatgen.core import Lattice, Molecule, Structure  # noqa: E402
from pymatgen.io.vasp import Poscar  # noqa: E402


def alloy(seed=0, pbc=True):
    rng = np.random.default_rng(seed)
    matrix = np.array([[7.0, 0.0, 0.0], [1.0, 7.5, 0.0], [0.5, 0.8, 8.0]])
    origin = np.array([1.0, -2.0, 0.5])
    n = 12
    system = mdapy.System(
        rng.random((n, 3)) @ matrix + origin,
        mdapy.Box(matrix, origin=origin, pbc=pbc),
        elements=["Cu", "Ni", "Al"] * 4,
    )
    return system.with_columns(
        velocities=rng.normal(size=(n, 3)),
        movable=rng.random((n, 3)) > 0.5,
        charge=rng.normal(size=n),
    ).with_info(energy=-12.5, config_type="random")


def test_round_trip():
    system = alloy(pbc=[True, True, False])
    structure = to_pymatgen(system)
    assert structure.lattice.pbc == (True, True, False)
    back = from_pymatgen(structure)
    assert back.box.pbc.tolist() == [True, True, False]
    np.testing.assert_allclose(back.box.matrix, system.box.matrix)
    # pymatgen has no origin: the positions start from the corner of the box
    np.testing.assert_allclose(
        back["positions"], system["positions"] - system.box.origin, atol=1e-12
    )
    assert back.table["element"].to_list() == system.table["element"].to_list()
    for name in ("velocities", "movable", "charge"):
        np.testing.assert_allclose(
            back.table[name].to_numpy(), system.table[name].to_numpy(), err_msg=name
        )
    assert dict(back.info) == dict(system.info)


def test_velocities_in_the_units_of_pymatgen():
    system = alloy()
    structure = to_pymatgen(system)
    np.testing.assert_allclose(
        structure.site_properties["velocities"],
        system.table["velocities"].to_numpy() / 1000,  # Å/ps to Å/fs
    )
    np.testing.assert_array_equal(
        structure.site_properties["selective_dynamics"],
        system.table["movable"].to_numpy(),
    )


def test_pymatgen_writes_what_mdapy_reads(tmp_path):
    system = alloy(seed=1)
    Poscar(to_pymatgen(system)).write_file(tmp_path / "POSCAR")
    read = mdapy.read(tmp_path / "POSCAR")
    np.testing.assert_allclose(
        read["positions"], system["positions"] - system.box.origin, atol=1e-6
    )
    assert read.table["element"].to_list() == system.table["element"].to_list()
    np.testing.assert_array_equal(
        read.table["movable"].to_numpy(), system.table["movable"].to_numpy()
    )
    np.testing.assert_allclose(
        read.table["velocities"].to_numpy(),
        system.table["velocities"].to_numpy(),
        atol=1e-6,
    )


def test_mdapy_writes_what_pymatgen_reads(tmp_path):
    system = alloy(seed=2)
    mdapy.write(tmp_path / "POSCAR", system)
    # a POSCAR holds the atoms of each element together
    elements = system.table["element"].to_list()
    system = system.gather(
        np.argsort([elements.index(e) for e in elements], kind="stable")
    )
    structure = Structure.from_file(tmp_path / "POSCAR")
    back = from_pymatgen(structure)
    np.testing.assert_allclose(back.box.matrix, system.box.matrix, atol=1e-10)
    np.testing.assert_allclose(
        back["positions"], system["positions"] - system.box.origin, atol=1e-8
    )
    assert back.table["element"].to_list() == system.table["element"].to_list()
    np.testing.assert_array_equal(
        back.table["movable"].to_numpy(), system.table["movable"].to_numpy()
    )


def test_a_molecule():
    water = Molecule(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    system = from_pymatgen(water)
    assert not system.box.pbc.any()
    assert system.table["element"].to_list() == ["O", "H", "H"]
    np.testing.assert_allclose(system["positions"], water.cart_coords)


def test_errors():
    with pytest.raises(MissingProperty, match="element"):
        to_pymatgen(mdapy.System([[0.0, 0.0, 0.0]], mdapy.Box(3.0)))
    disordered = Structure(Lattice.cubic(3.6), [{"Cu": 0.5, "Ni": 0.5}], [[0, 0, 0]])
    with pytest.raises(InvalidParameter, match="more than one element"):
        from_pymatgen(disordered)


def test_only_numeric_columns_become_site_properties():
    plain = mdapy.System([[0.0, 0, 0], [1.8, 1.8, 0]], mdapy.Box(3.6), elements="Cu")
    structure = to_pymatgen(plain.with_columns(type_label=["a", "b"]))
    assert structure.site_properties == {}
    structure.add_site_property("label", ["x", "y"])
    structure.add_site_property("magmom", [0.5, -0.5])
    back = from_pymatgen(structure)
    assert back.table.columns == ["positions", "element", "magmom"]
