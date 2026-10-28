# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""CIF files: crystals expanded by their symmetry operations, against
pymatgen, ASE and OVITO reading the same files."""

import numpy as np
import pytest

import mdapy
from mdapy.errors import FileFormatError


def atoms(elements, positions, matrix):
    """The atoms as a sorted list of elements and fractional coordinates."""
    fractions = np.asarray(positions) @ np.linalg.inv(matrix)
    fractions -= np.floor(fractions + 1e-9)
    fractions[np.isclose(fractions, 1.0, atol=1e-6)] = 0.0
    return sorted((str(e), *np.round(f, 5)) for e, f in zip(elements, fractions))


def crystals():
    """Crystals of several lattices with atoms on special positions."""
    from pymatgen.core import Lattice, Structure

    return {
        "rock salt": Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
        ),
        "rutile": Structure.from_spacegroup(
            "P4_2/mnm",
            Lattice.tetragonal(4.594, 2.959),
            ["Ti", "O"],
            [[0, 0, 0], [0.3053, 0.3053, 0]],
        ),
        "quartz": Structure.from_spacegroup(
            154,
            Lattice.hexagonal(4.913, 5.405),
            ["Si", "O"],
            [[0.4697, 0, 0], [0.4135, 0.2669, 0.1191]],
        ),
        "corundum": Structure.from_spacegroup(
            "R-3c",
            Lattice.hexagonal(4.759, 12.991),
            ["Al", "O"],
            [[0, 0, 0.35216], [0.30624, 0, 0.25]],
        ),
        "baddeleyite": Structure.from_spacegroup(
            "P2_1/c",
            Lattice.monoclinic(5.15, 5.21, 5.31, 99.2),
            ["Zr", "O", "O"],
            [
                [0.2754, 0.0395, 0.2083],
                [0.0700, 0.3317, 0.3447],
                [0.4416, 0.7569, 0.4792],
            ],
        ),
        "triclinic": Structure.from_spacegroup(
            "P-1",
            Lattice.from_parameters(4.1, 5.2, 6.3, 81.0, 95.0, 103.0),
            ["Ca", "S"],
            [[0.1, 0.2, 0.3], [0.5, 0.5, 0.5]],
        ),
    }


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    pytest.importorskip("pymatgen.core")
    from pymatgen.io.cif import CifWriter

    folder = tmp_path_factory.mktemp("cif")
    paths = {}
    for name, structure in crystals().items():
        path = folder / f"{name.replace(' ', '_')}.cif"
        CifWriter(structure, symprec=0.01).write_file(path)
        paths[name] = (path, structure)
    return paths


NAMES = ["rock salt", "rutile", "quartz", "corundum", "baddeleyite", "triclinic"]


# newer pymatgen tells when it rounds coordinates near special positions
@pytest.mark.filterwarnings("ignore:Issues encountered while parsing CIF:UserWarning")
@pytest.mark.parametrize("name", NAMES)
def test_as_pymatgen_ase_and_ovito(written, name):
    from pymatgen.io.cif import CifParser

    path, structure = written[name]
    ours = mdapy.read(path)
    assert mdapy.detect_format(path) == "cif"
    assert len(ours) == len(structure)
    # pymatgen reads the file in its own orientation: compare the lengths
    # and angles of the cell, and the fractional coordinates
    theirs = CifParser(path).parse_structures(primitive=False)[0]
    lengths = np.linalg.norm(ours.box.matrix, axis=1)
    np.testing.assert_allclose(lengths, theirs.lattice.abc, rtol=1e-12)
    np.testing.assert_allclose(
        abs(np.linalg.det(ours.box.matrix)), theirs.volume, rtol=1e-12
    )
    fractions = ours["positions"] @ np.linalg.inv(ours.box.matrix)
    assert atoms(ours["element"], fractions, np.eye(3)) == atoms(
        [s.specie.symbol for s in theirs], theirs.frac_coords, np.eye(3)
    )
    assert ours.box.pbc.all() and ours.info["name"]

    ase_io = pytest.importorskip("ase.io")
    theirs = ase_io.read(path)
    np.testing.assert_allclose(ours.box.matrix, theirs.cell.array, atol=1e-10)
    assert atoms(ours["element"], ours["positions"], ours.box.matrix) == atoms(
        theirs.get_chemical_symbols(), theirs.positions, theirs.cell.array
    )
    np.testing.assert_allclose(ours["positions"], theirs.positions, atol=1e-8)

    pytest.importorskip("ovito")
    from ovito.io import import_file

    data = import_file(str(path)).compute()
    types = data.particles.particle_types
    cell = np.asarray(data.cell)
    np.testing.assert_allclose(ours.box.matrix, cell[:, :3].T, atol=1e-10)
    assert atoms(ours["element"], ours["positions"], ours.box.matrix) == atoms(
        [types.type_by_id(t).name for t in types],
        data.particles.positions,
        cell[:, :3].T,
    )


def test_a_file_written_by_ase(tmp_path):
    """ASE writes every atom of the cell, its symmetry the identity."""
    ase_build = pytest.importorskip("ase.build")
    import ase.io

    crystal = ase_build.bulk("Cu", "fcc", a=3.615, cubic=True) * (2, 1, 1)
    crystal.rattle(0.05, seed=1)
    ase.io.write(tmp_path / "cu.cif", crystal)
    ours = mdapy.read(tmp_path / "cu.cif")
    crystal.wrap()  # as every reader of CIF files
    np.testing.assert_allclose(ours["positions"], crystal.positions, atol=1e-6)
    np.testing.assert_allclose(ours.box.matrix, crystal.cell.array, atol=1e-10)


SYNTAX = """# comments, quotes, text fields and uncertainties
data_first
_publ_section_title
;
A title over
two lines, with data_ and loop_ in it
;
_journal_name_full 'Acta Cryst. A's test'
_cell.length_a 4.0496(3)
_cell.length_b 4.0496(3)
_cell.length_c 4.0496(3)
_cell.angle_alpha 90
_cell.angle_beta 90.0
_cell.angle_gamma 90.00(0)
_space_group.IT_number 225
loop_
_space_group_symop.id
_space_group_symop.operation_xyz
1 'x,y,z' 2 '1/2+x,1/2+y,z' 3 "x+0.5, y, z+.5" 4 'x, +y+1/2, z+1/2'
loop_
_atom_site.label _atom_site.fract_x _atom_site.fract_y _atom_site.fract_z
_atom_site.occupancy
Al1 0.0 0.0 0.0 1.0   # aluminium
global_
data_second
_cell_length_a 3.0
_cell_length_b 3.0
_cell_length_c 3.0
loop_
_atom_site_type_symbol
_atom_site_Cartn_x
_atom_site_Cartn_y
_atom_site_Cartn_z
Fe2+ 0 0 0
FE 1.5 1.5 1.5
"""


def test_syntax_and_several_blocks(tmp_path):
    path = tmp_path / "two.cif"
    path.write_text(SYNTAX)
    first, second = mdapy.read_trajectory(path)
    assert first.info["name"] == "first" and len(first) == 4
    assert atoms(first["element"], first["positions"], first.box.matrix) == [
        ("Al", 0.0, 0.0, 0.0),
        ("Al", 0.0, 0.5, 0.5),
        ("Al", 0.5, 0.0, 0.5),
        ("Al", 0.5, 0.5, 0.0),
    ]
    np.testing.assert_allclose(first.box.matrix, 4.0496 * np.eye(3))
    # Cartesian coordinates, no symmetry operations: as they are
    assert second["element"].tolist() == ["Fe", "Fe"]
    np.testing.assert_allclose(second["positions"], [[0, 0, 0], [1.5, 1.5, 1.5]])
    np.testing.assert_allclose(second.box.matrix, 3.0 * np.eye(3))


@pytest.mark.parametrize(
    "text, message",
    [
        ("_cell_length_a 'open\n", "quote is not closed"),
        (";\nnever closed\n", "text field is not closed"),
        ("loop_\n_a\n_b\n1 2 3\n", "loop of 2 tags has 3 values"),
        ("_cell_length_a\n", "has no value"),
        ("stray\n", "unexpected 'stray'"),
        ("_cell_length_b 1\n_cell_length_c 1\n", "has no _cell_length_a"),
        ("_cell_length_a x\n_cell_length_b 1\n_cell_length_c 1\n", "is not a number"),
        ("_cell_length_a 0\n_cell_length_b 1\n_cell_length_c 1\n", "no cell of volume"),
        ("", "no atom sites"),
        (
            "loop_\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n0 0 0\n",
            "names no elements",
        ),
        (
            "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n"
            "_atom_site_fract_z\nQq1 0 0 0\n",
            "unknown element 'Qq1'",
        ),
        (
            "_atom_site_label Cu1\nloop_\n_atom_site_fract_x\n_atom_site_fract_y\n"
            "_atom_site_fract_z\n0 0 0 0.5 0.5 0.5\n",
            "differ in number",
        ),
        (
            "_symmetry_space_group_name_H-M 'F m -3 m'\n_atom_site_label Cu1\n"
            "_atom_site_fract_x 0\n_atom_site_fract_y 0\n_atom_site_fract_z 0\n",
            "but not its symmetry operations",
        ),
        (
            "_symmetry_equiv_pos_as_xyz 'x,y'\n_atom_site_label Cu1\n"
            "_atom_site_fract_x 0\n_atom_site_fract_y 0\n_atom_site_fract_z 0\n",
            "three expressions",
        ),
        (
            "_symmetry_equiv_pos_as_xyz 'x,y,z w'\n_atom_site_label Cu1\n"
            "_atom_site_fract_x 0\n_atom_site_fract_y 0\n_atom_site_fract_z 0\n",
            "cannot read",
        ),
    ],
)
def test_errors(tmp_path, text, message):
    cell = (
        ""
        if "_cell_length" in text
        else ("_cell_length_a 3\n_cell_length_b 3\n_cell_length_c 3\n")
    )
    path = tmp_path / "bad.cif"
    path.write_text("data_bad\n" + cell + text)
    with pytest.raises(FileFormatError, match=message):
        mdapy.read(path)


def test_no_data_block(tmp_path):
    path = tmp_path / "none.cif"
    path.write_text("# nothing\n")
    with pytest.raises(FileFormatError, match="no data block"):
        mdapy.read(path)
    # P1 named but without operations: the sites as they are
    path.write_text(
        "data_p1\n_cell_length_a 2\n_cell_length_b 2\n_cell_length_c 2\n"
        "_symmetry_space_group_name_H-M 'P 1'\n_atom_site_type_symbol O\n"
        "_atom_site_fract_x 0.25\n_atom_site_fract_y 0\n_atom_site_fract_z 0\n"
    )
    np.testing.assert_allclose(mdapy.read(path)["positions"], [[0.5, 0, 0]])
