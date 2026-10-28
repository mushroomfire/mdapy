# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""OUTCAR files: the ionic steps of files laid out as VASP writes them,
against the frames written and against ASE reading the same files."""

import re

import numpy as np
import pytest
from outcar_files import frames, outcar_text

import mdapy
from mdapy.errors import FileFormatError


def test_steps_as_frames(tmp_path):
    steps = frames()
    path = tmp_path / "OUTCAR"
    path.write_text(outcar_text(steps))
    assert mdapy.detect_format(path) == "outcar"
    assert mdapy.detect_format(tmp_path / "OUTCAR.relax") == "outcar"
    assert mdapy.detect_format(tmp_path / "relax.outcar") == "outcar"
    trajectory = mdapy.read_trajectory(path)
    assert len(trajectory) == 3
    for frame, (lattice, positions, forces, energy, virial) in zip(trajectory, steps):
        assert frame["element"].tolist() == ["Ag"] * 4  # no Pd: 0 of them
        np.testing.assert_allclose(frame.box.matrix, lattice, atol=1e-8)
        np.testing.assert_allclose(frame["positions"], positions, atol=1e-5)
        np.testing.assert_allclose(frame["forces"], forces, atol=1e-6)
        assert frame.info["energy"] == pytest.approx(energy, abs=1e-8)
        np.testing.assert_allclose(frame.info["virial"], virial, atol=1e-5)
        assert frame.info["converged"] is True
        assert frame.box.pbc.all()
    # one step alone, from the end
    assert mdapy.read(path, frame=-1).info["energy"] == pytest.approx(steps[-1][3])


@pytest.mark.filterwarnings("ignore:Setting the shape:DeprecationWarning")
def test_as_ase(tmp_path):
    ase_io = pytest.importorskip("ase.io")
    path = tmp_path / "OUTCAR"
    path.write_text(outcar_text(frames(seed=1), symbols=("Cu", "Ni"), counts=(2, 2)))
    for ours, theirs in zip(
        mdapy.read_trajectory(path), ase_io.read(path, ":", format="vasp-out")
    ):
        assert ours["element"].tolist() == theirs.get_chemical_symbols()
        np.testing.assert_array_equal(ours["positions"], theirs.positions)
        np.testing.assert_array_equal(ours.box.matrix, theirs.cell.array)
        np.testing.assert_array_equal(ours["forces"], theirs.get_forces())
        # TOTEN, of which the forces are the slope
        assert ours.info["energy"] == theirs.get_potential_energy(force_consistent=True)
        # ASE takes the stress in kB, rounded to 5 decimals
        np.testing.assert_allclose(
            -ours.info["virial"] / ours.box.volume,
            theirs.get_stress(voigt=False),
            atol=1e-7,
        )


def test_unconverged_steps_no_stress_and_learned_steps(tmp_path):
    steps = frames()
    text = outcar_text(steps, converged=[True, False, True], stress=False)
    # the steps of the machine-learned force field of VASP 6 are not steps
    text = text.replace(
        "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n",
        "  ML FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n"
        "  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)\n",
    )
    path = tmp_path / "OUTCAR"
    path.write_text(text)
    every = list(mdapy.read_trajectory(path))
    assert [f.info["converged"] for f in every] == [True, False, True]
    assert all("virial" not in f.info for f in every)
    assert [f.info["energy"] for f in every] == pytest.approx([s[3] for s in steps])


def test_training_set(tmp_path):
    """The converged steps of several calculations, without the steps of
    huge forces, written as extended XYZ with their energies and virials."""
    first, second = frames(count=2, seed=2), frames(count=1, seed=3)
    first[1] = (*first[1][:2], first[1][2] * 100, *first[1][3:])  # large forces
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "OUTCAR").write_text(outcar_text(first))
    (tmp_path / "b" / "OUTCAR").write_text(outcar_text(second))
    kept = [
        frame.with_info(converged=None)
        for path in sorted(tmp_path.glob("*/OUTCAR"))
        for frame in mdapy.read_trajectory(path)
        if frame.info["converged"] and np.abs(frame["forces"]).max() <= 30.0
    ]
    mdapy.write(tmp_path / "train.xyz", kept)
    back = list(mdapy.read_trajectory(tmp_path / "train.xyz"))
    assert len(back) == 2
    np.testing.assert_allclose(back[1].info["virial"], second[0][4], atol=1e-5)
    np.testing.assert_allclose(back[0]["forces"], first[0][2], atol=1e-6)
    assert "converged" not in back[0].info


def test_numbers_without_spaces(tmp_path):
    lines = outcar_text(frames(count=1)).splitlines()
    k = next(k for k, line in enumerate(lines) if "TOTAL-FORCE" in line) + 2
    lines[k] = "   1.00000-2.50000   3.00000  -0.123456-10.500000   7.000000"
    path = tmp_path / "OUTCAR"
    path.write_text("\n".join(lines) + "\n")
    frame = mdapy.read(path)
    np.testing.assert_array_equal(frame["positions"][0], [1.0, -2.5, 3.0])
    np.testing.assert_array_equal(frame["forces"][0], [-0.123456, -10.5, 7.0])


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda t: t.replace("ions per type =     0   4", "words"),
            "no line of ions per type",
        ),
        (
            lambda t: t.replace("ions per type =     0   4", "ions per type = 0 four"),
            "cannot read the ions per type",
        ),
        (
            lambda t: t.replace(" POTCAR:    PAW_PBE Ag 04Jan2005\n", "", 1),
            "3 POTCAR lines for 2 numbers",
        ),
        (lambda t: t.replace("PAW_PBE Ag", "PAW_PBE Qq"), "unknown element"),
        (
            # VASP writes stars for a number too long for its field
            lambda t: re.sub(r"(TOTEN  =)\s+\S+", r"\1 ****************", t),
            "expected 1 numbers",
        ),
        (
            lambda t: t.replace("  free  energy   TOTEN", "  free energy TOTEN"),
            "no free energy TOTEN",
        ),
        (
            lambda t: t.replace(" POSITION   ", " POSITIONS  "),
            "no positions and forces",
        ),
        (lambda t: t.replace("direct lattice vectors", "lattice"), "no direct lattice"),
        (
            lambda t: t.replace(
                "ions per type =     0   4", "ions per type =     0   9"
            ),
            "expected 6 numbers",
        ),
    ],
)
def test_errors(tmp_path, change, message):
    path = tmp_path / "OUTCAR"
    path.write_text(change(outcar_text(frames(count=1))))
    with pytest.raises(FileFormatError, match=message):
        mdapy.read(path)


def test_a_file_without_steps(tmp_path):
    """A calculation that stopped before its first step has no frames."""
    text = outcar_text(frames(count=1))
    path = tmp_path / "OUTCAR"
    path.write_text(text[: text.index(" POSITION")])
    assert len(mdapy.read_trajectory(path)) == 0
