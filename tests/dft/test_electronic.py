# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The electronic band structure and density of states of
mdapy.properties.electronic: in eV relative to the Fermi level, the bands of
pw.x 7.6 run live on the same path (silicon, and PAW iron with spin, whose
occupations of the projectors the calculator keeps); the path, its labels and
distances as those of the phonons; the bands do not depend on how many
k-points are solved together, nor the lower ones on how many are asked; k and
-k. The density of states from one k-point of each star of the symmetry
equals that of every k-point of the mesh; below the gap of silicon it holds
its 8 electrons; with Gaussians as wide as the smearing, on the mesh of the
field, it holds the occupations of iron, its electrons and its
magnetization. The errors."""

import os

import numpy as np
import pytest

import mdapy
from mdapy.calculators import DFT, LennardJones
from mdapy.calculators._dft import calculator as module
from mdapy.calculators._dft.calculator import BOHR, HARTREE
from mdapy.calculators._dft.upf import read_upf
from mdapy.errors import CalculationFailed, InvalidParameter
from mdapy.properties import _electronic, _paths
from mdapy.properties._phonons import _mesh
from mdapy.properties.electronic import (
    ElectronicBandStructure,
    ElectronicDOS,
    ElectronicStructure,
    electronic_structure,
)

from dft import espresso, gpu

_dft = gpu.module()
if _dft.device_count() == 0:
    pytest.skip("no GPU", allow_module_level=True)

FRACTIONS = np.array([[0.01, -0.02, 0.0], [0.27, 0.25, 0.23]])
H = 5.431 / 2
CELL = np.array([[0, H, H], [H, 0, H], [H, H, 0]])


def silicon():
    return mdapy.System(FRACTIONS @ CELL, mdapy.Box(CELL), elements=["Si", "Si"])


@pytest.fixture(scope="module")
def folder():
    espresso.pseudo("Si_std.upf")
    return os.environ["MDAPY_PSEUDO_DIR"]


@pytest.fixture(scope="module")
def dft(folder):
    return DFT(encut=12 * HARTREE, kpoints=(2, 2, 2), smearing="fixed", functional="pbe_pw",
               tolerance=1e-10, pseudopotentials=folder, symmetry=False)  # fmt: skip


@pytest.fixture(scope="module")
def structure(dft):
    return electronic_structure(silicon(), dft)


@pytest.fixture(scope="module")
def bands(structure):
    return structure.band_structure("fcc", points=3, bands=10)


@pytest.fixture(scope="module")
def diamond(folder):
    pytest.importorskip("spglib")
    system = mdapy.System(
        np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]) @ CELL, mdapy.Box(CELL),
        elements=["Si", "Si"],
    )  # fmt: skip
    dft = DFT(encut=12 * HARTREE, kpoints=(4, 4, 4), smearing="fixed", functional="pbe_pw",
              tolerance=1e-10, pseudopotentials=folder, symmetry=True)  # fmt: skip
    return electronic_structure(system, dft), dft


def test_silicon_against_pw(dft, structure, bands, folder):
    assert isinstance(structure, ElectronicStructure)
    assert isinstance(bands, ElectronicBandStructure)
    assert "2 atoms, 1 spin" in repr(structure)
    kpoints, weights = _dft.monkhorst_pack([2, 2, 2], [0.0, 0.0, 0.0], True)
    qe = espresso.run(
        CELL / BOHR, FRACTIONS, ["Si", "Si"], {"Si": espresso.pseudo("Si_std.upf")},
        12.0, kpoints, weights, conv_thr=1e-14, band_kpoints=bands.k, band_count=10,
    )  # fmt: skip
    assert dft.last["grid"] == qe["grid"]
    # the highest occupied level of the mesh of the field
    assert structure.fermi == pytest.approx(qe["fermi"] * HARTREE, abs=1e-5)
    assert bands.fermi == structure.fermi
    assert bands.energies.shape == (1, len(bands.k), 10)
    worst = np.abs(
        bands.energies[0] + bands.fermi - qe["band_eigenvalues"] * HARTREE
    ).max()
    print(f"\nSi through the calculator: max |e - e_QE| = {worst:.2e} eV")
    assert worst < 1e-5
    assert (np.diff(bands.energies, axis=2) >= 0).all()
    # four occupied bands, their top at Γ, a k-point of the mesh of the field,
    # and a gap above them
    assert bands.energies[0, :, :4].max() == pytest.approx(0.0, abs=1e-5)
    assert bands.energies[0, :, 4].min() > 0.05


def test_the_path_is_that_of_the_phonons(structure, bands):
    system = silicon()
    sections = _paths.standard_path(CELL, CELL, "fcc")
    k, distances, labels, _ = _paths.sample(
        sections, 3, 2 * np.pi * np.linalg.inv(CELL).T
    )
    np.testing.assert_array_equal(bands.k, k)
    np.testing.assert_array_equal(bands.distances, distances)
    assert bands.labels == labels
    assert [name for _, name in bands.labels] == ["Γ", "X", "U|K", "Γ", "L", "W", "X"]
    for array in (bands.k, bands.distances, bands.energies):
        assert not array.flags.writeable
    given = structure.band_structure(
        [[("Γ", [0, 0, 0]), ("X", [0.5, 0, 0.5])]], points=3, bands=10
    )
    np.testing.assert_allclose(given.energies, bands.energies[:, :3], atol=1e-7)
    assert len(system) == 2 and structure.system is not None


def test_auto_path(structure):
    pytest.importorskip("seekpath")
    auto = structure.band_structure("auto", points=2, bands=6)
    assert auto.labels[0][1] == "Γ" and auto.energies.shape[::2] == (1, 6)


def test_invariances(dft, structure, bands):
    k = np.array([0.13, 0.27, -0.41])
    e = structure.energies([k, -k, k + [1, 0, -1]], bands=10)
    assert e.shape == (1, 3, 10)
    np.testing.assert_allclose(e[0, 1], e[0, 0], atol=1e-7)
    np.testing.assert_allclose(e[0, 2], e[0, 0], atol=1e-7)
    one = structure.energies(k)
    assert one.shape == (1, structure.bands)
    np.testing.assert_allclose(one[0], e[0, 0, : structure.bands], atol=1e-7)
    # one k-point in each pass, as many as fit, and other numbers of bands
    ground = dft._fixed_density(silicon())
    apart = dft._bands(ground, bands.k, 10, memory=1.0) - ground["fermi"]
    np.testing.assert_allclose(apart, bands.energies, atol=1e-7)
    fewer = structure.band_structure("fcc", points=3, bands=6)
    np.testing.assert_allclose(fewer.energies, bands.energies[:, :, :6], atol=1e-7)


def test_dos_of_the_stars_is_that_of_the_mesh(diamond):
    structure, dft = diamond
    assert dft.last["operations"] == 48
    dos = structure.dos(4, bands=8)
    assert isinstance(dos, ElectronicDOS)
    assert dos.dos.shape == (1, 1001) and dos.fermi == structure.fermi
    np.testing.assert_array_equal(dos.mesh, [4, 4, 4])
    for array in (dos.energies, dos.dos, dos.mesh):
        assert not array.flags.writeable
    # every k-point of the mesh solved, without the symmetry
    full = structure.energies(_mesh(4), bands=8)
    reciprocal = 2 * np.pi * np.linalg.inv(CELL).T
    counts = np.array([4, 4, 4])
    reference = 2 * _electronic._tetrahedron_sum(full, counts, reciprocal, dos.energies)
    np.testing.assert_allclose(dos.dos, reference, rtol=0, atol=1e-4 * reference.max())
    # from 0.5 eV below the lowest band to the lowest energy of the highest
    assert dos.energies[0] == pytest.approx(full.min() - 0.5, abs=1e-5)
    assert dos.energies[-1] == pytest.approx(full[..., -1].min(), abs=1e-5)
    gauss = structure.dos([4, 4, 4], energies=dos.energies, sigma=0.2, bands=8)
    brute = 2 * _electronic._gaussian_sum(full, np.full(64, 1 / 64), dos.energies, 0.2)
    np.testing.assert_allclose(gauss.dos, brute, rtol=0, atol=1e-5 * brute.max())


def test_dos_holds_the_electrons_of_silicon(diamond):
    structure, _ = diamond
    # the default mesh, 0.15 1/Å apart at most: 2.004 / 0.15 = 13.4
    dos = structure.dos(energies=np.linspace(-14.0, 0.2, 40001), bands=8)
    np.testing.assert_array_equal(dos.mesh, [14, 14, 14])
    # the top of the valence bands at Γ, on both meshes, and a gap above it
    assert dos.dos[0, -1] == 0.0
    assert np.trapezoid(dos.dos[0], dos.energies) == pytest.approx(8.0, abs=2e-3)


def test_iron_with_spin_and_paw(folder):
    # the calculator hands the occupations of the projectors of its field to
    # the bands; QE's spin-polarized PBE takes a rounded gamma (test_spin.py)
    name = "pslibrary/Fe.pbe-spn-kjpaw_psl.1.0.0.UPF"
    file = espresso.pseudo(name)
    h = 2.87 / 2
    cell = np.array([[-h, h, h], [h, -h, h], [h, h, -h]])
    system = mdapy.System(np.zeros((1, 3)), mdapy.Box(cell), elements=["Fe"])
    dft = DFT(encut=20 * HARTREE, kpoints=(3, 3, 3), smearing="gaussian", sigma=0.01 * HARTREE,
              functional="pbe_pw", tolerance=1e-10, pseudopotentials={"Fe": file},
              magmoms={"Fe": 2.5}, symmetry=False)  # fmt: skip
    structure = electronic_structure(system, dft)
    bands = structure.band_structure("bcc", points=3, bands=12)
    assert structure.spins == 2 and bands.energies.shape == (2, len(bands.k), 12)
    kpoints, weights = _dft.monkhorst_pack([3, 3, 3], [0.0, 0.0, 0.0], True)
    qe = espresso.run(
        cell / BOHR, [[0.0, 0.0, 0.0]], ["Fe"], {"Fe": file}, 20.0, kpoints, weights,
        "gaussian", 0.01, conv_thr=1e-14, magmoms=[2.5], beta=0.3, maxstep=300,
        extra=f", ecutrho = {16 * 20.0:.10f}", band_kpoints=bands.k, band_count=12,
    )  # fmt: skip
    assert dft.last["grid"] == qe["grid"]
    assert structure.fermi == pytest.approx(qe["fermi"] * HARTREE, abs=1e-4)
    worst = np.abs(
        bands.energies + bands.fermi - qe["band_eigenvalues"] * HARTREE
    ).max()
    print(f"\nPAW Fe through the calculator: max |e - e_QE| = {worst:.2e} eV")
    assert worst < 1e-4
    # the exchange splitting of the d bands
    assert (bands.energies[1] - bands.energies[0]).max() > 1.0
    # Gaussians of the smearing, the occupations ½ erfc((e - E_F) / sigma), on
    # the mesh of the field: its electrons below the Fermi level, and its
    # magnetization
    dos = structure.dos(
        3, energies=np.linspace(-120.0, 0.0, 600001), sigma=0.01 * HARTREE / np.sqrt(2)
    )
    assert dos.dos.shape == (2, 600001)
    up, down = np.trapezoid(dos.dos, dos.energies, axis=1)
    assert up + down == pytest.approx(read_upf(file).valence, abs=1e-4)
    assert up - down == pytest.approx(dft.last["magnetization"], abs=1e-4)


def test_errors(dft, structure, folder, monkeypatch):
    with pytest.raises(InvalidParameter, match="needs the calculator DFT"):
        electronic_structure(silicon(), LennardJones({"Si": (0.4, 2.3)}, 5.0))
    plus_u = DFT(
        encut=12 * HARTREE, pseudopotentials=folder, hubbard={"Si": ("3P", 1.0)}
    )
    with pytest.raises(InvalidParameter, match="not yet for DFT\\+U"):
        electronic_structure(silicon(), plus_u)
    with pytest.raises(InvalidParameter, match="must be one of"):
        structure.band_structure("diamond")
    with pytest.raises(InvalidParameter, match="points must be at least 2"):
        structure.band_structure("fcc", points=1)
    with pytest.raises(InvalidParameter, match="bands must be at least 1"):
        structure.band_structure("fcc", bands=0)
    with pytest.raises(InvalidParameter, match="wave vectors must be"):
        structure.energies([0.0, 0.0])
    with pytest.raises(InvalidParameter, match="sigma must be positive"):
        structure.dos(2, sigma=0.0)
    with pytest.raises(InvalidParameter, match="mesh must be one or three positive"):
        structure.dos(0)
    with pytest.raises(InvalidParameter, match="kpoints must be"):
        dft._bands(dft._fixed_density(silicon()), np.zeros(3))
    # a residual that no band reaches
    monkeypatch.setattr(module, "_BAND_TOLERANCE", 1e-18)
    with pytest.raises(CalculationFailed, match="bands did not converge"):
        structure.energies([0.1, 0.2, 0.3])
