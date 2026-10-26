# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Writing EAM potentials, the potentials of Zhou et al., and average atoms."""

import itertools
import math

import numpy as np
import pytest

import mdapy
from mdapy.calculators import EAM
from mdapy.calculators._eam import _read_setfl
from mdapy.calculators._zhou import ELEMENTS, _parameters
from mdapy.errors import InvalidParameter

from .eam_files import alloy, write_setfl

ALL = ("energy", "atomic_energies", "forces", "virial", "atomic_virials")


def crystal(element, lattice, a, repeat=3):
    basis = {"fcc": [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
             "bcc": [[0, 0, 0], [0.5, 0.5, 0.5]]}[lattice]  # fmt: skip
    cells = np.array(list(itertools.product(range(repeat), repeat=3)))
    fractional = (cells[:, None, :] + np.array(basis)[None]).reshape(-1, 3) / repeat
    box = mdapy.Box(a * repeat)
    return mdapy.System(fractional @ box.matrix, box, elements=element)


def same_results(a, b):
    for name in ALL:
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


# ------------------------------------------------------------ writing


def test_written_files_read_back_the_same(tmp_path):
    original = EAM(write_setfl(tmp_path / "a.eam.alloy"))
    original.write(tmp_path / "b.eam.alloy")
    copy = EAM(tmp_path / "b.eam.alloy")
    a, b = _read_setfl(tmp_path / "a.eam.alloy"), _read_setfl(tmp_path / "b.eam.alloy")
    for key in ("elements", "numbers", "lattices", "drho", "dr", "cutoff", "comments"):
        assert a[key] == b[key]
    for key in ("masses", "lattice_constants", "embedding", "density", "pair"):
        np.testing.assert_array_equal(a[key], b[key])
    assert copy.elements == original.elements and copy.cutoff == original.cutoff
    system = alloy(seed=3)
    same_results(original.compute(system, ALL), copy.compute(system, ALL))


def test_element_lines_without_a_lattice(tmp_path):
    path = write_setfl(tmp_path / "a.eam.alloy")
    full = EAM(path)
    lines = path.read_text().splitlines()
    lines[5] = " ".join(lines[5].split()[:2])  # only the number and the mass
    path.write_text("\n".join(lines) + "\n")
    short = EAM(path)
    assert _read_setfl(path)["lattice_constants"][0] == 0.0
    same_results(short.compute(alloy(seed=1), ALL), full.compute(alloy(seed=1), ALL))


def test_names_other_than_elements(tmp_path):
    path = write_setfl(tmp_path / "a.eam.alloy")
    lines = path.read_text().splitlines()
    lines[3] = "2 Cu A"
    path.write_text("\n".join(lines) + "\n")
    potential = EAM(path)
    assert potential.elements == ("Cu",) and potential.masses.shape == (1,)
    copper = alloy(seed=2).with_columns(element=["Cu"] * 32)
    assert np.isfinite(potential.compute(copper).energy)
    with pytest.raises(InvalidParameter, match=r"no parameters for \['Ni'\]"):
        potential.compute(alloy(seed=2))


# ------------------------------------------------------------ Zhou et al.


def zhou_functions(element):
    """The density, pair potential and embedding energy of Zhou et al.
    (2004), equations 4, 5 and 6-8, written from the paper."""
    p = _parameters(element)

    def density(r):
        return (
            p["fe"]
            * math.exp(-p["beta1"] * (r / p["re"] - 1))
            / (1 + (r / p["re"] - p["ramda1"]) ** 20)
        )

    def pair(r):
        x = r / p["re"]
        return p["A"] * math.exp(-p["alpha"] * (x - 1)) / (
            1 + (x - p["cai"]) ** 20
        ) - p["B"] * math.exp(-p["beta"] * (x - 1)) / (1 + (x - p["ramda"]) ** 20)

    def embedding(rho):
        rho_n, rho_0 = p["rhol"] * p["rhoe"], p["rhoh"] * p["rhoe"]
        if rho == 0:
            return 0.0
        if rho < rho_n:
            return sum(p[f"Fi{i}"] * (rho / rho_n - 1) ** i for i in range(4))
        if rho < rho_0:
            fm3 = p["Fm3"] if rho < p["rhoe"] else p["Fm4"]
            terms = [p["Fm0"], p["Fm1"], p["Fm2"], fm3]
            return sum(f * (rho / p["rhoe"] - 1) ** i for i, f in enumerate(terms))
        x = rho / p["rhos"]
        return p["Fn"] * (1 - p["fnn"] * math.log(x)) * x ** p["fnn"]

    return density, pair, embedding


@pytest.mark.parametrize("elements", [["Cu"], ["Ni", "Al", "Cr"], ["W", "Mg"]])
def test_zhou_tables_follow_the_paper(tmp_path, elements):
    potential = EAM.zhou(elements, nr=500, nrho=400)
    potential.write(tmp_path / "z.eam.alloy")
    t = _read_setfl(tmp_path / "z.eam.alloy")
    assert t["elements"] == tuple(elements) and len(t["embedding"][0]) == 400
    lattice = max(math.sqrt(2) * _parameters(e)["re"] for e in elements)
    assert t["cutoff"] == pytest.approx(math.sqrt(10) / 2 * lattice, rel=1e-15)
    r = np.maximum(np.arange(500) * t["dr"], 0.5)
    rho = np.arange(400) * t["drho"]
    functions = [zhou_functions(e) for e in elements]
    pair = iter(t["pair"])
    for i, (density, phi, embedding) in enumerate(functions):
        np.testing.assert_allclose(t["density"][i], [density(x) for x in r], rtol=1e-13)
        np.testing.assert_allclose(
            t["embedding"][i], [embedding(x) for x in rho], rtol=1e-13, atol=1e-13
        )
        for j in range(i + 1):
            if j == i:
                expected = [x * phi(x) for x in r]
            else:  # the mixing rule of equation 9
                dj, pj = functions[j][0], functions[j][1]
                expected = [
                    x * 0.5 * (dj(x) / density(x) * phi(x) + density(x) / dj(x) * pj(x))
                    for x in r
                ]
            np.testing.assert_allclose(next(pair), expected, rtol=1e-12, atol=1e-13)
    assert t["masses"].tolist() == [_parameters(e)["mass"] for e in elements]


# Cohesive energy (eV) and lattice of Table I of Zhou et al. (2004).
COHESIVE = {"Cu": (3.54, "fcc"), "Ag": (2.85, "fcc"), "Au": (3.93, "fcc"),
            "Ni": (4.45, "fcc"), "Pd": (3.91, "fcc"), "Pt": (5.77, "fcc"),
            "Al": (3.58, "fcc"), "Pb": (2.02, "fcc"), "Fe": (4.29, "bcc"),
            "Mo": (6.81, "bcc"), "Ta": (8.09, "bcc"), "W": (8.76, "bcc")}  # fmt: skip


@pytest.mark.parametrize("element", COHESIVE)
def test_zhou_crystals_at_equilibrium(element):
    energy, lattice = COHESIVE[element]
    re = _parameters(element)["re"]  # the nearest-neighbor distance
    # in fcc the cutoff, sqrt(10)/2 a, falls on the fifth shell: a lattice
    # 1e-9 smaller keeps that shell inside for every atom
    a = re * math.sqrt(2) * (1 - 1e-9) if lattice == "fcc" else re * 2 / math.sqrt(3)
    system = crystal(element, lattice, a)
    result = EAM.zhou([element]).compute(system, ("energy", "forces", "virial"))
    assert result.energy / len(system) == pytest.approx(-energy, abs=2e-4)
    assert np.abs(result.stress(system)).max() < 0.005  # GPa
    assert np.abs(result.forces).max() < 1e-10


def test_zhou_errors():
    assert len(ELEMENTS) == 17 and "Cr" in ELEMENTS
    for elements, message in (
        (["Cu", "Xx"], "not \\['Xx'\\]"),
        ([], "Zhou potentials exist"),
        (["Cu", "Cu"], "repeat"),
    ):
        with pytest.raises(InvalidParameter, match=message):
            EAM.zhou(elements)
    with pytest.raises(InvalidParameter, match="at least 5"):
        EAM.zhou(["Cu"], nr=4)
    potential = EAM.zhou(["Cu", "Ni"])
    assert potential.path is None
    assert repr(potential) == f"EAM(elements=('Cu', 'Ni'), cutoff={potential.cutoff})"


# ------------------------------------------------------------ average atom


def test_an_average_of_one_element_is_that_element(tmp_path):
    potential = EAM(write_setfl(tmp_path / "a.eam.alloy"))
    copper = alloy(seed=4).with_columns(element=["Cu"] * 32)
    pure = potential.compute(copper, ALL)
    for concentrations in ({"Cu": 1.0}, [1.0, 0.0]):
        average = potential.average(concentrations)
        same_results(average.compute(copper, ALL), pure)
    # every atom is the average atom, whatever its element
    same_results(average.compute(alloy(seed=4), ALL), pure)
    assert average.elements is None and average.masses.tolist() == [63.546]


def test_average_tables(tmp_path):
    potential = EAM.zhou(["Cu", "Ni", "Al"], nr=300, nrho=300)
    c = np.array([0.5, 0.3, 0.2])
    average = potential.average({"Cu": 0.5, "Ni": 0.3, "Al": 0.2})
    potential.write(tmp_path / "base.eam.alloy")
    average.write(tmp_path / "average.eam.alloy")
    base, t = (
        _read_setfl(tmp_path / "base.eam.alloy"),
        _read_setfl(tmp_path / "average.eam.alloy"),
    )
    assert t["elements"] == ("Cu", "Ni", "Al", "A")
    np.testing.assert_allclose(t["masses"][3], c @ base["masses"], rtol=1e-15)
    for key in ("embedding", "density"):
        np.testing.assert_array_equal(t[key][:3], base[key])
        np.testing.assert_allclose(t[key][3], c @ base[key], rtol=1e-12, atol=1e-14)

    def pair(table, i, j):  # rows of the lower triangle
        i, j = max(i, j), min(i, j)
        return table["pair"][i * (i + 1) // 2 + j]

    for i in range(3):
        mixed = sum(c[a] * pair(base, a, i) for a in range(3))
        np.testing.assert_allclose(pair(t, 3, i), mixed, rtol=1e-12, atol=1e-14)
    both = sum(c[a] * c[b] * pair(base, a, b) for a in range(3) for b in range(3))
    np.testing.assert_allclose(pair(t, 3, 3), both, rtol=1e-12, atol=1e-14)
    # read back, the file holds the elements; the average atom is not one
    assert EAM(tmp_path / "average.eam.alloy").elements == ("Cu", "Ni", "Al")
    assert repr(average) == f"EAM(average atom, cutoff={average.cutoff})"


def test_average_errors(tmp_path):
    potential = EAM.zhou(["Cu", "Ni"])
    for concentrations, message in (
        ({"Cu": 0.5, "Al": 0.5}, r"has no \['Al'\]"),
        ([0.5, 0.3, 0.2], "expected 2 concentrations"),
        ({"Cu": 0.7}, "sum to 1"),
        ([1.2, -0.2], "non-negative"),
    ):
        with pytest.raises(InvalidParameter, match=message):
            potential.average(concentrations)
    with pytest.raises(InvalidParameter, match="already an average-atom"):
        potential.average([0.5, 0.5]).average([1.0])
