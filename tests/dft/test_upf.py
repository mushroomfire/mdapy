# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""UPF files read on the processor: files written here of every kind read_upf
takes come back as written, in hartree, and each part that is missing or short
is refused; public files of Quantum ESPRESSO's library and of GBRV against an
XML parse of their own numbers and against their sum rules, the norms of the
orbitals, the charge of the free atom, q_ij and the multipoles of PAW."""

import hashlib
import math
import re
from xml.etree import ElementTree

import numpy as np
import pytest

from mdapy.calculators._dft import radial
from mdapy.calculators._dft.upf import read_upf
from mdapy.errors import InvalidParameter

from dft import public, upf_files
from dft.upf_files import ANGULAR, CUT, MESH, POINTS, RINNER, VALENCE


def check_common(pp, data, path):
    """The parts every kind holds, against content: energies halved."""
    assert pp.element == "X" and pp.valence == VALENCE
    np.testing.assert_array_equal(pp.r, data["r"])
    np.testing.assert_array_equal(pp.rab, data["rab"])
    np.testing.assert_allclose(pp.local, 0.5 * data["local"], rtol=1e-15)
    assert [p.angular for p in pp.projectors] == list(ANGULAR)
    assert [p.cutoff_index for p in pp.projectors] == [CUT] * len(ANGULAR)
    for p, beta in zip(pp.projectors, data["betas"]):
        np.testing.assert_allclose(p.r_beta, beta, rtol=1e-15)
    np.testing.assert_allclose(pp.dij, 0.5 * data["dij"], rtol=1e-15)
    if data["core"] is None:
        assert pp.core is None
    else:
        np.testing.assert_allclose(pp.core, data["core"], rtol=1e-15)
    assert [(o.label, o.angular, o.occupation) for o in pp.orbitals] == list(
        upf_files.ORBITALS
    )
    for o, (*_, chi) in zip(pp.orbitals, data["orbitals"]):
        np.testing.assert_allclose(o.r_chi, chi, rtol=1e-15)
    np.testing.assert_allclose(pp.atom, data["atom"], rtol=1e-15)
    assert pp.digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert pp.lmax == max(ANGULAR)


@pytest.mark.parametrize("core", [True, False])
def test_norm_conserving(tmp_path, core):
    """The cutoff of the last projector, which the file does not give, is up
    to its last value that is not zero."""
    path = upf_files.write(tmp_path / "X.upf", "NC", core)
    pp = read_upf(path)
    check_common(pp, upf_files.content("NC", core), path)
    assert pp.functional == "SLA  PW   PBX  PBC"
    assert not pp.ultrasoft and pp.q is None and pp.augmentation is None
    assert pp.paw is None and pp.kkbeta == CUT
    # the orbitals hold one electron each, the atom the valence
    for o in pp.orbitals:
        assert radial.simpson(o.r_chi**2, pp.rab) == pytest.approx(1.0, abs=1e-8)
    assert radial.simpson(pp.atom, pp.rab) == pytest.approx(VALENCE, abs=1e-8)


def test_ultrasoft(tmp_path):
    path = upf_files.write(tmp_path / "X.upf", "US")
    pp = read_upf(path)
    data = upf_files.content("US")
    check_common(pp, data, path)
    assert pp.ultrasoft and pp.paw is None and pp.kkbeta == CUT
    np.testing.assert_allclose(pp.q, data["q"], rtol=1e-15)
    assert sorted(pp.augmentation) == upf_files.keys()
    for key, charge in pp.augmentation.items():
        np.testing.assert_allclose(charge, data["augmentation"][key], rtol=1e-15)
    # q_ij is the integral of the charge of L = 0
    for (i, j, ell), charge in pp.augmentation.items():
        if ell == 0:
            assert radial.simpson(charge, pp.rab) == pytest.approx(pp.q[i, j], abs=1e-9)


def test_paw(tmp_path):
    """The charges and the partial waves zero beyond the augmentation points,
    the term of a vanishing multipole left out, energies halved."""
    path = upf_files.write(tmp_path / "X.upf", "PAW")
    pp = read_upf(path)
    data = upf_files.content("PAW")
    check_common(pp, data, path)
    paw = pp.paw
    assert pp.ultrasoft and paw is not None and pp.kkbeta == POINTS
    assert paw.lmax_rho == 2 * max(ANGULAR) and paw.augmentation_points == POINTS
    assert (paw.dx, paw.xmin, paw.zmesh) == (
        upf_files.DX,
        upf_files.XMIN,
        upf_files.ZMESH,
    )
    np.testing.assert_allclose(paw.multipoles, data["multipoles"], rtol=1e-15)
    assert sorted(pp.augmentation) == sorted(
        set(upf_files.keys()) - {upf_files.VANISHING}
    )
    for key, charge in pp.augmentation.items():
        np.testing.assert_allclose(charge[:POINTS], data["augmentation"][key][:POINTS], rtol=1e-15)  # fmt: skip
        assert not charge[POINTS:].any()
    for read, written in [(paw.ae_waves, data["ae_waves"]), (paw.ps_waves, data["ps_waves"])]:  # fmt: skip
        assert read.shape == (len(ANGULAR), MESH)
        np.testing.assert_allclose(read[:, :POINTS], np.array(written)[:, :POINTS], rtol=1e-15)  # fmt: skip
        assert not read[:, POINTS:].any()
    np.testing.assert_array_equal(paw.occupations, data["occupations"])
    np.testing.assert_allclose(paw.ae_core, data["ae_core"], rtol=1e-15)
    np.testing.assert_allclose(paw.ae_local, 0.5 * data["ae_local"], rtol=1e-15)
    assert paw.core_energy == 0.5 * data["core_energy"]


@pytest.mark.parametrize("core", [True, False])
def test_norm_conserving_of_version_1(tmp_path, core):
    path = upf_files.write_v1(tmp_path / "X.upf", "NC", core)
    pp = read_upf(path)
    data = upf_files.content("NC", core)
    check_common(pp, data, path)
    # QE reads the first 20 characters of the line of the functional
    assert pp.functional == "SLA PW PBE PBE"
    assert not pp.ultrasoft and pp.augmentation is None


@pytest.mark.parametrize("nqf", [0, upf_files.NQF])
def test_ultrasoft_of_version_1(tmp_path, nqf):
    """One function r^2 Q_ij of each pair for all its L; with nqf > 0, inside
    rinner r^(L+2) sum_k c_k r^(2k), the series of r^(L+2) exp(-w r^2) here,
    and the file's function outside."""
    path = upf_files.write_v1(tmp_path / "X.upf", "US", nqf=nqf)
    pp = read_upf(path)
    data = upf_files.content("US")
    check_common(pp, data, path)
    np.testing.assert_allclose(pp.q, data["q"], rtol=1e-11)
    assert sorted(pp.augmentation) == upf_files.keys()
    r = pp.r
    inside = r < RINNER
    for (i, j, ell), charge in pp.augmentation.items():
        lowest = abs(ANGULAR[i] - ANGULAR[j])
        written = data["augmentation"][(i, j, lowest)]
        if nqf == 0:
            np.testing.assert_allclose(charge, written, rtol=1e-11)
            continue
        np.testing.assert_allclose(charge[~inside], written[~inside], rtol=1e-11)
        series = sum(upf_files.factor(i, j) * (-upf_files.WIDTH) ** k / math.factorial(k) * r[inside] ** (2 * k)
                     for k in range(nqf))  # fmt: skip
        np.testing.assert_allclose(charge[inside], r[inside] ** (ell + 2) * series, rtol=1e-11)  # fmt: skip
        if ell == lowest:
            # the series of the function itself, which keeps q_ij
            np.testing.assert_allclose(charge, written, rtol=1e-9, atol=0)
            if ell == 0:
                integral = radial.simpson(charge, pp.rab)
                assert integral == pytest.approx(pp.q[i, j], abs=1e-9)


def drop(tag):
    """A change of a file: the element tag taken out."""
    return lambda text: re.sub(rf"<{tag}\b.*?</{tag}>", "", text, count=1, flags=re.S)


def replace(old, new):
    def change(text):
        assert old in text
        return text.replace(old, new, 1)

    return change


def shorten(tag):
    """A change of a file: the element tag holding its first values only."""
    return lambda text: re.sub(rf"(<{tag}\b[^>]*>\s*\S+ \S+ \S+ \S+).*?(</{tag}>)", r"\1\n\2",
                               text, count=1, flags=re.S)  # fmt: skip


V2 = [
    ("NC", replace('<UPF version="2.0.1">', "<UPF>"), "not a UPF file of version 1 or 2"),
    ("NC", lambda t: re.sub(r"<PP_HEADER.*?/>", "", t, flags=re.S), "no PP_HEADER"),
    ("NC", replace('pseudo_type="NC"', 'pseudo_type="1/r"'), "norm-conserving, ultrasoft and PAW"),
    ("NC", replace('has_so="F"', 'has_so=".TRUE."'), "spin-orbit"),
    ("NC", drop("PP_RAB"), "no PP_RAB"),
    ("NC", shorten("PP_LOCAL"), "PP_LOCAL holds 4 of 801 values"),
    ("NC", replace('number_of_proj="4"', 'number_of_proj="5"'), "4 of 5 projectors"),
    ("NC", drop("PP_DIJ"), "no PP_DIJ"),
    ("US", drop("PP_AUGMENTATION"), "no PP_AUGMENTATION"),
    ("US", replace('q_with_l="T"', 'q_with_l="F"'), "lack their angular momenta"),
    ("US", replace('nqf="0"', 'nqf="8"'), "pseudized inside rinner"),
    ("US", drop("PP_Q"), "no PP_Q"),
    ("US", lambda t: re.sub(r"<PP_QIJL.*?</PP_QIJL\.\d+\.\d+\.\d+>", "", t, flags=re.S), "no PP_QIJL"),
    ("PAW", drop("PP_MULTIPOLES"), "no PP_MULTIPOLES"),
    ("PAW", replace(' zmesh="1.0"', ""), "PP_MESH lacks the logarithmic mesh"),
    ("PAW", drop("PP_FULL_WFC"), "no PP_FULL_WFC or PP_PAW"),
    ("PAW", drop("PP_PAW"), "no PP_FULL_WFC or PP_PAW"),
    ("PAW", drop("PP_AEWFC.4"), "3 PP_AEWFC of 4"),
    ("PAW", drop("PP_OCCUPATIONS"), "no PP_OCCUPATIONS"),
    ("PAW", shorten("PP_AE_NLCC"), "PP_AE_NLCC holds 4 of 801"),
]  # fmt: skip
V1 = [
    ("NC", replace("   NC   ", "   PAW   "), "norm-conserving and ultrasoft pseudopotentials are read"),
    ("NC", drop("PP_RAB"), "no PP_RAB"),
    ("NC", shorten("PP_LOCAL"), "PP_LOCAL holds 4 of 801 values"),
    ("NC", drop("PP_BETA"), "3 of 4 projectors"),
    ("US", drop("PP_RINNER"), "nqf > 0 without PP_RINNER"),
    ("US", drop("PP_QFCOEF"), "no PP_QFCOEF of 1 1"),
    ("US", lambda t: re.sub(r"(Q_int\n\S+ \S+ \S+ \S+).*?(\n    <PP_QFCOEF>)", r"\1\2", t, count=1, flags=re.S),
     "the Q of 1 1 is short"),
    ("US", lambda t: re.sub(r"(</PP_RINNER>).*?(</PP_QIJ>)", r"\1\n\2", t, flags=re.S),
     "no augmentation charges in PP_QIJ"),
]  # fmt: skip


@pytest.mark.parametrize(("kind", "change", "message"), V2)
def test_what_is_refused(tmp_path, kind, change, message):
    path = upf_files.write(tmp_path / "X.upf", kind)
    path.write_text(change(path.read_text()))
    with pytest.raises(InvalidParameter, match=message):
        read_upf(path)


@pytest.mark.parametrize(("kind", "change", "message"), V1)
def test_what_is_refused_of_version_1(tmp_path, kind, change, message):
    path = upf_files.write_v1(tmp_path / "X.upf", kind)
    path.write_text(change(path.read_text()))
    with pytest.raises(InvalidParameter, match=message):
        read_upf(path)


@pytest.mark.parametrize("version", [1, 2])
def test_a_local_potential_alone(tmp_path, version):
    """Without projectors there is no D_ij to read, and the cutoff of the
    projectors is the first point."""
    if version == 2:
        path = upf_files.write(tmp_path / "X.upf", "NC")
        text = re.sub(
            r"<PP_BETA\.\d.*?</PP_BETA\.\d>", "", path.read_text(), flags=re.S
        )
        text = text.replace('number_of_proj="4"', 'number_of_proj="0"')
    else:
        path = upf_files.write_v1(tmp_path / "X.upf", "NC")
        text = re.sub(r"<PP_BETA>.*?</PP_BETA>", "", path.read_text(), flags=re.S)
        text = replace("    3    4    ", "    3    0    ")(text)
    path.write_text(drop("PP_DIJ")(text))
    pp = read_upf(path)
    assert pp.projectors == () and pp.dij.shape == (0, 0)
    assert pp.kkbeta == 1 and pp.lmax == -1


def test_a_projector_of_zeros(tmp_path):
    """A projector of zeros without its cutoff in the file reaches the first
    point."""
    path = upf_files.write(tmp_path / "X.upf", "NC")
    text = re.sub(r"(<PP_BETA\.4[^>]*>).*?(</PP_BETA\.4>)",
                  lambda m: m[1] + "\n" + " ".join(["0.0"] * MESH) + "\n" + m[2],
                  path.read_text(), flags=re.S)  # fmt: skip
    path.write_text(text)
    assert read_upf(path).projectors[3].cutoff_index == 1


def xml_parts(path):
    """The elements of a UPF file of version 2 by an XML parser, PP_INFO left
    out (its text is not XML): the attributes of each and its numbers."""
    text = re.sub(r"<PP_INFO>.*?</PP_INFO>", "", path.read_text(), flags=re.S)
    root = ElementTree.fromstring(text)
    return {e.tag: (e.attrib, np.array((e.text or "").split(), dtype=float))
            for e in root.iter()}  # fmt: skip


def check_against_xml(pp, path):
    """read_upf against the numbers of the file as an XML parser reads them."""
    parts = xml_parts(path)
    head = parts["PP_HEADER"][0]
    mesh, count = int(head["mesh_size"]), int(head["number_of_proj"])
    assert pp.element == head["element"].strip()
    assert pp.valence == float(head["z_valence"])
    assert len(pp.r) == mesh and len(pp.projectors) == count
    np.testing.assert_array_equal(pp.r, parts["PP_R"][1][:mesh])
    np.testing.assert_array_equal(pp.local, 0.5 * parts["PP_LOCAL"][1][:mesh])
    for i, p in enumerate(pp.projectors):
        attributes, values = parts[f"PP_BETA.{i + 1}"]
        assert p.angular == int(attributes["angular_momentum"])
        assert p.cutoff_index == int(attributes["cutoff_radius_index"])
        np.testing.assert_array_equal(p.r_beta, values[:mesh])
    np.testing.assert_array_equal(
        pp.dij, 0.5 * parts["PP_DIJ"][1].reshape(count, count)
    )
    np.testing.assert_array_equal(pp.atom, parts["PP_RHOATOM"][1][:mesh])
    labels = [parts[f"PP_CHI.{k + 1}"][0]["label"] for k in range(len(pp.orbitals))]
    assert [o.label for o in pp.orbitals] == labels
    if pp.ultrasoft:
        np.testing.assert_array_equal(pp.q, parts["PP_Q"][1].reshape(count, count))
        for (i, j, ell), charge in pp.augmentation.items():
            values = parts[f"PP_QIJL.{i + 1}.{j + 1}.{ell}"][1][:mesh]
            if pp.paw is not None:
                values[pp.paw.augmentation_points :] = 0.0
            np.testing.assert_array_equal(charge, values)
    return parts


def test_norm_conserving_of_quantum_espresso():
    """Al.pz-vbc: the charge of the atom is that of its orbitals, which hold
    one electron each."""
    path = public.path("Al.pz-vbc.UPF")
    pp = read_upf(path)
    check_against_xml(pp, path)
    assert not pp.ultrasoft and pp.core is None
    for o in pp.orbitals:
        assert radial.simpson(o.r_chi**2, pp.rab) == pytest.approx(1.0, abs=1e-8)
    occupied = sum(o.occupation * o.r_chi**2 for o in pp.orbitals)
    np.testing.assert_allclose(pp.atom, occupied, rtol=0, atol=1e-10)
    assert radial.simpson(pp.atom, pp.rab) == pytest.approx(pp.valence, abs=1e-8)
    np.testing.assert_array_equal(pp.dij, pp.dij.T)


def test_ultrasoft_of_pslibrary():
    """The integrals of the charges of L = 0 are q_ij of the file, the others
    run from |l_i - l_j| to l_i + l_j by twos."""
    path = public.path("Al.pbe-n-rrkjus_psl.1.0.0.UPF")
    pp = read_upf(path)
    check_against_xml(pp, path)
    assert pp.ultrasoft and pp.paw is None and pp.core is not None
    angular = [p.angular for p in pp.projectors]
    assert angular == [0, 0, 1, 1, 2, 2]
    np.testing.assert_array_equal(pp.q, pp.q.T)
    for (i, j, ell), charge in pp.augmentation.items():
        assert abs(angular[i] - angular[j]) <= ell <= angular[i] + angular[j]
        assert (ell + angular[i] + angular[j]) % 2 == 0
        if ell == 0:
            integral = radial.simpson(charge, pp.rab)
            assert integral == pytest.approx(pp.q[i, j], abs=1e-10)
    for i in range(len(angular)):
        for j in range(i, len(angular)):
            if angular[i] != angular[j]:
                assert pp.q[i, j] == 0.0


def test_paw_of_pslibrary():
    """Fe: the multipoles of the file are those of the partial waves, AE less
    PS, and of the augmentation charges, to the augmentation points; q_ij is
    the multipole of L = 0; the occupations hold the valence."""
    path = public.path("Fe.pbe-spn-kjpaw_psl.1.0.0.UPF")
    pp = read_upf(path)
    parts = check_against_xml(pp, path)
    data = pp.paw
    assert data.lmax_rho == 4 and pp.kkbeta == data.augmentation_points == 868
    count = len(pp.projectors)
    multipoles = parts["PP_MULTIPOLES"][1].reshape(-1, count, count).transpose(2, 1, 0)
    np.testing.assert_array_equal(data.multipoles, multipoles)
    assert data.occupations.sum() == pytest.approx(pp.valence, abs=1e-12)
    assert data.core_energy == 0.5 * float(parts["PP_PAW"][0]["core_energy"])
    np.testing.assert_array_equal(data.ae_local, 0.5 * parts["PP_AE_VLOC"][1])
    angular = [p.angular for p in pp.projectors]
    r, n = pp.r, pp.kkbeta
    for i in range(count):
        np.testing.assert_array_equal(
            data.ae_waves[i, :n], parts[f"PP_AEWFC.{i + 1}"][1][:n]
        )
        np.testing.assert_array_equal(
            data.ps_waves[i, :n], parts[f"PP_PSWFC.{i + 1}"][1][:n]
        )
        for j in range(i, count):
            if angular[i] == angular[j]:
                assert data.multipoles[i, j, 0] == pp.q[i, j]
            difference = data.ae_waves[i] * data.ae_waves[j] - data.ps_waves[i] * data.ps_waves[j]  # fmt: skip
            for ell in range(
                abs(angular[i] - angular[j]), angular[i] + angular[j] + 1, 2
            ):
                moment = data.multipoles[i, j, ell]
                assert radial.simpson(difference * r**ell, pp.rab, n) == pytest.approx(moment, abs=1e-9)  # fmt: skip
                charge = pp.augmentation[(i, j, ell)]
                assert radial.simpson(charge * r**ell, pp.rab, n) == pytest.approx(moment, abs=2e-6)  # fmt: skip


def test_ultrasoft_of_gbrv():
    """A file of version 1 whose charges are pseudized inside rinner: they keep
    q_ij, the Q_int of the file, and inside rinner they are r^(L+2) times the
    polynomial of the coefficients of their pair."""
    path = public.path("al_pbe_v1.uspp.F.UPF")
    pp = read_upf(path)
    text = path.read_text()
    assert (
        pp.element == "Al" and pp.valence == 3.0 and pp.functional == "SLA PW PBE PBE"
    )
    assert len(pp.r) == 893 and pp.core is not None and pp.ultrasoft
    assert [p.angular for p in pp.projectors] == [0, 1, 2]
    assert [p.cutoff_index for p in pp.projectors] == [623] * 3
    assert [(o.label, o.angular, o.occupation) for o in pp.orbitals] == [("3S", 0, 2.0), ("3P", 1, 1.0)]  # fmt: skip
    q_int = [float(x) for x in re.findall(r"(\S+)\s+Q_int", text)]
    np.testing.assert_array_equal(np.sort(pp.q[np.triu_indices(3)]), np.sort(q_int))
    np.testing.assert_allclose(pp.dij, 0.5 * np.diag([7.84204084, 5.42275082714, 0.691720554313]))  # fmt: skip
    rinner = 0.9
    first = re.search(r"<PP_QFCOEF>(.*?)</PP_QFCOEF>", text, re.S)[1]
    c = np.array(first.split(), dtype=float).reshape(5, 8)  # nqlc of nqf
    inside = pp.r < rinner
    for (i, j, ell), charge in pp.augmentation.items():
        if ell == 0:
            assert radial.simpson(charge, pp.rab) == pytest.approx(pp.q[i, j], abs=1e-7)
        if (i, j) == (0, 0):
            x = pp.r[inside]
            series = sum(c[ell, k] * x ** (2 * k) for k in range(8))
            np.testing.assert_allclose(
                charge[inside], x ** (ell + 2) * series, rtol=1e-12
            )
