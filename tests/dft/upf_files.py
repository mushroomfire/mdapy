# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""UPF files written by the tests, of known content: a norm-conserving,
ultrasoft or PAW pseudopotential of version 2, or a norm-conserving or
ultrasoft one of version 1, on a logarithmic mesh, with Gaussians for its
radial functions. Content gives the values in rydberg, as the files hold
them; the tests take them back from read_upf in hartree."""

import math
from pathlib import Path

import numpy as np

MESH, DX, XMIN, ZMESH = 801, 0.0125, -7.0, 1.0  # r from 9e-4 to 20 bohr
CUT = 650  # the points of the projectors, up to r = 3 bohr
POINTS = 660  # the augmentation points of PAW
ANGULAR = (0, 0, 1, 2)  # the projectors
VALENCE = 9.0
ORBITALS = (("3S", 0, 2.0), ("3P", 1, 1.0), ("3D", 2, 6.0))
VANISHING = (2, 3, 3)  # a term of PAW whose multipole the file gives as zero
WIDTH = 4.0  # 1/bohr^2, of the augmentation charges, nothing beyond the projectors


def keys() -> list[tuple[int, int, int]]:
    """(i, j, L) of the augmentation charges, i <= j, L from |l_i - l_j| to
    l_i + l_j by twos."""
    n = len(ANGULAR)
    return [(i, j, ell) for i in range(n) for j in range(i, n)
            for ell in range(abs(ANGULAR[i] - ANGULAR[j]), ANGULAR[i] + ANGULAR[j] + 1, 2)]  # fmt: skip


def factor(i: int, j: int) -> float:
    """a of the augmentation charges r^2 Q^L_ij(r) = a r^(L+2) exp(-WIDTH r^2)."""
    return 0.2 + 0.1 * i + 0.05 * j


def content(kind: str = "NC", core: bool = True) -> dict:
    """The radial functions of a pseudopotential of kind NC, US or PAW, as
    the file holds them (rydberg)."""
    r = np.exp(XMIN + DX * np.arange(MESH)) / ZMESH
    erf = np.vectorize(math.erf)
    inside = np.arange(MESH) < CUT
    count = len(ANGULAR)
    data = {
        "r": r,
        "rab": DX * r,
        "local": -2 * VALENCE * erf(r) / r + 3.0 * np.exp(-r * r),
        "betas": [
            r ** (ell + 1) * np.exp(-r * r) * (1 + 0.3 * i) * inside
            for i, ell in enumerate(ANGULAR)
        ],
        "dij": np.zeros((count, count)),
        "core": 0.5 * np.exp(-2 * r * r) if core else None,
    }
    for i in range(count):
        for j in range(count):
            if ANGULAR[i] == ANGULAR[j]:
                data["dij"][i, j] = 1.5 - 0.5 * (i + j) + (0.25 if i == j else 0.0)
    # orbitals of unit norm, int (r chi)^2 dr = 1, and the atom their sum
    data["orbitals"] = [
        (label, ell, occupation,
         math.sqrt(2 / math.gamma(ell + 1.5)) * r ** (ell + 1) * np.exp(-r * r / 2))
        for label, ell, occupation in ORBITALS
    ]  # fmt: skip
    data["atom"] = sum(o * chi * chi for _, _, o, chi in data["orbitals"])
    if kind in ("US", "PAW"):
        # r^2 Q^L_ij(r) = a r^(L+2) exp(-w r^2), q_ij its integral for L = 0 and
        # the multipoles int r^L r^2 Q^L dr = a Gamma(L + 3/2) / (2 w^(L + 3/2))
        data["augmentation"] = {
            (i, j, ell): factor(i, j) * r ** (ell + 2) * np.exp(-WIDTH * r * r)
            for i, j, ell in keys()
        }
        q = np.zeros((count, count))
        multipoles = np.zeros((count, count, 2 * max(ANGULAR) + 1))
        for i, j, ell in data["augmentation"]:
            a = factor(i, j)
            if ell == 0:
                q[i, j] = q[j, i] = a * math.sqrt(math.pi) / (4 * WIDTH**1.5)
            if (i, j, ell) != VANISHING:
                multipoles[i, j, ell] = multipoles[j, i, ell] = (
                    a * math.gamma(ell + 1.5) / (2 * WIDTH ** (ell + 1.5))
                )
        data["q"], data["multipoles"] = q, multipoles
    if kind == "PAW":
        data["ae_waves"] = [r ** (ell + 1) * np.exp(-r * r / 2) * (1 + 0.2 * i)
                            for i, ell in enumerate(ANGULAR)]  # fmt: skip
        data["ps_waves"] = [r ** (ell + 1) * np.exp(-r * r) * (1 + 0.1 * i)
                            for i, ell in enumerate(ANGULAR)]  # fmt: skip
        data["occupations"] = np.array([2.0, 0.0, 1.0, 6.0])
        data["ae_core"] = 10.0 * np.exp(-4 * r * r)
        data["ae_local"] = -2 * 26.0 * erf(r) / r
        data["core_energy"] = -200.0
    return data


def _rows(values) -> str:
    return "\n".join(" ".join(f"{v:.16e}" for v in values[i : i + 4])
                     for i in range(0, len(values), 4))  # fmt: skip


def _block(tag: str, values, **attributes) -> str:
    extra = "".join(f' {k}="{v}"' for k, v in attributes.items())
    return f"<{tag}{extra}>\n{_rows(np.ravel(values))}\n</{tag}>"


def write(path: Path, kind: str = "NC", core: bool = True) -> Path:
    """A UPF file of version 2 of a pseudopotential of kind NC, US or PAW of
    content(kind, core), the cutoff of the last projector left for read_upf
    to find."""
    data = content(kind, core)
    count = len(ANGULAR)
    ultrasoft = "T" if kind in ("US", "PAW") else "F"
    head = (
        f'<PP_HEADER element="X" pseudo_type="{kind}" is_ultrasoft="{ultrasoft}"'
        f' is_paw="{"T" if kind == "PAW" else "F"}" has_so="F"'
        f' core_correction="{"T" if core else "F"}" functional=" SLA  PW   PBX  PBC"'
        f' z_valence="{VALENCE:.1f}" l_max_rho="{2 * max(ANGULAR)}"'
        f' mesh_size="{MESH}" number_of_proj="{count}"/>'
    )
    betas = []
    for i, (ell, beta) in enumerate(zip(ANGULAR, data["betas"])):
        extra = {"angular_momentum": ell}
        if i < count - 1:
            extra["cutoff_radius_index"] = CUT
        betas.append(_block(f"PP_BETA.{i + 1}", beta, index=i + 1, **extra))
    parts = [
        '<UPF version="2.0.1">',
        "<PP_INFO>input with a < sign & an ampersand</PP_INFO>",
        head,
        f'<PP_MESH dx="{DX}" mesh="{MESH}" xmin="{XMIN}" zmesh="{ZMESH}">',
        _block("PP_R", data["r"]),
        _block("PP_RAB", data["rab"]),
        "</PP_MESH>",
    ]
    if core:
        parts.append(_block("PP_NLCC", data["core"]))
    parts += [_block("PP_LOCAL", data["local"]), "<PP_NONLOCAL>", *betas]
    parts.append(_block("PP_DIJ", data["dij"]))
    if kind in ("US", "PAW"):
        attributes = 'q_with_l="T" nqf="0"'
        if kind == "PAW":
            attributes += f' cutoff_r_index="{POINTS}" augmentation_epsilon="1.d-12"'
        parts.append(f"<PP_AUGMENTATION {attributes}>")
        parts.append(_block("PP_Q", data["q"]))
        if kind == "PAW":
            # (i, j, L) in the order of Fortran
            parts.append(_block("PP_MULTIPOLES", data["multipoles"].ravel(order="F")))
        for (i, j, ell), charge in data["augmentation"].items():
            tag = f"PP_QIJL.{i + 1}.{j + 1}.{ell}"
            parts.append(_block(tag, charge, angular_momentum=ell))
        parts.append("</PP_AUGMENTATION>")
    parts += ["</PP_NONLOCAL>", "<PP_PSWFC>"]
    for k, (label, ell, occupation, chi) in enumerate(data["orbitals"]):
        parts.append(_block(f"PP_CHI.{k + 1}", chi, label=label, l=ell, occupation=occupation))  # fmt: skip
    parts.append("</PP_PSWFC>")
    if kind == "PAW":
        parts.append("<PP_FULL_WFC>")
        for i in range(count):
            parts.append(_block(f"PP_AEWFC.{i + 1}", data["ae_waves"][i], index=i + 1))
            parts.append(_block(f"PP_PSWFC.{i + 1}", data["ps_waves"][i], index=i + 1))
        parts.append("</PP_FULL_WFC>")
    parts.append(_block("PP_RHOATOM", data["atom"]))
    if kind == "PAW":
        parts += [
            f'<PP_PAW paw_data_format="2" core_energy="{data["core_energy"]}">',
            _block("PP_OCCUPATIONS", data["occupations"]),
            _block("PP_AE_NLCC", data["ae_core"]),
            _block("PP_AE_VLOC", data["ae_local"]),
            "</PP_PAW>",
        ]
    parts.append("</UPF>")
    path.write_text("\n".join(parts) + "\n")
    return path


RINNER = 0.3  # bohr, within which version 1 pseudizes the charges
NQF = 10  # the coefficients of the polynomials there


def write_v1(path: Path, kind: str = "US", core: bool = True, nqf: int = NQF) -> Path:
    """A UPF file of version 1 of a norm-conserving (NC) or ultrasoft (US)
    pseudopotential of content(kind, core): for US one function r^2 Q_ij(r)
    of each pair, that of its lowest L (all L of a pair share it in version 1),
    and with nqf > 0 inside RINNER the series of r^(L+2) exp(-WIDTH r^2) (times the
    factor of the pair) to nqf terms."""
    data = content(kind, core)
    count = len(ANGULAR)
    header = f"""<PP_HEADER>
   0                   Version Number
  X                    Element
   {kind}                  Ultrasoft pseudopotential
    {"T" if core else "F"}                  Nonlinear Core Correction
SLA  PW   PBE  PBE     PBE  Exchange-Correlation functional
    {VALENCE:.11f}      Z valence
   -6.43282807590      Total energy
    0.00000    0.00000 Suggested cutoff for wfc and rho
    {max(ANGULAR)}                  Max angular momentum component
  {MESH}                  Number of points in mesh
    {len(ORBITALS)}    {count}             Number of Wavefunctions, Number of Projectors
 Wavefunctions         nl  l   occ
{chr(10).join(f"                       {label}  {ell}  {o:.2f}" for label, ell, o in ORBITALS)}
</PP_HEADER>"""

    def block(tag, values, before=""):
        return f"<{tag}>\n{before}{_rows(values)}\n</{tag}>"

    parts = [
        "<PP_INFO>\nGenerated by hand\n</PP_INFO>",
        header,
        "<PP_MESH>",
        block("PP_R", data["r"]),
        block("PP_RAB", data["rab"]),
        "</PP_MESH>",
    ]
    if core:
        parts.append(block("PP_NLCC", data["core"]))
    parts += [block("PP_LOCAL", data["local"]), "<PP_NONLOCAL>"]
    for i, (ell, beta) in enumerate(zip(ANGULAR, data["betas"])):
        parts.append(block("PP_BETA", beta[:CUT], f"    {i + 1}    {ell}             Beta    L\n   {CUT}\n"))  # fmt: skip
    nonzero = [
        (i, j) for i in range(count) for j in range(i, count) if data["dij"][i, j]
    ]
    rows = "\n".join(
        f"    {i + 1}    {j + 1}  {data['dij'][i, j]:.11e}" for i, j in nonzero
    )
    parts.append(f"<PP_DIJ>\n    {len(nonzero)}                  Number of nonzero Dij\n{rows}\n</PP_DIJ>")  # fmt: skip
    if kind == "US":
        lines = [f"    {nqf}     nqf. If not zero, Qij's inside rinner are computed using qfcoef's"]  # fmt: skip
        nqlc = 2 * max(ANGULAR) + 1
        if nqf > 0:
            lines.append("    <PP_RINNER>")
            lines += [f"    {ell + 1}  {RINNER:.11e}" for ell in range(nqlc)]
            lines.append("    </PP_RINNER>")
        for i in range(count):
            for j in range(i, count):
                lowest = abs(ANGULAR[i] - ANGULAR[j])
                charge = data["augmentation"][(i, j, lowest)]
                a = factor(i, j)
                lines.append(
                    f"    {i + 1}    {j + 1}    {ANGULAR[j]}        i  j  (l(j))"
                )
                lines.append(f"  {data['q'][i, j]:.11e}    Q_int")
                lines.append(_rows(charge))
                if nqf > 0:
                    # c_k of L: a (-w)^k / k!, the series of exp(-w r^2)
                    c = [[a * (-WIDTH) ** k / math.factorial(k) for k in range(nqf)]
                         for _ in range(nqlc)]  # fmt: skip
                    lines += ["    <PP_QFCOEF>", _rows(np.ravel(c)), "    </PP_QFCOEF>"]
        parts.append("<PP_QIJ>\n" + "\n".join(lines) + "\n</PP_QIJ>")
    parts.append("</PP_NONLOCAL>")
    wavefunctions = [f"{label}    {ell}  {o:.2f}          Wavefunction\n{_rows(chi)}"
                     for label, ell, o, chi in data["orbitals"]]  # fmt: skip
    parts.append("<PP_PSWFC>\n" + "\n".join(wavefunctions) + "\n</PP_PSWFC>")
    parts.append(block("PP_RHOATOM", data["atom"]))
    path.write_text("\n".join(parts) + "\n")
    return path
