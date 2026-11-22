# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Quantum ESPRESSO's pw.x run live as the reference of the tests of the DFT:
the same cell, pseudopotentials, cutoffs, k-points (given one by one, without
symmetry) and smearing; its energies (hartree) and eigenvalues (hartree, per
k-point) read from its output. The tests skip without pw.x on the PATH or
without the pseudopotentials in MDAPY_PSEUDO_DIR."""

import functools
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

RY = 0.5  # hartree
EV = 1 / 27.211386245988  # hartree

_SMEARING = {
    "gaussian": "gaussian",
    "methfessel-paxton": "m-p",
    "marzari-vanderbilt": "m-v",
    "fermi-dirac": "f-d",
}


def pseudo(name: str) -> Path:
    """The path of a pseudopotential in MDAPY_PSEUDO_DIR, or a skip."""
    folder = os.environ.get("MDAPY_PSEUDO_DIR")
    if not folder or not (Path(folder) / name).is_file():
        pytest.skip(f"{name} not in MDAPY_PSEUDO_DIR")
    return Path(folder) / name


@functools.lru_cache(maxsize=None)
def _run(text: str, folder: str) -> tuple[str, str]:
    exe = shutil.which("pw.x")
    if exe is None:
        pytest.skip("no pw.x")
    with tempfile.TemporaryDirectory() as work:
        Path(work, "pw.in").write_text(text)
        out = subprocess.run(
            [exe, "-in", "pw.in"], cwd=work, capture_output=True, text=True,
            env={**os.environ, "OMP_NUM_THREADS": "4"}, check=True, timeout=1800,
        ).stdout  # fmt: skip
        if "convergence has been achieved" not in out:
            raise RuntimeError("pw.x did not converge:\n" + out[-3000:])
        xml = Path(work, "tmp", "pwscf.save", "data-file-schema.xml").read_text()
    return out, xml


def run(cell, positions, symbols, files, cutoff, kpoints, weights, smearing="fixed",
        sigma=0.0, conv_thr=1e-12, extra="", magmoms=None, beta=0.7, maxstep=100):  # fmt: skip
    """pw.x of a cell (rows, bohr), fractional positions, symbols and their UPF
    files (by element), a cutoff of the bands in hartree, k-points (fractions
    of the reciprocal vectors) and weights; sigma in hartree; with magmoms (Bohr
    magnetons per atom) collinear spins, a species for each element and moment.
    A dict of the energies and the eigenvalues ((2,) K, bands)."""
    from mdapy.calculators._dft.upf import read_upf

    folder = str(Path(next(iter(files.values()))).parent)
    occupations = (
        'occupations = "fixed"'
        if smearing == "fixed"
        else f'occupations = "smearing", smearing = "{_SMEARING[smearing]}", '
        f"degauss = {sigma / RY:.12f}"
    )
    # the species of QE: one per element and first moment
    moments = [0.0] * len(symbols) if magmoms is None else list(magmoms)
    kinds = list(dict.fromkeys(zip(symbols, moments)))
    labels = [f"{el}{i + 1}" for i, (el, _) in enumerate(kinds)]
    spin = ""
    if magmoms is not None:
        valence = {el: read_upf(f).valence for el, f in files.items()}
        spin = ", nspin = 2, " + ", ".join(
            f"starting_magnetization({i + 1}) = {m / valence[el]:.12f}"
            for i, (el, m) in enumerate(kinds)
        )
    b = np.linalg.inv(np.asarray(cell)).T  # rows, without 2 pi
    alat = np.linalg.norm(cell[0])
    cart = np.asarray(kpoints) @ b * alat  # units of 2 pi / alat
    lines = [
        "&control",
        f'  calculation = "scf", pseudo_dir = "{folder}", outdir = "./tmp",',
        "  verbosity = 'high', tprnfor = .true., tstress = .true.",
        "/",
        "&system",
        f"  ibrav = 0, celldm(1) = {alat:.15f}, nat = {len(symbols)}, "
        f"ntyp = {len(kinds)}, ecutwfc = {cutoff / RY:.10f},",
        f"  {occupations}, nosym = .true., noinv = .true. {spin} {extra}",
        "/",
        "&electrons",
        f"  conv_thr = {conv_thr}, mixing_beta = {beta}, electron_maxstep = {maxstep}",
        "/",
        "ATOMIC_SPECIES",
        *[
            f"{label} 1.0 {Path(files[el]).name}"
            for label, (el, _) in zip(labels, kinds)
        ],
        "CELL_PARAMETERS alat",
        *[" ".join(f"{x / alat:.15f}" for x in row) for row in cell],
        "ATOMIC_POSITIONS crystal",
        *[
            f"{labels[kinds.index((s, m))]} " + " ".join(f"{x:.15f}" for x in p)
            for s, m, p in zip(symbols, moments, positions)
        ],
        "K_POINTS tpiba",
        str(len(cart)),
        *[
            " ".join(f"{x:.15f}" for x in k) + f" {w:.15f}"
            for k, w in zip(cart, weights)
        ],
    ]
    out, xml = _run("\n".join(lines) + "\n", folder)
    return parse(out, xml, len(cart))


def parse(out: str, xml: str, nk: int) -> dict:
    """The energies (hartree, full precision from the XML data file), the Fermi
    level, the eigenvalues and occupations (K, bands), the plane waves of each
    k-point and the density grid."""
    root = ElementTree.fromstring(xml)
    energy = root.find("output/total_energy")

    def get(tag):
        node = energy.find(tag)
        return None if node is None else float(node.text)

    result = {
        "energy": get("etot"),
        "band": get("eband"),
        "hartree": get("ehart"),
        "xc": get("etxc"),
        "ewald": get("ewald"),
        "entropy": get("demet"),
        "hxc_density": None,
    }
    # the one-electron energy as QE prints it, the rest of the total
    rest = [result[k] or 0.0 for k in ("hartree", "xc", "ewald", "entropy")]
    result["one_electron"] = result["energy"] - sum(rest)
    bs = root.find("output/band_structure")
    level = bs.find("fermi_energy")
    if level is None:
        level = bs.find("highestOccupiedLevel")
    result["fermi"] = float(level.text)
    eigen, occ, pws = [], [], []
    for ks in bs.findall("ks_energies"):
        eigen.append([float(x) for x in ks.find("eigenvalues").text.split()])
        occ.append([float(x) for x in ks.find("occupations").text.split()])
        pws.append(int(ks.find("npw").text))
    eigen, occ = np.array(eigen[:nk]), np.array(occ[:nk])
    if bs.find("lsda").text.strip() == "true":
        # up then down at each k-point
        half = eigen.shape[1] // 2
        eigen = np.stack([eigen[:, :half], eigen[:, half:]])
        occ = np.stack([occ[:, :half], occ[:, half:]])
        magnet = root.find("output/magnetization")
        result["magnetization"] = float(magnet.find("total").text)
        result["absolute_magnetization"] = float(magnet.find("absolute").text)
    stress = root.find("output/stress")
    if stress is not None:  # QE's sign: -(1/Omega) dE/d epsilon
        result["stress"] = np.array([float(x) for x in stress.text.split()]).reshape(
            3, 3
        )
    forces = root.find("output/forces")
    if forces is not None:
        result["forces"] = np.array([float(x) for x in forces.text.split()]).reshape(
            -1, 3
        )
    result["eigenvalues"] = eigen
    result["occupations"] = occ
    result["plane_waves"] = pws[:nk]
    grid = re.search(
        r"Dense\s+grid:\s+(\d+) G-vectors\s+FFT dimensions: \(\s*(\d+),\s*(\d+),\s*(\d+)\)",
        out,
    )
    result["dense_g"] = int(grid.group(1))
    result["grid"] = tuple(int(grid.group(i)) for i in (2, 3, 4))
    smooth = re.search(
        r"Smooth grid:\s+(\d+) G-vectors\s+FFT dimensions: \(\s*(\d+),\s*(\d+),\s*(\d+)\)",
        out,
    )
    result["smooth_grid"] = (
        result["grid"]
        if smooth is None
        else tuple(int(smooth.group(i)) for i in (2, 3, 4))
    )
    # PAW: the one-center energy and its parts as QE prints them, rydberg
    found = re.findall(r"one-center paw contrib\.\s*=\s*(\S+)\s*Ry", out)
    if found:
        result["onecenter"] = float(found[-1]) * RY
        for key, label in (("hartree_ae", "PAW hartree energy AE"), ("hartree_ps", "PAW hartree energy PS"),
                           ("xc_ae", "PAW xc energy AE"), ("xc_ps", "PAW xc energy PS")):  # fmt: skip
            part = re.findall(rf"-> {label}\s*=\s*(\S+)\s*Ry", out)
            if part:
                result["paw_" + key] = float(part[-1]) * RY
    result["text"] = out
    return result


def _pw_g(rs, a, a1, b1, b2, b3, b4):
    q = 2 * a * (b1 * np.sqrt(rs) + b2 * rs + b3 * rs**1.5 + b4 * rs * rs)
    return -2 * a * (1 + a1 * rs) * np.log1p(1 / q)


def rounded_gamma(result, cell, positions, kinds, pseudos, density_cutoff) -> float:
    """E_c(gamma = 0.031091) - E_c(gamma = (1 - ln 2) / pi^2) of PBE on the
    PW92 correlation of the paper, on the spin densities of our result with
    the core charge. QE's spin-polarized PBE takes the rounded gamma and its
    paramagnetic PBE the exact one; the energy, stationary in the density,
    moves by this difference to first order."""
    import math

    from mdapy_dft_cuda12 import _dft

    from mdapy.calculators._dft import scf

    grid = result["grid"]
    miller, _, _, _ = _dft.sphere(cell, [0.0, 0.0, 0.0], density_cutoff, grid)
    _, core_g, _, _ = scf.radial_parts(
        np.asarray(cell), np.asarray(positions, dtype=float), kinds, pseudos, miller
    )
    up, down = result["density"]
    n = up + down + (0.0 if core_g is None else scf.on_grid(core_g, miller, grid))
    m = up - down
    # |grad n|^2 from the components of n on the sphere, as the GPU's
    index = np.mod(miller, grid)
    components = np.fft.fftn(n)[index[:, 0], index[:, 1], index[:, 2]] / n.size
    g = miller @ (2 * math.pi * np.linalg.inv(cell).T)
    grad2 = np.zeros(grid)
    for j in range(3):
        box = np.zeros(grid, dtype=complex)
        box[index[:, 0], index[:, 1], index[:, 2]] = 1j * g[:, j] * components
        grad2 += (np.fft.ifftn(box).real * n.size) ** 2
    keep = n > 1e-10
    volume = abs(np.linalg.det(cell))
    return float(
        gamma_difference(n[keep], m[keep], grad2[keep]).sum() * volume / up.size
    )


def gamma_difference(n, m, grad2):
    """n (H(gamma = 0.031091) - H(gamma = (1 - ln 2) / pi^2)) of PBE at points
    of density n > 0, magnetization m and |grad n|^2, with the PW92
    correlation of the paper."""
    import math

    zeta = np.clip(m / n, -1 + 1e-12, 1 - 1e-12)
    rs = (3 / (4 * math.pi * n)) ** (1 / 3)
    e0 = _pw_g(rs, 0.031091, 0.21370, 7.5957, 3.5876, 1.6382, 0.49294)
    e1 = _pw_g(rs, 0.015545, 0.20548, 14.1189, 6.1977, 3.3662, 0.62517)
    alpha = -_pw_g(rs, 0.016887, 0.11125, 10.357, 3.6231, 0.88026, 0.49671)
    fz = ((1 + zeta) ** (4 / 3) + (1 - zeta) ** (4 / 3) - 2) / (2 ** (4 / 3) - 2)
    ec = e0 + alpha * fz / 1.709921 * (1 - zeta**4) + (e1 - e0) * fz * zeta**4
    phi3 = (0.5 * ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3))) ** 3
    kf = (3 * math.pi**2 * n) ** (1 / 3)
    t2 = grad2 / (4 * phi3 ** (2 / 3) * (4 * kf / math.pi) * n * n)
    beta = 0.06672455060314922

    def h(gamma):
        a = beta / gamma / (np.exp(-ec / (gamma * phi3)) - 1)
        at2 = a * t2
        return (
            gamma
            * phi3
            * np.log1p(beta / gamma * t2 * (1 + at2) / (1 + at2 + at2 * at2))
        )

    exact = (1 - math.log(2)) / math.pi**2
    return n * (h(0.031091) - h(exact))
