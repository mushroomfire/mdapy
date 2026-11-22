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
        sigma=0.0, conv_thr=1e-12, extra=""):  # fmt: skip
    """pw.x of a cell (rows, bohr), fractional positions, symbols and their UPF
    files (by element), a cutoff of the bands in hartree, k-points (fractions
    of the reciprocal vectors) and weights; sigma in hartree. A dict of the
    energies and the eigenvalues (K, bands)."""
    folder = str(Path(next(iter(files.values()))).parent)
    occupations = (
        'occupations = "fixed"'
        if smearing == "fixed"
        else f'occupations = "smearing", smearing = "{_SMEARING[smearing]}", '
        f"degauss = {sigma / RY:.12f}"
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
        f"ntyp = {len(files)}, ecutwfc = {cutoff / RY:.10f},",
        f"  {occupations}, nosym = .true., noinv = .true. {extra}",
        "/",
        "&electrons",
        f"  conv_thr = {conv_thr}, mixing_beta = 0.7",
        "/",
        "ATOMIC_SPECIES",
        *[f"{el} 1.0 {Path(f).name}" for el, f in files.items()],
        "CELL_PARAMETERS alat",
        *[" ".join(f"{x / alat:.15f}" for x in row) for row in cell],
        "ATOMIC_POSITIONS crystal",
        *[
            f"{s} " + " ".join(f"{x:.15f}" for x in p)
            for s, p in zip(symbols, positions)
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
    result["eigenvalues"] = np.array(eigen[:nk])
    result["occupations"] = np.array(occ[:nk])
    result["plane_waves"] = pws[:nk]
    grid = re.search(
        r"Dense\s+grid:\s+(\d+) G-vectors\s+FFT dimensions: \(\s*(\d+),\s*(\d+),\s*(\d+)\)",
        out,
    )
    result["dense_g"] = int(grid.group(1))
    result["grid"] = tuple(int(grid.group(i)) for i in (2, 3, 4))
    result["text"] = out
    return result
