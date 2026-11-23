# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The magnetic elements of the Delta project (Lejaeghere et al., Science 351,
aad3000 (2016)): O, Cr and Mn antiferromagnetic, Fe, Co and Ni ferromagnetic, in
the structures of its CIF files (Mn in its simpler fcc one, antiferromagnetic
from (001) plane to plane; O between its two molecules; Cr from corner to
centre). The equation of state of each from 94 % to 106 % of the volume of
its CIF, with the protocol of benchmarks/dft_verification.py (Fermi-Dirac of
0.0045 Ry, a mesh of at most 0.06 1/Å at the smallest volume, PBE as Quantum
ESPRESSO's), the moments at every volume, against WIEN2k, the reference of the
project; and the inputs of pw.x for the same volumes, to compare the codes on
the same input.

    python benchmarks/dft_magnetic.py prepare FOLDER
    python benchmarks/dft_magnetic.py run FOLDER EL [EL ...] [--library sssp] [--hint normal]
    python benchmarks/dft_magnetic.py qe FOLDER EL [--library sssp] [--hint normal]
    python benchmarks/dft_magnetic.py readqe FOLDER EL [--library sssp] [--hint normal]
    python benchmarks/dft_magnetic.py report FOLDER [--library sssp] [--hint normal]

FOLDER is that of dft_verification.py, prepared by it (pseudo-dojo/ and
sssp/); prepare adds delta/, the CIF files and WIEN2k.txt of the Delta project
(Materials Cloud Archive, doi:10.24435/materialscloud:5e-mv); run writes
FOLDER/magnetic-TAG/EL.json, TAG dojo-high, dojo-normal or sssp; qe writes the
inputs of pw.x to FOLDER/magnetic-TAG/qe/EL-i.in, i the volume, and readqe
reads their outputs EL-i.out beside them into EL-qe.json; report prints the
table (markdown).
"""

import io
import json
import math
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dft_verification import (  # noqa: E402
    BOHR,
    HARTREE,
    SCALES,
    SMEARING,
    _option,
    birch_murnaghan,
    fit,
    mesh_of,
    nu,
)

DELTA = (
    "https://archive.materialscloud.org/records/gekn5-r6h66/files/"
    "Delta%20project%20web%20site%20archive.zip?download=1"
)
ELEMENTS = ("O", "Cr", "Mn", "Fe", "Co", "Ni")
ORDER = {"O": "AFM", "Cr": "AFM", "Mn": "AFM", "Fe": "FM", "Co": "FM", "Ni": "FM"}
START = {
    "O": 1.0,
    "Cr": 1.5,
    "Mn": 2.5,
    "Fe": 2.5,
    "Co": 1.7,
    "Ni": 0.7,
}  # Bohr magnetons


def prepare(folder: Path) -> None:
    target = folder / "delta"
    target.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DELTA, timeout=600) as response:
        outer = zipfile.ZipFile(io.BytesIO(response.read()))
    inner = zipfile.ZipFile(io.BytesIO(outer.read("Delta_v3-1_0.zip")))
    (target / "WIEN2k.txt").write_bytes(inner.read("WIEN2k.txt"))
    with tarfile.open(fileobj=io.BytesIO(inner.read("CIFs.tar.gz"))) as t:
        for member in t.getmembers():
            name = Path(member.name).name
            if member.isfile() and name[:-4] in ELEMENTS and name.endswith(".cif"):
                (target / name).write_bytes(t.extractfile(member).read())
    print(f"prepared {target}: {sorted(p.name for p in target.iterdir())}")


def read_cif(path: Path):
    """The cell (rows, Å, a along x, b in the xy plane), the symbols and the
    fractional positions of a CIF of the Delta project (P 1)."""
    text = path.read_text()

    def number(key):
        return float(re.search(rf"{key}\s+([\d.]+)", text).group(1))

    a, b, c = (number(f"_cell_length_{x}") for x in "abc")
    al, be, ga = (
        math.radians(number(f"_cell_angle_{x}")) for x in ("alpha", "beta", "gamma")
    )
    cx = math.cos(be)
    cy = (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cell = np.array([[a, 0.0, 0.0], [b * math.cos(ga), b * math.sin(ga), 0.0],
                     [c * cx, c * cy, c * math.sqrt(1.0 - cx * cx - cy * cy)]])  # fmt: skip
    rows = text.split("_atom_site_type_symbol", 1)[1].split("\n")
    sites = [r.split() for r in rows if len(r.split()) >= 8]
    frac = np.array([[float(x) for x in s[2:5]] for s in sites])
    return cell, [s[-1] for s in sites], frac


def moments_of(element: str, frac: np.ndarray) -> list[float]:
    """The first moments of the magnetic order of the project."""
    m = START[element]
    if element == "Cr":  # the corner up, the centre down
        return [m if np.allclose(f, 0.0) else -m for f in frac]
    if element == "Mn":  # (001) planes, z = 0 up, z = 1/2 down
        return [m if min(f[2], 1.0 - f[2]) < 0.25 else -m for f in frac]
    if element == "O":  # the molecule at y = 0 up, that at y = 1/2 down
        return [m if min(f[1], 1.0 - f[1]) < 0.25 else -m for f in frac]
    return [m] * len(frac)


def tag_of(library: str, hint: str) -> str:
    return "sssp" if library == "sssp" else f"dojo-{hint}"


def setup(folder: Path, element: str, library: str, hint: str):
    """The cell, symbols, fractions, moments, pseudopotential file, cutoff and
    density cutoff (hartree, None for 4 times) and mesh of an element."""
    cell, symbols, frac = read_cif(folder / "delta" / f"{element}.cif")
    if library == "sssp":
        sssp = json.loads(
            (folder / "sssp" / "SSSP_1.3.0_PBE_precision.json").read_text()
        )
        path = folder / "sssp" / sssp[element]["filename"]
        cutoff, density = (
            sssp[element]["cutoff_wfc"] / 2,
            sssp[element]["cutoff_rho"] / 2,
        )
    else:
        hints = json.loads((folder / "pseudo-dojo" / "standard.djson").read_text())
        path = folder / "pseudo-dojo" / f"{element}.upf"
        cutoff = float(hints["pseudos_metadata"][element]["hints"][hint]["ecut"])
        density = None
    mesh = mesh_of(cell * SCALES[0] ** (1 / 3))
    return cell, symbols, frac, moments_of(element, frac), path, cutoff, density, mesh


def run(
    folder: Path, element: str, library: str = "dojo", hint: str = "normal"
) -> None:
    from mdapy.calculators._dft import scf
    from mdapy.calculators._dft.upf import read_upf

    cell, symbols, frac, moments, path, cutoff, density, mesh = setup(
        folder, element, library, hint
    )
    pp = read_upf(path)
    n = len(symbols)
    out = {"element": element, "order": ORDER[element], "atoms": n, "moments_start": moments,
           "pseudopotential": path.name, "cutoff_hartree": cutoff, "density_cutoff_hartree": density,
           "mesh": mesh, "volumes": [], "free_energies_ev": [], "magnetization": [],
           "absolute_magnetization": [], "steps": [], "seconds": []}  # fmt: skip
    volume = abs(np.linalg.det(cell))
    for scale in SCALES:
        start = time.time()
        r = scf.run(
            cell * scale ** (1 / 3) / BOHR, frac, [0] * n, [pp], cutoff, kgrid=mesh,
            smearing="fermi-dirac", sigma=SMEARING, functional="pbe_pw", density_cutoff=density,
            magmoms=moments, beta=0.4, magnetic_beta=1.0, energy_tolerance=1e-9 * n,
            density_tolerance=1e-11, max_iterations=300, stress=False,
        )  # fmt: skip
        if not r["converged"]:
            raise RuntimeError(f"{element} at {scale}: no convergence")
        out["volumes"].append(volume * scale / n)
        out["free_energies_ev"].append(r["energy"] * HARTREE / n)
        out["magnetization"].append(float(r["magnetization"]) / n)
        out["absolute_magnetization"].append(float(r["absolute_magnetization"]) / n)
        out["steps"].append(int(r["iterations"]))
        out["seconds"].append(time.time() - start)
        print(f"{element} {scale:.2f}: {out['free_energies_ev'][-1]:.8f} eV/atom, "
              f"|m| {out['absolute_magnetization'][-1]:.4f}, {r['iterations']} steps, "
              f"{time.time() - start:.1f} s", flush=True)  # fmt: skip
    target = folder / f"magnetic-{tag_of(library, hint)}" / f"{element}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1))


def valence_of(path: Path) -> float:
    """z_valence of UPF 2, or the header line of UPF 1."""
    text = path.read_text(errors="replace")
    m = re.search(r'z_valence\s*=\s*"\s*([-+.\dEe]+)', text) or re.search(
        r"([-+.\dEe]+)\s+Z valence", text
    )
    return float(m.group(1))


def qe_inputs(
    folder: Path, element: str, library: str = "dojo", hint: str = "normal"
) -> None:
    cell, symbols, frac, moments, path, cutoff, density, mesh = setup(
        folder, element, library, hint
    )
    valence = valence_of(path)
    signs = sorted(set(moments), reverse=True)
    labels = [f"{element}{i + 1}" for i in range(len(signs))]
    target = folder / f"magnetic-{tag_of(library, hint)}" / "qe"
    target.mkdir(parents=True, exist_ok=True)
    for i, scale in enumerate(SCALES):
        rows = cell * scale ** (1 / 3)
        lines = [
            "&control", f"  calculation = 'scf', pseudo_dir = '{path.parent.resolve()}', outdir = './tmp',",
            "  disk_io = 'none'", "/", "&system",
            f"  ibrav = 0, nat = {len(symbols)}, ntyp = {len(signs)}, ecutwfc = {2 * cutoff},"
            f" ecutrho = {2 * (density or 4 * cutoff)}, input_dft = 'PBE',",
            f"  occupations = 'smearing', smearing = 'fd', degauss = {2 * SMEARING}, nspin = 2,",
            "  " + ", ".join(f"starting_magnetization({j + 1}) = {m / valence:.6f}" for j, m in enumerate(signs)),
            "/", "&electrons", "  conv_thr = 1e-11, mixing_beta = 0.3, electron_maxstep = 300", "/",
            "ATOMIC_SPECIES", *[f"{label} 1.0 {path.name}" for label in labels],
            "CELL_PARAMETERS angstrom", *[" ".join(f"{x:.12f}" for x in row) for row in rows],
            "ATOMIC_POSITIONS crystal",
            *[f"{labels[signs.index(m)]} " + " ".join(f"{x:.12f}" for x in f) for m, f in zip(moments, frac)],
            "K_POINTS automatic", " ".join(str(k) for k in mesh) + " 0 0 0", "",
        ]  # fmt: skip
        (target / f"{element}-{i}.in").write_text("\n".join(lines))
    print(f"wrote {target}/{element}-*.in")


def read_qe(
    folder: Path, element: str, library: str = "dojo", hint: str = "normal"
) -> None:
    cell, symbols, *_ = setup(folder, element, library, hint)
    n = len(symbols)
    base = folder / f"magnetic-{tag_of(library, hint)}"
    out = {
        "element": element,
        "volumes": [],
        "free_energies_ev": [],
        "absolute_magnetization": [],
    }
    for i, scale in enumerate(SCALES):
        text = (base / "qe" / f"{element}-{i}.out").read_text()
        if "convergence has been achieved" not in text:
            raise RuntimeError(f"pw.x did not converge for {element}-{i}")
        energy = (
            float(re.findall(r"!\s+total energy\s+=\s+(\S+)\s+Ry", text)[-1])
            * HARTREE
            / 2
        )
        absolute = float(re.findall(r"absolute magnetization\s+=\s+(\S+)", text)[-1])
        out["volumes"].append(abs(np.linalg.det(cell)) * scale / n)
        out["free_energies_ev"].append(energy / n)
        out["absolute_magnetization"].append(absolute / n)
    (base / f"{element}-qe.json").write_text(json.dumps(out, indent=1))


def delta(a, b) -> float:
    """The Delta of the project, meV/atom: the root of the mean squared
    difference of two equations of state, each from its minimum, over 94 % to
    106 % of their mean V0."""
    low, high = 0.94 * (a[0] + b[0]) / 2, 1.06 * (a[0] + b[0]) / 2
    t, w = np.polynomial.legendre.leggauss(400)
    v = low + (high - low) * (t + 1) / 2
    d = birch_murnaghan(v, *a) - birch_murnaghan(v, *b)
    return 1e3 * math.sqrt(float((w / 2 * d * d).sum()))


def report(folder: Path, library: str = "dojo", hint: str = "normal") -> None:
    wien = {}
    for line in (folder / "delta" / "WIEN2k.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] in ELEMENTS:  # V0 (Å^3/atom), B0 (GPa), B1
            wien[parts[0]] = (
                float(parts[1]),
                float(parts[2]) / 160.21766208,
                float(parts[3]),
            )
    base = folder / f"magnetic-{tag_of(library, hint)}"
    print(
        "| Element | Order | V0 (Å³) | WIEN2k V0 | B0 (GPa) | WIEN2k B0 | B1 | WIEN2k B1 "
        "| Δ (meV/atom) | ν | \\|m\\| at V0 (μB) | ν against pw.x | Δ against pw.x |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for element in ELEMENTS:
        path = base / f"{element}.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        v0, _, b0, b1 = fit(r["volumes"], r["free_energies_ev"])
        ours, ref = (v0, b0, b1), wien[element]
        moment = float(np.interp(v0, r["volumes"], r["absolute_magnetization"]))
        versus = "— | — |"
        qe = base / f"{element}-qe.json"
        if qe.exists():
            q = json.loads(qe.read_text())
            qv0, _, qb0, qb1 = fit(q["volumes"], q["free_energies_ev"])
            versus = f"{nu(ours, (qv0, qb0, qb1)):.4f} | {delta(ours, (qv0, qb0, qb1)):.4f} |"
        print(f"| {element} | {r['order']} | {v0:.3f} | {ref[0]:.3f} | {b0 * 160.21766208:.1f} | "
              f"{ref[1] * 160.21766208:.1f} | {b1:.2f} | {ref[2]:.2f} | {delta(ours, ref):.3f} | "
              f"{nu(ours, ref):.3f} | {moment:.3f} | {versus}")  # fmt: skip


if __name__ == "__main__":
    command, folder = sys.argv[1], Path(sys.argv[2])
    rest, library = _option(sys.argv[3:], "--library")
    rest, hint = _option(rest, "--hint")
    library, hint = library or "dojo", hint or "normal"
    if command == "prepare":
        prepare(folder)
    elif command == "run":
        for element in rest:
            run(folder, element, library, hint)
    elif command == "qe":
        qe_inputs(folder, rest[0], library, hint)
    elif command == "readqe":
        read_qe(folder, rest[0], library, hint)
    elif command == "report":
        report(folder, library, hint)
    else:
        raise SystemExit(__doc__)
