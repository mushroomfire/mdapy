# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Magnetic metals, alloys and oxides against the all-electron full-potential
LAPW of Elk (https://elk.sourceforge.io): the equation of state and the
moments of each structure below, collinear PBE (the valence scalar
relativistic in both), Fermi-Dirac smearing of 0.0045 Ry, seven volumes from
94 % to 106 % of the central one, a Gamma-centred mesh of k-points at most
0.06 1/Å apart (2 pi included) at the smallest volume for the metals and
alloys, 0.1 1/Å for the oxides, the same in both codes, the Birch-Murnaghan
fit and Delta of the Delta project. Elk with its
very-high-quality settings (vhighq) and its own species files; MDAPY with
PseudoDojo 0.4 (a hint) or SSSP 1.3 precision, the preconditioner of the
local density of states (Kerker's lets fcc Ni of GBRV fall to m = 0).

    python benchmarks/dft_allelectron.py elk FOLDER NAME SPECIES  # FOLDER/allelectron/elk/NAME-i/elk.in
    python benchmarks/dft_allelectron.py readelk FOLDER NAME      # their INFO.OUT into NAME-elk.json
    python benchmarks/dft_allelectron.py run FOLDER NAME [--library sssp] [--hint normal]
    python benchmarks/dft_allelectron.py report FOLDER [--library sssp] [--hint normal]

FOLDER is that of dft_verification.py, prepared by it (pseudo-dojo/ and
sssp/); SPECIES the folder of Elk's species files; NAME one of STRUCTURES or
"all". The moments start as given, the same in both codes (in Elk by fields
in the muffin-tins that fall to zero during the iterations; the radii of
the muffin-tins fixed for all the volumes).
"""

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dft_magnetic import delta  # noqa: E402
from dft_verification import (  # noqa: E402
    BOHR,
    HARTREE,
    SCALES,
    SMEARING,
    _option,
    fit,
)

S3 = math.sqrt(3.0)


def bcc(a):
    return a / 2 * np.array([[-1.0, 1, 1], [1, -1, 1], [1, 1, -1]])


def fcc(a):
    return a / 2 * np.array([[0.0, 1, 1], [1, 0, 1], [1, 1, 0]])


def tetragonal(
    a, c
):  # fcc of cube edge c as a = c / sqrt 2 body centred, layers along z
    return np.diag([a, a, c])


def afm2(a):  # rock salt of cube edge a, the (111) planes of the metal alternating
    return a * np.array([[1.0, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]])


AFM2 = [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]]
L12 = [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
# name: (cell at the central volume, Å; symbols; fractions; first moments, Bohr magnetons)
STRUCTURES = {
    "Fe-bcc": (bcc(2.833), ["Fe"], [[0, 0, 0]], [2.5]),
    "Co-hcp": (np.array([[2.49, 0, 0], [-1.245, 1.245 * S3, 0], [0, 0, 4.02]]), ["Co", "Co"],
               [[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]], [1.7, 1.7]),
    "Ni-fcc": (fcc(3.52), ["Ni"], [[0, 0, 0]], [0.7]),
    "Cr-bcc-afm": (np.eye(3) * 2.87, ["Cr", "Cr"], [[0, 0, 0], [0.5, 0.5, 0.5]], [1.2, -1.2]),
    "Mn-fcc-afm": (tetragonal(3.53 / math.sqrt(2), 3.53), ["Mn", "Mn"], [[0, 0, 0], [0.5, 0.5, 0.5]], [2.5, -2.5]),
    "Fe-fcc-afm": (tetragonal(3.45 / math.sqrt(2), 3.45), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]], [1.8, -1.8]),
    "FeCo-B2": (np.eye(3) * 2.84, ["Fe", "Co"], [[0, 0, 0], [0.5, 0.5, 0.5]], [2.8, 1.8]),
    "FeNi-L10": (tetragonal(3.58 / math.sqrt(2), 3.58), ["Fe", "Ni"], [[0, 0, 0], [0.5, 0.5, 0.5]], [2.7, 0.7]),
    "Ni3Fe-L12": (np.eye(3) * 3.55, ["Fe", "Ni", "Ni", "Ni"], L12, [3.0, 0.7, 0.7, 0.7]),
    "CrCoNi2": (np.eye(3) * 3.56, ["Cr", "Co", "Ni", "Ni"], L12, [-1.0, 1.6, 0.6, 0.6]),
    "NiO-afm2": (afm2(4.19), ["Ni", "Ni", "O", "O"], AFM2, [1.6, -1.6, 0, 0]),
    "MnO-afm2": (afm2(4.45), ["Mn", "Mn", "O", "O"], AFM2, [4.5, -4.5, 0, 0]),
    "CoO-afm2": (afm2(4.24), ["Co", "Co", "O", "O"], AFM2, [2.6, -2.6, 0, 0]),
    "FeO-afm2": (afm2(4.31), ["Fe", "Fe", "O", "O"], AFM2, [3.5, -3.5, 0, 0]),
}  # fmt: skip


def mesh_of(name: str) -> tuple[int, int, int]:
    """The mesh of a structure, of its smallest volume."""
    cell = STRUCTURES[name][0] * SCALES[0] ** (1 / 3)
    spacing = 0.1 if name.endswith("afm2") else 0.06
    b = 2 * math.pi * np.linalg.inv(cell).T
    return tuple(
        max(1, math.ceil(round(float(np.linalg.norm(row)) / spacing, 5))) for row in b
    )


def radius_of(name: str) -> float:
    """The muffin-tin radius of every atom in Elk, bohr: 0.97 of half the
    shortest distance between atoms at the smallest volume, at most 2.4, the
    same at every volume so that the basis does not change along the curve."""
    cell, _, frac, _ = STRUCTURES[name]
    rows = cell * SCALES[0] ** (1 / 3)
    r = np.asarray(frac) @ rows
    shifts = (
        np.array(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)]
        )
        @ rows
    )
    d = np.linalg.norm(
        r[:, None, None, :] + shifts[None, :, None, :] - r[None, None, :, :], axis=-1
    )
    shortest = d[d > 1e-6].min()
    return min(0.97 * shortest / 2 / BOHR, 2.4)


def names_of(name: str) -> list[str]:
    return list(STRUCTURES) if name == "all" else [name]


def base_of(folder: Path) -> Path:
    return folder / "allelectron"


def tag_of(library: str, hint: str) -> str:
    return "sssp" if library == "sssp" else f"dojo-{hint}"


def elk_inputs(folder: Path, name: str, species: Path) -> None:
    cell, symbols, frac, moments = STRUCTURES[name]
    mesh = mesh_of(name)
    kinds = list(dict.fromkeys(symbols))
    for i, scale in enumerate(SCALES):
        rows = cell * scale ** (1 / 3) / BOHR
        # the blocks of elk.in, each ended by a blank line as Elk reads them
        blocks = [["tasks", "  0"], ["xctype", "  20"], ["spinpol", "  .true."], ["vhighq", "  .true."],
                  ["stype", "  3"], ["swidth", f"  {SMEARING:.10f}"], ["maxscl", "  400"],
                  ["epsengy", "  1.0e-7"], ["reducebf", "  0.8"],
                  ["ngridk", "  " + " ".join(str(k) for k in mesh)],
                  ["avec", *["  " + " ".join(f"{x:.12f}" for x in row) for row in rows]],
                  ["rmtall", f"  {radius_of(name):.4f}"],
                  ["sppath", f"  '{species.resolve()}/'"]]  # fmt: skip
        atoms = ["atoms", f"  {len(kinds)}"]
        for kind in kinds:
            sites = [(f, m) for s, f, m in zip(symbols, frac, moments) if s == kind]
            atoms += [f"  '{kind}.in'", f"  {len(sites)}"]
            # a field in the muffin-tin against the moment wanted
            atoms += [
                f"  {f[0]:.12f} {f[1]:.12f} {f[2]:.12f}  0.0 0.0 {-0.01 * np.sign(m):.3f}"
                for f, m in sites
            ]
        blocks.append(atoms)
        lines = [line for block in blocks for line in [*block, ""]]
        target = base_of(folder) / "elk" / f"{name}-{i}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "elk.in").write_text("\n".join(lines) + "\n")
    print(f"wrote {base_of(folder) / 'elk'}/{name}-*/elk.in, mesh {mesh}")


def read_info(text: str) -> dict:
    """The free energy (hartree), the total moment and those of the
    muffin-tins of the last loop of an INFO.OUT, which must have converged."""
    if "Convergence targets achieved" not in text:
        raise RuntimeError("Elk did not converge")
    last = text.rsplit("Self-consistent loop started", 1)[-1]
    last = last.rsplit("Loop number :", 1)[-1]
    free = re.findall(r"free energy\s*:\s*(\S+)", last, flags=re.IGNORECASE)
    total = re.findall(r"total energy\s*:\s*(\S+)", last, flags=re.IGNORECASE)
    moment = re.findall(r"total moment\s*:\s*(\S+)", last)
    mt = re.findall(
        r"atom\s+\d+\s*:\s*(\S+)",
        last.split("Moments", 1)[-1].split("total moment", 1)[0],
    )
    return {"free_energy_hartree": float((free or total)[-1]), "total_energy_hartree": float(total[-1]),
            "moment": float(moment[-1]) if moment else 0.0, "muffin_tin_moments": [float(x) for x in mt]}  # fmt: skip


def read_elk(folder: Path, name: str) -> None:
    cell, symbols, *_ = STRUCTURES[name]
    n = len(symbols)
    out = {"name": name, "code": "Elk", "volumes": [], "free_energies_ev": [], "moments": [],
           "muffin_tin_moments": []}  # fmt: skip
    for i, scale in enumerate(SCALES):
        r = read_info(
            (base_of(folder) / "elk" / f"{name}-{i}" / "INFO.OUT").read_text()
        )
        out["volumes"].append(abs(np.linalg.det(cell)) * scale / n)
        out["free_energies_ev"].append(r["free_energy_hartree"] * HARTREE / n)
        out["moments"].append(r["moment"] / n)
        out["muffin_tin_moments"].append(r["muffin_tin_moments"])
    (base_of(folder) / f"{name}-elk.json").write_text(json.dumps(out, indent=1))


def pseudopotentials(folder: Path, kinds: list[str], library: str, hint: str):
    if library == "sssp":
        sssp = json.loads(
            (folder / "sssp" / "SSSP_1.3.0_PBE_precision.json").read_text()
        )
        paths = [folder / "sssp" / sssp[k]["filename"] for k in kinds]
        return (
            paths,
            max(sssp[k]["cutoff_wfc"] for k in kinds) / 2,
            max(sssp[k]["cutoff_rho"] for k in kinds) / 2,
        )
    hints = json.loads((folder / "pseudo-dojo" / "standard.djson").read_text())[
        "pseudos_metadata"
    ]
    paths = [folder / "pseudo-dojo" / f"{k}.upf" for k in kinds]
    return paths, max(float(hints[k]["hints"][hint]["ecut"]) for k in kinds), None


def run(folder: Path, name: str, library: str = "dojo", hint: str = "normal") -> None:
    from mdapy.calculators._dft import scf
    from mdapy.calculators._dft.upf import read_upf

    cell, symbols, frac, moments = STRUCTURES[name]
    kinds = list(dict.fromkeys(symbols))
    paths, cutoff, density = pseudopotentials(folder, kinds, library, hint)
    pps = [read_upf(p) for p in paths]
    mesh = mesh_of(name)
    n = len(symbols)
    out = {"name": name, "code": "MDAPY", "library": library, "hint": hint, "mesh": mesh,
           "pseudopotentials": [p.name for p in paths], "cutoff_hartree": cutoff, "density_cutoff_hartree": density,
           "volumes": [], "free_energies_ev": [], "moments": [], "absolute_moments": [], "steps": [], "seconds": []}  # fmt: skip
    for scale in SCALES:
        start = time.time()
        r = scf.run(
            cell * scale ** (1 / 3) / BOHR, frac, [kinds.index(s) for s in symbols], pps, cutoff, kgrid=mesh,
            smearing="fermi-dirac", sigma=SMEARING, functional="pbe_pw", density_cutoff=density,
            magmoms=moments, beta=0.4, magnetic_beta=1.0, energy_tolerance=1e-9 * n,
            density_tolerance=1e-11, max_iterations=300, stress=False, preconditioner="ldos",
        )  # fmt: skip
        if not r["converged"]:
            raise RuntimeError(f"{name} at {scale}: no convergence")
        out["volumes"].append(abs(np.linalg.det(cell)) * scale / n)
        out["free_energies_ev"].append(r["energy"] * HARTREE / n)
        out["moments"].append(float(r["magnetization"]) / n)
        out["absolute_moments"].append(float(r["absolute_magnetization"]) / n)
        out["steps"].append(int(r["iterations"]))
        out["seconds"].append(time.time() - start)
        print(f"{name} {scale:.2f}: {out['free_energies_ev'][-1]:.8f} eV/atom, m {out['moments'][-1]:.4f}, "
              f"|m| {out['absolute_moments'][-1]:.4f}, {r['iterations']} steps, {time.time() - start:.1f} s",
              flush=True)  # fmt: skip
    target = base_of(folder) / f"{name}-{tag_of(library, hint)}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1))


def report(folder: Path, library: str = "dojo", hint: str = "normal") -> None:
    tag = tag_of(library, hint)
    print(
        "| Structure | V0 Elk (Å³/atom) | V0 MDAPY | B0 Elk (GPa) | B0 MDAPY | B1 Elk | B1 MDAPY "
        "| Delta (meV/atom) | m Elk (μB/atom) | m MDAPY |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in STRUCTURES:
        paths = [
            base_of(folder) / f"{name}-elk.json",
            base_of(folder) / f"{name}-{tag}.json",
        ]
        if not all(p.exists() for p in paths):
            continue
        elk, ours = (json.loads(p.read_text()) for p in paths)
        fits = []
        for d in (elk, ours):
            v0, e0, b0, b1 = fit(
                np.asarray(d["volumes"]), np.asarray(d["free_energies_ev"])
            )
            fits.append((v0, b0, b1))
        d_mev = delta(fits[0], fits[1])
        gpa = 160.21766208
        print(f"| {name} | {fits[0][0]:.3f} | {fits[1][0]:.3f} | {fits[0][1] * gpa:.1f} | {fits[1][1] * gpa:.1f} "
              f"| {fits[0][2]:.2f} | {fits[1][2]:.2f} | {d_mev:.2f} | {elk['moments'][3]:.3f} "
              f"| {ours['moments'][3]:.3f} |")  # fmt: skip


if __name__ == "__main__":
    command, folder = sys.argv[1], Path(sys.argv[2])
    rest, library = _option(sys.argv[3:], "--library")
    rest, hint = _option(rest, "--hint")
    library, hint = library or "dojo", hint or "normal"
    if command == "elk":
        for name in names_of(rest[0]):
            elk_inputs(folder, name, Path(rest[1]))
    elif command == "readelk":
        for name in names_of(rest[0]):
            read_elk(folder, name)
    elif command == "run":
        for name in names_of(rest[0]):
            run(folder, name, library, hint)
    elif command == "report":
        report(folder, library, hint)
    else:
        raise SystemExit(__doc__)
