# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The second check of the verification with SSSP: pw.x of Quantum ESPRESSO on
the input of MDAPY's runs of dft_verification.py (the same files, cutoffs, mesh
and smearing, the seven volumes), for the structures where MDAPY and the
paper's Quantum ESPRESSO differ.

    python benchmarks/dft_verification_pwx.py inputs FOLDER El-Conf ...   # FOLDER/pwx/El-Conf/v0.in .. v6.in
    pw.x -nk 8 -in v0.in > v0.out                                         # in each, for each volume
    python benchmarks/dft_verification_pwx.py report FOLDER El-Conf ...   # nu of pw.x against MDAPY and the paper

FOLDER is that of dft_verification.py: acwf/ (the structures and the paper's
results), sssp/ (the files and SSSP_1.3.0_PBE_precision.json) and results-sssp/
(MDAPY's equations of state).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dft_verification as dv  # noqa: E402

PAPER = (
    "results-unaries-verification-PBE-v1-quantum_espresso-SSSP-1.3-PBE-precision.json"
)


def inputs(folder: Path, keys: list[str]) -> None:
    table = json.loads((folder / "sssp" / "SSSP_1.3.0_PBE_precision.json").read_text())
    for key in keys:
        element = key.split("-")[0]
        ours = json.loads((folder / "results-sssp" / f"{key}.json").read_text())
        cell, species, fractions = dv.read_xsf(folder / "acwf" / f"{key}.xsf")
        name = table[element]["filename"]
        if name != ours["pseudopotentials"][0]:
            raise ValueError(
                f"{key}: MDAPY ran {ours['pseudopotentials'][0]}, not {name}"
            )
        wfc, rho = 2 * ours["cutoff_hartree"], 2 * ours["density_cutoff_hartree"]  # Ry
        mesh = " ".join(str(n) for n in ours["mesh"])
        out = folder / "pwx" / key
        out.mkdir(parents=True, exist_ok=True)
        for i, scale in enumerate(dv.SCALES):
            c = cell * scale ** (1 / 3)
            rows = "\n".join(" ".join(f"{x:.12f}" for x in row) for row in c)
            atoms = "\n".join(
                f"{element} " + " ".join(f"{x:.12f}" for x in f) for f in fractions
            )
            (out / f"v{i}.in").write_text(
                f"""&control
  calculation = 'scf', pseudo_dir = '{(folder / "sssp").resolve()}', outdir = './tmp{i}', disk_io = 'none'
/
&system
  ibrav = 0, nat = {len(species)}, ntyp = 1, ecutwfc = {wfc:.6f}, ecutrho = {rho:.6f},
  input_dft = 'PBE', occupations = 'smearing', smearing = 'fd', degauss = {2 * dv.SMEARING}
/
&electrons
  conv_thr = {1e-11 * len(species):.1e}, mixing_beta = 0.3, electron_maxstep = 300
/
ATOMIC_SPECIES
{element} 1.0 {name}
CELL_PARAMETERS angstrom
{rows}
ATOMIC_POSITIONS crystal
{atoms}
K_POINTS automatic
{mesh} 0 0 0
"""
            )
        print(key, f"ecutwfc {wfc:g} Ry, ecutrho {rho:g} Ry, mesh {mesh}", flush=True)


def energy(path: Path) -> float | None:
    """The free energy in eV of a converged pw.x output, None otherwise."""
    text = path.read_text() if path.exists() else ""
    lines = [line for line in text.splitlines() if line.startswith("!")]
    if not lines or "convergence has been achieved" not in text:
        return None
    return float(lines[-1].split("=")[1].split()[0]) * dv.RY


def report(folder: Path, keys: list[str]) -> None:
    paper = json.loads((folder / "acwf" / PAPER).read_text())["BM_fit_data"]
    for key in keys:
        element, configuration = key.split("-")
        ours = json.loads((folder / "results-sssp" / f"{key}.json").read_text())
        e = [energy(folder / "pwx" / key / f"v{i}.out") for i in range(len(dv.SCALES))]
        if any(x is None for x in e):
            print(
                key,
                "not converged or missing:",
                [i for i, x in enumerate(e) if x is None],
            )
            continue
        v = np.asarray(ours["volumes"])
        pwx = dv.fit(v, np.asarray(e))
        mdapy = dv.fit(v, np.asarray(ours["free_energies_ev"]))
        p = paper[f"{element}-X/{configuration}"]
        published = (p["min_volume"], p["bulk_modulus_ev_ang3"], p["bulk_deriv"])
        a, b = (pwx[0], pwx[2], pwx[3]), (mdapy[0], mdapy[2], mdapy[3])
        de = (
            1e3 * (np.asarray(e) - np.asarray(ours["free_energies_ev"])) / ours["atoms"]
        )
        print(
            f"{key}: nu pw.x-MDAPY {dv.nu(a, b):.4f}, pw.x-paper {dv.nu(a, published):.4f}, "
            f"MDAPY-paper {dv.nu(b, published):.4f}; E pw.x - MDAPY {de.min():+.4f} to "
            f"{de.max():+.4f} meV/atom",
            flush=True,
        )


if __name__ == "__main__":
    command, folder, keys = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    if command == "inputs":
        inputs(folder, keys)
    elif command == "report":
        report(folder, keys)
    else:
        raise SystemExit(__doc__)
