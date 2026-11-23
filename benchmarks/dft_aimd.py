# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Ab initio molecular dynamics: MDAPY's dynamics with its DFT, each step from
the bands and the density of the last (restart), against Quantum ESPRESSO's
pw.x (calculation "md"), on the cases and inputs of dft_performance.py: 20
steps of NVE of 1 fs from 300 K, the forces only, a tolerance of 1e-6 eV
(conv_thr 7.35e-8 Ry). pw.x with its default extrapolation (the density of the
free atoms moved with them, the bands of the last step) and with second-order
extrapolation of the density and the bands.

    python benchmarks/dft_aimd.py prepare FOLDER
    python benchmarks/dft_aimd.py mdapy FOLDER CASE LABEL
    python benchmarks/dft_aimd.py qe FOLDER CASE OUTPUT LABEL
    python benchmarks/dft_aimd.py report FOLDER

FOLDER is that of dft_performance.py, prepared by it; prepare writes
FOLDER/qe/CASE-md.in and CASE-md2.in from its inputs of pw.x; mdapy runs a
case on the GPU into FOLDER/aimd/CASE-LABEL.json; qe reads an output of pw.x
into the same; report prints the table (markdown).
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from dft_performance import CASES, HARTREE, structure  # noqa: E402

STEPS = 20
TIMESTEP = 1.0  # fs
TEMPERATURE = 300.0  # K
TOLERANCE = 1e-6  # eV
RY_TIME = 4.8377687e-2  # fs, Rydberg atomic unit of time
MASSES = {"Si": 28.0855, "W": 183.84, "Fe": 55.845}  # u, as MDAPY's


def prepare(folder: Path) -> None:
    for name in CASES:
        source = folder / "qe" / (f"{name}-tf.in" if name == "Fe54" else f"{name}.in")
        text = source.read_text()
        text = text.replace('calculation = "scf"', 'calculation = "md"')
        element = CASES[name]["element"]
        text = text.replace(f"\n{element} 1.0 ", f"\n{element} {MASSES[element]} ")
        text = text.replace("tstress = .true.", "tstress = .false.")
        # the bands of the steps before, for the extrapolation, kept in memory
        text = text.replace("disk_io = 'none'", "disk_io = 'low'")
        text = re.sub(
            r"conv_thr = [^,\n]+", f"conv_thr = {TOLERANCE / (HARTREE / 2):.4e}", text
        )
        text = text.replace(
            "&control\n",
            f"&control\n  nstep = {STEPS}, dt = {TIMESTEP / RY_TIME:.6f},\n",
        )
        for label, extrapolation in (("md", None), ("md2", "second_order")):
            ions = [f"  ion_temperature = 'initial', tempw = {TEMPERATURE}"]
            if extrapolation:
                ions.append(
                    f"  pot_extrapolation = '{extrapolation}', "
                    f"wfc_extrapolation = '{extrapolation}'"
                )
            body = text.replace(
                "ATOMIC_SPECIES", "&ions\n" + ",\n".join(ions) + "\n/\nATOMIC_SPECIES"
            )
            (folder / "qe" / f"{name}-{label}.in").write_text(body)
    print(f"prepared {folder}/qe/*-md.in, *-md2.in")


def run_mdapy(folder: Path, name: str, label: str) -> None:
    import mdapy
    from mdapy.calculators import DFT
    from mdapy.simulation import MolecularDynamics

    case = CASES[name]
    cell, frac = structure(case)
    rows = []

    class Timed(DFT):
        """The DFT, each computation's steps and time kept."""

        def compute(self, system, properties=("energy", "forces"), *, neighbors=None):
            start = time.perf_counter()
            out = super().compute(system, properties, neighbors=neighbors)
            rows.append({"seconds": time.perf_counter() - start,
                         "steps": int(self.last["iterations"]),
                         "restarted": bool(self.last["restarted"])})  # fmt: skip
            return out

    pseudo = folder / "pseudo" / case["pseudo"]
    magmoms = None if "magmom" not in case else {case["element"]: case["magmom"]}
    density = case.get("density_cutoff")
    dft = Timed(case["cutoff"] * HARTREE,
                ecutrho=None if density is None else density * HARTREE,
                pseudopotentials={case["element"]: pseudo}, kpoints=case["kgrid"],
                smearing=case["smearing"],
                sigma=case["sigma"] * HARTREE, magmoms=magmoms, functional="pbe_pw",
                tolerance=TOLERANCE, restart=True)  # fmt: skip
    system = mdapy.System(
        frac @ cell, mdapy.Box(cell), elements=[case["element"]] * len(frac)
    )
    start = time.perf_counter()
    with MolecularDynamics(system, dft, timestep=TIMESTEP / 1000) as md:
        md.velocities(TEMPERATURE, seed=1)
        md.run(STEPS, ensemble="nve", thermo=STEPS)
    seconds = time.perf_counter() - start
    out = {
        "case": name,
        "code": "MDAPY",
        "label": label,
        "seconds": seconds,
        "steps": rows,
    }
    save(folder, out)


def read_qe(folder: Path, name: str, output: Path, label: str) -> None:
    text = output.read_text()
    steps = [
        int(n) for n in re.findall(r"convergence has been achieved in\s+(\d+)", text)
    ]
    end = re.search(r"PWSCF\s*:.*CPU\s+(.*?)\s*WALL", text)
    seconds = sum(float(a) * {"h": 3600, "m": 60, "s": 1}[u]
                  for a, u in re.findall(r"([\d.]+)([hms])", end.group(1).replace(" ", "")))  # fmt: skip
    out = {"case": name, "code": "Quantum ESPRESSO", "label": label, "seconds": seconds,
           "steps": [{"steps": n} for n in steps]}  # fmt: skip
    save(folder, out)


def save(folder: Path, out: dict) -> None:
    path = folder / "aimd" / f"{out['case']}-{out['label']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "steps"}),
          [r["steps"] for r in out["steps"]], flush=True)  # fmt: skip


def report(folder: Path) -> None:
    rows = [json.loads(p.read_text()) for p in sorted((folder / "aimd").glob("*.json"))]
    print(
        "| Case | Code | Label | SCF steps, first | SCF steps per MD step, mean of the rest | Wall time (s) | Per MD step (s) |"
    )
    print("|---|---|---|---:|---:|---:|---:|")
    for name in CASES:
        for r in sorted(
            (r for r in rows if r["case"] == name), key=lambda r: r["seconds"]
        ):
            n = [s["steps"] for s in r["steps"]]
            print(f"| {name} | {r['code']} | {r['label']} | {n[0]} | {np.mean(n[1:]):.1f} | "
                  f"{r['seconds']:.0f} | {r['seconds'] / len(n):.1f} |")  # fmt: skip


if __name__ == "__main__":
    command, folder = sys.argv[1], Path(sys.argv[2])
    if command == "prepare":
        prepare(folder)
    elif command == "mdapy":
        run_mdapy(folder, sys.argv[3], sys.argv[4])
    elif command == "qe":
        read_qe(folder, sys.argv[3], Path(sys.argv[4]), sys.argv[5])
    elif command == "report":
        report(folder)
    else:
        raise SystemExit(__doc__)
