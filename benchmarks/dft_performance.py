# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The time of MDAPY's plane-wave DFT against Quantum ESPRESSO's pw.x on the
same inputs: the same cells (perfect crystals with every fractional
coordinate displaced by a normal draw of 0.003, seed 0, so that neither code
finds a symmetry beyond time reversal), pseudopotentials, cutoffs, k-point
meshes, smearing, number of bands and starting moments, and the PBE of
Quantum ESPRESSO ("pbe_pw"), run to the ground state with forces and stress.

    python benchmarks/dft_performance.py prepare FOLDER
    python benchmarks/dft_performance.py mdapy FOLDER CASE [LABEL]
    mpirun -np 4 python benchmarks/dft_performance.py mdapy FOLDER CASE LABEL --mpi
    python benchmarks/dft_performance.py qe FOLDER CASE OUTPUT LABEL
    python benchmarks/dft_performance.py report FOLDER
    python benchmarks/dft_performance.py figure FOLDER IMAGE LABEL=NAME ...

prepare downloads the pseudopotentials (PseudoDojo 0.4 standard for the
norm-conserving cases, pslibrary 1.0.0 from the site of Quantum ESPRESSO for
PAW) and writes the inputs of pw.x, FOLDER/qe/CASE.in (and CASE-gamma.in
with QE's real wavefunctions at Gamma where the mesh is Gamma only; for the
magnetic case, where pw.x's default mixing of 0.7 diverges, a mixing of 0.3
in CASE.in and with QE's local Thomas-Fermi screening in CASE-tf.in); mdapy
runs the case on the GPU and writes FOLDER/results/CASE-LABEL.json (with
--mpi on as many GPUs as processes, sharing the pairs of spin and k-point,
each process on the GPU of its rank); qe reads
the output of pw.x into FOLDER/results/CASE-LABEL.json; report prints the
table of the times (markdown); figure draws the wall times of the results of
the labels given, in that order, under the names given; a label may list
others after |, the first of them that a case has counts.
"""

import json
import math
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

BOHR = 0.529177210903  # Å
HARTREE = 27.211386245988  # eV
RY = 0.5  # hartree
DOJO = "https://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_upf.tgz"
PSLIBRARY = "https://pseudopotentials.quantum-espresso.org/upf_files/"

# lattice constants in Å; cutoffs in hartree (the normal hint of PseudoDojo,
# the suggested minimum of pslibrary's file); sigma in hartree
CASES = {
    "Si64": dict(element="Si", lattice="diamond", a=5.431, repeat=2, pseudo="Si.upf",
                 cutoff=18.0, kgrid=(2, 2, 2), smearing="fixed", sigma=0.0),
    "Si216": dict(element="Si", lattice="diamond", a=5.431, repeat=3, pseudo="Si.upf",
                  cutoff=18.0, kgrid=(1, 1, 1), smearing="fixed", sigma=0.0),
    "W54": dict(element="W", lattice="bcc", a=3.17, repeat=3, pseudo="W.upf",
                cutoff=37.0, kgrid=(2, 2, 2), smearing="marzari-vanderbilt", sigma=0.01),
    "Fe54": dict(element="Fe", lattice="bcc", a=2.83, repeat=3,
                 pseudo="Fe.pbe-spn-kjpaw_psl.1.0.0.UPF", cutoff=35.5, density_cutoff=248.0,
                 kgrid=(2, 2, 2), smearing="gaussian", sigma=0.01, magmom=2.5, beta=0.3),
}  # fmt: skip
QE_SMEARING = {"gaussian": "gaussian", "marzari-vanderbilt": "m-v"}
BASIS = {
    "diamond": [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]],
    "bcc": [[0, 0, 0], [.5, .5, .5]],
}  # fmt: skip
ENERGY_TOLERANCE = 1e-6 / HARTREE  # change of the energy between steps
DENSITY_TOLERANCE = 1e-7
CONV_THR = 1e-8  # QE's estimated error of the energy, Ry


def download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=600) as response:
            path.write_bytes(response.read())
    return path


def structure(case: dict):
    """The cell (rows, Å) and the fractional positions of a case."""
    n = case["repeat"]
    base = np.array(BASIS[case["lattice"]], dtype=float)
    shifts = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    frac = ((base[None] + shifts[:, None]) / n).reshape(-1, 3)
    frac = frac + np.random.default_rng(0).normal(0.0, 0.003, frac.shape)
    return np.eye(3) * case["a"] * n, frac


def bands_of(case: dict, valence: float, atoms: int) -> int:
    """MDAPY's default number of bands, given to pw.x too."""
    electrons = valence * atoms
    occupied = math.ceil(electrons / 2)
    if case["smearing"] == "fixed":
        return occupied + max(4, math.ceil(0.1 * occupied))
    return max(math.ceil(1.2 * electrons / 2), occupied + 4)


def valence_of(path: Path) -> float:
    """The valence charge in the header of a UPF file."""
    m = re.search(r'z_valence\s*=\s*"\s*([-+.\dEe]+)', path.read_text(errors="replace"))
    return float(m.group(1))


def prepare(folder: Path) -> None:
    pseudo = folder / "pseudo"
    tgz = download(DOJO, folder / "downloads" / "dojo-upf.tgz")
    wanted = {c["pseudo"] for c in CASES.values()}
    with tarfile.open(tgz) as t:
        for member in t.getmembers():
            if member.isfile() and Path(member.name).name in wanted:
                pseudo.mkdir(parents=True, exist_ok=True)
                (pseudo / Path(member.name).name).write_bytes(
                    t.extractfile(member).read()
                )
    for name in wanted:
        if name.endswith(".UPF"):
            download(PSLIBRARY + name, pseudo / name)
    (folder / "qe").mkdir(parents=True, exist_ok=True)
    for name, case in CASES.items():
        cell, frac = structure(case)
        valence = valence_of(pseudo / case["pseudo"])
        bands = bands_of(case, valence, len(frac))
        if case["smearing"] == "fixed":
            occupations = 'occupations = "fixed"'
        else:
            occupations = (
                f'occupations = "smearing", smearing = "{QE_SMEARING[case["smearing"]]}", '
                f"degauss = {case['sigma'] / RY:.10f}"
            )
        spin = ""
        if "magmom" in case:
            spin = f", nspin = 2, starting_magnetization(1) = {case['magmom'] / valence:.10f}"
        rho = case.get("density_cutoff", 4 * case["cutoff"])
        head = [
            "&control",
            '  calculation = "scf", pseudo_dir = "../pseudo", outdir = "./tmp",',
            "  tprnfor = .true., tstress = .true., disk_io = 'none'",
            "/",
            "&system",
            f"  ibrav = 0, nat = {len(frac)}, ntyp = 1, ecutwfc = {case['cutoff'] / RY:.4f}, "
            f"ecutrho = {rho / RY:.4f}, nbnd = {bands},",
            f"  {occupations}{spin}",
            "/",
            "&electrons",
            f"  conv_thr = {CONV_THR}, mixing_beta = {case.get('beta', 0.7)}",
            "/",
            "ATOMIC_SPECIES",
            f"{case['element']} 1.0 {case['pseudo']}",
            "CELL_PARAMETERS angstrom",
            *[" ".join(f"{x:.12f}" for x in row) for row in cell],
            "ATOMIC_POSITIONS crystal",
            *[f"{case['element']} " + " ".join(f"{x:.12f}" for x in p) for p in frac],
        ]
        mesh = " ".join(str(k) for k in case["kgrid"])
        (folder / "qe" / f"{name}.in").write_text(
            "\n".join(head + ["K_POINTS automatic", f"{mesh} 0 0 0"]) + "\n"
        )
        if "magmom" in case:
            text = "\n".join(head + ["K_POINTS automatic", f"{mesh} 0 0 0"]) + "\n"
            (folder / "qe" / f"{name}-tf.in").write_text(
                text.replace("mixing_beta =", "mixing_mode = 'local-TF', mixing_beta =")
            )
        if case["kgrid"] == (1, 1, 1):
            (folder / "qe" / f"{name}-gamma.in").write_text(
                "\n".join(head + ["K_POINTS gamma"]) + "\n"
            )
    print(f"prepared {folder}")


def run_mdapy(folder: Path, name: str, label: str, mpi: bool = False) -> None:
    from mdapy.calculators._dft import scf
    from mdapy.calculators._dft.upf import read_upf

    comm, rank, size = None, 0, 1
    if mpi:
        from mpi4py import MPI

        from mdapy.calculators._dft import loader

        _dft = loader.module()

        comm, rank, size = (
            MPI.COMM_WORLD,
            MPI.COMM_WORLD.Get_rank(),
            MPI.COMM_WORLD.Get_size(),
        )
        _dft.set_device(rank % _dft.device_count())
        comm.Barrier()

    case = CASES[name]
    cell, frac = structure(case)
    pp = read_upf(folder / "pseudo" / case["pseudo"])
    magmoms = None if "magmom" not in case else [case["magmom"]] * len(frac)
    start = time.perf_counter()
    r = scf.run(
        cell / BOHR, frac, [0] * len(frac), [pp], case["cutoff"], kgrid=case["kgrid"],
        smearing=case["smearing"], sigma=case["sigma"], magmoms=magmoms, functional="pbe_pw",
        density_cutoff=case.get("density_cutoff"), energy_tolerance=ENERGY_TOLERANCE,
        density_tolerance=DENSITY_TOLERANCE, max_iterations=200, comm=comm,
    )  # fmt: skip
    seconds = time.perf_counter() - start
    if rank > 0:
        return
    import subprocess

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
        capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip
    out = {
        "case": name, "code": "MDAPY", "label": label, "hardware": gpu if size == 1 else f"{size} x {gpu}", "gpus": size,
        "seconds": seconds, "steps": int(r["iterations"]), "converged": bool(r["converged"]),
        "energy_ev": r["energy"] * HARTREE, "atoms": len(frac), "gamma": bool(r.get("gamma", False)),
        "bands": int(bands_of(case, pp.valence, len(frac))), "grid": list(r["grid"]),
        "timings": {k: round(v, 3) for k, v in r["timings"].items()},
        "terms_ev": {k: float(r[k]) * HARTREE for k in ("one_electron", "hartree", "xc", "ewald", "entropy")},
        "fermi_ev": float(r["fermi"]) * HARTREE, "magnetization": float(r.get("magnetization", 0.0)),
    }  # fmt: skip
    save(folder, out)


def read_qe(folder: Path, name: str, output: Path, label: str) -> None:
    text = output.read_text()
    if "convergence has been achieved" not in text:
        raise RuntimeError(f"{output}: pw.x did not converge")

    def seconds(field: str) -> float:
        # QE prints e.g. 1h 2m 3.45s, 6m19.74s or 12.34s before WALL
        m = re.search(rf"{field}\s*:.*CPU\s+(.*?)\s*WALL", text)
        value, total = m.group(1).replace(" ", ""), 0.0
        for amount, unit in re.findall(r"([\d.]+)([hms])", value):
            total += float(amount) * {"h": 3600, "m": 60, "s": 1}[unit]
        return total

    energy = float(re.findall(r"!\s+total energy\s+=\s+(\S+)\s+Ry", text)[-1])
    processes = re.search(r"Number of MPI processes:\s+(\d+)", text)
    threads = re.search(r"Threads/MPI process:\s+(\d+)", text)
    pools = re.search(r"K-points division:\s+npool\s+=\s+(\d+)", text)
    gpu = "GPU acceleration is ACTIVE" in text
    out = {
        "case": name, "code": "Quantum ESPRESSO", "label": label,
        "version": re.search(r"Program PWSCF v\.(\S+)", text).group(1),
        "hardware": label, "gpu": gpu, "seconds": seconds("PWSCF"),
        "electrons_seconds": seconds("electrons"),
        "steps": int(re.search(r"convergence has been achieved in\s+(\d+)", text).group(1)),
        "converged": True, "energy_ev": energy * RY * HARTREE,
        "processes": int(processes.group(1)) if processes else 1,
        "threads": int(threads.group(1)) if threads else 1,
        "pools": int(pools.group(1)) if pools else 1,
    }  # fmt: skip
    save(folder, out)


def save(folder: Path, out: dict) -> None:
    path = folder / "results" / f"{out['case']}-{out['label']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def report(folder: Path) -> None:
    rows = [
        json.loads(p.read_text()) for p in sorted((folder / "results").glob("*.json"))
    ]
    print(
        "| Case | Code | Hardware | Steps | Wall time (s) | Per step (s) | ΔE (meV/atom) |"
    )
    print("|---|---|---|---:|---:|---:|---:|")
    for name in CASES:
        mine = [r for r in rows if r["case"] == name]
        ours = [r for r in mine if r["code"] == "MDAPY"]
        ref = ours[0]["energy_ev"] if ours else None
        for r in sorted(mine, key=lambda r: (r.get("gpus", 1), r["seconds"])):
            de = (
                ""
                if ref is None
                else f"{(r['energy_ev'] - ref) / r.get('atoms', ours[0]['atoms']) * 1e3:.3f}"
            )
            where = r["hardware"] + (", real bands at Gamma" if r.get("gamma") else "")
            if r["code"] != "MDAPY" and not r["gpu"]:
                where = f"{r['label']} ({r['processes']} MPI × {r['threads']} threads, {r['pools']} pools)"
            print(
                f"| {name} | {r['code']} {r.get('version', '')} | {where} | {r['steps']} | "
                f"{r['seconds']:.1f} | {r['seconds'] / r['steps']:.2f} | {de} |"
            )


def figure(folder: Path, image: Path, series: list[tuple[str, str]]) -> None:
    from mdapy.visualization import plot

    rows = {
        (r["case"], r["label"]): r
        for r in (
            json.loads(p.read_text()) for p in (folder / "results").glob("*.json")
        )
    }
    cases = list(CASES)
    width = 0.8 / len(series)
    with plot.style():
        fig, ax = plot.figure("double", 70)
        for i, (label, name) in enumerate(series):
            x = [c + (i - (len(series) - 1) / 2) * width for c in range(len(cases))]
            y = []
            for case in cases:
                found = [
                    rows[(case, one)] for one in label.split("|") if (case, one) in rows
                ]
                y.append(found[0]["seconds"] if found else np.nan)
            ax.bar(x, y, width * 0.9, label=name)
        ax.set_xticks(range(len(cases)), cases)
        ax.set(yscale="log", ylabel="time to the ground state, forces and stress (s)")
        ax.legend(ncols=2, fontsize=6)
        plot.save(fig, str(image.with_suffix("")), formats=("png",))


if __name__ == "__main__":
    command, folder = sys.argv[1], Path(sys.argv[2])
    if command == "prepare":
        prepare(folder)
    elif command == "mdapy":
        mpi = "--mpi" in sys.argv
        rest = [a for a in sys.argv[3:] if a != "--mpi"]
        run_mdapy(folder, rest[0], rest[1] if len(rest) > 1 else "gpu", mpi)
    elif command == "qe":
        read_qe(folder, sys.argv[3], Path(sys.argv[4]), sys.argv[5])
    elif command == "report":
        report(folder)
    elif command == "figure":
        figure(
            folder, Path(sys.argv[3]), [tuple(a.split("=", 1)) for a in sys.argv[4:]]
        )
    else:
        raise SystemExit(__doc__)
