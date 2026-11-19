# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Molecular dynamics on one NVIDIA GPU: MDAPY against LAMMPS with KOKKOS, in
mixed and in double precision, at constant energy, for each potential of
MDAPY that KOKKOS also has.

    LMP_KOKKOS_MIXED=... LMP_KOKKOS_DOUBLE=... LAMMPS_POTENTIALS=... \\
        python benchmarks/gpu.py run --output gpu.json
    python benchmarks/gpu.py table gpu.json
    python benchmarks/gpu.py plot gpu.json gpu.png

LAMMPS_POTENTIALS is the potentials folder of LAMMPS, which holds the files
of ADP, Stillinger-Weber, Tersoff and MEAM. Each run builds the crystal, gives
it the velocities of a temperature, runs 50 steps and times the next 300.
LAMMPS reports the time of its loop; MDAPY that of MolecularDynamics.run,
which records the state at its start and end. KOKKOS runs with full neighbor
lists and Newton's third law off, its default on GPUs, and with half lists and
the law on; both are kept, and a setting a style does not take is left out. A
LAMMPS not given by its variable is left out. --rest waits that many seconds
before each run, for a GPU that slows down as it warms to cool first.
"""

import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

COPPER = ("fcc", 3.615, "Cu", 63.546)
SILICON = ("diamond", 5.431, "Si", 28.0855)
# name: crystal (structure, lattice constant, element, mass), temperature K,
# time step ps, sizes (repeats of the cubic cell), the LAMMPS interaction and
# the files of LAMMPS it reads
CASES = {
    "eam": {
        "title": "Cu, EAM of Zhou et al.",
        "crystal": COPPER,
        "temperature": 600.0,
        "timestep": 0.002,
        "repeats": (20, 40, 60, 80),
        "lammps": "pair_style eam/alloy\npair_coeff * * Cu_zhou.eam.alloy Cu",
        "files": (),
    },
    "lj": {
        "title": "Ar, Lennard-Jones",
        "crystal": ("fcc", 5.26, "Ar", 39.948),
        "temperature": 60.0,
        "timestep": 0.005,
        "repeats": (16, 32, 48, 64),
        "lammps": "pair_style lj/cut 8.5\npair_coeff 1 1 0.0104 3.40",
        "files": (),
    },
    "morse": {
        "title": "Cu, Morse",
        "crystal": COPPER,
        "temperature": 600.0,
        "timestep": 0.002,
        "repeats": (20, 40, 60, 80),
        "lammps": "pair_style morse 6.5\npair_coeff 1 1 0.3429 1.3588 2.866",
        "files": (),
    },
    "adp": {
        "title": "Ni, ADP",
        "crystal": ("fcc", 3.52, "Ni", 58.6934),
        "temperature": 600.0,
        "timestep": 0.002,
        "repeats": (20, 40, 60, 80),
        "lammps": "pair_style adp\npair_coeff * * Ni.adp Ni",
        "files": ("Ni.adp",),
    },
    "sw": {
        "title": "Si, Stillinger-Weber",
        "crystal": SILICON,
        "temperature": 1000.0,
        "timestep": 0.001,
        "repeats": (16, 32, 48, 64),
        "lammps": "pair_style sw\npair_coeff * * Si.sw Si",
        "files": ("Si.sw",),
    },
    "tersoff": {
        "title": "Si, Tersoff",
        "crystal": SILICON,
        "temperature": 1000.0,
        "timestep": 0.001,
        "repeats": (16, 32, 48, 64),
        "lammps": "pair_style tersoff\npair_coeff * * Si.tersoff Si",
        "files": ("Si.tersoff",),
    },
    "meam": {
        "title": "Cu, MEAM",
        "crystal": COPPER,
        "temperature": 600.0,
        "timestep": 0.002,
        "repeats": (20, 40, 60, 80),
        "lammps": "pair_style meam\npair_coeff * * library.meam Cu Cu.meam Cu",
        "files": ("library.meam", "Cu.meam"),
    },
}
WARM, STEPS = 50, 300
KOKKOS = {
    "full": "neigh full newton off",
    "half": "neigh half newton on",
}


def lammps_input(case: dict, repeat: int) -> str:
    structure, a, _, mass = case["crystal"]
    return f"""units metal
atom_style atomic
lattice {structure} {a}
region box block 0 {repeat} 0 {repeat} 0 {repeat}
create_box 1 box
create_atoms 1 box
mass 1 {mass}
{case["lammps"]}
velocity all create {case["temperature"]} 1 dist gaussian mom yes rot no
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
fix 1 all nve
timestep {case["timestep"]}
thermo {STEPS}
run {WARM}
run {STEPS}
"""


def run_lammps(executable: str, flags: list[str], text: str, folder: Path) -> float:
    """Seconds per step of the timed run."""
    (folder / "in.run").write_text(text)
    done = subprocess.run(
        [executable, *flags, "-in", "in.run", "-log", "none", "-echo", "none"],
        cwd=folder, capture_output=True, text=True, check=True,
    )  # fmt: skip
    loops = re.findall(r"Loop time of ([0-9.eE+-]+) on .* for (\d+) steps", done.stdout)
    seconds, steps = loops[-1]
    return float(seconds) / int(steps)


def potential(name: str, folder: Path, precision: str):
    """The potential of MDAPY of a case, on the GPU."""
    from mdapy.calculators import (
        ADP,
        EAM,
        MEAM,
        LennardJones,
        Morse,
        StillingerWeber,
        Tersoff,
    )

    options = {"device": "cuda", "precision": precision}
    if name == "eam":
        return EAM.zhou(["Cu"], **options)
    if name == "lj":
        return LennardJones((0.0104, 3.40), 8.5, **options)
    if name == "morse":
        return Morse({"Cu": (0.3429, 1.3588, 2.866)}, 6.5, **options)
    if name == "adp":
        return ADP(folder / "Ni.adp", **options)
    if name == "sw":
        return StillingerWeber(folder / "Si.sw", **options)
    if name == "tersoff":
        return Tersoff(folder / "Si.tersoff", **options)
    return MEAM(folder / "library.meam", ["Cu"], folder / "Cu.meam", **options)


def run_mdapy(
    name: str, case: dict, repeat: int, precision: str, folder: Path
) -> float:
    from mdapy.build import build_crystal
    from mdapy.simulation import MolecularDynamics

    structure, a, element, _ = case["crystal"]
    system = build_crystal(structure, a, elements=element, repeat=(repeat,) * 3)
    interaction = potential(name, folder, precision)
    with MolecularDynamics(system, interaction, timestep=case["timestep"]) as md:
        md.velocities(case["temperature"], seed=1)
        md.run(WARM, thermo=WARM)
        start = time.perf_counter()
        md.run(STEPS, thermo=STEPS)
        seconds = (time.perf_counter() - start) / STEPS
    del md, interaction
    gc.collect()
    return seconds


def machine() -> dict:
    """The GPU, its driver and the processor."""
    out = {}
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()  # fmt: skip
        out["gpu"] = query[0] if query else ""
    except (OSError, subprocess.CalledProcessError):
        out["gpu"] = ""
    try:
        lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        model = re.search(r"Model name:\s*(.*)", lscpu)
        out["cpu"] = model.group(1).strip() if model else ""
    except OSError:
        out["cpu"] = ""
    return out


def run(output: Path, rest: float, names: list[str]) -> None:
    from mdapy.calculators import EAM

    programs = {
        "kokkos mixed": os.environ.get("LMP_KOKKOS_MIXED"),
        "kokkos double": os.environ.get("LMP_KOKKOS_DOUBLE"),
    }
    potentials = os.environ.get("LAMMPS_POTENTIALS")
    info = machine()
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        EAM.zhou(["Cu"]).write(folder / "Cu_zhou.eam.alloy")
        for name in names:
            case = CASES[name]
            for file in case["files"]:
                if not potentials:
                    raise SystemExit(f"{name} needs LAMMPS_POTENTIALS for {file}")
                shutil.copy(Path(potentials) / file, folder / file)
            for repeat in case["repeats"]:
                structure = case["crystal"][0]
                atoms = (4 if structure == "fcc" else 8) * repeat**3
                text = lammps_input(case, repeat)
                runs = []
                for program, executable in programs.items():
                    if not executable:
                        continue
                    for setting, options in KOKKOS.items():
                        flags = ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos",
                                 *options.split()]  # fmt: skip
                        runs.append(
                            (
                                program,
                                setting,
                                lambda f=flags, e=executable, t=text: run_lammps(
                                    e, f, t, folder
                                ),
                            )
                        )
                for precision in ("mixed", "double"):
                    runs.append(
                        (
                            f"mdapy {precision}",
                            "",
                            lambda p=precision, n=name, c=case, r=repeat: run_mdapy(
                                n, c, r, p, folder
                            ),
                        )
                    )
                for program, setting, measure in runs:
                    time.sleep(rest)
                    try:
                        seconds = measure()
                    except subprocess.CalledProcessError as error:
                        said = (error.stdout or "") + (error.stderr or "")
                        tail = [s for s in said.splitlines() if "ERROR" in s]
                        tail = tail or said.strip().splitlines()[-1:]
                        print(f"{name} {atoms:>9} {program:<14} {setting:<5} failed: {tail}",
                              flush=True)  # fmt: skip
                        continue
                    records.append({"case": name, "program": program, "setting": setting,
                                    "atoms": atoms, "seconds_per_step": seconds})  # fmt: skip
                    print(
                        f"{name} {atoms:>9} {program:<14} {setting:<5} {seconds * 1e3:9.3f} ms/step",
                        flush=True,
                    )
                    output.write_text(
                        json.dumps({"machine": info, "records": records}, indent=1)
                    )


def best(records: list[dict]) -> dict:
    """The shortest time of each case, program and size, over settings."""
    out = {}
    for r in records:
        key = (r["case"], r["program"], r["atoms"])
        if key not in out or r["seconds_per_step"] < out[key]["seconds_per_step"]:
            out[key] = r
    return out


PROGRAMS = ("mdapy mixed", "kokkos mixed", "mdapy double", "kokkos double")
NAMES = {
    "mdapy mixed": "MDAPY, mixed",
    "kokkos mixed": "KOKKOS, mixed",
    "mdapy double": "MDAPY, double",
    "kokkos double": "KOKKOS, double",
}


def table(path: Path) -> None:
    fastest = best(json.loads(path.read_text())["records"])
    for name, case in CASES.items():
        sizes = sorted({k[2] for k in fastest if k[0] == name})
        if not sizes:
            continue
        programs = [
            p for p in PROGRAMS if any(k[1] == p for k in fastest if k[0] == name)
        ]
        print(f"\n{case['title']}, ms per step\n")
        print("| Atoms | " + " | ".join(NAMES[p] for p in programs) + " |")
        print("|---:|" + "---:|" * len(programs))
        for atoms in sizes:
            cells = []
            for p in programs:
                r = fastest.get((name, p, atoms))
                cells.append(f"{r['seconds_per_step'] * 1e3:.2f}" if r else "")
            print(f"| {atoms:,} | " + " | ".join(cells) + " |")


def plot(path: Path, image: Path) -> None:
    from mdapy.visualization import plot as mplot

    fastest = best(json.loads(path.read_text())["records"])
    names = [n for n in CASES if any(k[0] == n for k in fastest)]
    styles = {
        "mdapy mixed": ("C0", "-", "o"),
        "kokkos mixed": ("C1", "-", "s"),
        "mdapy double": ("C0", "--", "o"),
        "kokkos double": ("C1", "--", "s"),
    }
    columns = 4
    rows = -(-(len(names) + 1) // columns)
    with mplot.style():
        fig, axes = mplot.figure("double", nrows=rows, ncols=columns)
        axes = axes.ravel()
        for ax, name in zip(axes, names):
            for program in PROGRAMS:
                points = sorted((k[2], r["seconds_per_step"]) for k, r in fastest.items()
                                if k[0] == name and k[1] == program)  # fmt: skip
                if not points:
                    continue
                atoms = [p[0] for p in points]
                rate = [p[0] / p[1] / 1e6 for p in points]
                color, line, marker = styles[program]
                ax.plot(atoms, rate, color=color, linestyle=line, marker=marker,
                        label=NAMES[program])  # fmt: skip
            ax.set(xscale="log", title=CASES[name]["title"], ylim=(0, None))
        for ax in axes[len(names) :]:
            ax.axis("off")
        for ax in axes[::columns]:
            ax.set_ylabel("Matom-steps/s")
        for ax in axes[len(names) - columns : len(names)]:
            ax.set_xlabel("Atoms")
        handles, labels = axes[0].get_legend_handles_labels()
        axes[len(names)].legend(handles, labels, loc="center")
        mplot.save(fig, str(image.with_suffix("")), formats=("png",))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, default=Path("gpu.json"))
    r.add_argument("--rest", type=float, default=0.0)
    r.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    t = sub.add_parser("table")
    t.add_argument("path", type=Path)
    p = sub.add_parser("plot")
    p.add_argument("path", type=Path)
    p.add_argument("image", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.output, args.rest, args.cases)
    elif args.command == "table":
        table(args.path)
    else:
        plot(args.path, args.image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
