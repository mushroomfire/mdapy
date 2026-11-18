# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Molecular dynamics on one NVIDIA GPU: MDAPY against LAMMPS with KOKKOS, in
mixed and in double precision, and with its GPU package in mixed precision,
for copper with the EAM of Zhou et al. and argon with Lennard-Jones, at
constant energy.

    LMP_KOKKOS_MIXED=... LMP_KOKKOS_DOUBLE=... LMP_GPU=... \\
        python benchmarks/gpu.py run --output gpu.json
    python benchmarks/gpu.py table gpu.json
    python benchmarks/gpu.py plot gpu.json gpu.png

Each run builds the crystal, gives it the velocities of a temperature, runs 50
steps and times the next 300. LAMMPS reports the time of its loop; MDAPY
that of MolecularDynamics.run, which records the state at its start and end.
KOKKOS runs with full neighbor lists and Newton's third law off, its
default on GPUs, and with half lists and the law on; both are kept. A LAMMPS
not given by its variable is left out.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# name: crystal (structure, lattice constant, element, mass), temperature K,
# time step ps, sizes (repeats of the cubic cell), the LAMMPS interaction
CASES = {
    "eam": {
        "title": "Copper, EAM of Zhou et al.",
        "crystal": ("fcc", 3.615, "Cu", 63.546),
        "temperature": 600.0,
        "timestep": 0.002,
        "repeats": (20, 40, 60, 80),
        "lammps": "pair_style eam/alloy\npair_coeff * * Cu_zhou.eam.alloy Cu",
    },
    "lj": {
        "title": "Argon, Lennard-Jones cut at 8.5 Å",
        "crystal": ("fcc", 5.26, "Ar", 39.948),
        "temperature": 60.0,
        "timestep": 0.005,
        "repeats": (16, 32, 48, 64),
        "lammps": "pair_style lj/cut 8.5\npair_coeff 1 1 0.0104 3.40",
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


def run_mdapy(case: dict, repeat: int, precision: str) -> float:
    from mdapy.build import build_crystal
    from mdapy.calculators import EAM, LennardJones
    from mdapy.simulation import MolecularDynamics

    structure, a, element, _ = case["crystal"]
    system = build_crystal(structure, a, elements=element, repeat=(repeat,) * 3)
    if element == "Cu":
        potential = EAM.zhou(["Cu"], device="cuda", precision=precision)
    else:
        potential = LennardJones(
            (0.0104, 3.40), 8.5, device="cuda", precision=precision
        )
    with MolecularDynamics(system, potential, timestep=case["timestep"]) as md:
        md.velocities(case["temperature"], seed=1)
        md.run(WARM, thermo=WARM)
        start = time.perf_counter()
        md.run(STEPS, thermo=STEPS)
        return (time.perf_counter() - start) / STEPS


def run(output: Path) -> None:
    from mdapy.calculators import EAM

    programs = {
        "kokkos mixed": os.environ.get("LMP_KOKKOS_MIXED"),
        "kokkos double": os.environ.get("LMP_KOKKOS_DOUBLE"),
        "gpu package mixed": os.environ.get("LMP_GPU"),
    }
    records = []
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        EAM.zhou(["Cu"]).write(folder / "Cu_zhou.eam.alloy")
        for name, case in CASES.items():
            for repeat in case["repeats"]:
                atoms = 4 * repeat**3
                text = lammps_input(case, repeat)
                runs = []
                for program, executable in programs.items():
                    if not executable:
                        continue
                    if program.startswith("kokkos"):
                        for setting, options in KOKKOS.items():
                            flags = [
                                "-k",
                                "on",
                                "g",
                                "1",
                                "-sf",
                                "kk",
                                "-pk",
                                "kokkos",
                                *options.split(),
                            ]
                            runs.append(
                                (
                                    program,
                                    setting,
                                    lambda f=flags, e=executable: run_lammps(
                                        e, f, text, folder
                                    ),
                                )
                            )
                    else:
                        flags = ["-sf", "gpu", "-pk", "gpu", "1"]
                        runs.append(
                            (
                                program,
                                "",
                                lambda f=flags, e=executable: run_lammps(
                                    e, f, text, folder
                                ),
                            )
                        )
                for precision in ("mixed", "double"):
                    runs.append(
                        (
                            f"mdapy {precision}",
                            "",
                            lambda p=precision: run_mdapy(case, repeat, p),
                        )
                    )
                for program, setting, measure in runs:
                    seconds = measure()
                    records.append({"case": name, "program": program, "setting": setting,
                                    "atoms": atoms, "seconds_per_step": seconds})  # fmt: skip
                    print(
                        f"{name} {atoms:>9} {program:<18} {setting:<5} {seconds * 1e3:9.3f} ms/step",
                        flush=True,
                    )
    output.write_text(json.dumps({"records": records}, indent=1))


def best(records: list[dict]) -> dict:
    """The shortest time of each case, program and size, over settings."""
    out = {}
    for r in records:
        key = (r["case"], r["program"], r["atoms"])
        if key not in out or r["seconds_per_step"] < out[key]["seconds_per_step"]:
            out[key] = r
    return out


PROGRAMS = (
    "mdapy mixed",
    "kokkos mixed",
    "gpu package mixed",
    "mdapy double",
    "kokkos double",
)
NAMES = {
    "mdapy mixed": "MDAPY, mixed",
    "kokkos mixed": "LAMMPS KOKKOS, mixed",
    "gpu package mixed": "LAMMPS GPU package, mixed",
    "mdapy double": "MDAPY, double",
    "kokkos double": "LAMMPS KOKKOS, double",
}


def table(path: Path) -> None:
    fastest = best(json.loads(path.read_text())["records"])
    for name, case in CASES.items():
        sizes = sorted({k[2] for k in fastest if k[0] == name})
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
    styles = {
        "mdapy mixed": ("C0", "-", "o"),
        "kokkos mixed": ("C1", "-", "s"),
        "gpu package mixed": ("C2", "-", "^"),
        "mdapy double": ("C0", "--", "o"),
        "kokkos double": ("C1", "--", "s"),
    }
    with mplot.style():
        fig, axes = mplot.figure("double", ncols=2)
        for ax, (name, case) in zip(axes, CASES.items()):
            for program in PROGRAMS:
                points = sorted((k[2], r["seconds_per_step"]) for k, r in fastest.items()
                                if k[0] == name and k[1] == program)  # fmt: skip
                if not points:
                    continue
                atoms = [p[0] for p in points]
                rate = [p[0] / p[1] / 1e6 for p in points]
                color, line, marker = styles[program]
                ax.plot(
                    atoms,
                    rate,
                    color=color,
                    linestyle=line,
                    marker=marker,
                    label=NAMES[program],
                )
            ax.set(xscale="log", xlabel="Atoms", ylabel="Million atom-steps per second",
                   title=case["title"], ylim=(0, None))  # fmt: skip
        axes[0].legend()
        mplot.save(fig, str(image.with_suffix("")), formats=("png",))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--output", type=Path, default=Path("gpu.json"))
    t = sub.add_parser("table")
    t.add_argument("path", type=Path)
    p = sub.add_parser("plot")
    p.add_argument("path", type=Path)
    p.add_argument("image", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.output)
    elif args.command == "table":
        table(args.path)
    else:
        plot(args.path, args.image)
    return 0


if __name__ == "__main__":
    sys.exit(main())
