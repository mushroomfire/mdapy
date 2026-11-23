# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The robustness of MDAPY's plane-wave DFT: cold starts of hard cells with
the defaults of the DFT calculator (Gaussian smearing of 0.1 eV, a mesh of at
most 0.25 1/Å between k-points, a tolerance of 1e-6 eV, at most 100 steps,
its mixing), against Quantum ESPRESSO's pw.x with its own defaults (Broyden
mixing of 0.7, at most 100 steps) on the same cells, pseudopotentials,
cutoffs, k-points, smearing, bands and starting moments, and with its local
Thomas-Fermi mixing of 0.3 where its defaults fail. The energies, forces and
stresses are compared with pw.x converged to 1e-11 Ry (label qe-tight, mixing
of 0.3, local-TF where plain fails) where it was run, else with its default
run: at its default threshold pw.x stops early on magnetic iron, 5e-4 Ry from
its converged energy. Eight groups: magnetic
crystals (ferromagnetic, antiferromagnetic, PAW and ultrasoft), random
packings at the density of the liquid, slabs, molecules in a box, vacancies
and interstitials, compressed, stretched and sheared cells, ionic crystals,
cells of aluminium from generate_structures, and equiatomic CrCoNi, a random
fcc cell of 27 atoms with the ultrasoft files of SSSP 1.3 precision (GBRV's,
made for 40 Ry) at 45 and 360 Ry, at five volumes from 94 % to 112 %,
started ferromagnetic and with chromium reversed. Every cell is displaced
from symmetry by its own construction or by a normal draw of 0.05 Å.

The noise of the labels: the same cell moved rigidly by a quarter, a half and
three quarters of a spacing of the FFT grid along a, its energy and forces
against those at the start, and the net force the calculator takes out.

    python benchmarks/dft_robustness.py prepare FOLDER
    python benchmarks/dft_robustness.py mdapy FOLDER [CASE ...]
    python benchmarks/dft_robustness.py qe FOLDER CASE OUTPUT LABEL
    python benchmarks/dft_robustness.py translation FOLDER CASE
    python benchmarks/dft_robustness.py report FOLDER

prepare downloads the pseudopotentials (PseudoDojo 0.4 standard with the
normal cutoff of its hints; pslibrary 1.0.0 PAW and ultrasoft for iron and
nickel with the cutoffs their files suggest; the files of SSSP 1.3 precision
for CrCoNi), builds the cells into
FOLDER/cases/CASE.json and writes the inputs of pw.x, FOLDER/qe/CASE.in and
CASE-tf.in; mdapy runs the cases given (all by default) on the GPU into
FOLDER/results/CASE-mdapy.json; qe reads an output of pw.x into
FOLDER/results/CASE-LABEL.json; translation writes
FOLDER/results/CASE-translation.json; report prints the tables (markdown).
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
RY = 0.5 * HARTREE  # eV
DOJO = "https://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_upf.tgz"
HINTS = "https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-PBE-PDv0.4/master/standard.djson"
PSLIBRARY = "https://pseudopotentials.quantum-espresso.org/upf_files/"
PSL = {
    ("Fe", "paw"): "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
    ("Fe", "us"): "Fe.pbe-spn-rrkjus_psl.1.0.0.UPF",
    ("Ni", "paw"): "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF",
}
SSSP = "https://archive.materialscloud.org/api/records/rcyfm-68h65/files/SSSP_1.3.0_PBE_precision"
ALLOY_CUTOFFS = (22.5, 180.0)  # hartree, 45 and 360 Ry, of CrCoNi
KSPACING = 0.25  # 1/Å, with 2 pi: the default of the calculator
SIGMA = 0.1  # eV, Gaussian: the default of the calculator
TOLERANCE = 1e-6  # eV: the default of the calculator
VACUUM = 12.0  # Å
BOX = 10.0  # Å, of the molecules
RATTLE = 0.05  # Å
MOMENTS = {"Fe": 3.0, "Co": 2.0, "Ni": 1.0, "Cr": 2.0, "O": 1.0}
GROUPS = {
    "magnetic": "Magnetic crystals",
    "packing": "Random packings at the density of the liquid",
    "slab": "Slabs with 12 Å of vacuum",
    "molecule": "Molecules in a box of 10 Å",
    "defect": "Vacancies and interstitials",
    "strained": "Compressed, stretched and sheared",
    "ionic": "Ionic crystals",
    "generated": "Cells of aluminium from generate_structures",
    "alloy": "Equiatomic CrCoNi, 27 atoms of random fcc, SSSP 1.3 (GBRV ultrasoft)",
}
TRANSLATED = ("Si-vacancy", "W-packing", "Fe16-paw")


def download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=600) as response:
            path.write_bytes(response.read())
    return path


# cells: each a dict of cell (rows, Å), symbols, frac, and optionally
# moments (per atom), vacuum (the axes along which the mesh is 1), kind
# ("nc", "paw" or "us")


def crystal(structure, a, elements, repeat=(1, 1, 1), c=None, orientation=None):
    from mdapy.build import build_crystal

    s = build_crystal(structure, a, c=c, elements=elements, orientation=orientation,
                      repeat=repeat)  # fmt: skip
    cell = np.array(s.box.matrix, dtype=float)
    frac = np.asarray(s.box.to_fractional(s["positions"]), dtype=float) % 1.0
    return {
        "cell": cell,
        "symbols": s.table["element"].cast(str).to_list(),
        "frac": frac,
    }


def rattled(case, seed, sigma=RATTLE):
    cart = case["frac"] @ case["cell"]
    cart = cart + np.random.default_rng(seed).normal(0.0, sigma, cart.shape)
    return {**case, "frac": (cart @ np.linalg.inv(case["cell"])) % 1.0}


def scaled(case, volume):
    return {**case, "cell": case["cell"] * volume ** (1 / 3)}


def strained(case, strain):
    return {**case, "cell": case["cell"] @ (np.eye(3) + np.asarray(strain))}


def without(case, index):
    keep = [i for i in range(len(case["symbols"])) if i != index]
    return {
        **case,
        "symbols": [case["symbols"][i] for i in keep],
        "frac": case["frac"][keep],
    }


def plus(case, symbol, frac):
    return {**case, "symbols": [*case["symbols"], symbol],
            "frac": np.vstack([case["frac"], frac])}  # fmt: skip


def slab(case):
    """The crystal with VACUUM Å of vacuum along c (orthogonal cells)."""
    cell = case["cell"].copy()
    length = np.linalg.norm(cell[2])
    cell[2] *= (length + VACUUM) / length
    frac = case["frac"].copy()
    frac[:, 2] *= length / (length + VACUUM)
    return {**case, "cell": cell, "frac": frac, "vacuum": [2]}


def packing(symbol, atoms, volume, seed):
    """atoms at random in a cube of volume per atom (Å^3), none closer than
    0.8 volume^(1/3) (periodic)."""
    rng = np.random.default_rng(seed)
    side = (atoms * volume) ** (1 / 3)
    closest = 0.8 * volume ** (1 / 3) / side
    frac = []
    while len(frac) < atoms:
        p = rng.random(3)
        d = np.array(frac) - p if frac else np.zeros((0, 3))
        d -= np.round(d)
        if not frac or (np.linalg.norm(d, axis=1) > closest).all():
            frac.append(p)
    return {
        "cell": np.eye(3) * side,
        "symbols": [symbol] * atoms,
        "frac": np.array(frac),
    }


def molecule(symbols, positions):
    cart = np.asarray(positions, dtype=float)
    cart = cart - cart.mean(axis=0) + BOX / 2 + np.array([0.13, 0.07, 0.03])
    return {"cell": np.eye(3) * BOX, "symbols": list(symbols), "frac": cart / BOX,
            "vacuum": [0, 1, 2]}  # fmt: skip


def fcc_cell(a, n, elements, seed):
    """n x n x n primitive cells of fcc of the cubic lattice constant a, the
    sites shared equally among the elements at random."""
    cell = a / 2 * n * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
    frac = (
        np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)]) / n
    )
    symbols = [elements[i % len(elements)] for i in range(len(frac))]
    order = np.random.default_rng(seed).permutation(len(frac))
    return {"cell": cell, "symbols": [symbols[i] for i in order], "frac": frac}


def magnetic(case, moments=None):
    m = [MOMENTS.get(s, 0.0) for s in case["symbols"]] if moments is None else moments
    return {**case, "moments": [float(x) for x in m]}


def cases() -> dict:
    out = {}
    # magnetic crystals
    out["Fe-bcc"] = magnetic(rattled(crystal("bcc", 2.83, "Fe"), 1))
    out["Ni-fcc"] = magnetic(rattled(crystal("fcc", 3.52, "Ni"), 2))
    out["Co-hcp"] = magnetic(rattled(crystal("hcp", 2.49, "Co", c=4.02), 3))
    cr = rattled(crystal("bcc", 2.87, "Cr"), 4)
    out["Cr-afm"] = magnetic(cr, [2.0, -2.0])
    # the corners of the cubic cells up, the centres down; the layers of fcc
    # iron at z = 0 up, at z = 1/2 down
    cr16 = crystal("bcc", 2.87, "Cr", repeat=2)
    corner = [bool(np.allclose(f * 2, np.round(f * 2))) for f in cr16["frac"]]
    out["Cr16-afm"] = magnetic(rattled(cr16, 5), [2.0 if c else -2.0 for c in corner])
    fe_fcc = crystal("fcc", 3.45, "Fe")
    layer = [2.0 if f[2] < 0.25 else -2.0 for f in fe_fcc["frac"]]
    out["Fe-fcc-afm"] = magnetic(rattled(fe_fcc, 6), layer)
    out["Fe16-paw"] = {
        **magnetic(rattled(crystal("bcc", 2.83, "Fe", repeat=2), 7)),
        "kind": "paw",
    }
    out["Fe16-us"] = {
        **magnetic(rattled(crystal("bcc", 2.83, "Fe", repeat=2), 8)),
        "kind": "us",
    }
    feni = rattled(crystal("fcc", 3.55, "Fe", repeat=(2, 2, 1)), 9)
    order = np.random.default_rng(9).permutation(len(feni["symbols"]))
    feni["symbols"] = ["Ni" if i % 2 else "Fe" for i in order]
    out["FeNi16-paw"] = {**magnetic(feni), "kind": "paw"}
    # random packings at the density of the liquid (volume per atom, Å^3)
    for i, (symbol, volume) in enumerate([("Al", 18.2), ("Si", 18.8), ("Cu", 12.9),
                                          ("Fe", 13.0), ("Ni", 12.0), ("W", 17.5),
                                          ("Ti", 19.0), ("Mg", 25.5)]):  # fmt: skip
        case = packing(symbol, 16, volume, 100 + i)
        out[f"{symbol}-packing"] = magnetic(case) if symbol in MOMENTS else case
    # slabs
    o111 = [[1, -1, 0], [1, 1, -2], [1, 1, 1]]
    out["Al-100"] = slab(crystal("fcc", 4.04, "Al", repeat=(1, 1, 3)))
    out["Cu-111"] = slab(crystal("fcc", 3.63, "Cu", orientation=o111, repeat=(1, 1, 2)))
    out["Pt-111"] = slab(crystal("fcc", 3.97, "Pt", orientation=o111, repeat=(1, 1, 2)))
    out["Fe-100"] = magnetic(slab(crystal("bcc", 2.83, "Fe", repeat=(1, 1, 5))))
    out["Si-001"] = slab(crystal("diamond", 5.47, "Si", repeat=(1, 1, 2)))
    out["MgO-100"] = slab(crystal("rocksalt", 4.25, ["Mg", "O"], repeat=(1, 1, 2)))
    # molecules
    h = 0.629
    out["H2O"] = molecule(
        "OHH", [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]]
    )
    out["CO"] = molecule("CO", [[0, 0, 0], [0, 0, 1.128]])
    out["N2"] = molecule("NN", [[0, 0, 0], [0, 0, 1.098]])
    out["O2"] = magnetic(molecule("OO", [[0, 0, 0], [0, 0, 1.208]]))
    out["CH4"] = molecule(
        "CHHHH", [[0, 0, 0], [h, h, h], [h, -h, -h], [-h, h, -h], [-h, -h, h]]
    )
    # vacancies and interstitials
    out["Al-vacancy"] = rattled(without(crystal("fcc", 4.04, "Al", repeat=2), 0), 10)
    out["Si-vacancy"] = rattled(
        without(crystal("diamond", 5.47, "Si", repeat=2), 0), 11
    )
    out["W-vacancy"] = rattled(without(crystal("bcc", 3.17, "W", repeat=2), 0), 12)
    fe15 = rattled(without(crystal("bcc", 2.83, "Fe", repeat=2), 0), 13)
    out["Fe-vacancy-paw"] = {**magnetic(fe15), "kind": "paw"}
    si = crystal("diamond", 5.47, "Si", repeat=2)
    out["Si-interstitial"] = rattled(plus(si, "Si", [0.25, 0.25, 0.25]), 14)
    w = crystal("bcc", 3.17, "W", repeat=2)
    out["W-interstitial"] = rattled(plus(w, "W", [0.25, 0.125, 0.0]), 15)
    # compressed, stretched and sheared
    out["Al-0.6V"] = rattled(scaled(crystal("fcc", 4.04, "Al"), 0.6), 16)
    out["Cu-0.7V"] = rattled(scaled(crystal("fcc", 3.63, "Cu"), 0.7), 17)
    out["Si-1.4V"] = rattled(scaled(crystal("diamond", 5.47, "Si"), 1.4), 18)
    out["Fe-1.3V"] = magnetic(rattled(scaled(crystal("bcc", 2.83, "Fe"), 1.3), 19))
    out["Fe-0.75V"] = magnetic(rattled(scaled(crystal("bcc", 2.83, "Fe"), 0.75), 20))
    shear = [[0, 0.15, 0], [0.15, 0, 0], [0, 0, 0]]
    out["W-shear"] = rattled(strained(crystal("bcc", 3.17, "W"), shear), 21)
    # ionic crystals
    out["MgO"] = rattled(crystal("rocksalt", 4.25, ["Mg", "O"]), 22)
    out["NaCl"] = rattled(crystal("rocksalt", 5.69, ["Na", "Cl"]), 23)
    out["LiF"] = rattled(crystal("rocksalt", 4.07, ["Li", "F"]), 24)
    out["ZrO2"] = rattled(crystal("fluorite", 5.12, ["Zr", "O"]), 25)
    out["SiC"] = rattled(crystal("zincblende", 4.38, ["Si", "C"]), 26)
    # cells of aluminium from generate_structures
    from mdapy.mlip import generate_structures

    for i, s in enumerate(
        generate_structures("Al", 8, volume=(8.3, 33.2), seed=0).structures
    ):
        cell = np.array(s.box.matrix, dtype=float)
        frac = np.asarray(s.box.to_fractional(s["positions"]), dtype=float) % 1.0
        out[f"Al-generated-{i}"] = {
            "cell": cell,
            "symbols": ["Al"] * len(frac),
            "frac": frac,
        }
    # CrCoNi at five volumes, ferromagnetic and with chromium reversed
    alloy = rattled(fcc_cell(3.56, 3, ["Cr", "Co", "Ni"], 27), 28)
    for volume in (0.94, 1.00, 1.04, 1.08, 1.12):
        for start, cr in (("fm", 2.0), ("ferri", -2.0)):
            moments = [cr if s == "Cr" else MOMENTS[s] for s in alloy["symbols"]]
            out[f"CrCoNi-{volume:.2f}V-{start}"] = {
                **magnetic(scaled(alloy, volume), moments),
                "kind": "sssp",
            }
    groups = (["magnetic"] * 9 + ["packing"] * 8 + ["slab"] * 6 + ["molecule"] * 5
              + ["defect"] * 6 + ["strained"] * 6 + ["ionic"] * 5 + ["generated"] * 8
              + ["alloy"] * 10)  # fmt: skip
    for (name, case), group in zip(out.items(), groups, strict=True):
        case["group"] = group
    return out


def suggested(path: Path) -> tuple[float, float]:
    """The cutoffs of the bands and the density in hartree that a file of
    pslibrary suggests."""
    text = path.read_text(errors="replace")
    wfc = re.search(
        r"Suggested minimum cutoff for wavefunctions:\s*([\d.]+)\s*Ry", text
    )
    rho = re.search(
        r"Suggested minimum cutoff for charge density:\s*([\d.]+)\s*Ry", text
    )
    return float(wfc.group(1)) / 2, float(rho.group(1)) / 2


def valence_of(path: Path) -> float:
    """z_valence of UPF 2, or the header line of UPF 1."""
    text = path.read_text(errors="replace")
    m = re.search(r'z_valence\s*=\s*"\s*([-+.\dEe]+)', text) or re.search(
        r"([-+.\dEe]+)\s+Z valence", text
    )
    return float(m.group(1))


def mesh_of(case) -> list[int]:
    """The calculator's mesh, 1 along the axes of vacuum."""
    b = 2 * math.pi * np.linalg.inv(case["cell"]).T
    mesh = [max(1, math.ceil(float(np.linalg.norm(v)) / KSPACING - 1e-8)) for v in b]
    for axis in case.get("vacuum", []):
        mesh[axis] = 1
    return mesh


def bands_of(electrons: float) -> int:
    """The calculator's default with smearing, as pw.x's."""
    return max(math.ceil(1.2 * electrons / 2), math.ceil(electrons / 2) + 4)


def prepare(folder: Path) -> None:
    pseudo = folder / "pseudo"
    tgz = download(DOJO, folder / "downloads" / "dojo-upf.tgz")
    with tarfile.open(tgz) as t:
        for member in t.getmembers():
            if member.isfile() and member.name.endswith(".upf"):
                pseudo.mkdir(parents=True, exist_ok=True)
                (pseudo / Path(member.name).name).write_bytes(
                    t.extractfile(member).read()
                )
    hints = json.loads(download(HINTS, pseudo / "standard.djson").read_text())[
        "pseudos_metadata"
    ]
    for name in PSL.values():
        download(PSLIBRARY + name, pseudo / name)
    sssp = json.loads(
        download(SSSP + ".json/content", folder / "downloads" / "sssp.json").read_text()
    )
    wanted = {sssp[e]["filename"] for e in ("Cr", "Co", "Ni")}
    with tarfile.open(
        download(SSSP + ".tar.gz/content", folder / "downloads" / "sssp.tar.gz")
    ) as t:
        for member in t.getmembers():
            if member.isfile() and Path(member.name).name in wanted:
                (pseudo / Path(member.name).name).write_bytes(
                    t.extractfile(member).read()
                )
    (folder / "cases").mkdir(parents=True, exist_ok=True)
    (folder / "qe").mkdir(parents=True, exist_ok=True)
    for name, case in cases().items():
        kind = case.get("kind", "nc")
        elements = list(dict.fromkeys(case["symbols"]))
        if kind == "sssp":
            files = {e: sssp[e]["filename"] for e in elements}
        else:
            files = {
                e: (f"{e}.upf" if kind == "nc" else PSL[(e, kind)]) for e in elements
            }
        if kind == "nc":
            cutoff = max(float(hints[e]["hints"]["normal"]["ecut"]) for e in elements)
            density = 4 * cutoff
        elif kind == "sssp":
            cutoff, density = ALLOY_CUTOFFS
        else:
            pairs = [suggested(pseudo / f) for f in files.values()]
            cutoff, density = max(p[0] for p in pairs), max(p[1] for p in pairs)
        valence = {e: valence_of(pseudo / f) for e, f in files.items()}
        electrons = sum(valence[s] for s in case["symbols"])
        record = {
            "name": name, "group": case["group"], "kind": kind, "cell": case["cell"].tolist(),
            "symbols": case["symbols"], "frac": case["frac"].tolist(), "moments": case.get("moments"),
            "pseudopotentials": files, "cutoff_hartree": cutoff, "density_cutoff_hartree": density,
            "kpoints": mesh_of(case), "bands": bands_of(electrons), "electrons": electrons,
        }  # fmt: skip
        (folder / "cases" / f"{name}.json").write_text(json.dumps(record, indent=1))
        text = qe_input(record, valence)
        (folder / "qe" / f"{name}.in").write_text(text)
        (folder / "qe" / f"{name}-tf.in").write_text(
            text.replace(
                "mixing_beta = 0.7", "mixing_beta = 0.3, mixing_mode = 'local-TF'"
            )
        )
    print(f"prepared {folder}: {len(list((folder / 'cases').glob('*.json')))} cases")


def qe_input(case: dict, valence: dict) -> str:
    moments = case["moments"] or [0.0] * len(case["symbols"])
    species = list(dict.fromkeys(zip(case["symbols"], moments)))
    labels = [f"{e}{i + 1}" for i, (e, _) in enumerate(species)]
    spin = ""
    if case["moments"]:
        spin = ", nspin = 2, " + ", ".join(
            f"starting_magnetization({i + 1}) = {max(-1.0, min(1.0, m / valence[e])):.10f}"
            for i, (e, m) in enumerate(species)
        )
    mesh = " ".join(str(k) for k in case["kpoints"])
    lines = [
        "&control",
        '  calculation = "scf", pseudo_dir = "../pseudo", outdir = "./tmp",',
        "  tprnfor = .true., tstress = .true., disk_io = 'none'",
        "/",
        "&system",
        f"  ibrav = 0, nat = {len(case['symbols'])}, ntyp = {len(species)}, "
        f"ecutwfc = {2 * case['cutoff_hartree']:.4f}, ecutrho = {2 * case['density_cutoff_hartree']:.4f},",
        f"  nbnd = {case['bands']}, input_dft = 'PBE', occupations = 'smearing', "
        f"smearing = 'gaussian', degauss = {SIGMA / RY:.10f}{spin}",
        "/",
        "&electrons",
        f"  conv_thr = {TOLERANCE / RY:.4e}, mixing_beta = 0.7",
        "/",
        "ATOMIC_SPECIES",
        *[
            f"{label} 1.0 {case['pseudopotentials'][e]}"
            for label, (e, _) in zip(labels, species)
        ],
        "CELL_PARAMETERS angstrom",
        *[" ".join(f"{x:.12f}" for x in row) for row in case["cell"]],
        "ATOMIC_POSITIONS crystal",
        *[
            f"{labels[species.index((s, m))]} " + " ".join(f"{x:.12f}" for x in p)
            for s, m, p in zip(case["symbols"], moments, case["frac"])
        ],  # fmt: skip
        "K_POINTS automatic",
        f"{mesh} 0 0 0",
    ]
    return "\n".join(lines) + "\n"


def calculator(folder: Path, case: dict, **extra):
    from mdapy.calculators import DFT

    moments = case["moments"]
    return DFT(
        case["cutoff_hartree"] * HARTREE, ecutrho=case["density_cutoff_hartree"] * HARTREE,
        pseudopotentials={e: folder / "pseudo" / f for e, f in case["pseudopotentials"].items()},
        kpoints=case["kpoints"], magmoms=moments, functional="pbe_pw", **extra,
    )  # fmt: skip


def system_of(case: dict, shift=(0.0, 0.0, 0.0)):
    from mdapy import System

    cell = np.array(case["cell"])
    positions = np.array(case["frac"]) @ cell + np.asarray(shift)
    return System(positions, cell, elements=case["symbols"])


def run_mdapy(folder: Path, names: list[str]) -> None:
    from mdapy.errors import CalculationFailed

    for name in names:
        case = json.loads((folder / "cases" / f"{name}.json").read_text())
        dft = calculator(folder, case)
        start = time.perf_counter()
        out = {
            "case": name,
            "code": "MDAPY",
            "label": "mdapy",
            "atoms": len(case["symbols"]),
        }
        try:
            r = dft.compute(system_of(case), ("energy", "forces", "virial"))
            out.update(converged=True, steps=dft.last["iterations"], energy_ev=r.energy,
                       davidson=dft.last["davidson"], operations=dft.last["operations"],
                       magnetization=dft.last.get("magnetization"),
                       absolute_magnetization=dft.last.get("absolute_magnetization"),
                       forces_ev_per_a=np.asarray(r.forces).tolist(),
                       virial_ev=np.asarray(r.virial).tolist())  # fmt: skip
        except CalculationFailed as error:
            out.update(converged=False, steps=dft.max_iterations, error=str(error))
        out["seconds"] = time.perf_counter() - start
        save(folder, out)


def translation(folder: Path, name: str) -> None:
    """The energies and forces of the cell moved by 0, 1/4, 1/2 and 3/4 of a
    spacing of the grid along a."""
    case = json.loads((folder / "cases" / f"{name}.json").read_text())
    dft = calculator(folder, case, tolerance=1e-8, max_iterations=200, symmetry=False)
    cell = np.array(case["cell"])
    rows = []
    for quarter in range(4):
        grid = dft.last["grid"][0] if dft.last else None
        shift = cell[0] / grid * quarter / 4 if quarter else np.zeros(3)
        r = dft.compute(system_of(case, shift), ("energy", "forces"))
        rows.append({"shift": quarter / 4, "energy_ev": r.energy, "forces": np.asarray(r.forces).tolist(),
                     "net_force": np.asarray(dft.last["net_force"]).tolist(),
                     "grid": list(dft.last["grid"])})  # fmt: skip
        print(
            f"{name} shift {quarter / 4:.2f} of a grid spacing: {r.energy:.8f} eV",
            flush=True,
        )
    f0 = np.array(rows[0]["forces"])
    out = {
        "case": name, "code": "MDAPY", "label": "translation", "atoms": len(f0), "rows": rows,
        "energy_mev_per_atom": max(abs(r["energy_ev"] - rows[0]["energy_ev"]) for r in rows) / len(f0) * 1e3,
        "force_mev_per_a": max(float(np.abs(np.array(r["forces"]) - f0).max()) for r in rows) * 1e3,
        "net_force_mev_per_a": max(float(np.linalg.norm(r["net_force"])) for r in rows) * 1e3,
    }  # fmt: skip
    save(folder, out)


def read_qe(folder: Path, name: str, output: Path, label: str) -> None:
    text = output.read_text()
    converged = "convergence has been achieved" in text
    steps = len(re.findall(r"^\s+iteration #\s*\d+", text, re.M))
    out = {
        "case": name, "code": "Quantum ESPRESSO", "label": label, "converged": converged,
        "steps": int(re.search(r"convergence has been achieved in\s+(\d+)", text).group(1)) if converged else steps,
        "version": (re.search(r"Program PWSCF v\.(\S+)", text) or [None, None])[1],
    }  # fmt: skip
    if converged:
        out["energy_ev"] = (
            float(re.findall(r"!\s+total energy\s+=\s+(\S+)\s+Ry", text)[-1]) * RY
        )
        m = re.findall(r"total magnetization\s+=\s+(\S+)", text)
        out["magnetization"] = float(m[-1]) if m else None
        # the forces (Ry/bohr, the first block, of the atoms in their order)
        # and the stress (kbar, pressure positive), in eV/Å and GPa
        block = text.split("Forces acting on atoms")[-1]
        forces = re.findall(
            r"atom\s+\d+\s+type\s+\d+\s+force =\s+(\S+)\s+(\S+)\s+(\S+)", block
        )
        n = len(json.loads((folder / "cases" / f"{name}.json").read_text())["symbols"])
        out["forces_ev_per_a"] = (
            np.array(forces[:n], dtype=float) * RY / BOHR
        ).tolist()
        rows = re.search(r"total\s+stress.*\n(.*)\n(.*)\n(.*)", text)
        if rows:
            kbar = np.array(
                [[float(x) for x in rows.group(i).split()[3:6]] for i in (1, 2, 3)]
            )
            out["stress_gpa"] = (kbar / 10).tolist()
    wall = re.search(r"PWSCF\s*:.*CPU\s+(.*?)\s*WALL", text)
    if wall:
        out["seconds"] = sum(float(a) * {"h": 3600, "m": 60, "s": 1}[u]
                             for a, u in re.findall(r"([\d.]+)([hms])", wall.group(1).replace(" ", "")))  # fmt: skip
    save(folder, out)


def save(folder: Path, out: dict) -> None:
    path = folder / "results" / f"{out['case']}-{out['label']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(
        json.dumps({k: v for k, v in out.items() if k not in ("rows", "error")}),
        flush=True,
    )


def report(folder: Path) -> None:
    results = {}
    for p in (folder / "results").glob("*.json"):
        r = json.loads(p.read_text())
        results[(r["case"], r["label"])] = r
    records = [
        json.loads(p.read_text()) for p in sorted((folder / "cases").glob("*.json"))
    ]
    order = {name: i for i, name in enumerate(cases_order(records))}
    records.sort(key=lambda c: order[c["name"]])

    def steps(r):
        if r is None:
            return "–"
        return f"{r['steps']}" if r["converged"] else f"not in {r['steps']}"

    totals = {"mdapy": [0, 0], "qe": [0, 0], "qe-tf": [0, 0]}
    for group, title in GROUPS.items():
        mine = [c for c in records if c["group"] == group]
        if not mine:
            continue
        print(f"\n**{title}**\n")
        print(
            "| Cell | Atoms | k-points | MDAPY steps | pw.x steps | pw.x, local-TF 0.3 "
            "| ΔE (meV/atom) | ΔF (meV/Å) | Δσ (GPa) |"
        )
        print("|---|---:|---|---:|---:|---:|---:|---:|---:|")
        for c in mine:
            ours, qe, tf, tight = (
                results.get((c["name"], label))
                for label in ("mdapy", "qe", "qe-tf", "qe-tight")
            )
            for key, r in (("mdapy", ours), ("qe", qe), ("qe-tf", tf)):
                if r is not None:
                    totals[key][0] += r["converged"]
                    totals[key][1] += 1
            reference = qe if qe and qe["converged"] else tf
            if tight and tight["converged"]:
                reference = tight
            de = df = ds = ""
            if ours and ours["converged"] and reference and reference["converged"]:
                de = f"{(ours['energy_ev'] - reference['energy_ev']) / len(c['symbols']) * 1e3:.3f}"
                if "forces_ev_per_a" in ours and reference.get("forces_ev_per_a"):
                    d = np.array(ours["forces_ev_per_a"]) - np.array(
                        reference["forces_ev_per_a"]
                    )
                    df = f"{np.abs(d).max() * 1e3:.2f}"
                if "virial_ev" in ours and "stress_gpa" in reference:
                    # QE's stress is the virial over the volume, pressure positive
                    volume = abs(np.linalg.det(np.array(c["cell"])))
                    sigma = np.array(ours["virial_ev"]) / volume * 160.21766208
                    ds = (
                        f"{np.abs(sigma - np.array(reference['stress_gpa'])).max():.3f}"
                    )
            mesh = "×".join(str(k) for k in c["kpoints"])
            print(f"| {c['name']} | {len(c['symbols'])} | {mesh} | {steps(ours)} | {steps(qe)} | "
                  f"{steps(tf) if qe and not qe['converged'] else ''} | {de} | {df} | {ds} |")  # fmt: skip
    print(
        "\nConverged: " + ", ".join(f"{k} {v[0]} of {v[1]}" for k, v in totals.items())
    )
    rows = [
        results[(name, "translation")]
        for name in TRANSLATED
        if (name, "translation") in results
    ]
    if rows:
        print(
            "\n| Cell | Atoms | Energy (meV/atom) | Force (meV/Å) | Net force (meV/Å) |"
        )
        print("|---|---:|---:|---:|---:|")
        for r in rows:
            print(f"| {r['case']} | {r['atoms']} | {r['energy_mev_per_atom']:.4f} | "
                  f"{r['force_mev_per_a']:.3f} | {r['net_force_mev_per_a']:.3f} |")  # fmt: skip


def cases_order(records):
    groups = list(GROUPS)
    return [c["name"] for c in sorted(records, key=lambda c: groups.index(c["group"]))]


if __name__ == "__main__":
    command, folder = sys.argv[1], Path(sys.argv[2])
    if command == "prepare":
        prepare(folder)
    elif command == "mdapy":
        names = sys.argv[3:] or [
            p.stem for p in sorted((folder / "cases").glob("*.json"))
        ]
        run_mdapy(folder, names)
    elif command == "qe":
        read_qe(folder, sys.argv[3], Path(sys.argv[4]), sys.argv[5])
    elif command == "translation":
        translation(folder, sys.argv[3])
    elif command == "report":
        report(folder)
    else:
        raise SystemExit(__doc__)
