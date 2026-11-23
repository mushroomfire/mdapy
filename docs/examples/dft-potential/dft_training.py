# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A NEP model of aluminium trained on labels of MDAPY's DFT, and its properties
against those of the DFT itself.

    python dft_training.py sets                 # train.xyz, near.xyz and test.xyz, unlabelled
    python dft_training.py atom                 # atom.json: the energy of the isolated atom
    python dft_training.py label SET PART PARTS # labels of the frames i % PARTS == PART of SET
    python dft_training.py train SEED           # nep_SEED/nep_best.txt, its loss.out
    python dft_training.py errors               # errors.json: the models on the test frames
    python dft_training.py properties WHO       # properties-WHO.json, WHO = dft, dft-fine or nep_SEED

The structures are the "generated + crystals" set of the page on generated
training sets for aluminium: two thirds of the atoms from generate_structures,
one third perturbed fcc, bcc and hcp crystals, 4800 atoms; with them fcc
crystals near equilibrium, 864 atoms; the test frames another generation and
other crystals, 1200 atoms. The models learn the energies relative to the
isolated atom.
"""

import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

import numpy as np

from mdapy import System
from mdapy.build import build_crystal
from mdapy.io import read_trajectory, write
from mdapy.mlip import generate_structures, perturb
from mdapy.transform import replicate

DOJO = "https://www.pseudo-dojo.org/pseudos/nc-sr-04_pbe_standard_upf.tgz"
VOLUME = (8.3, 33.2)
fcc = build_crystal("fcc", 4.05, elements="Al", repeat=(2, 2, 2))
bcc = build_crystal("bcc", 3.24, elements="Al", repeat=(3, 3, 3))
hcp = build_crystal("hcp", 2.86, elements="Al", repeat=(3, 2, 2))
NEP_IN = """type       1 Al
version    4
cutoff     6 4
n_max      6 6
basis_size 8 8
l_max      4 2 0
neuron     50
epoch      600
batch      {batch}
lr         0.01
stage2     1
"""


def pseudopotential() -> Path:
    path = Path("Al.upf")
    if not path.exists():
        with urllib.request.urlopen(DOJO, timeout=600) as response:
            Path("dojo.tgz").write_bytes(response.read())
        with tarfile.open("dojo.tgz") as t:
            member = next(m for m in t.getmembers() if Path(m.name).name == "Al.upf")
            path.write_bytes(t.extractfile(member).read())
    return path


KSPACING, SMEARING, SIGMA = 0.15, "marzari-vanderbilt", 0.25  # 1/Å, eV


def dft(kspacing=KSPACING, **extra):
    from mdapy.calculators import DFT

    return DFT(
        600.0,
        pseudopotentials={"Al": pseudopotential()},
        kspacing=kspacing,
        smearing=SMEARING,
        sigma=SIGMA,
        **extra,
    )


def first(frames, atoms):
    out, count = [], 0
    for frame in frames:
        if count >= atoms:
            break
        out.append(frame)
        count += len(frame)
    return out


def scaled(system, factor):
    return system.with_box(system.box.matrix * factor ** (1 / 3), scale_positions=True)


def crystals(seed, atoms):
    rng = np.random.default_rng(seed)
    kinds = [(fcc, 0.1), (fcc, 0.25), (bcc, 0.15), (hcp, 0.15)]
    frames = []
    for k in range(160):
        crystal, displacement = kinds[k % 4]
        volume = np.exp(rng.uniform(np.log(0.85), np.log(1.15)))
        frames += perturb(
            scaled(crystal, volume), 1, displacement=displacement, strain=0.05, seed=rng
        )
    return first(frames, atoms)


def near(seed, cells):
    """fcc near equilibrium: volumes from 90 to 110 %, strains up to 2 %, atoms
    displaced by 0.03 Å; cells of 4 and of 32 atoms in turn."""
    rng = np.random.default_rng(seed)
    unit = build_crystal("fcc", 4.04, elements="Al")
    frames = []
    for k in range(cells):
        crystal = unit if k % 2 == 0 else replicate(unit, (2, 2, 2))
        volume = rng.uniform(0.90, 1.10)
        frames += perturb(
            scaled(crystal, volume), 1, displacement=0.03, strain=0.02, seed=rng
        )
    return frames


def mixed(parts, shares, budget):
    taken, atoms, out = [0] * len(parts), [0] * len(parts), []
    while sum(atoms) < budget:
        k = min(range(len(parts)), key=lambda k: atoms[k] / shares[k])
        if taken[k] == len(parts[k]):
            break
        out.append(parts[k][taken[k]])
        taken[k] += 1
        atoms[k] += len(out[-1])
    return out


def sets():
    generated = generate_structures("Al", volume=VOLUME, seed=0).structures
    write("train.xyz", mixed([generated, crystals(50, 4800)], [2, 1], 4800))
    write("near.xyz", near(51, 48))
    other = generate_structures("Al", volume=VOLUME, seed=7).structures
    write("test.xyz", mixed([other, crystals(57, 1200)], [2, 1], 1200))
    for name in ("train", "near", "test"):
        frames = list(read_trajectory(f"{name}.xyz"))
        print(
            name,
            len(frames),
            "frames",
            sum(len(f) for f in frames),
            "atoms",
            flush=True,
        )


def atom():
    """The energy of one atom with spin in a box of 12 Å, Γ only, by the DFT of
    the labels otherwise: the energy every atom of a cell is counted from."""
    single = System([[6.0, 6.0, 6.0]], np.eye(3) * 12.0, elements="Al")
    result = dft(kpoints=(1, 1, 1), magmoms={"Al": 1.0}).compute(single)
    Path("atom.json").write_text(json.dumps({"energy": float(result.energy)}))
    print("the atom:", result.energy, "eV", flush=True)


def label(name, part, parts):
    calculator = dft()
    out = []
    for i, frame in enumerate(read_trajectory(f"{name}.xyz")):
        if i % parts != part:
            continue
        start = time.time()
        result = calculator.compute(frame, ("energy", "forces", "virial"))
        out.append(result.attach(frame).with_info(index=i))
        print(f"{name} {i}: {len(frame)} atoms, {calculator.last['iterations']} steps, "
              f"{time.time() - start:.1f} s", flush=True)  # fmt: skip
    write(f"labelled/{name}_{part}.xyz", out)


def merged(name):
    """The labelled frames of a set, their energies counted from the atom's."""
    e_atom = json.loads(Path("atom.json").read_text())["energy"]
    frames = []
    for path in sorted(Path("labelled").glob(f"{name}_*.xyz")):
        frames += list(read_trajectory(path))
    frames = sorted(frames, key=lambda f: f.info["index"])
    return [f.with_info(energy=f.info["energy"] - len(f) * e_atom) for f in frames]


def train(seed):
    """NEP of seed `seed` on the training frames, a tenth of them held out to
    choose the best epoch; the test frames are not seen."""
    from torchnep import train_nep

    folder = Path(f"nep_{seed}")
    folder.mkdir(exist_ok=True)
    frames = merged("train") + merged("near")
    energies = [f.info["energy"] / len(f) for f in frames]
    low = min(energies) + 0.5
    frames = [
        f.with_info(weight=3.0) if e < low else f for f, e in zip(frames, energies)
    ]
    write(folder / "train.xyz", frames)
    atoms = sum(len(f) for f in frames)
    (folder / "nep.in").write_text(
        NEP_IN.format(batch=max(1, round(150 * len(frames) / atoms)))
    )
    start = time.time()
    train_nep(str(folder / "nep.in"), str(folder / "train.xyz"), output_dir=str(folder),
              device="cuda", run_seed=seed, valid_ratio=0.1)  # fmt: skip
    print(f"trained nep_{seed} in {time.time() - start:.0f} s", flush=True)


def errors():
    """The errors of each model on the test frames."""
    from mdapy.calculators import NEP

    frames = merged("test")
    out = {}
    for folder in sorted(Path(".").glob("nep_*")):
        nep = NEP(str(folder / "nep_best.txt"))
        de, df, dv, pe, pf = [], [], [], [], []
        for frame in frames:
            r = nep.compute(frame, ("energy", "forces", "virial"))
            n = len(frame)
            de.append((r.energy - frame.info["energy"]) / n)
            df.append((r.forces - frame["forces"]).ravel())
            dv.append(
                (
                    np.asarray(r.virial)
                    - np.asarray(frame.info["virial"]).reshape(3, 3)
                ).ravel()
                / n
            )
            pe.append((frame.info["energy"] / n, r.energy / n))
            pf.append(
                np.stack(
                    [np.asarray(frame["forces"]).ravel(), np.asarray(r.forces).ravel()],
                    axis=1,
                )
            )
        np.savez(
            f"predictions-{folder.name}.npz",
            energies=np.array(pe),
            forces=np.concatenate(pf),
        )

        def rmse(x):
            return float(
                np.sqrt(
                    np.mean(np.square(np.concatenate([np.atleast_1d(v) for v in x])))
                )
            )

        out[folder.name] = {"energy_mev_per_atom": 1e3 * rmse(de), "force_ev_per_a": rmse(df),
                            "virial_mev_per_atom": 1e3 * rmse(dv), "frames": len(frames)}  # fmt: skip
    Path("errors.json").write_text(json.dumps(out, indent=1))
    print(out)


def properties(who):
    """The properties of the DFT or of a model, each written as soon as it is
    known into properties-WHO.json, those already there kept."""
    from mdapy.calculators import NEP
    from mdapy.properties.defects import surface_energy, vacancy_formation_energy
    from mdapy.properties.elastic import elastic_constants
    from mdapy.properties.phonons import phonons, primitive_cell

    # dft-fine: k-points twice as dense as the labels
    calc = (
        dft()
        if who == "dft"
        else dft(KSPACING / 2)
        if who == "dft-fine"
        else NEP(f"{who}/nep_best.txt")
    )
    path = Path(f"properties-{who}.json")
    out = json.loads(path.read_text()) if path.exists() else {}

    def keep(**values):
        out.update(values)
        path.write_text(json.dumps(out))
        print({k: v for k, v in values.items() if not isinstance(v, list)}, flush=True)

    cell = build_crystal("fcc", 4.04, elements="Al")  # four atoms
    if "a0" not in out:  # seven volumes, Birch-Murnaghan as the Delta project
        t, v, e = time.time(), [], []
        for s in np.linspace(0.94, 1.06, 7):
            c = scaled(cell, s)
            v.append(c.box.volume / 4)
            e.append(calc.compute(c).energy / 4)
        p = np.polyfit(np.asarray(v) ** (-2 / 3), e, 3)
        roots = [
            r.real
            for r in np.roots(np.polyder(p))
            if abs(r.imag) < 1e-12 and r.real > 0
        ]
        roots = [r for r in roots if np.polyval(np.polyder(p, 2), r) > 0]  # minima
        x = min(roots, key=lambda r: abs(r - np.mean(np.asarray(v) ** (-2 / 3))))
        d2 = np.polyval(np.polyder(p, 2), x)
        keep(volumes=v, energies=e, a0=float((4 * x**-1.5) ** (1 / 3)),
             B0_GPa=float(4 / 9 * x**5 * d2 / x**1.5 * 160.21766208), eos_seconds=time.time() - t)  # fmt: skip
    crystal = scaled(cell, out["a0"] ** 3 / cell.box.volume)
    if "elastic_GPa" not in out:
        t = time.time()
        el = elastic_constants(crystal, calc, max_force=1e-3)
        keep(
            elastic_GPa=np.asarray(el.constants).tolist(),
            elastic_seconds=time.time() - t,
        )
    if "vacancy_eV" not in out:
        t = time.time()
        vac = vacancy_formation_energy(
            replicate(crystal, (3, 3, 3)), calc, max_force=0.01
        )
        keep(vacancy_eV=float(vac.energy), vacancy_seconds=time.time() - t)
    if "surface_100_J_m2" not in out:
        t = time.time()
        surf = surface_energy(crystal, calc, axis=2, max_force=0.01)
        keep(surface_100_J_m2=float(surf.energy), surface_seconds=time.time() - t)
    if "phonon_frequencies_THz" not in out:
        t = time.time()
        bands = phonons(primitive_cell(crystal), calc, supercell=4).band_structure(
            "fcc"
        )
        keep(phonon_labels=[[float(d), str(name)] for d, name in bands.labels],
             phonon_distances=np.asarray(bands.distances).tolist(),
             phonon_frequencies_THz=np.asarray(bands.frequencies).tolist(), phonon_seconds=time.time() - t)  # fmt: skip


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "sets":
        sets()
    elif command == "atom":
        atom()
    elif command == "label":
        Path("labelled").mkdir(exist_ok=True)
        label(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif command == "train":
        train(int(sys.argv[2]))
    elif command == "errors":
        errors()
    elif command == "properties":
        properties(sys.argv[2])
