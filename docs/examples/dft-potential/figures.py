# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The figures of the page "A potential from MDAPY's DFT", from the files of
dft_training.py and kconv.py in FOLDER: the training of the model of seed 0
(torchnep's loss curves and the energies, forces and stresses of the training
frames and of the tenth held out against the DFT); the equations of state and the phonons of
the DFT and of the two models; and the elastic constants of the DFT against
the mesh of k-points.

    python figures.py FOLDER IMAGES
"""

import json
import sys
from pathlib import Path

import numpy as np
from torchnep.plot import NEPPlotter

from mdapy.visualization import plot

folder, images = Path(sys.argv[1]), Path(sys.argv[2])
who = {
    "dft": "DFT",
    "dft-fine": "DFT, k-points twice as dense",
    "nep_0": "NEP, seed 0",
    "nep_1": "NEP, seed 1",
}
style = {
    "dft": ("k", "-"),
    "dft-fine": ("0.55", ":"),
    "nep_0": ("C0", "--"),
    "nep_1": ("C1", "--"),
}
props = {k: json.loads((folder / f"properties-{k}.json").read_text()) for k in who}

NEPPlotter().dashboard(
    str(folder / "nep_0"), out=str(images / "dft-potential-training.png")
)

with plot.style():
    fig, (eos, bands) = plot.figure("double", 62, nrows=1, ncols=2)
    for k, p in props.items():
        color, line = style[k]
        v, e = np.asarray(p["volumes"]), np.asarray(p["energies"])
        eos.plot(v, 1e3 * (e - e.min()), line, marker="o", ms=3, c=color, lw=0.9, label=who[k])  # fmt: skip
        x, y = np.asarray(p["phonon_distances"]), np.asarray(p["phonon_frequencies_THz"])  # fmt: skip
        for branch in range(y.shape[1]):
            bands.plot(x, y[:, branch], line, c=color, lw=0.9, label=who[k] if branch == 0 else None)  # fmt: skip
    eos.set(xlabel="volume per atom (Å³)", ylabel="energy above the minimum (meV/atom)")
    eos.legend(fontsize=6)
    # the high-symmetry points: (distance, name)
    ticks = props["dft"]["phonon_labels"]
    bands.set_xticks([d for d, _ in ticks], [name for _, name in ticks])
    for d, _ in ticks:
        bands.axvline(d, c="0.85", lw=0.5, zorder=0)
    bands.set(xlim=(0, ticks[-1][0]), ylim=(0, None), ylabel="frequency (THz)")
    plot.panel_label(eos, "a")
    plot.panel_label(bands, "b")
    plot.save(fig, str(images / "dft-potential"), formats=("png",))

    fig, ax = plot.figure("single", 62)
    for name, smearing, line in [("kconv-gaussian", "Gaussian 0.1 eV", "--"),
                                 ("kconv-marzari-vanderbilt0.25", "Marzari–Vanderbilt 0.25 eV", "-")]:  # fmt: skip
        rows = {}
        for path in folder.glob(f"{name}*.json"):
            rows.update({int(n): r for n, r in json.loads(path.read_text()).items()})
        n = sorted(rows)
        for key, color in [("C11", "C0"), ("C12", "C1"), ("C44", "C2")]:
            ax.plot(n, [rows[m][key] for m in n], line, marker="o", ms=3, c=color, lw=0.9,
                    label=f"{key}, {smearing}")  # fmt: skip
    ax.set(
        xlabel="k-points along each axis of the cubic cell",
        ylabel="elastic constant (GPa)",
    )
    ax.set_ylim(0, 215)
    ax.legend(fontsize=5.5, ncols=2, loc="upper center")
    plot.save(fig, str(images / "dft-potential-kpoints"), formats=("png",))
