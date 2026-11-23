# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The figures of the page "A potential from MDAPY's DFT", from the files of
dft_training.py and kconv.py in FOLDER: the forces of the test frames, the
equations of state and the phonons of the DFT and of the two models; and the
elastic constants of the DFT against the mesh of k-points.

    python figures.py FOLDER IMAGES
"""

import json
import sys
from pathlib import Path

import numpy as np

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

with plot.style():
    fig, (a, b, c) = plot.figure("double", 62, nrows=1, ncols=3)
    d = np.load(folder / "predictions-nep_0.npz")
    f = d["forces"]
    a.plot(f[:, 0], f[:, 1], ".", ms=1.5, alpha=0.4, c="C0", rasterized=True)
    lim = np.abs(f).max() * 1.05
    a.plot([-lim, lim], [-lim, lim], "-", c="0.5", lw=0.6)
    a.set(
        xlim=(-lim, lim),
        ylim=(-lim, lim),
        xlabel="force of the DFT (eV/Å)",
        ylabel="force of NEP (eV/Å)",
    )
    for k, p in props.items():
        v, e = np.asarray(p["volumes"]), np.asarray(p["energies"])
        color, line = style[k]
        b.plot(
            v,
            1e3 * (e - e.min()),
            line,
            marker="o",
            ms=3,
            c=color,
            lw=0.9,
            label=who[k],
        )
    b.set(xlabel="volume per atom (Å³)", ylabel="energy above the minimum (meV/atom)")
    b.legend(fontsize=6)
    for k, p in props.items():
        color, line = style[k]
        x = np.asarray(p["phonon_distances"])
        y = np.asarray(p["phonon_frequencies_THz"])
        for branch in range(y.shape[1]):
            c.plot(
                x,
                y[:, branch],
                line,
                c=color,
                lw=0.9,
                label=who[k] if branch == 0 else None,
            )
    labels = props["dft"]["phonon_labels"]
    c.set_xticks([t for t, _ in labels], [name.replace("|", "|") for _, name in labels])
    for t, _ in labels:
        c.axvline(t, c="0.85", lw=0.5, zorder=0)
    c.set(xlim=(0, labels[-1][0]), ylim=(0, None), ylabel="frequency (THz)")
    for ax, tag in zip((a, b, c), "abc"):
        plot.panel_label(ax, tag)
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
