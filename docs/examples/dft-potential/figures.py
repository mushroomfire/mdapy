# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The figures of the page "A potential from MDAPY's DFT", from the files of
dft_training.py and kconv.py in FOLDER: the training of the model of seed 0
(torchnep's loss curves and the energies, forces and stresses of the training
frames and of the tenth held out against the DFT); the equations of state and the phonons of
the DFT and of the two models; the elastic constants of the DFT against
the mesh of k-points; and far from the crystal, the stacking fault and the
compression of the DFT and of the models and their Hugoniots against that
measured.

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
# the Hugoniot of aluminium 1100 measured: its density (g/cm³), the shock and the
# particle velocities (km/s); Marsh (ed.), LASL Shock Hugoniot Data (1980), p. 165
ALUMINIUM_1100 = np.array([
    (2.713, 5.934, 0.426), (2.713, 5.804, 0.428), (2.713, 6.254, 0.526), (2.713, 6.132, 0.537),
    (2.706, 6.347, 0.804), (2.713, 6.558, 0.884), (2.713, 6.624, 0.890), (2.706, 6.862, 1.164),
    (2.713, 7.054, 1.197), (2.713, 6.979, 1.214), (2.713, 7.187, 1.327), (2.706, 7.348, 1.448),
    (2.713, 7.534, 1.600), (2.706, 7.489, 1.606), (2.713, 7.785, 1.812),
])  # fmt: skip

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

    # far from the crystal: the stacking fault, the compression and the Hugoniot
    fig, (fault, pv, shock) = plot.figure("double", 52, nrows=1, ncols=3)
    for k in ("dft", "nep_0", "nep_1"):
        color, line = style[k]
        p = props[k]
        fault.plot(p["gsfe_fraction"], 1e3 * np.asarray(p["gsfe_J_m2"]), line, marker="o", ms=2.5, c=color,
                   lw=0.9, label=who[k])  # fmt: skip
        pv.plot(p["compression_ratios"], p["compression_GPa"], line, c=color, lw=0.9, label=f"{who[k]}, 0 K")  # fmt: skip
        if "hugoniot" in p:
            h = p["hugoniot"]
            pv.plot([r["volume_ratio"] for r in h], [r["pressure_GPa"] for r in h], "s", ms=3, c=color,
                    mfc="none", label=f"{who[k]}, Hugoniot")  # fmt: skip
            shock.plot([r["particle_velocity_km_s"] for r in h], [r["shock_velocity_km_s"] for r in h], "s",
                       ms=3, c=color, mfc="none", label=who[k])  # fmt: skip
    rho0, us, up = ALUMINIUM_1100.T
    pv.plot(1 - up / us, rho0 * us * up, "o", ms=2.5, c="0.4", label="experiment")
    shock.plot(up, us, "o", ms=2.5, c="0.4", label="experiment")
    u = np.linspace(0, 2.25, 10)
    shock.plot(u, 5.392 + 1.341 * u, "-", c="0.75", lw=0.8, zorder=0, label="its fit")
    fault.set(
        xlabel=r"slip along [11$\bar{2}$], of a√6/2",
        ylabel="energy (mJ/m²)",
        xlim=(0, 1),
    )
    fault.legend(fontsize=5.5)
    pv.set(xlabel=r"$V\,/\,V_0$", ylabel="pressure (GPa)")
    pv.legend(fontsize=5)
    shock.set(xlabel="particle velocity (km/s)", ylabel="shock velocity (km/s)")
    shock.legend(fontsize=5.5)
    for ax, label in zip((fault, pv, shock), "abc"):
        plot.panel_label(ax, label)
    plot.save(fig, str(images / "dft-potential-far"), formats=("png",))
