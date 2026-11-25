# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The Frenkel pairs left by the cascades of cascade.py against the energy
of the PKA and the damage energy, with the NRT and arc-dpa models and the
power laws fitted by Liu et al. (J. Nucl. Mater. 2025), and the clustered
fractions of the defects.

    python cascade_figures.py FOLDER IMAGE
"""

import glob
import json
import sys
from collections import defaultdict

import numpy as np

from mdapy.visualization import plot

# E_d (eV), b, c of arc-dpa: W of Nordlund et al. (2018); the alloy's b and c of
# Liu et al. with E_d its mean threshold displacement energy, 51 eV in their Table 2
ARC = {"W": (70.0, -0.56, 0.12), "HEA": (51.0, -0.73, 0.21)}
# N_FP = a E^b, E the energy of the PKA in keV, of Liu et al.: W in two parts,
# below and above where the two cross (29 keV), the alloy in one
POWER = {"W": [(2.14, 0.58), (0.16, 1.35)], "HEA": [(1.67, 0.94)]}


def power_law(material, e):
    """The fit of Liu et al. at the energies e (keV): the larger of the parts."""
    return np.max([a * e**b for a, b in POWER[material]], axis=0)


def nrt(t, e_d):
    return np.where(t < e_d, 0.0, np.where(t < 2 * e_d / 0.8, 1.0, 0.8 * t / (2 * e_d)))


def arc_dpa(t, e_d, b, c):
    xi = (1 - c) / (2 * e_d / 0.8) ** b * t**b + c
    return np.where(t < 2 * e_d / 0.8, nrt(t, e_d), nrt(t, e_d) * xi)


def load(folder):
    runs = defaultdict(list)
    for path in glob.glob(f"{folder}/*.json"):
        r = json.load(open(path))
        runs[(r["material"], r["energy"])].append(r)
    return runs


def clustered(sizes):
    """The share of the defects in clusters of two or more."""
    total = sum(sizes)
    return np.nan if total == 0 else sum(s for s in sizes if s >= 2) / total


def summary(runs):
    rows = []
    for (material, energy), rs in sorted(runs.items()):
        final = np.array([r["final"] for r in rs], dtype=float)
        rows.append({
            "material": material, "energy": energy, "runs": len(rs),
            "damage": float(np.mean([r["damage_energy"] for r in rs])),
            "pairs": float(final.mean()), "error": float(final.std(ddof=1) / np.sqrt(len(rs))) if len(rs) > 1 else 0.0,
            "peak": float(np.mean([r["peak"] for r in rs])),
            "vacancies_clustered": float(np.nanmean([clustered(r["vacancy_clusters"]) for r in rs])),
            "interstitials_clustered": float(np.nanmean([clustered(r["interstitial_clusters"]) for r in rs])),
        })  # fmt: skip
    return rows


def figure(rows, image):
    with plot.style():
        fig, (left, right) = plot.figure("double", 70, nrows=1, ncols=2)
        for material, marker in [("W", "o"), ("HEA", "s")]:
            mine = [r for r in rows if r["material"] == material]
            if not mine:
                continue
            e = np.array([r["energy"] for r in mine]) / 1e3
            t = np.array([r["damage"] for r in mine])
            n = np.array([r["pairs"] for r in mine])
            err = np.array([r["error"] for r in mine])
            label = "W" if material == "W" else "MoNbTaVW"
            line = left.errorbar(
                e, n, err, fmt=marker, ms=4, capsize=2, label=f"{label}, MDAPY"
            )[0]
            grid = np.geomspace(0.8, 1.2 * e.max(), 80)
            left.plot(
                grid,
                power_law(material, grid),
                "--",
                c=line.get_color(),
                lw=0.8,
                label=f"{label}, fit of Liu et al.",
            )
            right.errorbar(
                t / 1e3,
                n,
                err,
                fmt=marker,
                ms=4,
                capsize=2,
                c=line.get_color(),
                label=label,
            )
            g = np.geomspace(300, 1.2 * t.max(), 80)
            e_d, bb, cc = ARC[material]
            right.plot(g / 1e3, arc_dpa(g, e_d, bb, cc), "-", c=line.get_color(), lw=0.8,
                       label=f"arc-dpa, $E_d$ = {e_d:.0f} eV")  # fmt: skip
            right.plot(
                g / 1e3, nrt(g, e_d), ":", c=line.get_color(), lw=0.8, label="NRT"
            )
        left.set(
            xscale="log",
            yscale="log",
            xlabel="energy of the PKA (keV)",
            ylabel="Frenkel pairs",
        )
        right.set(
            xscale="log",
            yscale="log",
            xlabel="damage energy (keV)",
            ylabel="Frenkel pairs",
        )
        from matplotlib.ticker import FuncFormatter

        plain = FuncFormatter(
            lambda value, _: f"{value:g}" if f"{value:g}"[0] in "1235" else ""
        )
        for ax in (left, right):
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_formatter(plain)
                axis.set_minor_formatter(plain)
        left.legend(fontsize=6)
        right.legend(fontsize=6)
        plot.panel_label(left, "a")
        plot.panel_label(right, "b")
        plot.save(fig, image, formats=("png",))


if __name__ == "__main__":
    rows = summary(load(sys.argv[1]))
    print(
        "| Material | E_PKA (keV) | Runs | T_d (keV) | Frenkel pairs | Peak | Vacancies in clusters | Interstitials in clusters |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['material']} | {r['energy'] / 1e3:g} | {r['runs']} | {r['damage'] / 1e3:.2f} | "
              f"{r['pairs']:.1f} ± {r['error']:.1f} | {r['peak']:.0f} | {100 * r['vacancies_clustered']:.0f} % | "
              f"{100 * r['interstitials_clustered']:.0f} % |")  # fmt: skip
    figure(rows, sys.argv[2])
