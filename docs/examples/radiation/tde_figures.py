# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Figures 3 and 4 of Liu et al. (J. Nucl. Mater. 2025) from the runs of
tde.py: maps of the threshold displacement energy over the directions of the
standard triangle [001]-[101]-[111] of the cubic crystal, in stereographic
projection, interpolated linearly between the directions computed, and the
distributions of E_d.

    python tde_figures.py FOLDER IMAGE_PREFIX
"""

import glob
import json
import sys
from itertools import permutations, product

import matplotlib.tri as mtri
import numpy as np
from matplotlib.patches import Polygon

from mdapy.visualization import plot

METALS = ["Mo", "Nb", "Ta", "V", "W"]


def load(folder):
    """{(material, element): (directions (n, 3), E_d (n,))}, directions where a
    pair stayed below 400 eV."""
    out = {}
    for path in sorted(glob.glob(f"{folder}/*.json")):
        d = json.load(open(path))
        key = (d["material"], d["element"])
        keep = [(v, e) for v, e in zip(d["directions"], d["tde"]) if e is not None]
        if not keep:
            continue
        v, e = map(np.array, zip(*keep))
        old = out.get(key)
        out[key] = (
            (v, e)
            if old is None
            else (np.vstack([old[0], v]), np.concatenate([old[1], e]))
        )
    return out


def images(v):
    """The 48 images of the directions v (n, 3) under the cubic point group,
    with the index of the direction of each."""
    out, index = [], []
    for p in permutations(range(3)):
        for s in product((1, -1), repeat=3):
            out.append(v[:, p] * s)
            index.append(np.arange(len(v)))
    return np.vstack(out), np.concatenate(index)


def project(v):
    """Stereographic projection from -z of unit vectors with z > 0."""
    return v[:, 0] / (1 + v[:, 2]), v[:, 1] / (1 + v[:, 2])


def triangle():
    """The outline of the standard triangle: [001] to [101] along y = 0, the
    arc x = z to [111], back along x = y."""
    t = np.linspace(0, 1, 60)
    arc = np.stack(
        [np.ones_like(t), t, np.ones_like(t)], axis=1
    )  # x = z from [101] to [111]
    arc /= np.linalg.norm(arc, axis=1)[:, None]
    x, y = project(arc)
    return np.vstack([[[0.0, 0.0]], np.stack([x, y], axis=1), [[0.0, 0.0]]])


def draw_map(ax, v, e, levels):
    w, index = images(v / np.linalg.norm(v, axis=1)[:, None])
    up = w[:, 2] > 0.2
    x, y = project(w[up])
    near = (x > -0.15) & (x < 0.65) & (y > -0.15) & (y < 0.55)
    tri = mtri.Triangulation(x[near], y[near])
    fill = ax.tricontourf(
        tri, e[index[up]][near], levels=levels, cmap="viridis", extend="both"
    )
    outline = triangle()
    clip = Polygon(outline, closed=True, transform=ax.transData, facecolor="none")
    for c in fill.collections if hasattr(fill, "collections") else [fill]:
        c.set_clip_path(clip)
    ax.plot(outline[:, 0], outline[:, 1], c="k", lw=0.6)
    px, py = project(reduce(v))
    ax.plot(px, py, ".", ms=1.5, c="w", alpha=0.6)
    for label, (lx, ly), ha, va in [
        ("[001]", (0, 0), "right", "top"),
        ("[101]", (0.4142, 0), "left", "top"),
        ("[111]", (0.366, 0.366), "left", "bottom"),
    ]:
        ax.annotate(label, (lx, ly), ha=ha, va=va, fontsize=6)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_xlim(-0.04, 0.46)
    ax.set_ylim(-0.04, 0.40)
    return fill


def reduce(v):
    """The directions in the standard triangle: z >= x >= y >= 0."""
    a = np.sort(np.abs(v), axis=1)
    out = np.stack([a[:, 1], a[:, 0], a[:, 2]], axis=1)
    return out / np.linalg.norm(out, axis=1)[:, None]


def figure(data, material, image, title):
    keys = [(m if material == "pure" else material, m) for m in METALS]
    levels = np.linspace(20, 200, 19)
    with plot.style():
        fig, axes = plot.figure("double", 110, nrows=2, ncols=3)
        axes = axes.ravel()
        fill = None
        for ax, key, label in zip(axes, keys, "abcde"):
            if key not in data:
                ax.set_axis_off()
                continue
            v, e = data[key]
            fill = draw_map(ax, v, e, levels)
            ax.set_title(
                f"{key[1]}: {e.mean():.0f} eV, {len(e)} directions", fontsize=7
            )
            plot.panel_label(ax, label)
        if fill is not None:
            fig.colorbar(fill, ax=axes[:5].tolist(), shrink=0.6, label="$E_d$ (eV)")
        ax = axes[5]
        bins = np.arange(0, 260, 10)
        for key in keys:
            if key in data:
                ax.hist(
                    data[key][1], bins=bins, density=True, histtype="step", label=key[1]
                )
        ax.set(xlabel="$E_d$ (eV)", ylabel="probability density")
        ax.legend()
        plot.panel_label(ax, "f")
        plot.save(fig, image, formats=("png",))


if __name__ == "__main__":
    data = load(sys.argv[1])
    for key, (v, e) in sorted(data.items()):
        print(
            key,
            len(e),
            f"mean {e.mean():.1f} median {np.median(e):.1f} min {e.min():.0f} max {e.max():.0f}",
        )
    figure(data, "pure", sys.argv[2] + "-metals", "pure metals")
    figure(data, "HEA", sys.argv[2] + "-hea", "MoNbTaVW")
