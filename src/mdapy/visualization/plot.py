# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Plot templates for figures of journals, and plots of the results of mdapy.
Needs matplotlib."""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mdapy.errors import InvalidParameter

MM = 1 / 25.4
"""Inches per millimetre."""
SINGLE = 88 * MM
"""Width of a single column, in inches."""
DOUBLE = 180 * MM
"""Width of a double column, in inches."""

#: Colors safe for color blindness, after Okabe and Ito.
COLORS = (
    "#0072b2",
    "#e69f00",
    "#009e73",
    "#d55e00",
    "#cc79a7",
    "#56b4e9",
    "#f0e442",
    "#000000",
)

#: Settings of matplotlib for figures of journals: sans-serif text of 5 to 7
#: points, thin lines, no top and right spines, vector fonts.
STYLE: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 2,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.fontset": "stixsans",
}


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a test dependency
        raise ImportError("plots need matplotlib: pip install matplotlib") from None
    return plt


@contextmanager
def style() -> Iterator[None]:
    """A context in which matplotlib uses :data:`STYLE` and :data:`COLORS`."""
    plt = _pyplot()
    from cycler import cycler

    settings = dict(STYLE) | {"axes.prop_cycle": cycler(color=list(COLORS))}
    with plt.rc_context(settings):
        yield


def figure(
    width: str | float = "single",
    height: float | None = None,
    *,
    nrows: int = 1,
    ncols: int = 1,
    **options: Any,
) -> tuple[Any, Any]:
    """A figure of the width of a column of a journal.

    Parameters
    ----------
    width : {"single", "double"} or float, optional
        One column (88 mm), two (180 mm), or a width in mm. Default single.
    height : float, optional
        Height in mm. Default 3/4 of the width of one panel times the rows.
    nrows, ncols : int, optional
        Rows and columns of panels. Default 1.
    **options
        Passed to ``matplotlib.pyplot.subplots``.

    Returns
    -------
    tuple
        The figure and its axes, as ``matplotlib.pyplot.subplots`` returns.
    """
    plt = _pyplot()
    if width == "single":
        inches = SINGLE
    elif width == "double":
        inches = DOUBLE
    elif isinstance(width, int | float) and width > 0:
        inches = float(width) * MM
    else:
        raise InvalidParameter(
            f"width must be single, double or a positive width in mm, not {width!r}"
        )
    tall = 0.75 * inches / ncols * nrows if height is None else float(height) * MM
    options.setdefault("constrained_layout", True)
    return plt.subplots(nrows, ncols, figsize=(inches, tall), **options)


def panel_label(ax: Any, text: str, *, x: float = -0.12, y: float = 1.02) -> None:
    """A bold lower-case label of a panel, at the top left of its axes."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def save(
    fig: Any, path: str | PathLike, formats: Sequence[str] = ("pdf", "svg", "png")
) -> list[Path]:
    """Saves a figure in several formats: vector PDF and SVG with text kept as
    text, and PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    path : str or path-like
        The file, its suffix replaced by each format.
    formats : sequence of str, optional
        Default PDF, SVG and PNG.

    Returns
    -------
    list of pathlib.Path
        The files written.
    """
    plt = _pyplot()
    written = []
    for fmt in formats:
        target = Path(path).with_suffix("." + fmt)
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(target, dpi=300 if fmt == "png" else None)
        written.append(target)
    return written


def _axes(ax: Any) -> Any:
    return figure()[1] if ax is None else ax


def parity(
    reference: ArrayLike,
    predicted: ArrayLike,
    *,
    ax: Any = None,
    label: str = "",
    unit: str = "",
    scale: float = 1.0,
    density: bool = False,
    color: str | None = None,
) -> Any:
    """A parity plot of predicted against reference values, with the root
    mean square error in the legend.

    Parameters
    ----------
    reference, predicted : array_like
        The values, of the same size.
    ax : matplotlib.axes.Axes, optional
        The axes; by default a new figure.
    label : str, optional
        Name of the set in the legend.
    unit : str, optional
        Unit of the error in the legend.
    scale : float, optional
        Factor of the error in the legend, such as 1000 for meV from eV.
    density : bool, optional
        A hexagonal density map instead of points, for many values.
    color : str, optional
        Color of the points.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    ax = _axes(ax)
    r = np.asarray(reference, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if r.shape != p.shape:
        raise InvalidParameter("reference and predicted must have the same size")
    keep = np.isfinite(r) & np.isfinite(p)
    r, p = r[keep], p[keep]
    rmse = float(np.sqrt(np.mean((p - r) ** 2))) * scale if len(r) else float("nan")
    text = f"{label + ': ' if label else ''}RMSE {rmse:.3g} {unit}".strip()
    if density:
        ax.hexbin(r, p, gridsize=80, mincnt=1, bins="log", cmap="viridis")
        ax.plot([], [], " ", label=text)
    else:
        ax.scatter(r, p, s=3, alpha=0.6, lw=0, color=color, label=text, rasterized=True)
    if len(r):
        low, high = min(r.min(), p.min()), max(r.max(), p.max())
        pad = 0.03 * (high - low or 1.0)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "k--", lw=0.6)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", handlelength=0.8)
    return ax


def scatter(
    system: Any,
    x: str,
    y: str,
    *,
    ax: Any = None,
    atoms: ArrayLike | None = None,
    max_points: int = 20000,
    color: str | None = None,
) -> Any:
    """Two columns of the atoms against each other, a point for each atom.

    Parameters
    ----------
    system : System
        The atoms.
    x, y : str
        A column of numbers, a component such as ``"positions.z"`` or an
        expression, as ``system[x]`` takes it; the labels of the axes.
    ax : matplotlib.axes.Axes, optional
        The axes; by default a new figure.
    atoms : array_like, optional
        A boolean mask or indices of the atoms drawn. Default all of them.
    max_points : int, optional
        At most this many atoms drawn, picked at random with a fixed seed
        from those with finite values. Default 20000.
    color : str, optional
        Color of the points.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    from mdapy.core._atoms import _mask

    if max_points < 1:
        raise InvalidParameter(f"max_points must be positive, not {max_points}")
    xs = np.asarray(system[x], dtype=float)
    ys = np.asarray(system[y], dtype=float)
    if xs.ndim != 1 or ys.ndim != 1:
        raise InvalidParameter("x and y must be columns of one component")
    keep = np.isfinite(xs) & np.isfinite(ys)
    if atoms is not None:
        keep &= _mask(atoms, len(system))
    rows = np.flatnonzero(keep)
    if len(rows) > max_points:
        rows = np.sort(np.random.default_rng(0).choice(rows, max_points, replace=False))
    ax = _axes(ax)
    ax.scatter(xs[rows], ys[rows], s=3, alpha=0.6, lw=0, color=color, rasterized=True)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def rdf(result: Any, *, ax: Any = None, partial: bool = False) -> Any:
    """The radial distribution function, and optionally its partial
    functions, of a :class:`mdapy.analysis.RadialDistributionResult`."""
    ax = _axes(ax)
    if partial:
        for (a, b), g in result.partial.items():
            ax.plot(result.r, g, label=f"{a}–{b}")
        ax.legend()
    else:
        ax.plot(result.r, result.g)
    ax.axhline(1.0, color="0.6", lw=0.5, zorder=0)
    ax.set_xlabel(r"$r$ (Å)")
    ax.set_ylabel(r"$g(r)$")
    ax.set_xlim(0, result.edges[-1])
    return ax


def structure_factor(result: Any, *, ax: Any = None) -> Any:
    """The structure factor of a :class:`mdapy.analysis.StructureFactorResult`."""
    ax = _axes(ax)
    ax.plot(result.q, result.s)
    ax.set_xlabel(r"$q$ (Å$^{-1}$)")
    ax.set_ylabel(r"$S(q)$")
    return ax


def bond_angles(result: Any, *, ax: Any = None) -> Any:
    """The bond angle distribution of a
    :class:`mdapy.analysis.BondAngleResult`."""
    ax = _axes(ax)
    ax.plot(result.angle, result.density)
    ax.set_xlabel("bond angle (°)")
    ax.set_ylabel("density (1/°)")
    ax.set_xlim(0, 180)
    return ax


def msd(
    result: Any,
    *,
    timestep: float = 1.0,
    ax: Any = None,
    fit: tuple[int, int] | None = None,
) -> Any:
    """The mean squared displacement of a
    :class:`mdapy.analysis.DisplacementResult` against time, with a fit of
    the diffusion coefficient.

    Parameters
    ----------
    result : DisplacementResult
        The mean squared displacement.
    timestep : float, optional
        Time between frames in ps. Default 1.
    ax : matplotlib.axes.Axes, optional
        The axes.
    fit : tuple of int, optional
        (first, last) lags of a straight line fitted to the MSD, whose slope
        over 6 is the diffusion coefficient, shown in the legend.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    ax = _axes(ax)
    t = result.lag * timestep
    ax.plot(t, result.msd, label="MSD")
    if fit is not None:
        first, last = fit
        slope, intercept = np.polyfit(t[first:last], result.msd[first:last], 1)
        ax.plot(
            t[first:last],
            slope * t[first:last] + intercept,
            "k--",
            lw=0.6,
            label=f"D = {slope / 6:.3g} Å²/ps",
        )
        ax.legend()
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("MSD (Å²)")
    return ax


def stacking_fault(result: Any, *, ax: Any = None) -> Any:
    """The generalized stacking fault energy of a
    :class:`mdapy.simulation.StackingFaultResult`, in mJ/m²."""
    ax = _axes(ax)
    ax.plot(result.fraction, 1000 * result.energy, marker="o")
    ax.set_xlabel("slip / slip vector")
    ax.set_ylabel(r"$\gamma$ (mJ/m$^2$)")
    ax.set_xlim(0, 1)
    return ax


def band_structure(result: Any, *, ax: Any = None, color: str | None = None) -> Any:
    """The phonon frequencies of a
    :class:`mdapy.lattice_dynamics.BandStructure` along its path, with the
    special points marked and labeled.

    Parameters
    ----------
    result : BandStructure
        The band structure; with ``connected=True`` each branch is one line
        through its crossings.
    ax : matplotlib.axes.Axes, optional
        The axes.
    color : str, optional
        One color for every branch; by default each branch its own.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    ax = _axes(ax)
    # a line for each straight piece, so that breaks of the path stay open
    distances, frequencies = result.distances, result.frequencies
    breaks = np.flatnonzero(np.diff(distances) == 0) + 1
    for k, branch in enumerate(frequencies.T):
        options = {"color": color if color is not None else f"C{k % 10}", "lw": 1.0}
        for piece_d, piece_f in zip(
            np.split(distances, breaks), np.split(branch, breaks)
        ):
            ax.plot(piece_d, piece_f, **options)
    for x, _ in result.labels:
        ax.axvline(x, color="0.75", lw=0.5, zorder=0)
    ax.axhline(0.0, color="0.75", lw=0.5, zorder=0)
    ax.set_xticks([x for x, _ in result.labels], [name for _, name in result.labels])
    ax.set_xlim(distances[0], distances[-1])
    ax.set_ylabel("frequency (THz)")
    return ax


def phonon_dos(result: Any, *, ax: Any = None, projected: bool = False) -> Any:
    """The phonon density of states of a
    :class:`mdapy.lattice_dynamics.PhononDOS`, frequencies upward, to put
    beside a band structure.

    Parameters
    ----------
    result : PhononDOS
        The density of states.
    ax : matplotlib.axes.Axes, optional
        The axes.
    projected : bool, optional
        Also the density of each atom of the cell, when computed.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    ax = _axes(ax)
    ax.plot(result.dos, result.frequencies, color="k", lw=1.0, label="total")
    if projected and result.projected is not None:
        for k, part in enumerate(result.projected.T):
            ax.plot(part, result.frequencies, lw=0.8, label=f"atom {k}")
        ax.legend()
    ax.set_xlabel("DOS (1/THz)")
    ax.set_xlim(0, None)
    return ax


def structure_counts(result: Any, *, ax: Any = None) -> Any:
    """The number of atoms of each type of a structure identification, in
    the colors of the types."""
    ax = _axes(ax)
    counts = result.counts()
    names = list(counts)
    ax.bar(
        names,
        [counts[k] for k in names],
        color=[result.colors[result.names.index(k)] for k in names],
        edgecolor="0.3",
    )
    ax.set_ylabel("atoms")
    return ax
