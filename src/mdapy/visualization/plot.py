# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Plot templates for figures of journals, and plots of the results of mdapy.
Needs matplotlib."""

from collections.abc import Iterator, Mapping, Sequence
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
    "mathtext.default": "regular",
}


def _pyplot() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - matplotlib is a test dependency
        raise ImportError("plots need matplotlib: pip install matplotlib") from None
    return plt


@contextmanager
def style(
    *,
    font_size: float | None = None,
    line_width: float | None = None,
    colors: str | Sequence[str] | None = None,
    settings: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """A context in which matplotlib uses ``STYLE`` and ``COLORS``, changed by
    the options.

    Parameters
    ----------
    font_size : float64, optional
        Size of the text in points; tick labels and legends 6/7 of it.
        Default 7.
    line_width : float64, optional
        Width of the lines of data in points. Default 1.
    colors : str or sequence of str, optional
        The colors that lines and points take in turn. Default ``COLORS``.
    settings : mapping, optional
        Any other settings of matplotlib (its rcParams) by their names, such
        as ``{"axes.grid": True}``; they take precedence over the options.

    Raises
    ------
    InvalidParameter
        For a size or width that is not positive, no colors, a color or a
        setting that matplotlib does not know.
    """
    plt = _pyplot()
    from cycler import cycler

    chosen = dict(STYLE)
    if font_size is not None:
        if not font_size > 0:
            raise InvalidParameter(f"font_size must be positive, not {font_size!r}")
        small = float(font_size) * 6 / 7
        for key in ("font.size", "axes.titlesize", "axes.labelsize"):
            chosen[key] = float(font_size)
        for key in ("xtick.labelsize", "ytick.labelsize", "legend.fontsize"):
            chosen[key] = small
    if line_width is not None:
        if not line_width > 0:
            raise InvalidParameter(f"line_width must be positive, not {line_width!r}")
        chosen["lines.linewidth"] = float(line_width)
    try:
        if colors is None:
            palette = list(COLORS)
        else:
            palette = [colors] if isinstance(colors, str) else list(colors)
    except TypeError:
        raise InvalidParameter(
            f"colors must be a color or a sequence of colors, not {colors!r}"
        ) from None
    if not palette:
        raise InvalidParameter("colors must hold at least one color")
    chosen["axes.prop_cycle"] = cycler(color=palette)
    try:
        chosen |= dict(settings or {})
    except (TypeError, ValueError):
        raise InvalidParameter(
            f"settings must map names of matplotlib's settings to values, not {settings!r}"
        ) from None
    try:  # settings of matplotlib check each value, as rc_context would
        plt.matplotlib.RcParams(chosen)
    except (KeyError, ValueError) as error:
        raise InvalidParameter(f"matplotlib: {error}") from None
    try:
        with plt.rc_context(chosen):
            try:
                yield
                # a notebook draws its figures after the cell, when the style
                # no longer holds: they are shown here, in the style
                if "inline" in plt.get_backend():
                    plt.show()
            finally:
                # a notebook loads its backend at the first figure, which turns
                # interactive mode on: leaving the context must not turn it off,
                # or later figures of the notebook do not show by themselves
                interactive = plt.isinteractive()
    finally:
        plt.ion() if interactive else plt.ioff()


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
    width : {"single", "double"} or float64, optional
        One column (88 mm), two (180 mm), or a width in mm. Default single.
    height : float64, optional
        Height in mm. Default 3/4 of the width of one panel times the rows.
    nrows, ncols : int32, optional
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
    # axes in 3D place text in the plane of the figure with text2D
    write = ax.text2D if getattr(ax, "name", None) == "3d" else ax.text
    write(
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
    fig: Any,
    path: str | PathLike,
    *,
    formats: Sequence[str] = ("pdf", "svg", "png"),
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
    reference, predicted : array_like of float64
        The values, of the same size.
    ax : matplotlib.axes.Axes, optional
        The axes; by default a new figure.
    label : str, optional
        Name of the set in the legend.
    unit : str, optional
        Unit of the error in the legend.
    scale : float64, optional
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
    ax.set_xlabel("reference")
    ax.set_ylabel("predicted")
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
    max_points : int32, optional
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
    functions, of a ``mdapy.analysis.RadialDistributionResult``."""
    ax = _axes(ax)
    if partial:
        for (a, b), g in result.partial.items():
            ax.plot(result.r, g, label=f"{a}–{b}")
        ax.legend()
    else:
        ax.plot(result.r, result.g)
    ax.axhline(1.0, color="0.6", lw=0.5, zorder=0)
    ax.set_xlabel("r (Å)")
    ax.set_ylabel("g(r)")
    ax.set_xlim(0, result.edges[-1])
    return ax


def warren_cowley(
    result: Any,
    *,
    ax: Any = None,
    shell: int | None = None,
    vmin: float = -2.0,
    vmax: float = 1.0,
    cmap: str = "GnBu",
) -> Any:
    """The Warren-Cowley parameters as a matrix of colors with their values
    written in, the central element along x and its neighbor along y.

    Parameters
    ----------
    result : mdapy.analysis.WarrenCowleyResult
        The parameters.
    ax : matplotlib.axes.Axes, optional
        The axes; by default a new figure.
    shell : int32, optional
        The shell to draw, from 0, of parameters of several shells.
    vmin, vmax : float64, optional
        The values at the two ends of the colors. Default -2 and 1.
    cmap : str, optional
        The name of a colormap of matplotlib. Default "GnBu".

    Returns
    -------
    matplotlib.axes.Axes
        The axes.

    Raises
    ------
    InvalidParameter
        Without a shell that exists for parameters of several shells, or with
        one for parameters of one shell.
    """
    alpha = result.alpha
    if alpha.ndim == 2:
        if shell is not None:
            raise InvalidParameter("the parameters are of one shell")
    elif shell is None:
        raise InvalidParameter(
            f"the parameters are of {len(alpha)} shells: choose one with shell"
        )
    elif not 0 <= shell < len(alpha):
        raise InvalidParameter(f"shell {shell} is not among {len(alpha)} shells")
    else:
        alpha = alpha[shell]
    ax = _axes(ax)
    image = ax.imshow(alpha.T, vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    positions = np.arange(len(result.elements))
    ax.set_xticks(positions, result.elements)
    ax.set_yticks(positions, result.elements)
    for a, b in np.ndindex(alpha.shape):
        value = alpha[a, b]
        dark = value > vmin + 0.75 * (vmax - vmin)
        ax.text(
            a,
            b,
            # no atoms to tell, and no -0.00
            "–" if np.isnan(value) else f"{round(value, 2) + 0.0:.2f}",
            ha="center",
            va="center",
            color="w" if dark else "k",
        )
    ax.set_xlabel("Central element")
    ax.set_ylabel("Neighboring element")
    bar = ax.figure.colorbar(image, ax=ax, ticks=[vmin, 0.0, vmax])
    bar.set_label("Warren–Cowley α")
    return ax


def structure_factor(result: Any, *, ax: Any = None) -> Any:
    """The structure factor of a ``mdapy.analysis.StructureFactorResult``."""
    ax = _axes(ax)
    ax.plot(result.q, result.s)
    ax.set_xlabel("q (1/Å)")
    ax.set_ylabel("S(q)")
    return ax


def bond_angles(result: Any, *, ax: Any = None) -> Any:
    """The bond angle distribution of a
    ``mdapy.analysis.BondAngleResult``."""
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
    ``mdapy.analysis.DisplacementResult`` against time, with a fit of
    the diffusion coefficient.

    Parameters
    ----------
    result : DisplacementResult
        The mean squared displacement.
    timestep : float64, optional
        Time between frames in ps. Default 1.
    ax : matplotlib.axes.Axes, optional
        The axes.
    fit : tuple of int32, optional
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


def stacking_fault(result: Any, *, ax: Any = None, **options: Any) -> Any:
    """The generalized stacking fault energy of a
    ``mdapy.simulation.StackingFaultResult``, in mJ/m²: a point for each
    slip computed, joined by a line. ``options`` go to ``ax.plot``, such as
    ``label``."""
    ax = _axes(ax)
    ax.plot(result.fraction, 1000 * result.energy, **({"marker": "o"} | options))
    ax.set_xlabel("slip / slip vector")
    ax.set_ylabel("γ (mJ/m²)")
    ax.set_xlim(0, 1)
    return ax


def band_structure(result: Any, *, ax: Any = None, color: str | None = None) -> Any:
    """The phonon frequencies of a
    ``mdapy.lattice_dynamics.BandStructure`` along its path, with the
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
    ``mdapy.lattice_dynamics.PhononDOS``, frequencies upward, to put
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


def phase_diagram(result: Any, *, ax: Any = None, labels: bool = True) -> Any:
    """The boundaries of the phases of a
    ``mdapy.simulation.BinaryPhaseDiagram``: where each tie line ends, against
    the temperature, with the name of each phase in the middle of where it is the
    only one.

    Parameters
    ----------
    result : BinaryPhaseDiagram
        The phase diagram.
    ax : matplotlib.axes.Axes, optional
        The axes.
    labels : bool, optional
        Write the names of the phases. Default True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    ax = _axes(ax)
    ends: dict[tuple[str, str, int], list[tuple[float, float]]] = {}
    for line in result.tie_lines:
        pair = (line.first, line.second)
        ends.setdefault((*pair, 0), []).append((line.first_fraction, line.temperature))
        ends.setdefault((*pair, 1), []).append((line.second_fraction, line.temperature))
    for points in ends.values():
        x, t = np.array(points).T
        # a new line where the boundary jumps: a gap that closed and opened
        for piece_x, piece_t in zip(
            np.split(x, np.flatnonzero(np.abs(np.diff(x)) > 0.1) + 1),
            np.split(t, np.flatnonzero(np.abs(np.diff(x)) > 0.1) + 1),
        ):
            ax.plot(piece_x, piece_t, color="black", lw=1.0)
    if labels:
        for k, name in enumerate(result.phases):
            rows, columns = np.nonzero(result.stable == k)
            if len(rows):
                ax.text(
                    result.fractions[columns].mean(),
                    result.temperatures[rows].mean(),
                    name,
                    ha="center",
                    va="center",
                )
    ax.set_xlim(0, 1)
    ax.set_ylim(result.temperatures.min(), result.temperatures.max())
    ax.set_xlabel(f"fraction of {result.elements[1]}")
    ax.set_ylabel("temperature (K)")
    return ax


def pressure_temperature(result: Any, *, ax: Any = None, labels: bool = True) -> Any:
    """The boundaries of the phases of a
    ``mdapy.simulation.PressureTemperatureDiagram``: the temperature of each
    against the pressure, with the name of each phase in the middle of where it
    is stable.

    Parameters
    ----------
    result : PressureTemperatureDiagram
        The phase diagram.
    ax : matplotlib.axes.Axes, optional
        The axes.
    labels : bool, optional
        Write the names of the phases. Default True.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    ax = _axes(ax)
    for boundary in result.boundaries:
        ax.plot(boundary.pressures, boundary.temperatures, color="black", marker="o")
    if labels:
        for k, name in enumerate(result.phases):
            rows, columns = np.nonzero(result.stable == k)
            if len(rows):
                ax.text(
                    result.pressures[rows].mean(),
                    result.temperatures[columns].mean(),
                    name,
                    ha="center",
                    va="center",
                )
    ax.set_xlabel("Pressure (GPa)")
    ax.set_ylabel("Temperature (K)")
    return ax


#: The quantities of elastic constants along directions that ``elastic_surface``
#: and ``elastic_section`` draw: their labels, units, and whether they take a
#: second direction across the first.
_ELASTIC_QUANTITIES = {
    "young_modulus": ("Young's modulus", "GPa", False),
    "linear_compressibility": ("linear compressibility", "1/GPa", False),
    "shear_modulus": ("shear modulus", "GPa", True),
    "poisson_ratio": ("Poisson ratio", "", True),
}


def _elastic_quantity(quantity: str, extreme: str | None) -> str:
    """The label of a quantity, its unit in parentheses after it."""
    if quantity not in _ELASTIC_QUANTITIES:
        raise InvalidParameter(
            f"quantity must be one of {list(_ELASTIC_QUANTITIES)}, not {quantity!r}"
        )
    name, _, across = _ELASTIC_QUANTITIES[quantity]
    if across:
        if extreme not in ("min", "max"):
            raise InvalidParameter(
                f'{quantity} takes a second direction: extreme must be "min" or "max", '
                f"its smallest or largest value across each direction, not {extreme!r}"
            )
        name = ("smallest " if extreme == "min" else "largest ") + name
    elif extreme is not None:
        raise InvalidParameter(f"{quantity} takes no extreme, not {extreme!r}")
    return name


def _across(directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to each unit vector and to each other."""
    axis = np.eye(3)[np.argmin(np.abs(directions), axis=-1)]
    first = np.cross(directions, axis)
    first /= np.linalg.norm(first, axis=-1, keepdims=True)
    return first, np.cross(directions, first)


def _elastic_values(
    result: Any, quantity: str, extreme: str | None, directions: np.ndarray
) -> np.ndarray:
    """A quantity along unit vectors (N, 3); of a quantity of two directions,
    its smallest or largest value over the directions across each."""
    if not _ELASTIC_QUANTITIES[quantity][2]:
        return getattr(result, quantity)(directions)
    # across a direction, the inverse of the shear modulus and the Poisson ratio
    # are quadratic forms of the second direction: a + b cos 2t + c sin 2t over
    # the angle t in the plane across, known from three angles
    first, second = _across(directions)
    if quantity == "shear_modulus":
        values = [
            1 / result.shear_modulus(normal=directions, direction=m)
            for m in (first, second, first + second)
        ]
    else:
        values = [
            result.poisson_ratio(directions, m) for m in (first, second, first + second)
        ]
    mean = (values[0] + values[1]) / 2
    swing = np.hypot((values[0] - values[1]) / 2, values[2] - mean)
    low, high = mean - swing, mean + swing
    if quantity == "shear_modulus":
        low, high = 1 / high, 1 / low
    return low if extreme == "min" else high


def elastic_surface(
    result: Any,
    quantity: str = "young_modulus",
    *,
    ax: Any = None,
    extreme: str | None = None,
    points: int = 60,
    colormap: str = "viridis",
) -> Any:
    """A quantity of an ``mdapy.lattice_dynamics.ElasticResult`` along every
    direction, as a surface in 3D: along each direction it lies as far from
    the origin as the size of the value there, in the color of the value.

    Parameters
    ----------
    result : ElasticResult
        The elastic constants.
    quantity : {"young_modulus", "linear_compressibility", "shear_modulus", "poisson_ratio"}, optional
        Default "young_modulus".
    ax : matplotlib.axes.Axes, optional
        Axes in 3D, made with ``projection="3d"``; by default a new figure.
    extreme : {"min", "max"}, optional
        Of the shear modulus and the Poisson ratio, which take a second
        direction across the first: their smallest or largest value over it.
    points : int32, optional
        Directions from the z axis to its opposite; twice as many around it.
        Default 60.
    colormap : str, optional
        The name of a colormap of matplotlib. Default "viridis".

    Returns
    -------
    matplotlib.axes.Axes
        The axes.

    Raises
    ------
    InvalidParameter
        For another quantity, no extreme of a quantity of two directions or one
        of another, fewer than 3 points, a colormap that matplotlib does not
        know, or axes not in 3D.
    """
    plt = _pyplot()
    name = _elastic_quantity(quantity, extreme)
    unit = _ELASTIC_QUANTITIES[quantity][1]
    if points < 3:
        raise InvalidParameter(f"points must be at least 3, not {points}")
    try:
        colors = plt.matplotlib.colormaps[colormap]
    except KeyError:
        raise InvalidParameter(f"matplotlib has no colormap {colormap!r}") from None
    if ax is None:
        ax = figure(subplot_kw={"projection": "3d"})[1]
    elif getattr(ax, "name", None) != "3d":
        raise InvalidParameter('ax must be axes in 3D, made with projection="3d"')
    polar, azimuth = np.meshgrid(
        np.linspace(0, np.pi, points),
        np.linspace(0, 2 * np.pi, 2 * points),
        indexing="ij",
    )
    directions = np.stack(
        [
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ],
        axis=-1,
    )
    values = _elastic_values(
        result, quantity, extreme, directions.reshape(-1, 3)
    ).reshape(polar.shape)
    surface = np.abs(values)[..., None] * directions
    scale = plt.matplotlib.colors.Normalize(values.min(), values.max())
    # no shading: the color of each face is its value on the color bar
    ax.plot_surface(
        *surface.transpose(2, 0, 1),
        facecolors=colors(scale(values)),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    reach = float(np.abs(values).max())
    for axis in "xyz":
        getattr(ax, f"set_{axis}lim")(-reach, reach)
        getattr(ax, f"set_{axis}label")(f"{axis} ({unit})" if unit else axis)
        getattr(ax, f"{axis}axis").pane.fill = False
        getattr(ax, f"{axis}axis").set_major_locator(
            plt.matplotlib.ticker.MaxNLocator(4)
        )
    ax.set_box_aspect((1, 1, 1))
    bar = ax.figure.colorbar(
        plt.matplotlib.cm.ScalarMappable(scale, colors), ax=ax, shrink=0.6, pad=0.12
    )
    bar.set_label(f"{name} ({unit})" if unit else name)
    return ax


def _indices(vector: np.ndarray) -> str:
    """A direction as small whole numbers, such as [2 -1 -1], when it is one."""
    v = vector / np.abs(vector).max()
    for scale in range(1, 7):
        whole = np.round(v * scale)
        if np.allclose(v * scale, whole, atol=1e-6):
            return "[" + " ".join(f"{int(k)}" for k in whole + 0.0) + "]"
    return "[" + " ".join(f"{k:.2f}" for k in np.round(vector, 2) + 0.0) + "]"


def elastic_section(
    result: Any,
    quantity: str = "young_modulus",
    normal: ArrayLike = (0, 0, 1),
    *,
    ax: Any = None,
    extreme: str | None = None,
    points: int = 360,
    color: str | None = None,
    label: str | None = None,
) -> Any:
    """A quantity of an ``mdapy.lattice_dynamics.ElasticResult`` along the
    directions of a plane: a curve as far from the origin along each direction
    as the size of the value there, dashed where the value is negative.

    Parameters
    ----------
    result : ElasticResult
        The elastic constants.
    quantity : {"young_modulus", "linear_compressibility", "shear_modulus", "poisson_ratio"}, optional
        Default "young_modulus".
    normal : array_like, optional
        The normal (3,) float64 of the plane, in the axes of the box. Default
        (0, 0, 1). The first axis of the plot is the axis of the box most
        nearly in the plane, laid into it.
    ax : matplotlib.axes.Axes, optional
        The axes; by default a new figure.
    extreme : {"min", "max"}, optional
        Of the shear modulus and the Poisson ratio, which take a second
        direction across the first: their smallest or largest value over it,
        in any direction across, not only in the plane.
    points : int32, optional
        Directions around the plane. Default 360.
    color : str, optional
        Color of the curve.
    label : str, optional
        Name of the curve in a legend.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.

    Raises
    ------
    InvalidParameter
        For another quantity, no extreme of a quantity of two directions or one
        of another, a normal of another shape, zero or not finite, or fewer
        than 3 points.
    """
    _elastic_quantity(quantity, extreme)
    unit = _ELASTIC_QUANTITIES[quantity][1]
    n = np.asarray(normal, dtype=np.float64)
    if n.shape != (3,) or not np.all(np.isfinite(n)) or not np.any(n):
        raise InvalidParameter(
            f"normal must be a vector (3,), finite and not zero, not {normal!r}"
        )
    if points < 3:
        raise InvalidParameter(f"points must be at least 3, not {points}")
    n = n / np.linalg.norm(n)
    nearest = np.eye(3)[np.argmin(np.abs(n))]
    first = nearest - (nearest @ n) * n
    first /= np.linalg.norm(first)
    second = np.cross(n, first)
    angles = np.linspace(0, 2 * np.pi, points + 1)
    directions = np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    values = _elastic_values(result, quantity, extreme, directions)
    x, y = np.abs(values) * np.cos(angles), np.abs(values) * np.sin(angles)
    ax = _axes(ax)
    negative = values < 0
    # each part reaches into the next, so that the curve has no gap
    near = negative | np.roll(negative, 1) | np.roll(negative, -1)
    if not negative.all():
        keep = ~negative | np.roll(~negative, 1) | np.roll(~negative, -1)
        line = ax.plot(
            np.where(keep, x, np.nan),
            np.where(keep, y, np.nan),
            color=color,
            label=label,
        )
        color = line[0].get_color()
        label = None
    if negative.any():
        ax.plot(
            np.where(near, x, np.nan),
            np.where(near, y, np.nan),
            color=color,
            ls="--",
            label=label,
        )
    ax.axhline(0, color="0.8", lw=0.5, zorder=0)
    ax.axvline(0, color="0.8", lw=0.5, zorder=0)
    ax.set_aspect("equal", adjustable="datalim")
    suffix = f" ({unit})" if unit else ""
    ax.set_xlabel(_indices(first) + suffix)
    ax.set_ylabel(_indices(second) + suffix)
    return ax
