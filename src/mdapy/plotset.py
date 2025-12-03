# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is part of the mdapy project, released under the BSD 3-Clause License.

"""
Plotting Utilities for Scientific Publications
===============================================

This module provides utilities for creating publication-quality figures with Matplotlib.
It includes functions for configuring plot styles, creating figures with consistent
formatting, and saving figures with uniform margins.

The module is designed to produce figures suitable for scientific papers and presentations,
with sensible defaults for font sizes, line widths, tick marks, and color schemes.

Note: This module requires matplotlib to be installed. Install it with:
    pip install matplotlib

Functions
---------
set_figure : Create a figure with scientific style settings
save_figure : Save a figure with uniform whitespace margins
_pltset : Configure global Matplotlib style (internal)
_cm2inch : Convert centimeters to inches (internal)
_ensure_matplotlib : Check matplotlib availability (internal)

Examples
--------
Basic usage for creating a simple plot:

>>> import numpy as np
>>> fig, ax = set_figure(figsize=(8.5, 7.0))
>>> x = np.linspace(0, 2 * np.pi, 100)
>>> ax.plot(x, np.sin(x), label="sin(x)")
>>> ax.set_xlabel("x")
>>> ax.set_ylabel("y")
>>> ax.legend()
>>> save_figure(fig, "output.png")

Creating a multi-panel figure:

>>> fig, axes = set_figure(figsize=(17, 7), nrow=1, ncol=2)
>>> for i, ax in enumerate(axes):
...     ax.plot(x, np.sin(x * (i + 1)))
...     ax.set_xlabel("x")
...     ax.set_ylabel("y")
>>> save_figure(fig, "multi_panel.pdf")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, List, Tuple, Any, Literal
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def _ensure_matplotlib() -> None:
    """
    Ensure matplotlib and its dependencies are available.

    This function performs lazy import of matplotlib and cycler, only loading
    them when plotting functions are actually called. This allows the parent
    package to be installed without matplotlib as a hard dependency.

    Raises
    ------
    ImportError
        If matplotlib or cycler is not installed, with instructions on how to install.

    Notes
    -----
    This function imports matplotlib modules into the local namespace of each
    calling function, not into the module namespace.
    """
    try:
        import matplotlib.pyplot  # noqa: F401
        from matplotlib.figure import Figure  # noqa: F401
        from matplotlib.axes import Axes  # noqa: F401
        from matplotlib.transforms import Bbox  # noqa: F401
        from cycler import cycler  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Matplotlib is required for plotting functionality but is not installed.\n"
            "Please install it using one of the following methods:\n"
            "  pip install matplotlib\n"
            f"Original error: {e}"
        ) from e


def _pltset(
    color_cycler: Optional[Union[List[str], Tuple[str, ...]]] = None, **kwargs: Any
) -> None:
    """
    Configure global Matplotlib style optimized for scientific publications.

    Parameters
    ----------
    color_cycler : list of str or tuple of str, optional
        Custom color palette for plot lines.
    **kwargs : Any
        Additional keyword arguments to override specific rcParams settings.
    """
    _ensure_matplotlib()

    # Import here after ensuring matplotlib is available
    import matplotlib.pyplot as plt
    from cycler import cycler

    plt.rcParams.clear()

    if color_cycler is None:
        color_cycler = [
            "#4477AA",  # Blue
            "#EE6677",  # Red
            "#228833",  # Green
            "#CCBB44",  # Yellow
            "#66CCEE",  # Cyan
            "#AA3377",  # Purple
            "#BBBBBB",  # Gray
        ]

    plt.rcParams["axes.prop_cycle"] = cycler("color", color_cycler)

    plt.rcParams.update(
        {
            # X-axis tick configuration
            "xtick.direction": "in",
            "xtick.major.size": 3,
            "xtick.major.width": 0.6,
            "xtick.minor.size": 1.5,
            "xtick.minor.width": 0.6,
            "xtick.top": True,
            "xtick.minor.visible": False,
            # Y-axis tick configuration
            "ytick.direction": "in",
            "ytick.major.size": 3,
            "ytick.major.width": 0.6,
            "ytick.minor.size": 1.5,
            "ytick.minor.width": 0.6,
            "ytick.right": True,
            "ytick.minor.visible": False,
            # Line and axes styling
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.2,
            "lines.markersize": 3,
            # Font configuration
            "font.weight": "normal",
            "font.size": 10.0,
            "axes.labelweight": "normal",
            "legend.frameon": False,
            "legend.fontsize": 9.0,
            "axes.titlesize": 9.0,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Arial", "cmr10"],
            # Mathematical text
            "axes.formatter.use_mathtext": True,
            "mathtext.fontset": "cm",
        }
    )

    # Apply custom overrides
    for key, value in kwargs.items():
        if key in plt.rcParams:
            plt.rcParams[key] = value
        else:
            print(f"Warning: '{key}' is not a valid rcParam key and will be ignored.")


def _cm2inch(value: Union[float, int]) -> float:
    """
    Convert centimeters to inches for Matplotlib figure sizing.

    Parameters
    ----------
    value : float or int
        Size in centimeters to convert.

    Returns
    -------
    float
        Equivalent size in inches.
    """
    return value / 2.54


def set_figure(
    figsize: Tuple[Union[float, int], Union[float, int]] = (8.5, 7.0),
    figdpi: int = 150,
    nrow: int = 1,
    ncol: int = 1,
    color_cycler: Optional[Union[List[str], Tuple[str, ...]]] = None,
    **kwargs: Any,
) -> tuple[Figure, Union[Axes, List[Axes], List[List[Axes]]]]:
    """
    Create a Matplotlib figure and axes with scientific publication style.

    Parameters
    ----------
    figsize : tuple of (float or int, float or int), default=(8.5, 7.0)
        Figure dimensions in centimeters as (width, height).
    figdpi : int, default=150
        Figure resolution in dots per inch.
    nrow : int, default=1
        Number of subplot rows.
    ncol : int, default=1
        Number of subplot columns.
    color_cycler : list of str or tuple of str, optional
        Custom color palette for the figure.
    **kwargs : Any
        Additional keyword arguments passed to `_pltset`.

    Returns
    -------
    fig : Figure
        The created figure object.
    ax : Axes or numpy.ndarray of Axes
        Single Axes or array of Axes objects.
    """
    _ensure_matplotlib()

    # Import here after ensuring matplotlib is available
    import matplotlib.pyplot as plt

    _pltset(color_cycler=color_cycler, **kwargs)
    figsize_inches = tuple(_cm2inch(size) for size in figsize)

    fig, ax = plt.subplots(
        nrow,
        ncol,
        figsize=figsize_inches,
        dpi=figdpi,
        constrained_layout=True,
    )

    # Ensure uniform whitespace margins (in inches)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    if isinstance(ax, np.ndarray):
        ax = ax.tolist()

    return fig, ax


def save_figure(
    fig: Figure,
    filename: str,
    dpi: int = 300,
    format: Literal["png", "pdf", "svg", "eps", "tiff"] = "png",
    transparent: bool = True,
    pad_scale: float = 1.02,
) -> None:
    """
    Save a figure with uniform whitespace margins on all sides.

    Parameters
    ----------
    fig : Figure
        The figure object to save.
    filename : str
        Output filename.
    dpi : int, default=300
        Resolution in dots per inch.
    format : {'png', 'pdf', 'svg', 'eps', 'tiff'}, default='png'
        Output file format.
    transparent : bool, default=True
        Whether to use a transparent background.
    pad_scale : float, default=1.02
        Scale factor to expand the bounding box.
    """
    _ensure_matplotlib()

    # Import here after ensuring matplotlib is available
    from matplotlib.transforms import Bbox

    # Add file extension if not present
    if "." not in filename:
        filename = f"{filename}.{format}"

    # Force a draw to ensure all elements are rendered
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get tight bounding boxes for all axes (in pixel coordinates)
    bboxes = [ax.get_tightbbox(renderer) for ax in fig.axes]
    if not bboxes:
        raise ValueError("No Axes found in figure.")

    # Union all bounding boxes and expand uniformly
    tight_bbox = Bbox.union(bboxes)
    tight_bbox = tight_bbox.expanded(pad_scale, pad_scale)

    # Convert from pixel coordinates to inch coordinates
    tight_bbox_inch = tight_bbox.transformed(fig.dpi_scale_trans.inverted())

    # Save the figure with the computed bounding box
    fig.savefig(
        filename,
        dpi=dpi,
        format=format,
        transparent=transparent,
        bbox_inches=tight_bbox_inch,
    )


if __name__ == "__main__":
    fig, axes = set_figure(
        figsize=(17, 14),
        ncol=2,
        nrow=2,
        **{
            "font.size": 10.0,
            "lines.linewidth": 1.4,
        },
    )
    x = np.linspace(0, 7, 100)

    for i, j in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        ax = axes[j[0]][j[1]]
        ax.plot(x, np.sin(x) * (i + 1), label=f"{i + 1}·sin(x)")
        ax.set_xlabel("X (arb)")
        ax.set_ylabel("Y (arb)")
        ax.legend()

    # save_figure(fig, "test_uniform.png", transparent=False)
    import matplotlib.pyplot as plt

    plt.show()
