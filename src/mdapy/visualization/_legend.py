# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A legend on a picture: what the colors of its atoms mean."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from mdapy.visualization._scene import _color, colormap

# sizes relative to the height of the picture, in pixels of its font
_FONT = 0.03
_SMALLEST_FONT = 10.0
_BAR_LENGTH = 7.0
_PANEL_ALPHA = 0.75


def _drawn(figure_size: tuple[int, int], draw: Any) -> np.ndarray:
    """(H, W, 4) floats from 0 to 1 of what draw puts on empty axes in pixels."""
    try:
        from matplotlib import rc_context
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        raise ImportError("a legend needs matplotlib: pip install matplotlib") from None
    height, width = figure_size
    fonts = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    }
    with rc_context(fonts):
        figure = Figure(figsize=(width / 100, height / 100), dpi=100)
        canvas = FigureCanvasAgg(figure)
        figure.patch.set_alpha(0.0)
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_axis_off()
        draw(axes, canvas.get_renderer())
        axes.set_xlim(0, width)
        axes.set_ylim(height, 0)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba(), dtype=float) / 255.0


def _panel(axes: Any, renderer: Any, parts: list, margin: float, ink: Any) -> None:
    """A panel in the color of the background behind what is drawn."""
    from matplotlib.patches import Rectangle

    boxes = [part.get_window_extent(renderer) for part in parts]
    right = max(box.x1 for box in boxes) + 0.5 * margin
    bottom = max(axes.figure.bbox.height - box.y0 for box in boxes) + 0.5 * margin
    axes.add_patch(
        Rectangle(
            (0.5 * margin, 0.5 * margin),
            right - 0.5 * margin,
            bottom - 0.5 * margin,
            facecolor=1.0 - ink,
            alpha=_PANEL_ALPHA,
            edgecolor="none",
            zorder=0,
        )
    )


def _with_legend(
    image: np.ndarray,
    background: Any,
    *,
    names: Sequence[str] = (),
    colors: Any = (),
    title: str = "",
    bar: tuple[float, float, str] | None = None,
) -> np.ndarray:
    """The picture with a legend at its top left: names with colors, or a bar
    from the low to the high value of a colormap."""
    height, width = image.shape[:2]
    font = max(_SMALLEST_FONT, _FONT * height)
    points = font * 0.72  # at 100 dots per inch
    behind = np.asarray(_color(background), dtype=float)
    ink = np.zeros(3) if behind.mean() > 0.5 else np.ones(3)
    x, y = font, font

    def draw(axes: Any, renderer: Any) -> None:
        from matplotlib.patches import Circle

        # the same scale before anything is measured
        axes.set_xlim(0, width)
        axes.set_ylim(height, 0)
        parts, top = [], y
        if title:
            parts.append(
                axes.text(x, top, title, fontsize=points, color=ink, va="top", zorder=2)
            )
            top += 1.4 * font
        for k, name in enumerate(names):
            middle = top + (k + 0.5) * 1.3 * font
            axes.add_patch(
                Circle(
                    (x + 0.4 * font, middle),
                    0.38 * font,
                    facecolor=_color(colors[k]),
                    edgecolor=ink,
                    linewidth=0.5,
                    zorder=2,
                )
            )
            parts.append(
                axes.text(
                    x + 1.1 * font,
                    middle,
                    name,
                    fontsize=points,
                    color=ink,
                    va="center",
                    zorder=2,
                )
            )
        if bar is not None:
            low, high, name = bar
            length = _BAR_LENGTH * font
            steps = np.linspace(1.0, 0.0, 256)
            ramp = colormap(steps, name, (0.0, 1.0))[:, None, :]
            axes.imshow(
                ramp,
                extent=(x, x + font, top + length, top),
                aspect="auto",
                zorder=2,
            )
            for value, where, align in (
                (high, top, "top"),
                (low, top + length, "bottom"),
            ):
                parts.append(
                    axes.text(
                        x + 1.3 * font,
                        where,
                        f"{value:.3g}",
                        fontsize=points,
                        color=ink,
                        va=align,
                        zorder=2,
                    )
                )
            # the text of the bar is not lower than the bar, and imshow set limits
            axes.set_xlim(0, width)
            axes.set_ylim(height, 0)
        _panel(axes, renderer, parts, font, ink)

    overlay = _drawn((height, width), draw)
    alpha = overlay[:, :, 3:]
    result = image.astype(float) / 255.0
    result[:, :, :3] = overlay[:, :, :3] * alpha + result[:, :, :3] * (1.0 - alpha)
    result[:, :, 3] = np.maximum(result[:, :, 3], alpha[:, :, 0])
    return np.rint(255.0 * result).astype(np.uint8)
