# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A legend beside a picture: what the colors of its atoms mean."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from mdapy.visualization._scene import _color, colormap

# sizes relative to the height of the picture, in pixels of its font
_FONT = 0.03
_SMALLEST_FONT = 10.0
_BAR_LENGTH = 7.0
_FONTS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}


def _matplotlib() -> tuple[Any, Any, Any]:
    try:
        from matplotlib import rc_context
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        raise ImportError("a legend needs matplotlib: pip install matplotlib") from None
    return rc_context, FigureCanvasAgg, Figure


def _font(height: int) -> float:
    """The size of the text of the legend of a picture, in pixels."""
    return max(_SMALLEST_FONT, _FONT * height)


def _drawn(figure_size: tuple[int, int], draw: Any) -> np.ndarray:
    """(H, W, 4) floats from 0 to 1 of what draw puts on empty axes in pixels."""
    rc_context, FigureCanvasAgg, Figure = _matplotlib()
    height, width = figure_size
    with rc_context(_FONTS):
        figure = Figure(figsize=(width / 100, height / 100), dpi=100)
        canvas = FigureCanvasAgg(figure)
        figure.patch.set_alpha(0.0)
        axes = figure.add_axes((0.0, 0.0, 1.0, 1.0))
        axes.set_axis_off()
        draw(axes)
        axes.set_xlim(0, width)
        axes.set_ylim(height, 0)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba(), dtype=float) / 255.0


def _text_width(texts: Sequence[str], points: float) -> float:
    """The width in pixels of the widest of the texts in this size."""
    rc_context, FigureCanvasAgg, Figure = _matplotlib()
    with rc_context(_FONTS):
        figure = Figure(figsize=(1.0, 1.0), dpi=100)
        renderer = FigureCanvasAgg(figure).get_renderer()
        return max(
            (
                figure.text(0.0, 0.0, text, fontsize=points)
                .get_window_extent(renderer)
                .width
                for text in texts
            ),
            default=0.0,
        )


def _legend_width(
    height: int,
    *,
    names: Sequence[str] = (),
    colors: Any = (),
    title: str = "",
    bar: tuple[float, float, str] | None = None,
) -> int:
    """The width in pixels of the column of a legend beside a picture of
    this height, its margins included."""
    font = _font(height)
    points = font * 0.72  # at 100 dots per inch
    widths = [_text_width([title], points)] if title else []
    if names:
        widths.append(1.1 * font + _text_width(names, points))
    if bar is not None:
        low, high, _ = bar
        widths.append(1.3 * font + _text_width([f"{low:.3g}", f"{high:.3g}"], points))
    return int(np.ceil(2.0 * font + max(widths, default=0.0)))


def _with_legend(
    image: np.ndarray,
    background: Any,
    *,
    names: Sequence[str] = (),
    colors: Any = (),
    title: str = "",
    bar: tuple[float, float, str] | None = None,
    column: int,
) -> np.ndarray:
    """The picture with a column of this width at its left, in the color of
    the background, and a legend at the top left: names with colors, or a bar
    from the low to the high value of a colormap."""
    height = image.shape[0]
    side = np.zeros((height, column, 4), dtype=image.dtype)
    if background is not None:
        side[:, :, :3] = np.rint(255.0 * _color(background))
        side[:, :, 3] = 255
    image = np.concatenate([side, image], axis=1)
    width = image.shape[1]
    font = _font(height)
    points = font * 0.72  # at 100 dots per inch
    behind = np.ones(3) if background is None else _color(background)
    ink = np.zeros(3) if behind.mean() > 0.5 else np.ones(3)
    x, y = font, font

    def draw(axes: Any) -> None:
        from matplotlib.patches import Circle
        from matplotlib.patheffects import withStroke

        # a thin edge in the color of the background keeps text over atoms legible
        words = dict(
            fontsize=points,
            color=ink,
            zorder=2,
            path_effects=[withStroke(linewidth=0.15 * font, foreground=1.0 - ink)],
        )
        top = y
        if title:
            axes.text(x, top, title, va="top", **words)
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
            axes.text(x + 1.1 * font, middle, name, va="center", **words)
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
                axes.text(x + 1.3 * font, where, f"{value:.3g}", va=align, **words)

    overlay = _drawn((height, width), draw)
    alpha = overlay[:, :, 3:]
    result = image.astype(float) / 255.0
    result[:, :, :3] = overlay[:, :, :3] * alpha + result[:, :, :3] * (1.0 - alpha)
    result[:, :, 3] = np.maximum(result[:, :, 3], alpha[:, :, 0])
    return np.rint(255.0 * result).astype(np.uint8)
