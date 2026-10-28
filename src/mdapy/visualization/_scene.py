# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Pictures of atoms: a scene of spheres, cylinders and polygons, rendered by
ray tracing."""

import struct
import zlib
from collections.abc import Sequence
from os import PathLike
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from mdapy._core import visualization as _native
from mdapy.core.box import Box
from mdapy.core.threads import _native_token
from mdapy.elements import COVALENT_RADII, SYMBOLS, VDW_RADII
from mdapy.errors import InvalidParameter
from mdapy.visualization._elements import (
    DEFAULT,
    ELEMENTS,
    MISSING_COVALENT,
    MISSING_VDW,
)

#: The looks of a picture: lights, shadows, ambient occlusion and outlines.
STYLES: dict[str, dict[str, Any]] = {
    # quick, as the viewports of OVITO: lit in sRGB by an ambient part, a
    # light from the viewer and a broad highlight from the upper left, with
    # neither occlusion nor shadows
    "fast": {
        "srgb": True,
        "ambient": 0.4,
        "diffuse": 0.6,
        "key": (0.0, 0.0, 1.0),
        "highlight": (-1.8, 1.5, -0.2),
        "specular": 0.25,
        "shininess": 6.0,
        "fill_intensity": 0.0,
        "ao_samples": 0,
        "shadows": False,
    },
    # matte spheres, after the pictures of the Tachyon ray tracer as OVITO
    # sets it up: lit in sRGB by a bright sky that ambient occlusion darkens,
    # a weak light from the viewer with sharp shadows and a little light
    # everywhere
    "matte": {
        "srgb": True,
        "base_light": 0.3,
        "ambient": 0.64,
        "ao_strength": 1.0,
        "ao_samples": 12,
        "ao_distance": float("inf"),
        "diffuse": 0.144,
        "key": (-0.2, 0.2, 1.0),
        "shadow_strength": 1.0,
        "fill_intensity": 0.0,
        "specular": 0.0,
        "samples": 3,
    },
    # shiny spheres with a metallic sheen, after the pictures of OSPRay in
    # OVITO: light from a sky and a wide key light that bounces once between
    # the atoms, soft shadows and small bright highlights
    "shiny": {
        "bounces": 1,
        "samples": 32,
        "light_size": 0.087,
        "ambient": 0.7,
        "diffuse": 0.5,
        "key": (-0.5, 0.5, 1.0),
        "specular": 0.7,
        "shininess": 40.0,
        "metalness": 0.4,
    },
    # plain colors with every atom outlined, as ASE draws atoms
    "flat": {
        "flat": True,
        "ao_samples": 0,
        "shadows": False,
        "outline_width": 1.0,
        "outline_depth": 0.0,
    },
}

_DEFAULTS = {
    "ambient": 0.35,
    "diffuse": 0.75,
    "specular": 0.25,
    "shininess": 40.0,
    "metalness": 0.0,
    "key": (-0.45, 0.6, 0.66),
    "highlight": None,  # the direction of the key light
    "fill": (0.6, -0.2, 0.77),
    "fill_intensity": 0.25,
    "ao_samples": 8,
    "ao_distance": None,  # 3 times the median radius
    "ao_strength": 0.7,
    "shadows": True,
    "shadow_strength": 0.45,
    "flat": False,
    "toon_levels": 0,
    "outline_width": 0.0,
    "outline_depth": None,  # half the median radius
    "outline_color": (0.0, 0.0, 0.0),
    "fog": 0.0,
    "supersample": 2,
    "bounces": 0,
    "samples": 1,
    "light_size": 0.0,
    "base_light": 0.0,
    "srgb": False,
}

#: Directions of view: (forward, up).
VIEWS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "top": ((0, 0, -1), (0, 1, 0)),
    "bottom": ((0, 0, 1), (0, -1, 0)),
    "front": ((0, 1, 0), (0, 0, 1)),
    "back": ((0, -1, 0), (0, 0, 1)),
    "left": ((1, 0, 0), (0, 0, 1)),
    "right": ((-1, 0, 0), (0, 0, 1)),
    "iso": ((-1, -1, -1), (0, 0, 1)),
}

# Colormaps, sRGB from 0 to 1 at equal steps.
_COLORMAPS = {
    "viridis": ["#440154", "#482878", "#3e4a89", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6dcd59", "#b4de2c", "#fde725"],
    "plasma": ["#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"],
    "coolwarm": ["#3b4cc0", "#6688ee", "#88bbff", "#b8d0f9", "#dddddd", "#f5c4ad", "#f49a7b", "#de604d", "#b40426"],
    "jet": ["#00007f", "#0000ff", "#007fff", "#00ffff", "#7fff7f", "#ffff00", "#ff7f00", "#ff0000", "#7f0000"],
    "gray": ["#000000", "#ffffff"],
}  # fmt: skip


def _hex(code: str) -> tuple[float, float, float]:
    return tuple(int(code[k : k + 2], 16) / 255 for k in (1, 3, 5))  # type: ignore[return-value]


def colormap(
    values: ArrayLike, name: str = "viridis", value_range: Sequence[float] | None = None
) -> np.ndarray:
    """Colors of values through a colormap.

    Parameters
    ----------
    values : array_like
        (N,) the values.
    name : str, optional
        One of viridis, plasma, coolwarm, jet and gray. Default viridis.
    value_range : sequence of float, optional
        (low, high) values of the two ends; by default the smallest and the
        largest finite value.

    Returns
    -------
    numpy.ndarray
        (N, 3) sRGB colors from 0 to 1.
    """
    if name not in _COLORMAPS:
        raise InvalidParameter(
            f"colormap must be one of {tuple(_COLORMAPS)}, not {name!r}"
        )
    v = np.asarray(values, dtype=float).reshape(-1)
    if value_range is None:
        finite = v[np.isfinite(v)]
        low, high = (finite.min(), finite.max()) if len(finite) else (0.0, 1.0)
    else:
        low, high = (float(x) for x in value_range)
    span = high - low if high > low else 1.0
    t = np.clip(np.nan_to_num((v - low) / span), 0.0, 1.0)
    table = np.array([_hex(c) for c in _COLORMAPS[name]])
    x = np.linspace(0, 1, len(table))
    return np.stack([np.interp(t, x, table[:, c]) for c in range(3)], axis=1)


def _linear(srgb: np.ndarray) -> np.ndarray:
    """Linear RGB of sRGB colors from 0 to 1."""
    c = np.clip(np.asarray(srgb, dtype=float), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _color(value: Any) -> np.ndarray:
    """An sRGB color from 0 to 1: a name of the form #rrggbb or a triplet."""
    if isinstance(value, str):
        if len(value) == 7 and value.startswith("#"):
            return np.array(_hex(value))
        named = {"white": "#ffffff", "black": "#000000", "gray": "#808080", "red": "#d62728", "blue": "#1f77b4", "green": "#2ca02c"}  # fmt: skip
        if value in named:
            return np.array(_hex(named[value]))
        raise InvalidParameter(f"unknown color {value!r}")
    color = np.asarray(value, dtype=float).reshape(-1)
    if color.shape != (3,):
        raise InvalidParameter(f"a color is three numbers from 0 to 1, not {value!r}")
    return color


def _per_element(elements: Sequence[str] | pl.Series, values: Any) -> np.ndarray:
    """values(element) of each element, looked up once per kind."""
    series = elements if isinstance(elements, pl.Series) else pl.Series(list(elements))
    series = series.cast(pl.String)
    kinds = series.unique(maintain_order=True).to_list()
    table = np.array([values(k) for k in kinds], dtype=float)
    if not kinds:
        return table
    index = series.replace_strict(kinds, list(range(len(kinds))), return_dtype=pl.Int64)
    return table[index.to_numpy()]


def element_colors(elements: Sequence[str] | pl.Series) -> np.ndarray:
    """(N, 3) default sRGB colors of elements, from 0 to 1."""
    colors = _per_element(elements, lambda e: ELEMENTS.get(e, DEFAULT)[:3]) / 255
    return colors.reshape(-1, 3)


_RADII = {
    "display": None,
    "covalent": (COVALENT_RADII, MISSING_COVALENT),
    "vdw": (VDW_RADII, MISSING_VDW),
}
_NUMBER = {symbol: z for z, symbol in enumerate(SYMBOLS)}


def element_radii(
    elements: Sequence[str] | pl.Series, kind: str = "display"
) -> np.ndarray:
    """(N,) radii of elements in Å.

    Parameters
    ----------
    elements : sequence of str
        Chemical symbols.
    kind : {"display", "covalent", "vdw"}, optional
        The radii pictures show, the covalent radii or the van der Waals
        radii. Default display.

    Returns
    -------
    numpy.ndarray
        (N,) radii; elements without a value take 1.5 Å as covalent radius
        and 2 Å as van der Waals radius.
    """
    if kind not in _RADII:
        raise InvalidParameter(f"kind must be display, covalent or vdw, not {kind!r}")
    if kind == "display":

        def radius(element: str) -> float:
            return ELEMENTS.get(element, DEFAULT)[3]

    else:
        table, missing = _RADII[kind]

        def radius(element: str) -> float:
            value = table[_NUMBER.get(element, 0)]
            return missing if np.isnan(value) else float(value)

    return _per_element(elements, radius).reshape(-1)


def style_settings(style: str = "matte") -> dict[str, Any]:
    """Every setting of a style: the options of :meth:`Scene.render` it
    changes, and the defaults of the others.

    Parameters
    ----------
    style : str, optional
        One of ``STYLES``. Default matte.

    Returns
    -------
    dict
        Each setting by name; ``ao_distance`` and ``outline_depth`` are
        None, which a picture takes from the radii of its atoms, and
        ``highlight`` None, the direction of the key light.

    Raises
    ------
    InvalidParameter
        For an unknown style.
    """
    if style not in STYLES:
        raise InvalidParameter(f"style must be one of {tuple(STYLES)}, not {style!r}")
    return dict(_DEFAULTS) | STYLES[style]


def gpu_available() -> bool:
    """Whether pictures can be traced on a GPU: with Metal on macOS.

    Returns
    -------
    bool
        True when a GPU is found and its shader compiles.
    """
    return bool(_native.gpu_available())


def write_png(path: str | PathLike, image: np.ndarray) -> None:
    """Writes an RGBA image to a PNG file.

    Parameters
    ----------
    path : str or path-like
        The file.
    image : numpy.ndarray
        (H, W, 4) uint8 image, rows from the top.
    """
    image = np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 4:
        raise InvalidParameter("the image must be (height, width, 4) RGBA")
    height, width = image.shape[:2]
    raw = np.zeros((height, 1 + 4 * width), dtype=np.uint8)  # filter 0 on each row
    raw[:, 1:] = image.reshape(height, -1)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)))
        f.write(chunk(b"IEND", b""))


class Scene:
    """Spheres, cylinders and polygons to render.

    Colors are sRGB from 0 to 1, lengths in Å.
    """

    def __init__(self) -> None:
        self.__spheres: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        self.__cylinders: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self.__triangles: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = []
        self.__next_id = 0

    def __ids(self, count: int, ids: ArrayLike | None) -> np.ndarray:
        if ids is None:
            ids = np.arange(self.__next_id, self.__next_id + count)
        ids = np.asarray(ids, dtype=np.int64).reshape(-1)
        if len(ids) != count or (count and ids.min() < 0):
            raise InvalidParameter("ids must be one non-negative integer per primitive")
        if count:
            self.__next_id = max(self.__next_id, int(ids.max()) + 1)
        return ids.astype(np.int32)

    @classmethod
    def __colors(cls, count: int, value: Any) -> np.ndarray:
        if isinstance(value, str):
            value = _color(value)
        return cls.__per(count, value, 3, "colors")

    @staticmethod
    def __per(count: int, value: ArrayLike, width: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == (0 if width == 1 else 1):
            array = np.broadcast_to(array, (count,) if width == 1 else (count, width))
        expected = (count,) if width == 1 else (count, width)
        if array.shape != expected:
            raise InvalidParameter(
                f"{name} must have shape {expected}, not {array.shape}"
            )
        return np.array(array, dtype=float)

    def add_spheres(
        self,
        centers: ArrayLike,
        radii: ArrayLike,
        colors: ArrayLike,
        *,
        ids: ArrayLike | None = None,
    ) -> None:
        """Adds spheres, such as atoms.

        Parameters
        ----------
        centers : array_like
            (N, 3) centers.
        radii : float or array_like
            Radius of every sphere, or (N,) each.
        colors : array_like
            One color, as three numbers from 0 to 1 or a name such as
            "#rrggbb", or (N, 3) each.
        ids : array_like, optional
            (N,) non-negative ids; outlines separate different ids. Default
            new ids, one per sphere.
        """
        c = np.asarray(centers, dtype=float).reshape(-1, 3)
        n = len(c)
        self.__spheres.append(
            (
                c,
                self.__per(n, radii, 1, "radii"),
                self.__colors(n, colors),
                self.__ids(n, ids),
            )
        )

    def add_cylinders(
        self,
        starts: ArrayLike,
        ends: ArrayLike,
        radii: ArrayLike,
        colors: ArrayLike,
        *,
        ids: ArrayLike | None = None,
    ) -> None:
        """Adds cylinders without caps, such as bonds.

        Parameters
        ----------
        starts, ends : array_like
            (M, 3) the two ends of each cylinder.
        radii : float or array_like
            Radius of every cylinder, or (M,) each.
        colors : array_like
            One color, or (M, 3) each.
        ids : array_like, optional
            (M,) non-negative ids. Default new ids.
        """
        a = np.asarray(starts, dtype=float).reshape(-1, 3)
        b = np.asarray(ends, dtype=float).reshape(-1, 3)
        if a.shape != b.shape:
            raise InvalidParameter("starts and ends must have the same shape")
        m = len(a)
        self.__cylinders.append(
            (
                np.hstack([a, b]),
                self.__per(m, radii, 1, "radii"),
                self.__colors(m, colors),
                self.__ids(m, ids),
            )
        )

    def add_triangles(
        self,
        vertices: ArrayLike,
        colors: ArrayLike,
        *,
        alpha: ArrayLike = 1.0,
        ids: ArrayLike | None = None,
    ) -> None:
        """Adds triangles, seen from both sides.

        Parameters
        ----------
        vertices : array_like
            (T, 3, 3) the corners of each triangle.
        colors : array_like
            One color, or (T, 3) each.
        alpha : float or array_like, optional
            Opacity from 0 to 1 of every triangle, or (T,) each. Default 1.
        ids : array_like, optional
            (T,) non-negative ids. Default new ids.
        """
        v = np.asarray(vertices, dtype=float).reshape(-1, 9)
        t = len(v)
        opacity = self.__per(t, alpha, 1, "alpha")
        if ((opacity < 0) | (opacity > 1)).any():
            raise InvalidParameter("alpha must be from 0 to 1")
        self.__triangles.append(
            (v, self.__colors(t, colors), opacity, self.__ids(t, ids))
        )

    def add_polygons(
        self,
        vertices: ArrayLike,
        corner_offsets: ArrayLike,
        corners: ArrayLike,
        colors: ArrayLike,
        *,
        alpha: ArrayLike = 1.0,
        ids: ArrayLike | None = None,
    ) -> None:
        """Adds convex polygons, such as the faces of Voronoi cells.

        Parameters
        ----------
        vertices : array_like
            (V, 3) points.
        corner_offsets : array_like
            (F + 1,) the corners of polygon f are
            ``corners[corner_offsets[f]:corner_offsets[f + 1]]``.
        corners : array_like
            (C,) the vertex of each corner, in order around its polygon.
        colors : array_like
            One color, or (F, 3) each.
        alpha : float or array_like, optional
            Opacity of every polygon, or (F,) each. Default 1.
        ids : array_like, optional
            (F,) ids, the same for the triangles of a polygon. Default new
            ids.
        """
        points = np.asarray(vertices, dtype=float).reshape(-1, 3)
        offsets = np.asarray(corner_offsets, dtype=np.int64)
        loops = np.asarray(corners, dtype=np.int64)
        f = len(offsets) - 1
        colors = self.__colors(f, colors)
        opacity = self.__per(f, alpha, 1, "alpha")
        polygon_ids = self.__ids(f, ids)
        # a fan from the first corner of each polygon
        size = np.diff(offsets)
        if (size < 3).any():
            raise InvalidParameter("a polygon has fewer than three corners")
        owner = np.repeat(np.arange(f), size - 2)
        step = np.arange(len(owner)) - np.repeat(
            np.cumsum(size - 2) - (size - 2), size - 2
        )
        first = loops[offsets[owner]]
        second = loops[offsets[owner] + 1 + step]
        third = loops[offsets[owner] + 2 + step]
        triangles = np.stack([points[first], points[second], points[third]], axis=1)
        self.add_triangles(
            triangles, colors[owner], alpha=opacity[owner], ids=polygon_ids[owner]
        )

    def add_box(
        self, box: Box, *, radius: float = 0.15, color: Any = (0.0, 0.0, 0.0)
    ) -> None:
        """Adds the twelve edges of a box, with a sphere on each corner.

        Parameters
        ----------
        box : Box
            The box.
        radius : float, optional
            Radius of the edges in Å. Default 0.15.
        color : color, optional
            Default black.
        """
        h, o = box.matrix, box.origin
        corners = np.array(
            [
                o + i * h[0] + j * h[1] + k * h[2]
                for i in (0, 1)
                for j in (0, 1)
                for k in (0, 1)
            ]
        )
        edges = [
            (a, b)
            for a in range(8)
            for b in range(a + 1, 8)
            if bin(a ^ b).count("1") == 1
        ]
        start = np.array([corners[a] for a, _ in edges])
        end = np.array([corners[b] for _, b in edges])
        rgb = _color(color)
        edge_id = self.__next_id
        self.add_cylinders(start, end, radius, rgb, ids=np.full(12, edge_id))
        self.add_spheres(corners, radius, rgb, ids=np.full(8, edge_id))

    def _arrays(self, linear: bool = True) -> dict[str, np.ndarray]:
        def join(parts: list, k: int, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
            if not parts:
                return np.zeros(shape, dtype=dtype)
            return np.ascontiguousarray(
                np.concatenate([p[k] for p in parts]), dtype=dtype
            )

        f32 = np.float32

        # colors in linear light, or as given in sRGB
        def to(c: np.ndarray) -> np.ndarray:
            return _linear(c) if linear else np.clip(c, 0.0, 1.0)

        return {
            "sphere_center": join(self.__spheres, 0, (0, 3), f32),
            "sphere_radius": join(self.__spheres, 1, (0,), f32),
            "sphere_color": np.ascontiguousarray(
                to(join(self.__spheres, 2, (0, 3), float)), dtype=f32
            ),
            "sphere_id": join(self.__spheres, 3, (0,), np.int32),
            "cylinder_end": join(self.__cylinders, 0, (0, 6), f32),
            "cylinder_radius": join(self.__cylinders, 1, (0,), f32),
            "cylinder_color": np.ascontiguousarray(
                to(join(self.__cylinders, 2, (0, 3), float)), dtype=f32
            ),
            "cylinder_id": join(self.__cylinders, 3, (0,), np.int32),
            "triangle_vertex": join(self.__triangles, 0, (0, 9), f32),
            "triangle_color": np.ascontiguousarray(
                to(join(self.__triangles, 1, (0, 3), float)), dtype=f32
            ),
            "triangle_alpha": join(self.__triangles, 2, (0,), f32),
            "triangle_id": join(self.__triangles, 3, (0,), np.int32),
        }

    def __points(self) -> tuple[np.ndarray, np.ndarray]:
        """Points of the scene with the radius around each, for the camera."""
        points, pad = [np.zeros((0, 3))], [np.zeros(0)]
        for c, r, _, _ in self.__spheres:
            points.append(c)
            pad.append(r)
        for ends, r, _, _ in self.__cylinders:
            points += [ends[:, :3], ends[:, 3:]]
            pad += [r, r]
        for v, _, _, _ in self.__triangles:
            points.append(v.reshape(-1, 3))
            pad.append(np.zeros(len(v) * 3))
        return np.concatenate(points), np.concatenate(pad)

    def render(
        self,
        path: str | PathLike | None = None,
        *,
        width: int = 1200,
        height: int = 900,
        view: str | ArrayLike = "iso",
        up: ArrayLike | None = None,
        perspective: bool = False,
        fov: float = 30.0,
        zoom: float = 1.0,
        focus: ArrayLike | None = None,
        span: float | None = None,
        aperture: float = 0.0,
        focal_distance: float | None = None,
        style: str = "matte",
        background: Any = "white",
        device: str = "cpu",
        progress: Any = None,
        cancel: Any = None,
        **options: Any,
    ) -> np.ndarray:
        """Renders the scene, framed to fill the picture.

        Parameters
        ----------
        path : str or path-like, optional
            A PNG file to write the picture to.
        width, height : int, optional
            Size of the picture in pixels. Default 1200 by 900.
        view : str or array_like, optional
            One of top, bottom, front, back, left, right and iso, or the
            (3,) direction of view. Default iso.
        up : array_like, optional
            (3,) the direction up in the picture. Default that of the view,
            or z.
        perspective : bool, optional
            A perspective camera, or orthographic. Default False.
        fov : float, optional
            Vertical angle of view of a perspective camera in degrees.
            Default 30.
        zoom : float, optional
            Magnification of the framed scene. Default 1.
        focus : array_like, optional
            (3,) the point in Å at the middle of the picture. Default the
            middle of the scene.
        span : float, optional
            The length in Å the picture shows from its bottom to its top at
            the focus, in place of the framing and zoom. Default the scene
            framed.
        aperture : float, optional
            Radius in Å of the lens of a perspective camera: 0 for a picture
            sharp everywhere, more for depth of field. Default 0.
        focal_distance : float, optional
            Distance in Å from the camera along the view to the plane that
            stays sharp. Default that of the focus.
        style : str, optional
            One of ``STYLES``. Default matte.
        background : color or None, optional
            The background, or None for transparent. Default white.
        device : {"cpu", "gpu", "auto"}, optional
            Trace the rays on the processor, on the GPU (see
            :func:`gpu_available`), or on the GPU when there is one.
            Pictures on the GPU differ from those on the processor by
            rounding only. Default cpu.
        progress, cancel : optional
            Progress reports and cancellation.
        **options
            Settings of the style to change: ambient, diffuse, specular,
            shininess, metalness, key, highlight, fill, fill_intensity,
            base_light, srgb, ao_samples, ao_distance,
            ao_strength, shadows, shadow_strength, flat, toon_levels,
            outline_width, outline_depth, outline_color, fog, supersample,
            samples, bounces, light_size.

        Returns
        -------
        numpy.ndarray
            (height, width, 4) uint8 RGBA picture, rows from the top.
        """
        if style not in STYLES:
            raise InvalidParameter(
                f"style must be one of {tuple(STYLES)}, not {style!r}"
            )
        unknown = set(options) - set(_DEFAULTS)
        if unknown:
            raise InvalidParameter(f"unknown options {sorted(unknown)}")
        if width < 1 or height < 1:
            raise InvalidParameter("width and height must be positive")
        if focus is not None:
            focus = np.asarray(focus, dtype=float)
            if focus.shape != (3,) or not np.isfinite(focus).all():
                raise InvalidParameter("focus must be a point of three numbers")
        if span is not None and not (np.isfinite(span) and span > 0):
            raise InvalidParameter(f"span must be positive, not {span!r}")
        if not (np.isfinite(aperture) and aperture >= 0):
            raise InvalidParameter(f"aperture must be 0 or more, not {aperture!r}")
        if aperture > 0 and not perspective:
            raise InvalidParameter("depth of field needs a perspective camera")
        if focal_distance is not None and not (
            np.isfinite(focal_distance) and focal_distance > 0
        ):
            raise InvalidParameter(
                f"focal_distance must be positive, not {focal_distance!r}"
            )
        if device not in ("cpu", "gpu", "auto"):
            raise InvalidParameter(f"device must be cpu, gpu or auto, not {device!r}")
        gpu = device == "gpu" or (device == "auto" and gpu_available())
        if gpu and not gpu_available():
            raise InvalidParameter("there is no GPU to trace pictures on")
        settings = dict(_DEFAULTS) | STYLES[style] | options
        if int(settings["samples"]) < 1:
            raise InvalidParameter(
                f"samples must be at least 1, not {settings['samples']!r}"
            )
        if int(settings["bounces"]) < 0:
            raise InvalidParameter(
                f"bounces must be at least 0, not {settings['bounces']!r}"
            )
        if isinstance(view, str):
            if view not in VIEWS:
                raise InvalidParameter(
                    f"view must be one of {tuple(VIEWS)}, not {view!r}"
                )
            forward, default_up = (np.array(v, dtype=float) for v in VIEWS[view])
        else:
            forward, default_up = (
                np.asarray(view, dtype=float),
                np.array([0.0, 0.0, 1.0]),
            )
        forward = forward / np.linalg.norm(forward)
        up_vector = default_up if up is None else np.asarray(up, dtype=float)
        if np.linalg.norm(np.cross(forward, up_vector)) < 1e-9:
            up_vector = (
                np.array([0.0, 1.0, 0.0])
                if abs(forward[1]) < 0.9
                else np.array([1.0, 0.0, 0.0])
            )
        right = np.cross(forward, up_vector)
        right /= np.linalg.norm(right)
        true_up = np.cross(right, forward)
        points, pad = self.__points()
        radii = (
            np.concatenate([s[1] for s in self.__spheres])
            if self.__spheres
            else np.ones(1)
        )
        scale = float(np.median(radii)) if len(radii) else 1.0
        if settings["ao_distance"] is None:
            settings["ao_distance"] = 3.0 * scale
        if settings["outline_depth"] is None:
            settings["outline_depth"] = 0.5 * scale
        aspect = width / height
        if len(points):
            x, y, z = points @ right, points @ true_up, points @ forward
            lo = np.array([(x - pad).min(), (y - pad).min(), (z - pad).min()])
            hi = np.array([(x + pad).max(), (y + pad).max(), (z + pad).max()])
        else:
            lo, hi = -np.ones(3), np.ones(3)
        middle = 0.5 * (lo + hi)
        half_w, half_h = 0.5 * (hi[0] - lo[0]), 0.5 * (hi[1] - lo[1])
        half_h = max(half_h, half_w / aspect, 1e-6) * 1.04 / zoom
        center = middle[0] * right + middle[1] * true_up + middle[2] * forward
        if focus is not None:
            center = focus
        if perspective and span is not None:
            # the eye as far from the focus as shows the span there
            tangent = np.tan(np.radians(fov) / 2)
            camera_at = center - forward * (0.5 * span / tangent)
            half = (tangent, tangent * aspect)
        elif perspective:
            tangent = np.tan(np.radians(fov) / 2)
            radius = (
                float(np.sqrt(((points - center) ** 2).sum(axis=1).max()))
                if len(points)
                else 1.0
            )
            distance = max(half_h / tangent, radius) + 0.5 * (hi[2] - lo[2])
            camera_at = center - forward * distance
            half = (tangent / zoom, tangent * aspect / zoom)
        else:
            if span is not None:
                half_h = 0.5 * span
            camera_at = center
            half = (half_h, half_h * aspect)
        if focal_distance is None:
            focal_distance = max(float((center - camera_at) @ forward), 1e-6)
        srgb = bool(settings["srgb"])
        if background is None:
            back = np.array([1.0, 1.0, 1.0, 0.0])
        else:
            back = np.append(
                _color(background) if srgb else _linear(_color(background)), 1.0
            )
        f32 = np.float32
        image = _native.render(
            **self._arrays(linear=not srgb),
            center=camera_at.astype(f32),
            forward=forward.astype(f32),
            up=true_up.astype(f32),
            half_width=float(half[1]),
            half_height=float(half[0]),
            perspective=bool(perspective),
            aperture=float(aperture),
            focal_distance=float(focal_distance),
            width=int(width),
            height=int(height),
            ambient=float(settings["ambient"]),
            diffuse=float(settings["diffuse"]),
            specular=float(settings["specular"]),
            shininess=float(settings["shininess"]),
            metalness=float(settings["metalness"]),
            key=np.asarray(settings["key"], dtype=f32),
            highlight=np.asarray(
                settings["key"]
                if settings["highlight"] is None
                else settings["highlight"],
                dtype=f32,
            ),
            fill=np.asarray(settings["fill"], dtype=f32),
            fill_intensity=float(settings["fill_intensity"]),
            ao_samples=int(settings["ao_samples"]),
            ao_distance=float(settings["ao_distance"]),
            ao_strength=float(settings["ao_strength"]),
            shadows=bool(settings["shadows"]),
            shadow_strength=float(settings["shadow_strength"]),
            flat=bool(settings["flat"]),
            toon_levels=int(settings["toon_levels"]),
            outline_width=float(settings["outline_width"]),
            outline_depth=float(settings["outline_depth"]),
            outline_color=(
                _color(settings["outline_color"])
                if srgb
                else _linear(_color(settings["outline_color"]))
            ).astype(f32),
            fog=float(settings["fog"]),
            background=back.astype(f32),
            supersample=int(settings["supersample"]),
            bounces=int(settings["bounces"]),
            samples=int(settings["samples"]),
            light_size=float(settings["light_size"]),
            base_light=float(settings["base_light"]),
            srgb=srgb,
            gpu=gpu,
            progress=progress,
            cancel=_native_token(cancel),
        )
        if path is not None:
            write_png(path, image)
        return image
