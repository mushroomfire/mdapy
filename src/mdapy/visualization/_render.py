# Copyright (c) 2022-2026, Yongchao Wu
# This file is part of mdapy, released under the BSD 3-Clause License.

"""A picture of a system in one call."""

from os import PathLike
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from mdapy.core.system import System
from mdapy.errors import InvalidParameter
from mdapy.neighbor import NeighborList
from mdapy.visualization._scene import (
    Scene,
    _color,
    _hex,
    colormap,
    element_colors,
    element_radii,
)

# colors of the faces of Voronoi cells by their number of edges, 3 to 9
_ORDER_COLORS = np.array(
    [
        _hex(c)
        for c in (
            "#f2a541",
            "#3b75af",
            "#6a3d9a",
            "#2ca02c",
            "#e6007e",
            "#8c564b",
            "#555555",
        )
    ]
)


def atom_colors(
    system: System,
    color: Any = None,
    *,
    colormap_name: str = "viridis",
    value_range: Any = None,
) -> np.ndarray:
    """(N, 3) sRGB colors of the atoms of a system.

    Parameters
    ----------
    system : System
        The atoms.
    color : optional
        None for the colors of the elements; one color for all; (N, 3)
        colors; a column name or (N,) values through the colormap; or a
        structure result, with ``structure`` and ``colors``.
    colormap_name : str, optional
        The colormap of values. Default viridis.
    value_range : sequence of float, optional
        The values at the two ends of the colormap.

    Returns
    -------
    numpy.ndarray
        (N, 3) colors from 0 to 1.
    """
    n = len(system)
    if color is None:
        names = (
            system.table["element"].to_list()
            if "element" in system.table.columns
            else ["X"] * n
        )
        return element_colors(names)
    if hasattr(color, "structure") and hasattr(color, "colors"):
        palette = np.asarray(color.colors, dtype=float)
        return palette[np.asarray(color.structure, dtype=np.int64)]
    if isinstance(color, str) and color in system.table.columns:
        return colormap(system.table[color].to_numpy(), colormap_name, value_range)
    array = np.asarray(color) if not isinstance(color, str) else None
    if array is not None and array.shape == (n, 3):
        return np.asarray(array, dtype=float)
    if array is not None and array.shape == (n,) and n != 3:
        return colormap(array, colormap_name, value_range)
    return np.broadcast_to(_color(color), (n, 3)).copy()


def render(
    system: System,
    path: str | PathLike | None = None,
    *,
    atoms: ArrayLike | None = None,
    color: Any = None,
    colormap: str = "viridis",
    value_range: Any = None,
    radius: Any = None,
    bonds: Any = None,
    bond_radius: float = 0.25,
    box: bool = True,
    box_radius: float | None = None,
    polyhedra: Any = None,
    polyhedra_alpha: float = 0.6,
    polyhedra_color: str = "atom",
    polyhedra_edges: float | None = None,
    **options: Any,
) -> np.ndarray:
    """A picture of a system: its atoms, optionally bonds, the box and
    Voronoi cells.

    Parameters
    ----------
    system : System
        The atoms.
    path : str or path-like, optional
        A PNG file to write the picture to.
    atoms : array_like, optional
        A boolean mask or indices of the atoms shown. Default all.
    color : optional
        The colors of the atoms, see :func:`atom_colors`. Default those of
        the elements.
    colormap : str, optional
        The colormap of values. Default viridis.
    value_range : sequence of float, optional
        The values at the two ends of the colormap.
    radius : float or array_like, optional
        Radius of the atoms in Å, one or (N,) each. Default those of the
        elements.
    bonds : float or NeighborList, optional
        Bonds between the atoms closer than this cutoff in Å, or the pairs
        of a neighbor list. Default none.
    bond_radius : float, optional
        Radius of the bonds in Å. Default 0.25.
    box : bool, optional
        Draw the edges of the box. Default True.
    box_radius : float, optional
        Radius of the edges in Å. Default a tenth of the median atom radius.
    polyhedra : VoronoiPolyhedra, optional
        Voronoi cells to draw for the atoms shown.
    polyhedra_alpha : float, optional
        Opacity of the cells. Default 0.6.
    polyhedra_color : {"atom", "edges"}, optional
        Color the faces as their atoms, or by their number of edges.
        Default atom.
    polyhedra_edges : float, optional
        Radius in Å of the edges of the cells; 0 draws none. Default a
        twentieth of the median atom radius.
    **options
        The options of :meth:`Scene.render`, such as width, height, view,
        style, zoom, perspective and background.

    Returns
    -------
    numpy.ndarray
        (height, width, 4) uint8 RGBA picture.
    """
    n = len(system)
    shown = np.ones(n, dtype=bool)
    if atoms is not None:
        mask = np.asarray(atoms)
        if mask.dtype == bool:
            if mask.shape != (n,):
                raise InvalidParameter(f"a mask of atoms must have {n} entries")
            shown = mask.copy()
        else:
            shown = np.zeros(n, dtype=bool)
            shown[mask.astype(np.int64)] = True
    colors = atom_colors(system, color, colormap_name=colormap, value_range=value_range)
    names = (
        system.table["element"].to_list()
        if "element" in system.table.columns
        else ["X"] * n
    )
    radii = (
        element_radii(names)
        if radius is None
        else np.broadcast_to(np.asarray(radius, dtype=float), (n,)).copy()
    )
    index = np.flatnonzero(shown)
    scene = Scene()
    scene.add_spheres(system.positions[index], radii[index], colors[index], ids=index)
    if bonds is not None:
        nl = (
            bonds
            if isinstance(bonds, NeighborList)
            else NeighborList.build(system, float(bonds))
        )
        i, j = nl.i.astype(np.int64), nl.j.astype(np.int64)
        both = shown[i] & shown[j]
        inside = (nl.shifts == 0).all(axis=1)
        # each half of a bond in the color of its atom; across the boundary
        # each atom draws its own half only, as the pair comes both ways
        half = both & ((inside & (i < j)) | ~inside)
        start = system.positions[i[half]]
        middle = start + 0.5 * nl.vectors[half]
        scene.add_cylinders(
            start, middle, bond_radius, colors[i[half]], ids=np.full(int(half.sum()), n)
        )
        other = both & inside & (i < j)
        end = system.positions[j[other]]
        middle = end - 0.5 * nl.vectors[other]
        scene.add_cylinders(
            end, middle, bond_radius, colors[j[other]], ids=np.full(int(other.sum()), n)
        )
    if box:
        scale = float(np.median(radii[index])) if len(index) else 1.0
        scene.add_box(
            system.box, radius=0.1 * scale if box_radius is None else box_radius
        )
    if polyhedra is not None:
        faces = []
        for i in index:
            faces.extend(
                range(polyhedra.face_offsets[i], polyhedra.face_offsets[i + 1])
            )
        faces = np.array(faces, dtype=np.int64)
        if len(faces):
            owner = np.repeat(np.arange(n), np.diff(polyhedra.face_offsets))[faces]
            starts = polyhedra.corner_offsets[faces]
            sizes = polyhedra.corner_offsets[faces + 1] - starts
            corners = np.concatenate(
                [polyhedra.corners[s : s + k] for s, k in zip(starts, sizes)]
            )
            offsets = np.concatenate([[0], np.cumsum(sizes)])
            if polyhedra_color == "atom":
                face_colors = colors[owner]
            elif polyhedra_color == "edges":
                face_colors = _ORDER_COLORS[np.clip(sizes, 3, 9) - 3]
            else:
                raise InvalidParameter(
                    f"polyhedra_color must be atom or edges, not {polyhedra_color!r}"
                )
            scene.add_polygons(
                polyhedra.vertices,
                offsets,
                corners,
                face_colors,
                alpha=polyhedra_alpha,
                ids=n + 1 + faces,
            )
            scale = float(np.median(radii[index]))
            edge_radius = (
                0.05 * scale if polyhedra_edges is None else float(polyhedra_edges)
            )
            if edge_radius > 0:
                # each edge of a cell once: consecutive corners of its faces
                following = np.concatenate(
                    [
                        np.roll(corners[a:b], -1)
                        for a, b in zip(offsets[:-1], offsets[1:])
                    ]
                )
                pairs = np.unique(
                    np.sort(np.stack([corners, following], axis=1), axis=1), axis=0
                )
                points = polyhedra.vertices
                scene.add_cylinders(
                    points[pairs[:, 0]],
                    points[pairs[:, 1]],
                    edge_radius,
                    (0.15, 0.15, 0.15),
                    ids=np.full(len(pairs), n),
                )
                joints = np.unique(pairs)  # round corners
                scene.add_spheres(
                    points[joints],
                    edge_radius,
                    (0.15, 0.15, 0.15),
                    ids=np.full(len(joints), n),
                )
    return scene.render(path, **options)
