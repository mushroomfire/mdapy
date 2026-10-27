# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""A picture of a system in one call."""

from os import PathLike
from typing import Any

import numpy as np
import polars as pl
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
            system.table["element"] if "element" in system.table.columns else ["X"] * n
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


#: The ways atoms and bonds are drawn: the radius of the atoms, the bonds
#: drawn and the radius of the bonds, each unless given.
REPRESENTATIONS: dict[str, tuple[Any, Any, float]] = {
    # atoms as spheres of their display radius
    "spheres": ("display", None, 0.25),
    # small balls, a quarter of the van der Waals radius, and covalent bonds
    "ball_and_stick": (("vdw", 0.25), "covalent", 0.15),
    # atoms as spheres of their van der Waals radius
    "space_filling": ("vdw", None, 0.25),
    # covalent bonds as sticks with round ends
    "licorice": (None, "covalent", 0.2),
}

# added to the sum of two covalent radii for a bond, as in Jmol
_BOND_TOLERANCE = 0.45


def covalent_bonds(system: System) -> NeighborList:
    """The covalent bonds of a system: the pairs of atoms closer than the sum
    of their covalent radii and 0.45 Å.

    Parameters
    ----------
    system : System
        The atoms, with an element column.

    Returns
    -------
    NeighborList
        The bonds, each pair both ways.
    """
    if "element" not in system.table.columns:
        raise InvalidParameter("covalent bonds need an element column")
    names = system.table["element"].cast(pl.String)
    kinds = names.unique(maintain_order=True).to_list()
    types = names.replace_strict(kinds, list(range(len(kinds))), return_dtype=pl.Int32)
    types = types.to_numpy()
    radii = element_radii(kinds, "covalent")
    limits = radii[:, None] + radii[None, :] + _BOND_TOLERANCE
    largest = float(limits.max(initial=_BOND_TOLERANCE))
    return NeighborList.build(system, largest)._cut(largest, limits, types)


def render(
    system: System,
    path: str | PathLike | None = None,
    *,
    atoms: ArrayLike | None = None,
    color: Any = None,
    colormap: str = "viridis",
    value_range: Any = None,
    representation: str = "spheres",
    radius: Any = None,
    bonds: Any = None,
    bond_radius: float | None = None,
    box: bool = True,
    box_radius: float | None = None,
    polyhedra: Any = None,
    polyhedra_alpha: float = 0.6,
    polyhedra_color: str = "atom",
    polyhedra_edges: float | None = None,
    mesh: Any = None,
    mesh_color: Any = "#8fb0d8",
    mesh_alpha: float = 1.0,
    dislocations: Any = None,
    dislocation_radius: float = 0.5,
    **options: Any,
) -> np.ndarray:
    """A picture of a system: its atoms, optionally bonds, the box,
    Voronoi cells and a surface.

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
    representation : str, optional
        How atoms and bonds are drawn, one of ``REPRESENTATIONS``: spheres,
        ball_and_stick, space_filling or licorice. It sets the defaults of
        radius, bonds and bond_radius. Default spheres.
    radius : float, array_like or str, optional
        Radius of the atoms in Å, one or (N,) each, or the radii of the
        elements: "display", "covalent" or "vdw". Default that of the
        representation.
    bonds : float, NeighborList or "covalent", optional
        Bonds between the atoms closer than this cutoff in Å, the pairs of a
        neighbor list, or covalent bonds (see :func:`covalent_bonds`).
        Default that of the representation.
    bond_radius : float, optional
        Radius of the bonds in Å. Default that of the representation.
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
    mesh : TriangleMesh, optional
        A surface to draw, such as that of
        :class:`~mdapy.analysis.SurfaceMesh`. Triangles across a periodic
        boundary are cut there, the part outside moved in through the
        opposite face.
    mesh_color : optional
        One color for the surface, or (F, 3) colors of its triangles.
        Default light blue.
    mesh_alpha : float, optional
        Opacity of the surface. Default 1.
    dislocations : DislocationResult, optional
        Dislocation lines to draw, such as those of
        :class:`~mdapy.analysis.DislocationAnalysis`: tubes in the color of
        the family of their Burgers vector, cut at the periodic boundaries,
        the parts outside moved in through the opposite faces.
    dislocation_radius : float, optional
        Radius of the tubes in Å. Default 0.5.
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
    if representation not in REPRESENTATIONS:
        raise InvalidParameter(
            f"representation must be one of {tuple(REPRESENTATIONS)}, "
            f"not {representation!r}"
        )
    default_radius, default_bonds, default_bond_radius = REPRESENTATIONS[representation]
    bond_radius = default_bond_radius if bond_radius is None else float(bond_radius)
    if bonds is None:
        bonds = default_bonds
    if radius is None:
        radius = bond_radius if default_radius is None else default_radius
    colors = atom_colors(system, color, colormap_name=colormap, value_range=value_range)
    names = system.table["element"] if "element" in system.table.columns else ["X"] * n
    if isinstance(radius, tuple) and len(radius) == 2 and isinstance(radius[0], str):
        radii = element_radii(names, radius[0]) * radius[1]  # a fraction of them
    elif isinstance(radius, str):
        radii = element_radii(names, radius)
    else:
        radii = np.broadcast_to(np.asarray(radius, dtype=float), (n,)).copy()
    index = np.flatnonzero(shown)
    scene = Scene()
    scene.add_spheres(
        system["positions"][index], radii[index], colors[index], ids=index
    )
    if bonds is not None:
        if isinstance(bonds, NeighborList):
            nl = bonds
        elif isinstance(bonds, str):
            if bonds != "covalent":
                raise InvalidParameter(
                    f"bonds must be a cutoff, a NeighborList or covalent, not {bonds!r}"
                )
            nl = covalent_bonds(system)
        else:
            nl = NeighborList.build(system, float(bonds))
        i, j = nl.i.astype(np.int64), nl.j.astype(np.int64)
        both = shown[i] & shown[j]
        inside = (nl.shifts == 0).all(axis=1)
        # each half of a bond in the color of its atom; across the boundary
        # each atom draws its own half only, as the pair comes both ways
        half = both & ((inside & (i < j)) | ~inside)
        start = system["positions"][i[half]]
        middle = start + 0.5 * nl.vectors[half]
        scene.add_cylinders(
            start, middle, bond_radius, colors[i[half]], ids=np.full(int(half.sum()), n)
        )
        other = both & inside & (i < j)
        end = system["positions"][j[other]]
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
    if mesh is not None:
        corners, triangle = mesh.pieces()
        rgb = np.asarray(
            _color(mesh_color) if isinstance(mesh_color, str) else mesh_color,
            dtype=float,
        )
        if rgb.shape == (len(mesh), 3):
            rgb = rgb[triangle]
        elif rgb.shape != (3,):
            raise InvalidParameter(
                f"mesh_color must be one color or {len(mesh)} colors, "
                f"not shape {rgb.shape}"
            )
        # one id for the whole surface, after those of the atoms and cells
        surface = n + 1 + (0 if polyhedra is None else len(polyhedra.j))
        scene.add_triangles(
            corners, rgb, alpha=mesh_alpha, ids=np.full(len(corners), surface)
        )
    if dislocations is not None and dislocations.lines:
        from mdapy.analysis import DislocationAnalysis

        palette = DislocationAnalysis.FAMILY_COLORS[dislocations.crystal]
        starts, ends, rgb = [], [], []
        for line in dislocations.lines:
            a, b = _inside(line.points, system.box)
            starts.append(a)
            ends.append(b)
            rgb.append(np.tile(palette[line.family], (len(a), 1)))
        starts, ends, rgb = map(np.concatenate, (starts, ends, rgb))
        radius = float(dislocation_radius)
        if not radius > 0:
            raise InvalidParameter(
                f"dislocation_radius must be positive, not {dislocation_radius!r}"
            )
        ids = np.full(len(starts), n)
        scene.add_cylinders(starts, ends, radius, rgb, ids=ids)
        # round joints
        scene.add_spheres(starts, radius, rgb, ids=ids)
        scene.add_spheres(ends, radius, rgb, ids=ids)
    return scene.render(path, **options)


def _inside(points: np.ndarray, box: Any) -> tuple[np.ndarray, np.ndarray]:
    """The pieces of a polyline in the box: each segment cut where it
    crosses a periodic boundary, each piece moved into the box; the starts
    and ends of the pieces."""
    starts, ends = [], []
    for p, q in zip(points[:-1], points[1:]):
        f, g = (p - box.origin) @ box.inverse, (q - box.origin) @ box.inverse
        # the fractions of the segment where it crosses a periodic face
        cuts = [0.0, 1.0]
        for k in np.flatnonzero(box.pbc):
            low, high = sorted((f[k], g[k]))
            for face in range(int(np.floor(low)) + 1, int(np.floor(high)) + 1):
                cuts.append((face - f[k]) / (g[k] - f[k]))
        cuts = np.unique(np.clip(cuts, 0.0, 1.0))
        for s, t in zip(cuts[:-1], cuts[1:]):
            a, b = f + s * (g - f), f + t * (g - f)
            image = np.where(box.pbc, np.floor(0.5 * (a + b)), 0.0)
            starts.append((a - image) @ box.matrix + box.origin)
            ends.append((b - image) @ box.matrix + box.origin)
    return np.array(starts).reshape(-1, 3), np.array(ends).reshape(-1, 3)
