# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Grains of a polycrystal from the lattice orientations of its atoms."""

import colorsys
from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import identify as _native
from mdapy.core.indices import index_dtype
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.identify._ptm import PolyhedralTemplateMatching, PTMResult
from mdapy.neighbor import NeighborList

# neighbors in the template of each PTM structure: other, fcc, hcp, bcc,
# ico, sc, cubic diamond, hexagonal diamond, graphene
_TEMPLATE_NEIGHBORS = np.array([0, 12, 12, 14, 12, 6, 16, 16, 9])
# nearest neighbors of an atom of no structure, along which the atoms in no
# grain join one
_DISORDERED_NEIGHBORS = 8


@dataclass(frozen=True)
class GrainResult(Result):
    """The grains of a polycrystal.

    Attributes
    ----------
    grain : numpy.ndarray
        (N,) int64 grain of each atom, an index into the arrays of the
        grains, -1 for none.
    size : numpy.ndarray
        (G,) number of atoms of each grain, largest first.
    structure : numpy.ndarray
        (G,) int8 structure of each grain, an index into ``names``.
    names : tuple of str
        Names of the structures, those of polyhedral template matching.
    orientation : numpy.ndarray
        (G, 4) mean lattice orientation of each grain as a quaternion
        (x, y, z, w).
    center : numpy.ndarray
        (G, 3) center of mass of each grain in Å, inside the box.
    radius : numpy.ndarray
        (G,) radius in Å of the smallest sphere about the center that holds
        the grain.
    radius_of_gyration : numpy.ndarray
        (G,) radius of gyration in Å.
    colors : numpy.ndarray
        (G, 3) a color for each grain, RGB from 0 to 1.
    threshold : float
        The threshold applied: the log merge distance of graph clustering,
        or the misorientation in degrees of the spanning tree.
    merge_distance : numpy.ndarray
        (M,) merge distance (graph clustering) or misorientation in degrees
        (spanning tree) of each merge of two clusters of at least 20 atoms,
        in the order of the merges.
    merge_size : numpy.ndarray
        (M,) number of atoms of the smaller cluster of each of these merges.
    """

    grain: np.ndarray
    size: np.ndarray
    structure: np.ndarray
    names: tuple[str, ...]
    orientation: np.ndarray
    center: np.ndarray
    radius: np.ndarray
    radius_of_gyration: np.ndarray
    colors: np.ndarray
    threshold: float
    merge_distance: np.ndarray
    merge_size: np.ndarray

    @property
    def count(self) -> int:
        """Number of grains."""
        return len(self.size)

    def atom_colors(
        self, none: tuple[float, float, float] = (0.8, 0.8, 0.8)
    ) -> np.ndarray:
        """(N, 3) the color of the grain of each atom, ``none`` for atoms in
        no grain."""
        colors = np.empty((len(self.grain), 3))
        colors[:] = none
        inside = self.grain >= 0
        colors[inside] = self.colors[self.grain[inside]]
        return colors

    def columns(self) -> dict[str, np.ndarray]:
        return {"grain": self.grain}


def _grain_colors(count: int) -> np.ndarray:
    """Distinct colors: hues spaced by the golden ratio."""
    colors = np.empty((count, 3))
    for k in range(count):
        hue = (0.11 + 0.6180339887498949 * k) % 1.0
        saturation = 0.55 + 0.3 * ((k * 7) % 5) / 4
        value = 0.95 - 0.2 * ((k * 3) % 4) / 3
        colors[k] = colorsys.hsv_to_rgb(hue, saturation, value)
    return colors


class GrainSegmentation:
    """Grains of a polycrystal: groups of crystalline atoms of nearly the
    same lattice orientation, from polyhedral template matching, as OVITO's
    Grain segmentation modifier finds them.

    Parameters
    ----------
    threshold : float, optional
        With ``method="graph"``, the largest log merge distance of the merges
        kept; by default chosen from the merges. With
        ``method="spanning_tree"``, the largest misorientation in degrees,
        needed.
    method : str, optional
        ``"graph"``: clusters merged by node pair sampling of the graph of
        bonds weighted by their misorientation; ``"spanning_tree"``: merged
        along the bonds of least misorientation. Default ``"graph"``.
    min_grain_size : int, optional
        Clusters of fewer atoms are no grains. Default 100.
    stacking_faults : bool, optional
        The atoms of stacking faults and twin boundaries, hcp in fcc and
        hexagonal in cubic diamond or the other way round, join the grains
        around them. Default True.
    adopt_orphans : bool, optional
        Atoms in no grain join the grain nearest along their bonds. Default
        True.
    structures : iterable of str, optional
        The structures polyhedral template matching looks for. Default fcc,
        hcp and bcc.
    rmsd_cutoff : float, optional
        Atoms whose root-mean-square deviation from the best template is
        larger are of no structure; 0 for no cutoff. Default 0.1.
    """

    METHODS = ("graph", "spanning_tree")

    def __init__(
        self,
        threshold: float | None = None,
        *,
        method: str = "graph",
        min_grain_size: int = 100,
        stacking_faults: bool = True,
        adopt_orphans: bool = True,
        structures: tuple[str, ...] = ("fcc", "hcp", "bcc"),
        rmsd_cutoff: float = 0.1,
    ):
        if method not in self.METHODS:
            raise InvalidParameter(
                f"method must be one of {self.METHODS}, got {method!r}"
            )
        if threshold is not None and not np.isfinite(threshold):
            raise InvalidParameter(f"threshold must be finite, got {threshold!r}")
        if method == "spanning_tree" and (threshold is None or not threshold > 0):
            raise InvalidParameter(
                "the spanning tree needs a threshold: the largest misorientation in degrees"
            )
        if (
            isinstance(min_grain_size, bool)
            or int(min_grain_size) != min_grain_size
            or min_grain_size < 1
        ):
            raise InvalidParameter(
                f"min_grain_size must be a positive integer, got {min_grain_size!r}"
            )
        self.threshold = None if threshold is None else float(threshold)
        self.method = method
        self.min_grain_size = int(min_grain_size)
        self.stacking_faults = bool(stacking_faults)
        self.adopt_orphans = bool(adopt_orphans)
        self.__ptm = PolyhedralTemplateMatching(
            structures, rmsd_cutoff=rmsd_cutoff, template_neighbors=True
        )
        self.structures = self.__ptm.structures
        self.rmsd_cutoff = self.__ptm.rmsd_cutoff

    def compute(
        self,
        system: System,
        *,
        ptm: PTMResult | None = None,
        neighbors: NeighborList | None = None,
        progress: Any = None,
        cancel: Any = None,
    ) -> GrainResult:
        """The grains of system.

        Parameters
        ----------
        system : System
            The atoms, in a box at least twice as wide as their bonds along
            its periodic directions.
        ptm : PTMResult, optional
            Polyhedral template matching of system already done, with
            orientations and the neighbors in the order of the templates;
            done here when not given.
        neighbors : NeighborList, optional
            Neighbors of system with at least the 18 nearest of each atom,
            for the template matching.

        Returns
        -------
        GrainResult
            The grain of each atom and the grains.

        Raises
        ------
        InvalidParameter
            If ptm is not of system or has no neighbors in the order of the
            templates, or if the box is too short for the bonds.
        """
        n = len(system)
        if ptm is None:
            ptm = self.__ptm.compute(
                system, neighbors=neighbors, progress=progress, cancel=cancel
            )
        elif ptm.neighbors is None or len(ptm.structure) != n:
            raise InvalidParameter(
                "ptm must be of this system, from PolyhedralTemplateMatching("
                "template_neighbors=True)"
            )
        kinds = np.asarray(ptm.structure, dtype=np.int8)
        rows = np.array(ptm.neighbors, dtype=np.int64)
        counts = _TEMPLATE_NEIGHBORS[kinds]
        disordered = np.flatnonzero(kinds == 0)
        if len(disordered) and n > 1:
            k = min(_DISORDERED_NEIGHBORS, n - 1)
            near = NeighborList.build(system, num_neighbors=k).sorted()
            starts = near.offsets[disordered]
            take = np.minimum(near.counts[disordered], k)
            columns = np.arange(k)
            within = columns < take[:, None]
            picked = near.j[
                np.minimum(starts[:, None] + columns, max(len(near.j) - 1, 0))
            ]
            rows[disordered, :k] = np.where(within, picked, -1)
            counts[disordered] = take
        mask = np.arange(rows.shape[1]) < counts[:, None]
        order = rows[mask]
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        positions = np.ascontiguousarray(system["positions"], dtype=np.float64)
        owners = np.repeat(np.arange(n), counts)
        # bonds are shorter than half the box, checked below: the image of
        # nearest fractional coordinates is theirs
        pbc = np.asarray(system.box.pbc, dtype=bool)
        fractional = (positions[order] - positions[owners]) @ system.box.inverse
        fractional[:, pbc] -= np.round(fractional[:, pbc])
        lengths = np.linalg.norm(fractional @ system.box.matrix, axis=1)
        if len(lengths) and pbc.any():
            thickness = system.box.thickness[pbc]
            if 2 * lengths.max() >= thickness.min():
                raise InvalidParameter(
                    "the box is too short for the bonds between neighbors along a "
                    "periodic direction: replicate it first"
                )
        masses = None
        if "mass" in system.table.columns:
            masses = np.ascontiguousarray(system["mass"], dtype=np.float64)
        automatic = self.method == "graph" and self.threshold is None
        found = _native.grain_segmentation(
            np.ascontiguousarray(kinds),
            # in single precision, as OVITO keeps orientations: the atoms
            # between two nearly equal choices go the way OVITO sends them
            np.asarray(ptm.orientation, dtype=np.float32).astype(np.float64),
            offsets,
            np.ascontiguousarray(order, dtype=index_dtype),
            np.ascontiguousarray(lengths),
            positions,
            np.ascontiguousarray(system.box.matrix, dtype=np.float64),
            np.ascontiguousarray(system.box.origin, dtype=np.float64),
            np.ascontiguousarray(pbc),
            masses,
            self.method == "spanning_tree",
            automatic,
            0.0 if self.threshold is None else self.threshold,
            self.min_grain_size,
            self.stacking_faults,
            self.adopt_orphans,
            progress,
            _native_token(cancel),
        )
        size = found["size"]
        return GrainResult(
            grain=found["grain"],
            size=size,
            structure=found["structure"],
            names=PolyhedralTemplateMatching.NAMES,
            orientation=found["orientation"].reshape(-1, 4),
            center=found["center"].reshape(-1, 3),
            radius=found["radius"],
            radius_of_gyration=found["gyration"],
            colors=_grain_colors(len(size)),
            threshold=float(found["threshold"]),
            merge_distance=found["merge_distance"],
            merge_size=found["merge_size"],
        )

    def __repr__(self) -> str:
        return (
            f"GrainSegmentation({self.threshold!r}, method={self.method!r}, "
            f"min_grain_size={self.min_grain_size}, stacking_faults={self.stacking_faults}, "
            f"adopt_orphans={self.adopt_orphans})"
        )
