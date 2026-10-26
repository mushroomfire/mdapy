# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The surface mesh of a solid of atoms, from the alpha shape of their
Delaunay tessellation."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.mesh import TriangleMesh
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter

# the atoms are moved at random by up to this much (Å) before the
# tessellation, which makes that of a perfect lattice unique
_JITTER = 1e-4


@dataclass(frozen=True)
class SurfaceRegions(Result):
    """The regions of connected solid or empty space the surface separates.

    Attributes
    ----------
    solid : numpy.ndarray
        (R,) bool whether each region is solid.
    volume : numpy.ndarray
        (R,) volume of each region in Å³; infinite for the exterior one.
    area : numpy.ndarray
        (R,) area of the surface around each region in Å².
    exterior : numpy.ndarray
        (R,) bool whether the region is the empty space around the atoms,
        out of a box that is not periodic.
    periodic : numpy.ndarray
        (R,) bool whether the region runs across a periodic boundary into
        its own periodic image, without end.
    faces : numpy.ndarray
        (F, 2) int64 the solid and the empty region on the two sides of
        each triangle of the mesh.
    """

    solid: np.ndarray
    volume: np.ndarray
    area: np.ndarray
    exterior: np.ndarray
    periodic: np.ndarray
    faces: np.ndarray

    @property
    def voids(self) -> np.ndarray:
        """(K,) int64 the empty regions enclosed by the solid on every side."""
        return np.flatnonzero(~self.solid & ~self.exterior & ~self.periodic)


@dataclass(frozen=True)
class SurfaceMeshResult(Result):
    """The surface of the solid.

    Attributes
    ----------
    mesh : TriangleMesh
        The surface, its normals pointing out of the solid.
    atoms : numpy.ndarray
        (V,) int64 the atom at each vertex of the mesh.
    area : float
        Area of the surface in Å².
    solid_volume : float
        Volume of the solid in Å³.
    regions : SurfaceRegions or None
        The solid and empty regions, when asked for.
    surface : numpy.ndarray or None
        (N,) bool whether each atom lies on the surface, when asked for.
    """

    mesh: TriangleMesh
    atoms: np.ndarray
    area: float
    solid_volume: float
    regions: SurfaceRegions | None = None
    surface: np.ndarray | None = None

    def columns(self) -> dict[str, np.ndarray]:
        return {} if self.surface is None else {"surface": self.surface}


class SurfaceMesh:
    """The surface of the solid the atoms make, by the alpha-shape method:
    the space a probe sphere cannot enter is solid.

    Parameters
    ----------
    radius : float, optional
        Radius of the probe sphere in Å: the tetrahedra of the Delaunay
        tessellation of the atoms whose circumscribed sphere is smaller are
        solid. Larger than the distance between neighbors, and smaller than
        the voids looked for. Default 4.
    smoothing : int, optional
        Number of smoothing steps of the mesh, which round its shape and
        keep the volumes. Default 8.
    regions : bool, optional
        Also give the regions of solid and empty space, and the voids among
        them. Default False.
    surface : bool, optional
        Also mark the atoms on the surface. Default False.
    """

    def __init__(
        self,
        radius: float = 4.0,
        smoothing: int = 8,
        *,
        regions: bool = False,
        surface: bool = False,
    ):
        radius = float(radius)
        if not np.isfinite(radius) or radius <= 0:
            raise InvalidParameter(f"radius must be positive, not {radius!r}")
        if isinstance(smoothing, bool) or int(smoothing) != smoothing or smoothing < 0:
            raise InvalidParameter(
                f"smoothing must be an integer of at least 0, not {smoothing!r}"
            )
        self.radius = radius
        self.smoothing = int(smoothing)
        self.regions = bool(regions)
        self.surface = bool(surface)

    def compute(
        self, system: System, *, progress: Any = None, cancel: Any = None
    ) -> SurfaceMeshResult:
        """The surface of a system.

        Parameters
        ----------
        system : System
            The atoms, no two at the same position. Out of a box that is not
            periodic is empty space.

        Returns
        -------
        SurfaceMeshResult
            The surface, its area and the volume of the solid.

        Raises
        ------
        InvalidParameter
            For a periodic box so short that an atom meets its own periodic
            image: replicate the system first.
        """
        box = system.box
        n = len(system)
        jittered = system["positions"] + np.random.default_rng(20261023).uniform(
            -_JITTER, _JITTER, (n, 3)
        )
        # into the box along its periodic directions, both alike
        fractional = (jittered - box.origin) @ box.inverse
        image = np.where(box.pbc, np.floor(fractional), 0.0)
        moved = image @ box.matrix
        jittered = jittered - moved
        positions = system["positions"] - moved
        # the walls of the open directions a probe beyond the atoms, so that
        # no tetrahedron that may be solid falls outside
        matrix, origin = box.matrix.copy(), box.origin.copy()
        if n:
            for axis in np.flatnonzero(~box.pbc):
                margin = (self.radius + 1.0) / box.thickness[axis]
                low = fractional[:, axis].min() - margin
                high = fractional[:, axis].max() + margin
                origin += low * box.matrix[axis]
                matrix[axis] = (high - low) * box.matrix[axis]
        try:
            atoms, vertices, faces, shifts, solid_volume, found = _native.surface_mesh(
                np.ascontiguousarray(positions),
                np.ascontiguousarray(jittered),
                np.ascontiguousarray(matrix),
                np.ascontiguousarray(origin),
                np.ascontiguousarray(box.pbc),
                self.radius,
                self.smoothing,
                self.regions,
                progress,
                _native_token(cancel),
            )
        except ValueError as error:
            raise InvalidParameter(str(error)) from None
        except RuntimeError as error:  # cells that touch their own images
            raise InvalidParameter(
                f"{error}: replicate the system along its short periodic directions"
            ) from None
        mesh = TriangleMesh(
            vertices.reshape(-1, 3), faces.reshape(-1, 3), shifts.reshape(-1, 3, 3), box
        )
        areas = mesh.areas
        regions = None
        if found is not None:
            sides, solid, exterior, periodic, volume = found
            sides = sides.reshape(-1, 2)
            exterior = exterior.astype(bool)
            volume[exterior] = np.inf
            count = len(solid)
            regions = SurfaceRegions(
                solid=solid.astype(bool),
                volume=volume,
                area=np.bincount(sides.ravel(), np.repeat(areas, 2), count).astype(
                    float
                ),
                exterior=exterior,
                periodic=periodic.astype(bool),
                faces=sides,
            )
        on_surface = None
        if self.surface:
            on_surface = np.zeros(n, dtype=bool)
            on_surface[atoms] = True
        return SurfaceMeshResult(
            mesh=mesh,
            atoms=atoms,
            area=float(areas.sum()),
            solid_volume=float(solid_volume),
            regions=regions,
            surface=on_surface,
        )

    def __repr__(self) -> str:
        return (
            f"SurfaceMesh(radius={self.radius!r}, smoothing={self.smoothing!r}, "
            f"regions={self.regions!r}, surface={self.surface!r})"
        )
