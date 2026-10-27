# Copyright (c) 2022-2026, Yongchao Wu
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""The dislocation lines of a crystal by the dislocation extraction
algorithm: Burgers circuits on the mesh between the good crystal and the
rest, swept along the lines."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mdapy._core import analysis as _native
from mdapy.core.mesh import TriangleMesh
from mdapy.core.result import Result
from mdapy.core.system import System
from mdapy.core.threads import _native_token
from mdapy.errors import InvalidParameter
from mdapy.identify import PolyhedralTemplateMatching

# the atoms are moved at random by up to this much (Å) before the
# tessellation, which makes that of a perfect lattice unique
_JITTER = 1e-4

# the PTM type of each crystal
_CRYSTALS = {"fcc": 1, "hcp": 2, "bcc": 3, "cubic_diamond": 6, "hexagonal_diamond": 7}

# the families of Burgers vectors of the cubic lattices: their components
# in units of the lattice constant, in any order and sign
_CUBIC_FAMILIES = {
    "fcc": {
        "1/2<110>": (0, 1 / 2, 1 / 2),
        "1/6<112>": (1 / 6, 1 / 6, 1 / 3),
        "1/6<110>": (0, 1 / 6, 1 / 6),
        "1/3<100>": (0, 0, 1 / 3),
        "1/3<111>": (1 / 3, 1 / 3, 1 / 3),
    },
    "bcc": {"1/2<111>": (1 / 2, 1 / 2, 1 / 2), "<100>": (0, 0, 1), "<110>": (0, 1, 1)},
    "cubic_diamond": {
        "1/2<110>": (0, 1 / 2, 1 / 2),
        "1/6<112>": (1 / 6, 1 / 6, 1 / 3),
        "1/6<110>": (0, 1 / 6, 1 / 6),
        "1/3<111>": (1 / 3, 1 / 3, 1 / 3),
    },
}

# those of the hexagonal lattices: the Miller-Bravais components in the
# basal plane in any order and sign, and the one along c
_HEXAGONAL_FAMILIES = {
    "1/3<11-20>": ((1 / 3, 1 / 3, 2 / 3), 0),
    "<0001>": ((0, 0, 0), 1),
    "<1-100>": ((0, 1, 1), 0),
    "1/3<1-100>": ((0, 1 / 3, 1 / 3), 0),
    "1/3<11-23>": ((1 / 3, 1 / 3, 2 / 3), 1),
}

# the lattice parameters of the hexagonal frames, c along z and a1 along x,
# in units of the cubic lattice constant of the same nearest neighbors
_A = np.sqrt(0.5)
_C = np.sqrt(4 / 3)


def _family(burgers: np.ndarray, crystal: str) -> str:
    """The family of a Burgers vector in the lattice frame of a cluster of
    the crystal, "other" for none."""
    if crystal in _CUBIC_FAMILIES:
        key = np.sort(np.abs(burgers))
        for name, components in _CUBIC_FAMILIES[crystal].items():
            if np.allclose(key, components, atol=1e-3):
                return name
        return "other"
    # the Miller-Bravais indices: b = U a1 + V a2 + W c
    v = burgers[1] / (_A * np.sqrt(3) / 2)
    u = burgers[0] / _A + v / 2
    w = burgers[2] / _C
    basal = np.sort(np.abs([(2 * u - v) / 3, (2 * v - u) / 3, -(u + v) / 3]))
    for name, (components, along) in _HEXAGONAL_FAMILIES.items():
        if crystal == "hexagonal_diamond" and name == "1/3<11-23>":
            continue
        if np.allclose(basal, components, atol=1e-3) and np.isclose(
            abs(w), along, atol=1e-3
        ):
            return name
    return "other"


def _families(crystal: str) -> tuple[str, ...]:
    """The families of Burgers vectors of a crystal, then "other"."""
    if crystal in _CUBIC_FAMILIES:
        return (*_CUBIC_FAMILIES[crystal], "other")
    names = [
        name
        for name in _HEXAGONAL_FAMILIES
        if not (crystal == "hexagonal_diamond" and name == "1/3<11-23>")
    ]
    return (*names, "other")


@dataclass(frozen=True)
class DislocationLine:
    """A dislocation line between two junctions or free ends, or closed on
    itself.

    Attributes
    ----------
    points : numpy.ndarray
        (P, 3) points along the line in Å, unwrapped: consecutive points
        are close also across a periodic boundary.
    burgers : numpy.ndarray
        (3,) the Burgers vector in the lattice frame of its crystal cluster,
        in units of the lattice constant, such as (0.5, 0.5, 0) for
        a/2[110]: the hexagonal lattices with c along z and a1 along x, in
        units of the cubic lattice constant with the same nearest
        neighbors.
    spatial_burgers : numpy.ndarray
        (3,) the Burgers vector in Å along the axes of the box, as the
        crystal cluster is turned and stretched.
    cluster : int
        The crystal cluster of the frame of the Burgers vector.
    family : str
        The family of the Burgers vector, such as "1/2<110>", or "other".
    length : float
        Length in Å.
    loop : bool
        Whether the line closes on itself: a loop, or a line through
        periodic boundaries.
    infinite : bool
        Whether the line closes through periodic boundaries.
    junctions : tuple of int
        The junctions at the first and the last point, -1 for a free end
        and for the ends of a closed line.
    """

    points: np.ndarray = field(repr=False)
    burgers: np.ndarray
    spatial_burgers: np.ndarray
    cluster: int
    family: str
    length: float
    loop: bool
    infinite: bool
    junctions: tuple[int, int]


@dataclass(frozen=True)
class CrystalClusters(Result):
    """The crystal clusters: atoms of one structure joined through their
    neighbors into one lattice frame.

    Attributes
    ----------
    structure : numpy.ndarray
        (C,) int8 structure of each cluster, an index into the names of the
        result.
    size : numpy.ndarray
        (C,) int64 number of atoms of each.
    orientation : numpy.ndarray
        (C, 3, 3) the matrix that takes vectors of the lattice frame of
        each, in units of the lattice constant, to vectors in Å along the
        axes of the box, from the neighbors of its atoms.
    """

    structure: np.ndarray
    size: np.ndarray
    orientation: np.ndarray


@dataclass(frozen=True)
class DislocationResult(Result):
    """The dislocations of a crystal.

    Attributes
    ----------
    crystal : str
        The crystal analyzed.
    lines : tuple of DislocationLine
        The dislocation lines.
    junctions : numpy.ndarray
        (J, 3) the points in Å where three or more lines meet, in the box.
    structure : numpy.ndarray
        (N,) int8 structure of each atom, an index into ``names``.
    names : tuple of str
        The structures: other, fcc, hcp, bcc, cubic_diamond and
        hexagonal_diamond.
    colors : tuple of tuple of float
        Default RGB color of each structure, from 0 to 1.
    crystal_cluster : numpy.ndarray
        (N,) int64 crystal cluster of each atom, -1 for none.
    clusters : CrystalClusters
        The crystal clusters.
    defect_mesh : TriangleMesh
        The surface around the defects that are not dislocations, such as
        grain boundaries, free surfaces and voids, closed at the free ends
        of the lines, its normals pointing out of the crystal.
    defect_atoms : numpy.ndarray
        (V,) int64 the atom at each vertex of the defect mesh, -1 at the
        end of a line.
    interface_mesh : TriangleMesh or None
        The mesh between the tetrahedra of the good crystal and the rest,
        its normals pointing into the good crystal, when asked for.
    interface_atoms : numpy.ndarray or None
        (V,) int64 the atom at each vertex of the interface mesh, when asked
        for.
    """

    crystal: str
    lines: tuple[DislocationLine, ...]
    junctions: np.ndarray
    structure: np.ndarray
    names: tuple[str, ...]
    colors: tuple[tuple[float, float, float], ...]
    crystal_cluster: np.ndarray
    clusters: CrystalClusters
    defect_mesh: TriangleMesh
    defect_atoms: np.ndarray
    interface_mesh: TriangleMesh | None = None
    interface_atoms: np.ndarray | None = None

    @property
    def total_length(self) -> float:
        """Total length of the lines in Å."""
        return float(sum(line.length for line in self.lines))

    def counts(self) -> dict[str, int]:
        """Number of atoms of each structure, by name."""
        found = np.bincount(self.structure, minlength=len(self.names))
        return {name: int(c) for name, c in zip(self.names, found)}

    def lengths(self) -> dict[str, float]:
        """Total length in Å of the lines of each family of Burgers vectors
        of the crystal, "other" last."""
        found = dict.fromkeys(_families(self.crystal), 0.0)
        for line in self.lines:
            found[line.family] += line.length
        return found

    def columns(self) -> dict[str, np.ndarray]:
        return {"structure": self.structure, "crystal_cluster": self.crystal_cluster}


class DislocationAnalysis:
    """Dislocations of a crystal by the dislocation extraction algorithm
    (DXA): Burgers circuits on the mesh between the tetrahedra of the atoms
    where the crystal maps onto the perfect lattice and the rest.

    Parameters
    ----------
    crystal : str
        The crystal: fcc, hcp, bcc, cubic_diamond or hexagonal_diamond.
    circuit_length : int, optional
        Number of edges of the longest Burgers circuit that finds a line;
        at least 3. Default 14.
    stretch : int, optional
        Number of edges more a circuit may grow to follow a line. Default 9.
    perfect_only : bool, optional
        Find perfect dislocations only: the stacking faults of fcc and hcp,
        or of the diamonds, count as defects, without partial dislocations
        at their edges. Default False.
    point_separation : float, optional
        Sparseness of the points along the lines: each point is the mean of
        a run of the points found, longer where the lines are wider; 0
        keeps them all. Default 2.5.
    smoothing : int, optional
        Number of smoothing steps of the lines. Default 1.
    mesh_smoothing : int, optional
        Number of smoothing steps of the defect mesh. Default 8.
    interface_mesh : bool, optional
        Also give the interface mesh, between the tetrahedra of the good
        crystal and the rest, on which the circuits run. Default False.
    """

    #: Names of the structure types.
    NAMES = ("other", "fcc", "hcp", "bcc", "cubic_diamond", "hexagonal_diamond")
    #: The color of the lines of each family of Burgers vectors, by crystal.
    FAMILY_COLORS = {
        "fcc": {
            "1/2<110>": (0.2, 0.2, 1.0),
            "1/6<112>": (0.0, 1.0, 0.0),
            "1/6<110>": (1.0, 0.0, 1.0),
            "1/3<100>": (1.0, 1.0, 0.0),
            "1/3<111>": (0.0, 1.0, 1.0),
            "other": (0.9, 0.2, 0.2),
        },
        "bcc": {
            "1/2<111>": (0.0, 1.0, 0.0),
            "<100>": (1.0, 0.3, 0.8),
            "<110>": (0.2, 0.5, 1.0),
            "other": (0.9, 0.2, 0.2),
        },
        "cubic_diamond": {
            "1/2<110>": (0.2, 0.2, 1.0),
            "1/6<112>": (0.0, 1.0, 0.0),
            "1/6<110>": (1.0, 0.0, 1.0),
            "1/3<111>": (0.0, 1.0, 1.0),
            "other": (0.9, 0.2, 0.2),
        },
    }
    FAMILY_COLORS["hcp"] = {
        "1/3<11-20>": (0.0, 1.0, 0.0),
        "<0001>": (0.2, 0.2, 1.0),
        "<1-100>": (1.0, 0.0, 1.0),
        "1/3<1-100>": (1.0, 0.5, 0.0),
        "1/3<11-23>": (1.0, 1.0, 0.0),
        "other": (0.9, 0.2, 0.2),
    }
    FAMILY_COLORS["hexagonal_diamond"] = {
        k: v for k, v in FAMILY_COLORS["hcp"].items() if k != "1/3<11-23>"
    }
    #: Their default colors.
    COLORS = tuple(
        PolyhedralTemplateMatching.COLORS[PolyhedralTemplateMatching.NAMES.index(name)]
        for name in NAMES
    )

    def __init__(
        self,
        crystal: str,
        *,
        circuit_length: int = 14,
        stretch: int = 9,
        perfect_only: bool = False,
        point_separation: float = 2.5,
        smoothing: int = 1,
        mesh_smoothing: int = 8,
        interface_mesh: bool = False,
    ):
        if crystal not in _CRYSTALS:
            raise InvalidParameter(
                f"crystal must be one of {', '.join(_CRYSTALS)}, not {crystal!r}"
            )
        for name, value, low in [
            ("circuit_length", circuit_length, 3),
            ("stretch", stretch, 0),
            ("smoothing", smoothing, 0),
            ("mesh_smoothing", mesh_smoothing, 0),
        ]:
            if isinstance(value, bool) or int(value) != value or value < low:
                raise InvalidParameter(
                    f"{name} must be an integer of at least {low}, not {value!r}"
                )
        point_separation = float(point_separation)
        if not np.isfinite(point_separation) or point_separation < 0:
            raise InvalidParameter(
                f"point_separation must be 0 or more, not {point_separation!r}"
            )
        self.crystal = crystal
        self.circuit_length = int(circuit_length)
        self.stretch = int(stretch)
        self.perfect_only = bool(perfect_only)
        self.point_separation = point_separation
        self.smoothing = int(smoothing)
        self.mesh_smoothing = int(mesh_smoothing)
        self.interface_mesh = bool(interface_mesh)

    def compute(
        self, system: System, *, progress: Any = None, cancel: Any = None
    ) -> DislocationResult:
        """The dislocations of a system.

        Parameters
        ----------
        system : System
            The atoms, no two at the same position.

        Returns
        -------
        DislocationResult
            The dislocation lines, the structure and crystal cluster of each
            atom.

        Raises
        ------
        InvalidParameter
            For a box so short along a periodic direction that an atom meets
            its own periodic image or an edge of the mesh spans half of it:
            replicate the system first.
        """
        box = system.box
        n = len(system)
        jittered = system["positions"] + np.random.default_rng(20261023).uniform(
            -_JITTER, _JITTER, (n, 3)
        )
        # into the box along its periodic directions, both alike
        fractional = (jittered - box.origin) @ box.inverse
        moved = np.where(box.pbc, np.floor(fractional), 0.0) @ box.matrix
        try:
            found = _native.dislocation_analysis(
                np.ascontiguousarray(system["positions"] - moved),
                np.ascontiguousarray(jittered - moved),
                np.ascontiguousarray(box.matrix),
                np.ascontiguousarray(box.inverse),
                np.ascontiguousarray(box.origin),
                np.ascontiguousarray(box.pbc),
                _CRYSTALS[self.crystal],
                not self.perfect_only,
                self.circuit_length,
                self.stretch,
                self.point_separation,
                self.smoothing,
                self.mesh_smoothing,
                self.interface_mesh,
                progress,
                _native_token(cancel),
            )
        except (ValueError, RuntimeError) as error:  # periodic images too close
            raise InvalidParameter(
                f"{error}: replicate the system along its short periodic directions"
            ) from None
        (
            structure,
            crystal_cluster,
            cluster_structure,
            size,
            orientation,
            points,
            offsets,
            burgers,
            line_cluster,
            ends,
            closed,
            count,
            mesh_atoms,
            mesh_vertices,
            mesh_faces,
            mesh_shifts,
            interface_atoms,
            interface_vertices,
            interface_faces,
            interface_shifts,
        ) = found
        # the PTM types as indices into NAMES
        index = np.zeros(8, dtype=np.int8)
        for k, name in enumerate(self.NAMES[1:], start=1):
            index[_CRYSTALS[name]] = k
        orientation = orientation.reshape(-1, 3, 3)
        points = points.reshape(-1, 3)
        burgers = burgers.reshape(-1, 3)
        ends = ends.reshape(-1, 2)
        junctions = np.zeros((count, 3))
        lines = []
        for s in range(len(line_cluster)):
            p = points[offsets[s] : offsets[s + 1]]
            p.flags.writeable = False
            b = burgers[s].copy()
            spatial = orientation[line_cluster[s]] @ b
            b.flags.writeable = spatial.flags.writeable = False
            loop = bool(closed[s])
            for end, at in zip(ends[s], (p[0], p[-1])):
                if end >= 0:
                    junctions[end] = at
            lines.append(
                DislocationLine(
                    points=p,
                    burgers=b,
                    spatial_burgers=spatial,
                    cluster=int(line_cluster[s]),
                    family=_family(b, self.crystal),
                    length=float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()),
                    loop=loop,
                    infinite=loop and not np.allclose(p[0], p[-1], atol=1e-4),
                    junctions=(int(ends[s, 0]), int(ends[s, 1])),
                )
            )
        if count:
            fraction = (junctions - box.origin) @ box.inverse
            junctions -= np.where(box.pbc, np.floor(fraction), 0.0) @ box.matrix
        return DislocationResult(
            crystal=self.crystal,
            lines=tuple(lines),
            junctions=junctions,
            structure=index[structure],
            names=self.NAMES,
            colors=self.COLORS,
            crystal_cluster=crystal_cluster,
            clusters=CrystalClusters(
                structure=index[cluster_structure], size=size, orientation=orientation
            ),
            defect_mesh=TriangleMesh(
                mesh_vertices.reshape(-1, 3),
                mesh_faces.reshape(-1, 3),
                mesh_shifts.reshape(-1, 3, 3),
                box,
            ),
            defect_atoms=mesh_atoms,
            interface_mesh=None
            if not self.interface_mesh
            else TriangleMesh(
                interface_vertices.reshape(-1, 3),
                interface_faces.reshape(-1, 3),
                interface_shifts.reshape(-1, 3, 3),
                box,
            ),
            interface_atoms=interface_atoms if self.interface_mesh else None,
        )

    def __repr__(self) -> str:
        return (
            f"DislocationAnalysis({self.crystal!r}, circuit_length={self.circuit_length!r}, "
            f"stretch={self.stretch!r}, perfect_only={self.perfect_only!r}, "
            f"point_separation={self.point_separation!r}, smoothing={self.smoothing!r}, "
            f"mesh_smoothing={self.mesh_smoothing!r}, "
            f"interface_mesh={self.interface_mesh!r})"
        )
