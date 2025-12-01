# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""

This module provides efficient Voronoi tessellation analysis for atomic systems,
enabling calculation of Voronoi cells, neighbor identification, volumes, and
detailed geometric information. The implementation uses OpenMP parallelization
through the Voro++ library for high performance.

The module supports both orthogonal and triclinic simulation boxes with
periodic boundary conditions.

References
----------
.. [1] Lu J, Lazar E A, Rycroft C H. An extension to
    Voro++ for multithreaded computation of Voronoi cells[J].
    Computer Physics Communications, 2023, 291: 108832.
    https://doi.org/10.1016/j.cpc.2023.108832
"""

from mdapy import _voronoi
from mdapy.box import Box
import numpy as np
import polars as pl
import os
from typing import Tuple, List, Union
import mdapy.tool_function as tool
from dataclasses import dataclass


class Voronoi:
    """Voronoi tessellation analysis for atomic systems.

    This class provides methods to compute Voronoi cells and their properties
    for a system of atoms/particles. It supports both orthogonal and triclinic
    simulation boxes with periodic boundary conditions.

    Parameters
    ----------
    box : Box
        The simulation box object containing box vectors, boundary conditions,
        and origin information.
    data : pl.DataFrame
        A Polars DataFrame containing atomic positions with columns 'x', 'y', 'z'.

    Attributes
    ----------
    box : Box
        The simulation box object.
    data : pl.DataFrame
        The atomic position data.
    _enlarge_data : pl.DataFrame, optional
        Internal replicated atomic data when periodic extension is required.
    _enlarge_box : Box, optional
        The enlarged simulation box corresponding to replicated atoms.

    Notes
    -----
    The implementation automatically handles small systems by replicating atoms
    to ensure sufficient neighbors for accurate Voronoi tessellation. For
    triclinic boxes, automatic rotation to LAMMPS-compatible format is performed
    when necessary.

    """

    def __init__(self, box: Box, data: pl.DataFrame):
        self.box = box
        self.data = data

    def get_neighbor(
        self, a_face_area_threshold: float = -1.0, r_face_area_threshold: float = -1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Voronoi neighbors and face properties for all atoms.

        This method performs Voronoi tessellation to identify nearest neighbors
        based on shared Voronoi cell faces. It can filter neighbors based on
        face area thresholds to exclude insignificant contacts.

        Parameters
        ----------
        a_face_area_threshold : float, optional
            Absolute face area threshold.
            Faces with area below this value are ignored as neighbors.
            Default is -1.0 (no filtering).
        r_face_area_threshold : float, optional
            Relative face area threshold (fraction of average face area).
            Faces with relative area below this value are ignored.
            Default is -1.0 (no filtering).

        Returns
        -------
        verlet_list : np.ndarray
            2D array of shape (N, max_neighbors) containing neighbor indices
            for each atom. Unused entries are filled with -1.
        distance_list : np.ndarray
            2D array of shape (N, max_neighbors) containing distances to
            each neighbor (in simulation units).
        face_area : np.ndarray
            2D array of shape (N, max_neighbors) containing the area of
            the Voronoi face shared with each neighbor.
        neighbor_number : np.ndarray
            1D array of shape (N,) containing the coordination number
            (number of Voronoi neighbors) for each atom.

        Notes
        -----
        - If both thresholds are provided, the larger effective threshold is used.
        - The method automatically handles periodic boundary conditions.
        - For small systems (<50 atoms), automatic replication is performed
          to ensure accurate neighbor identification.
        - OpenMP parallelization is used for performance with large systems.

        """
        num_t = os.cpu_count()
        repeat = [1, 1, 1]
        N = self.data.shape[0]
        nopbc = False
        if N < 50:
            if sum(self.box.boundary) > 0:
                while np.prod(repeat) * N < 50:
                    for i in range(3):
                        if self.box.boundary[i] == 1:
                            repeat[i] += 1
            else:
                assert N > 1, "system with all free boundary must has at least 2 atoms."
                nopbc = True
        data = self.data
        box = self.box
        if sum(repeat) != 3:
            # Small box: replicate atoms to find enough neighbors
            self._enlarge_data, self._enlarge_box = tool._replicate_pos(
                data, box, *repeat
            )
            data = self._enlarge_data
            box = self._enlarge_box

        if box.triclinic and not nopbc:
            need_rotation = False
            if (
                abs(box.box[0, 1]) > 1e-10
                or abs(box.box[0, 2]) > 1e-10
                or abs(box.box[1, 2]) > 1e-10
                or box.box[0, 0] < 0
                or box.box[1, 1] < 0
                or box.box[2, 2] < 0
            ):
                need_rotation = True
            box, rotation = box.align_to_lammps_box()
            for i in range(3):
                if box.boundary[i] == 0:
                    box.box[i] *= 3
            verlet_list, distance_list, face_area, neighbor_number = (
                _voronoi.get_voronoi_neighbor_tri(
                    data["x"].to_numpy(allow_copy=False),
                    data["y"].to_numpy(allow_copy=False),
                    data["z"].to_numpy(allow_copy=False),
                    box.box,
                    box.origin,
                    box.boundary,
                    rotation,
                    need_rotation,
                    a_face_area_threshold,
                    r_face_area_threshold,
                    num_t,
                )
            )
        else:
            verlet_list, distance_list, face_area, neighbor_number = (
                _voronoi.get_voronoi_neighbor(
                    data["x"].to_numpy(allow_copy=False),
                    data["y"].to_numpy(allow_copy=False),
                    data["z"].to_numpy(allow_copy=False),
                    box.box,
                    box.origin,
                    box.boundary,
                    a_face_area_threshold,
                    r_face_area_threshold,
                    num_t,
                )
            )
        return verlet_list, distance_list, face_area, neighbor_number

    def get_cell_info(
        self,
    ) -> Tuple[
        List[List[List[int]]],
        List[List[List[float]]],
        List[float],
        List[float],
        List[List[float]],
    ]:
        """Retrieve detailed Voronoi polygon information for all atoms.

        This method provides comprehensive geometric information about each
        Voronoi cell, including face vertices, positions, and areas. Due to
        the detailed nature of the output, this method is computationally
        intensive and memory-demanding, making it suitable primarily for
        small systems.

        Returns
        -------
        face_vertices_indices : List[List[List[int]]]
            For each atom, a list of faces, where each face is defined by
            a list of vertex indices that form the face polygon.
        face_vertices_positions : List[List[List[float]]]
            For each atom, a list of faces, where each face contains the
            3D positions (x, y, z) of all vertices forming that face.
        volume : List[float]
            The volume of each atom's Voronoi cell (in cubic simulation units).
        radius : List[float]
            The cavity radius for each atom - the distance from the atom to
            the farthest vertex of its Voronoi cell (in simulation units).
        face_areas : List[List[float]]
            For each atom, a list containing the area of each face of its
            Voronoi cell (in square simulation units).

        Raises
        ------
        AssertionError
            If the box is triclinic (only orthogonal boxes are supported).
        AssertionError
            If the system contains less than 2 atoms.

        Warnings
        --------
        This method is computationally expensive and requires significant memory,
        especially for large systems. It should primarily be used for small systems.

        For routine analysis of large systems, use `get_neighbor()` or
        `get_volume()` instead.

        """
        assert not self.box.triclinic, "Only support orthogonal box."
        assert self.data.shape[0] > 1, "At least has one atom."
        num_t = os.cpu_count()
        face_vertices_indices, face_vertices_positions, volume, radius, face_areas = (
            _voronoi.get_cell_info(
                self.data["x"].to_numpy(allow_copy=False),
                self.data["y"].to_numpy(allow_copy=False),
                self.data["z"].to_numpy(allow_copy=False),
                self.box.box,
                self.box.origin,
                self.box.boundary,
                num_t,
            )
        )
        return (
            face_vertices_indices,
            face_vertices_positions,
            volume,
            radius,
            face_areas,
        )

    def get_volume(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Voronoi cell volumes, coordination numbers, and cavity radii.

        This method efficiently computes key Voronoi cell properties without
        storing detailed geometric information, making it suitable for large-scale
        analysis of atomic systems.

        Returns
        -------
        volume : np.ndarray
            1D array of shape (N,) containing the volume of each atom's
            Voronoi cell (in cubic simulation units of length).
        neighbor_number : np.ndarray
            1D array of shape (N,) containing the coordination number for
            each atom. This is the number of faces of the Voronoi cell,
            which equals the number of nearest neighbors.
        cavity_radius : np.ndarray
            1D array of shape (N,) containing the cavity radius for each atom.
            This is the distance from the atom to the farthest vertex of its
            Voronoi cell, representing the radius of the largest empty sphere
            (containing no other atoms) that touches the atom.

        """
        num_t = os.cpu_count()
        volume = np.zeros(self.data.shape[0])
        neighbor_number = np.zeros(self.data.shape[0], np.int32)
        cavity_radius = np.zeros(self.data.shape[0])

        if self.box.triclinic:
            need_rotation = False
            if (
                abs(self.box.box[0, 1]) > 1e-10
                or abs(self.box.box[0, 2]) > 1e-10
                or abs(self.box.box[1, 2]) > 1e-10
                or self.box.box[0, 0] < 0
                or self.box.box[1, 1] < 0
                or self.box.box[2, 2] < 0
            ):
                need_rotation = True

            box, rotation = self.box.align_to_lammps_box()
            for i in range(3):
                if box.boundary[i] == 0:
                    box.box[i] *= 3
            _voronoi.get_voronoi_volume_number_radius_tri(
                self.data["x"].to_numpy(allow_copy=False),
                self.data["y"].to_numpy(allow_copy=False),
                self.data["z"].to_numpy(allow_copy=False),
                box.box,
                box.origin,
                box.boundary,
                rotation,
                volume,
                neighbor_number,
                cavity_radius,
                need_rotation,
                num_t,
            )
        else:
            _voronoi.get_voronoi_volume_number_radius(
                self.data["x"].to_numpy(allow_copy=False),
                self.data["y"].to_numpy(allow_copy=False),
                self.data["z"].to_numpy(allow_copy=False),
                self.box.box,
                self.box.origin,
                self.box.boundary,
                volume,
                neighbor_number,
                cavity_radius,
                num_t,
            )
        return volume, neighbor_number, cavity_radius


@dataclass
class Cell:
    """
    A lightweight container representing a single Voronoi cell and all its
    geometric properties.


    Parameters
    ----------
    face_vertices : List[List[int]]
        A list of faces, where each face is represented by a list of vertex
        indices referring to positions in the `vertices` array.
    vertices : np.ndarray
        Array of shape (M, 3) containing the 3D coordinates of all polygon
        vertices that form the Voronoi cell.
    volume : float
        The volume of the Voronoi cell.
    cavity_radius : float
        The distance from the particle to the farthest vertex of its Voronoi
        cell, representing the largest empty-sphere radius.
    face_areas : np.ndarray
        Array containing the area of each face of the Voronoi cell.
    pos : np.ndarray
        The (x, y, z) coordinates of the particle associated with this Voronoi
        cell.


    Notes
    -----
    This class stores only geometric results and contains no computational
    methods. Instances of this class are typically created internally by the
    `Container` class.
    """

    face_vertices: List[List[int]]
    vertices: np.ndarray
    volume: float
    cavity_radius: float
    face_areas: np.ndarray
    pos: np.ndarray


class Container:
    """
    High-level container that stores Voronoi cell geometry for all atoms in
    a system.


    This class wraps the `Voronoi` computation and provides Python‑friendly
    access to each atom's Voronoi cell, represented by a `Cell` object.


    Parameters
    ----------
    data : Union[pl.DataFrame, np.ndarray]
        Atomic coordinates. Accepted formats:

        - A Polars DataFrame with columns 'x', 'y', 'z'.
        - A NumPy array of shape (N, 3).
    box : Box
        The simulation box defining boundaries and coordinate transformation.


    Attributes
    ----------
    _data : List[Cell]
        A list containing a `Cell` instance for each atom in the input system.


    Examples
    --------
    >>> container = Container(pos_array, box)
    >>> cell_0 = container[0]
    >>> print(cell_0.volume)


    The object behaves like a Python list of `Cell` objects.
    """

    def __init__(self, data: Union[pl.DataFrame, np.ndarray], box: Box):
        if isinstance(data, np.ndarray):
            assert data.ndim == 2
            assert data.shape[1] == 3
            data = pl.from_numpy(data, schema=["x", "y", "z"])
        vor = Voronoi(box, data)
        (
            face_vertices_indices,
            face_vertices_positions,
            volume,
            radius,
            face_areas,
        ) = vor.get_cell_info()
        self._data: List[Cell] = []
        for i in range(data.shape[0]):
            self._data.append(
                Cell(
                    face_vertices_indices[i],
                    np.array(face_vertices_positions[i], np.float64),
                    volume[i],
                    radius[i],
                    np.array(face_areas[i], np.float64),
                    np.array([data[i, "x"], data[i, "y"], data[i, "z"]], np.float64),
                )
            )

    def __getitem__(self, index: int):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for i in self._data:
            yield i


if __name__ == "__main__":
    pass
