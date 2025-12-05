# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from typing import Tuple, Optional, Union, Iterable
import numpy as np
import polars as pl
from mdapy.voronoi import Container, Cell
from mdapy.system import System
from mdapy.build_lattice import build_crystal
from mdapy.box import Box
from mdapy import _polycrystal, _neighbor
from mdapy.tool_function import _replicate_pos
from time import time


class CreatePolycrystal:
    """
    Create polycrystalline structures using Voronoi tessellation method.

    This class generates polycrystalline atomic structures by:
    1. Creating Voronoi cells from seed positions
    2. Filling each cell with rotated unit cell atoms
    3. Optionally decorating grain boundaries with graphene layers
    4. Removing overlapping atoms based on distance criteria

    Parameters
    ----------
    unitcell : System
        The unit cell structure to be replicated in each grain.
    box : Union[int, float, Iterable[float], np.ndarray, Box]
        Simulation box dimensions. Can be a single value (cubic),
        array of 3 values, or Box object.
    seed_number : int
        Number of grains (Voronoi seeds) to generate.
    seed_position : Optional[np.ndarray], default=None
        Positions of Voronoi seeds with shape (seed_number, 3).
        If None, randomly generated within the box.
    theta_list : Optional[np.ndarray], default=None
        Rotation angles (in degrees) for each grain with shape (seed_number, 3).
        Represents rotations around x, y, z axes. If None, randomly generated.
    randomseed : Optional[int], default=None
        Random seed for reproducibility. If None, randomly chosen.
    metal_overlap_dis : Optional[float], default=None
        Minimum allowed distance between metal atoms (Angstrom).
        Atoms closer than this will be removed.
    add_graphene : bool, default=False
        Whether to add graphene layers at grain boundaries.
    metal_gra_overlap_dis : float, default=3.0
        Minimum allowed distance between metal and carbon atoms (Angstrom).
    face_threshold : float, default=0.0
        Minimum face area (Angstrom²) for graphene decoration.
        Faces smaller than this will be skipped.
    need_rotation : bool, default=True
        Whether to apply random rotations to grains.

    Attributes
    ----------
    unitcell : System
        The unit cell structure.
    box : Box
        The simulation box.
    seed_number : int
        Number of grains.
    seed_position : np.ndarray
        Voronoi seed positions.
    theta_list : np.ndarray
        Rotation angles for each grain.
    con : Container
        Voronoi container with all cells.
    randomseed : int
        Random seed used.

    Examples
    --------
    >>> from mdapy.build_lattice import build_crystal
    >>> unit = build_crystal("Al", "fcc", 4.05)
    >>> poly = CreatePolycrystal(unit, box=100, seed_number=10, metal_overlap_dis=2.0)
    >>> system = poly.compute()
    >>> system.write_xyz("polycrystal.xyz")

    Notes
    -----
    - Triclinic boxes are not supported
    - Free boundary conditions are not supported
    """

    def __init__(
        self,
        unitcell: System,
        box: Union[int, float, Iterable[float], np.ndarray, Box],
        seed_number: int,
        seed_position: Optional[np.ndarray] = None,
        theta_list: Optional[np.ndarray] = None,
        randomseed: Optional[int] = None,
        metal_overlap_dis: Optional[float] = None,
        add_graphene: bool = False,
        metal_gra_overlap_dis: float = 3.0,
        face_threshold: float = 0.0,
        need_rotation: bool = True,
    ) -> None:
        """Initialize CreatePolycrystal with given parameters."""
        self.unitcell = unitcell
        self.box = Box(box)
        if sum(self.box.boundary) != 3:
            raise ValueError("Free boundary condition is not supported.")
        if self.box.triclinic:
            raise ValueError("Triclinic box is not supported")

        self.seed_number = seed_number
        self.metal_overlap_dis = metal_overlap_dis
        self.add_graphene = add_graphene
        self.metal_gra_overlap_dis = metal_gra_overlap_dis
        self.need_rotation = need_rotation
        self.face_threshold = face_threshold

        # Initialize random number generator
        if randomseed is None:
            randomseed = np.random.randint(0, 1_000_000_000)
        self.randomseed = int(randomseed)
        self.rng = np.random.default_rng(self.randomseed)

        # Initialize seed positions
        if seed_position is None:
            box_lengths = np.diag(self.box.box)
            self.seed_position = self.rng.random((self.seed_number, 3)) * box_lengths
        else:
            if seed_position.shape != (self.seed_number, 3):
                raise ValueError(
                    f"seed_position shape must be ({self.seed_number}, 3), "
                    f"got {seed_position.shape}"
                )
            self.seed_position = seed_position

        # Initialize rotation angles
        if theta_list is None:
            self.theta_list = self.rng.uniform(-180, 180, (self.seed_number, 3))
        else:
            if theta_list.shape != (self.seed_number, 3):
                raise ValueError(
                    f"theta_list shape must be ({self.seed_number}, 3), "
                    f"got {theta_list.shape}"
                )
            self.theta_list = theta_list

        # Voronoi container (will be created in compute())
        self.con: Optional[Container] = None

    @staticmethod
    def _get_rotation_matrix(
        theta_deg: float, axis_tuple: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Calculate rotation matrix using axis-angle representation.

        Uses Rodrigues' rotation formula to compute the rotation matrix
        that rotates vectors by theta_deg degrees around the given axis.

        Parameters
        ----------
        theta_deg : float
            Rotation angle in degrees.
        axis_tuple : Tuple[float, float, float]
            Rotation axis as (x, y, z) tuple.

        Returns
        -------
        np.ndarray
            3x3 rotation matrix.

        Raises
        ------
        ValueError
            If the rotation axis has zero length.

        Notes
        -----
        The rotation matrix R is computed as:
        R = I + sin(θ)K + (1-cos(θ))K²
        where K is the skew-symmetric cross-product matrix of the normalized axis.
        """
        theta = np.radians(theta_deg)
        axis = np.array(axis_tuple, dtype=float)
        norm = np.linalg.norm(axis)

        if norm == 0:
            raise ValueError("Rotation axis must be non-zero")

        x, y, z = axis / norm
        c = np.cos(theta)
        s = np.sin(theta)
        C = 1 - c

        return np.array(
            [
                [c + C * x * x, C * x * y - s * z, C * x * z + s * y],
                [C * y * x + s * z, c + C * y * y, C * y * z - s * x],
                [C * z * x - s * y, C * z * y + s * x, c + C * z * z],
            ],
            dtype=float,
        )

    def _get_plane_equation_coeffs_for_cell(self, cell: Cell) -> np.ndarray:
        """
        Calculate plane equation coefficients for all faces of a Voronoi cell.

        For each face, computes the plane equation: ax + by + cz + d = 0
        where the normal vector (a, b, c) points inward to the cell.

        Parameters
        ----------
        cell : Cell
            Voronoi cell object containing vertices and face information.

        Returns
        -------
        np.ndarray
            Array of shape (n_faces, 4) where each row is [a, b, c, d].

        Raises
        ------
        ValueError
            If face vertices are degenerate (collinear).

        Notes
        -----
        The normal vector is computed using cross product of two edge vectors,
        then normalized. The direction is adjusted to point inward.
        """
        n_faces = len(cell.face_vertices)
        face_index = np.array([face[:3] for face in cell.face_vertices])
        plane_pos = np.array([cell.vertices[face_index[i]] for i in range(n_faces)])

        coeffs = np.zeros((n_faces, 4))

        for i in range(n_faces):
            p1, p2, p3 = plane_pos[i]

            # Calculate normal vector using cross product
            v1, v2 = p2 - p1, p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)

            if norm_n < 1e-10:
                raise ValueError(f"Degenerate face vertices at face {i}")

            n = n / norm_n
            d = -np.dot(n, p1)

            # Ensure normal points inward to the cell
            if np.dot(n, cell.pos) + d > 0:
                n, d = -n, -d

            coeffs[i, :3] = n
            coeffs[i, 3] = d

        return coeffs

    def _generate_grain_atoms(
        self, grain_idx: int, cell: Cell, data: pl.DataFrame, coeffs: np.ndarray
    ) -> np.ndarray:
        """
        Generate atoms for a single grain by rotating and filtering unit cell.

        Process:
        1. Apply rotation matrices (Rx * Ry * Rz) to unit cell
        2. Translate to grain center
        3. Filter atoms inside Voronoi cell using plane equations

        Parameters
        ----------
        grain_idx : int
            Index of the grain (0 to seed_number-1).
        cell : Cell
            Voronoi cell for this grain.
        data : pl.DataFrame
            Replicated unit cell data with columns ['x', 'y', 'z'].
        coeffs : np.ndarray
            Plane equation coefficients for cell faces.

        Returns
        -------
        np.ndarray
            Filtered atomic positions with shape (n_atoms, 3).

        """
        if self.need_rotation:
            R_x = self._get_rotation_matrix(
                self.theta_list[grain_idx, 0], (1.0, 0.0, 0.0)
            )
            R_y = self._get_rotation_matrix(
                self.theta_list[grain_idx, 1], (0.0, 1.0, 0.0)
            )
            R_z = self._get_rotation_matrix(
                self.theta_list[grain_idx, 2], (0.0, 0.0, 1.0)
            )
            rotate_matrix = R_x @ R_y @ R_z
        else:
            rotate_matrix = self._get_rotation_matrix(0, (1.0, 0.0, 0.0))

        pos_center = data.select("x", "y", "z").mean().to_numpy().ravel()

        filtered_pos = _polycrystal.transform_and_filter(
            data["x"].to_numpy(allow_copy=False),
            data["y"].to_numpy(allow_copy=False),
            data["z"].to_numpy(allow_copy=False),
            rotate_matrix,
            pos_center,
            cell.pos,
            coeffs,
        )

        return filtered_pos

    def _generate_gra_atoms(
        self, cell: Cell, gra_data: pl.DataFrame, coeffs: np.ndarray
    ) -> np.ndarray:
        """
        Generate graphene atoms on grain boundary faces.

        Strategy:
        1. For each face larger than face_threshold
        2. Rotate graphene sheet to align with face normal
        3. Translate to face center
        4. Filter atoms inside face polygon

        Parameters
        ----------
        cell : Cell
            Voronoi cell containing face information.
        gra_data : pl.DataFrame
            Graphene sheet data with columns ['x', 'y', 'z'].
        coeffs : np.ndarray
            Plane equation coefficients for cell faces.

        Returns
        -------
        np.ndarray
            Graphene atomic positions with shape (n_atoms, 3).

        Raises
        ------
        AssertionError
            If no graphene atoms are generated for any face.

        Notes
        -----
        - Initial graphene normal is [0, 0, 1] (xy-plane)
        - Uses 2D projection and ray-casting for polygon filtering
        - Z-tolerance of 0.5 Å is used for face proximity
        """
        gra_normal = np.array([0.0, 0.0, 1.0])  # Initial graphene normal (xy-plane)
        gra_positions_list = []

        gra_pos = gra_data.select("x", "y", "z").to_numpy()

        for face_idx in range(coeffs.shape[0]):
            # Skip faces smaller than threshold
            if cell.face_areas[face_idx] <= self.face_threshold:
                continue

            face_vertices = cell.vertices[cell.face_vertices[face_idx]]
            face_normal = coeffs[face_idx, :3]
            face_normal = face_normal / np.linalg.norm(face_normal)
            face_center = np.mean(face_vertices, axis=0)

            # Step 1: Compute rotation matrix to align graphene with face
            rotation_matrix = self._compute_rotation_to_align_normals(
                gra_normal, face_normal
            )

            # Step 2: Rotate graphene and translate to face center
            rotated_gra_pos = gra_pos @ rotation_matrix.T
            gra_centered = rotated_gra_pos - np.mean(rotated_gra_pos, axis=0)
            gra_on_face = gra_centered + face_center

            # Step 3: Filter atoms inside face polygon
            atoms_in_face = self._filter_atoms_in_polygon(
                gra_on_face, face_vertices, face_normal
            )

            if len(atoms_in_face) > 0:
                gra_positions_list.append(atoms_in_face)

        assert len(gra_positions_list) > 0, "No graphene atoms generated"

        return np.vstack(gra_positions_list)

    def _compute_rotation_to_align_normals(
        self, source_normal: np.ndarray, target_normal: np.ndarray
    ) -> np.ndarray:
        """
        Compute rotation matrix to align source normal with target normal.

        Uses Rodrigues' rotation formula to find the rotation that maps
        source_normal to target_normal.

        Parameters
        ----------
        source_normal : np.ndarray
            Initial normal vector (will be normalized).
        target_normal : np.ndarray
            Target normal vector (will be normalized).

        Returns
        -------
        np.ndarray
            3x3 rotation matrix.

        Notes
        -----
        Special cases:
        - If normals are already aligned: returns identity matrix
        - If normals are opposite: rotates 180° around a perpendicular axis
        - General case: rotates around cross product of normals
        """
        # Normalize
        v1 = source_normal / np.linalg.norm(source_normal)
        v2 = target_normal / np.linalg.norm(target_normal)

        # Check if already aligned
        dot_product = np.dot(v1, v2)

        if np.isclose(dot_product, 1.0, atol=1e-6):
            # Already aligned, return identity
            return np.eye(3)

        if np.isclose(dot_product, -1.0, atol=1e-6):
            # Opposite directions, rotate 180° around perpendicular axis
            if abs(v1[0]) < 0.9:
                perp_axis = np.array([1.0, 0.0, 0.0])
            else:
                perp_axis = np.array([0.0, 1.0, 0.0])
            rotation_axis = np.cross(v1, perp_axis)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            return self._get_rotation_matrix(180.0, tuple(rotation_axis))

        # General case: compute rotation axis and angle
        rotation_axis = np.cross(v1, v2)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Rotation angle
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        return self._get_rotation_matrix(angle_deg, tuple(rotation_axis))

    def _filter_atoms_in_polygon(
        self, points: np.ndarray, polygon_vertices: np.ndarray, face_normal: np.ndarray
    ) -> np.ndarray:
        """
        Filter atoms that lie inside a 3D polygon face.

        Method:
        1. Build local coordinate system with face normal as z-axis
        2. Project polygon and points to 2D plane
        3. Use ray-casting algorithm to test if points are inside polygon
        4. Check z-coordinate tolerance to ensure points are near face

        Parameters
        ----------
        points : np.ndarray
            Points to test with shape (n_points, 3).
        polygon_vertices : np.ndarray
            Polygon vertices with shape (n_vertices, 3).
        face_normal : np.ndarray
            Normal vector of the face.

        Returns
        -------
        np.ndarray
            Filtered points inside polygon with shape (n_filtered, 3).

        Notes
        -----
        - Z-tolerance is set to 0.5 Angstrom
        - Uses vectorized ray-casting for efficiency
        """
        # Build local coordinate system
        # z-axis: face normal
        local_z = face_normal / np.linalg.norm(face_normal)

        # x-axis: from face center to first vertex (projected on face)
        face_center = np.mean(polygon_vertices, axis=0)
        temp_x = polygon_vertices[0] - face_center
        temp_x = temp_x - np.dot(temp_x, local_z) * local_z  # Project onto face

        if np.linalg.norm(temp_x) < 1e-8:
            # First vertex is at center, try another
            temp_x = polygon_vertices[1] - face_center
            temp_x = temp_x - np.dot(temp_x, local_z) * local_z

        local_x = temp_x / np.linalg.norm(temp_x)

        # y-axis: right-hand rule
        local_y = np.cross(local_z, local_x)

        # Transform matrix (world -> local)
        transform_matrix = np.array([local_x, local_y, local_z])

        # Project to 2D
        vertices_local = (polygon_vertices - face_center) @ transform_matrix.T
        polygon_2d = vertices_local[:, :2]  # Take only x, y

        points_local = (points - face_center) @ transform_matrix.T
        points_2d = points_local[:, :2]

        # Check z-coordinate (points must be close to face)
        z_tolerance = 0.5  # Angstrom
        close_to_face = np.abs(points_local[:, 2]) < z_tolerance

        # Ray-casting algorithm
        in_polygon = self._points_in_polygon_2d(polygon_2d, points_2d)

        # Both conditions must be satisfied
        valid_mask = close_to_face & in_polygon

        return points[valid_mask]

    def _points_in_polygon_2d(
        self, polygon: np.ndarray, points: np.ndarray
    ) -> np.ndarray:
        """
        Test if 2D points are inside a 2D polygon using ray-casting.

        Casts a ray from each point to the right and counts intersections
        with polygon edges. Odd number of intersections means inside.

        Parameters
        ----------
        polygon : np.ndarray
            Polygon vertices with shape (n_vertices, 2).
        points : np.ndarray
            Points to test with shape (n_points, 2).

        Returns
        -------
        np.ndarray
            Boolean array indicating if each point is inside (n_points,).

        Notes
        -----
        Vectorized implementation for efficiency. Handles edge cases:
        - Points on vertices
        - Points on edges
        - Ray passing through vertices
        """
        polygon = np.asarray(polygon, dtype=np.float32)
        points = np.asarray(points, dtype=np.float32)

        # Build polygon edges (current vertex -> next vertex)
        p1 = polygon
        p2 = np.roll(polygon, -1, axis=0)

        # Vectorized computation
        # Expand dimensions: points [N, 2] -> [N, 1, 2], polygon [V, 2] -> [1, V, 2]
        pts = points[:, np.newaxis, :]  # [N, 1, 2]
        v1 = p1[np.newaxis, :, :]  # [1, V, 2]
        v2 = p2[np.newaxis, :, :]  # [1, V, 2]

        # Check if points are on vertices
        on_vertex = np.any(np.all(np.isclose(pts, v1, atol=1e-6), axis=2), axis=1)

        # Ray-casting: cast ray to the right, count intersections
        # Condition 1: edge crosses point's y-coordinate
        y_cross = (v1[:, :, 1] > pts[:, :, 1]) != (v2[:, :, 1] > pts[:, :, 1])

        # Condition 2: intersection is to the right of point
        x_intersect = (v2[:, :, 0] - v1[:, :, 0]) * (pts[:, :, 1] - v1[:, :, 1]) / (
            v2[:, :, 1] - v1[:, :, 1] + 1e-10
        ) + v1[:, :, 0]
        right_side = pts[:, :, 0] < x_intersect

        # Count intersections
        intersections = np.sum(y_cross & right_side, axis=1)

        # Odd number of intersections -> inside polygon
        inside = (intersections % 2 == 1) | on_vertex

        return inside

    def _get_pos(
        self, verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate atomic positions for all grains.

        Process:
        1. Determine required unit cell replication
        2. Generate metal atoms for each grain
        3. Optionally generate graphene atoms for grain boundaries
        4. Assign grain IDs and atom types

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            - all_pos: atomic positions (n_atoms, 3)
            - all_grain_ids: grain ID for each atom (n_atoms,)
            - all_types: atom type (1=metal, 2=carbon) (n_atoms,)

        Notes
        -----
        - Unit cell is replicated based on maximum Voronoi cell radius
        - Graphene lattice constant is 1.42 Angstrom
        - Type 1 = metal, Type 2 = carbon
        """
        # Calculate required unit cell replication
        r_max = max(cell.cavity_radius for cell in self.con)
        thickness = self.unitcell.box.get_thickness()
        replicate_nums = np.ceil(r_max / thickness).astype(int)

        # Replicate unit cell
        data, _ = _replicate_pos(self.unitcell.data, self.unitcell.box, *replicate_nums)

        # Prepare graphene if needed
        if self.add_graphene:
            gra_lattice = 1.42  # Angstrom
            x1 = int(np.ceil(r_max / (gra_lattice * 3)))
            y1 = int(np.ceil(r_max / (gra_lattice * 3**0.5)))
            gra = build_crystal("C", "graphene", gra_lattice, nx=x1, ny=y1, nz=1)

        # Preallocate lists
        pos_list = []
        grain_id_list = []
        type_list = []

        # Generate atoms for each grain
        for n, cell in enumerate(self.con):
            if verbose:
                print(
                    f"  Grain {n + 1:>3}/{len(self.con)}: "
                    f"Volume = {cell.volume:>10.2f} A^3  ",
                    end="",
                )

            coeffs = self._get_plane_equation_coeffs_for_cell(cell)

            # Generate metal atoms
            pos = self._generate_grain_atoms(n, cell, data, coeffs)
            pos_list.append(pos)
            N_metal = pos.shape[0]
            type_list.extend([1] * N_metal)

            # Generate graphene atoms if enabled
            if self.add_graphene:
                gra_pos = self._generate_gra_atoms(cell, gra.data, coeffs)
                pos_list.append(gra_pos)
                N_carbon = gra_pos.shape[0]
                type_list.extend([2] * N_carbon)
                N_total = N_metal + N_carbon
                if verbose:
                    print(f"Metal = {N_metal:>6}, Carbon = {N_carbon:>6}")
            else:
                N_total = N_metal
                if verbose:
                    print(f"Atoms = {N_metal:>6}")

            # Assign grain IDs
            grain_ids = np.full(N_total, n + 1, dtype=np.int32)
            grain_id_list.append(grain_ids)

        # Concatenate all atoms
        all_pos = np.vstack(pos_list)
        all_grain_ids = np.concatenate(grain_id_list)
        all_types = np.array(type_list, dtype=np.int32)

        return all_pos, all_grain_ids, all_types

    def compute(self, verbose: bool = True) -> System:
        """
        Compute and generate the polycrystalline structure.

        Main workflow:

        1. Generate Voronoi tessellation from seeds
        2. Fill each cell with rotated unit cell atoms
        3. Optionally add graphene at grain boundaries
        4. Remove overlapping atoms
        5. Wrap atoms into simulation box

        Parameters
        ----------
        verbose : bool, default=True
            If True, print detailed progress information.
            If False, suppress all output.

        Returns
        -------
        System
            The generated polycrystalline system with attributes:

            - data: polars DataFrame with columns ['element', 'x', 'y', 'z', 'grain_id', 'type']
            - box: simulation box

        Examples
        --------
        >>> poly = CreatePolycrystal(unitcell, box=100, seed_number=10)
        >>> system = poly.compute(verbose=True)
        >>> system.write_xyz("output.xyz")

        """
        if verbose:
            start = time()
            print("=" * 70)
            print(" " * 20 + "POLYCRYSTAL GENERATION")
            print("=" * 70)

        # Generate Voronoi tessellation
        if verbose:
            print("[1/5] Generating Voronoi tessellation...")
        origin = self.box.origin.copy()
        self.con = Container(self.seed_position, Box(self.box.box))

        # Statistical information
        volumes = np.array([cell.volume for cell in self.con])
        ave_grain_volume = np.mean(volumes)

        if verbose:
            print(f"  Number of grains: {self.seed_number}")
            print(f"  Average volume:   {ave_grain_volume:>10.2f} A^3")
            print(
                f"  Volume range:     {volumes.min():>10.2f} - {volumes.max():<10.2f} A^3"
            )
            if self.add_graphene:
                print(
                    f"  Graphene enabled: Yes (threshold = {self.face_threshold} A^2)"
                )
            print(f"  Random seed:      {self.randomseed}")

        # Generate atomic positions
        if verbose:
            print(f"[2/5] Generating atoms for {self.seed_number} grains...")
        pos, grain_id, type_list = self._get_pos(verbose)

        if verbose:
            print(f"  Total atoms generated: {len(pos):,}")

        # Create dataframe
        if verbose:
            print("[3/5] Creating atomic structure...")
        data = pl.DataFrame(
            {
                "x": pos[:, 0] + origin[0],
                "y": pos[:, 1] + origin[1],
                "z": pos[:, 2] + origin[2],
                "grain_id": grain_id,
                "type": type_list,
            }
        )

        # Remove overlapping atoms
        if verbose:
            print("[4/5] Removing overlapping atoms...")

        if self.add_graphene:
            if verbose:
                print("  Filtering: Metal-Metal, C-C, Metal-C overlaps")
                print(
                    f"    Metal-Metal distance: {self.metal_overlap_dis or 2.0:.2f} Å"
                )
                print("    C-C distance:         1.40 Å")
                print(f"    Metal-C distance:     {self.metal_gra_overlap_dis:.2f} Å")

            metal_metal_distance = (
                self.metal_overlap_dis if self.metal_overlap_dis is not None else 2.0
            )
            cc_distance = 1.4  # C-C minimum distance (Angstrom)
            metal_c_distance = self.metal_gra_overlap_dis

            filter_mask = _neighbor.filter_overlap_atom_with_grain(
                data["x"].to_numpy(allow_copy=False),
                data["y"].to_numpy(allow_copy=False),
                data["z"].to_numpy(allow_copy=False),
                data["type"].to_numpy(allow_copy=False),
                data["grain_id"].to_numpy(allow_copy=False),
                self.box.box,
                self.box.origin,
                self.box.boundary,
                metal_metal_distance,
                cc_distance,
                metal_c_distance,
            )
            data = data.filter(filter_mask)
        elif self.metal_overlap_dis is not None:
            if verbose:
                print("  Filtering: Metal-Metal overlaps")
                print(f"    Metal-Metal distance: {self.metal_overlap_dis:.2f} Å")

            filter_mask = _neighbor.filter_overlap_atom(
                data["x"].to_numpy(allow_copy=False),
                data["y"].to_numpy(allow_copy=False),
                data["z"].to_numpy(allow_copy=False),
                self.box.box,
                self.box.origin,
                self.box.boundary,
                self.metal_overlap_dis,
            )
            data = data.filter(filter_mask)

        if verbose:
            removed = len(pos) - len(data)
            print(f"  Atoms removed: {removed:,} ({removed / len(pos) * 100:.2f}%)")
            print(f"  Atoms remaining: {len(data):,}")

        # Add element information
        if "element" in self.unitcell.data.columns:
            element = self.unitcell.data["element"][0]
            type2ele = {1: element, 2: "C"}
            data = data.with_columns(
                pl.col("type").replace_strict(type2ele).alias("element")
            ).select("element", "x", "y", "z", "grain_id", "type")

        # Create system and wrap atoms
        if verbose:
            print("[5/5] Finalizing structure...")
        system = System(data=data, box=self.box)
        system.wrap_pos()

        # Final statistics
        if verbose:
            print("=" * 70)
            print(" ✓ Polycrystal generation completed successfully!")
            end = time()
            print(f" ✓ Execution time: {end - start:.2f} seconds")
            print("=" * 70 + "")

        return system


if __name__ == "__main__":
    pass
