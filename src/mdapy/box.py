# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

from __future__ import annotations
from typing import Tuple, Union, Optional, Iterable
import numpy as np


class Box:
    """
    The Box class represents a parallelepiped simulation cell that can be either
    orthogonal or triclinic (non-orthogonal). It supports periodic boundary conditions
    and provides methods for coordinate transformations and box manipulations.
    
    Parameters
    ----------
    box : int, float, Iterable[float], np.ndarray, or Box
        Defines the box vectors or size:
        
        * **int/float**: Creates a cubic box with given edge length
        * **Iterable[float] of length 3**: Creates an orthogonal box with given dimensions [Lx, Ly, Lz]
        * **np.ndarray of shape (3,3)**: Full box matrix with vectors as rows
        * **np.ndarray of shape (4,3)**: Box matrix with origin as 4th row
        * **Box**: Creates a deep copy of an existing Box object
        
    boundary : Iterable[int] or np.ndarray, optional
        Boundary condition flags for each dimension (x, y, z):
        
        * 1: Periodic boundary condition
        * 0: Fixed (non-periodic) boundary
        
        Defaults to [1, 1, 1] (all periodic).
        
    origin : Iterable[float] or np.ndarray, optional
        The origin position of the box in 3D space.
        Defaults to [0.0, 0.0, 0.0].
    
    Attributes
    ----------
    box : np.ndarray
        The 3x3 box matrix where each row represents a box vector.
    origin : np.ndarray
        The origin position of the box.
    boundary : np.ndarray
        Boundary condition flags for each dimension.
    triclinic : bool
        True if the box is triclinic (non-orthogonal), False otherwise.
    volume : float
        The volume of the simulation box.
    inverse_box : np.ndarray
        The inverse of the box matrix for coordinate transformations.
    
    Examples
    --------
    Creating different types of boxes:
    
    .. code-block:: python
    
        import numpy as np
        from mdapy.box import Box
        
        # Cubic box with edge length 10
        cubic_box = Box(10)
        
        # Orthogonal box with different dimensions
        ortho_box = Box([10, 20, 30])
        
        # Triclinic box from full matrix
        matrix = np.array([[10, 0, 0],
                          [5, 10, 0],
                          [0, 0, 10]])
        triclinic_box = Box(matrix)
        
        # Box with custom origin and boundary conditions
        custom_box = Box([20, 20, 20], 
                        boundary=[1, 1, 0],  # z-direction non-periodic
                        origin=[5, 5, 5])
        
        # Copy an existing box
        box_copy = Box(cubic_box)
    
    Applying periodic boundary conditions:
    
    .. code-block:: python
    
        # Create a box
        box = Box(10)
        
        # Apply PBC to a displacement vector
        rij = np.array([12, -8, 5])  # Vector that may cross boundaries
        rij_pbc = box.pbc(rij)
        print(f"Original: {rij}")
        print(f"After PBC: {rij_pbc}")
    
    Notes
    -----
    The box matrix convention follows the standard where box vectors are rows:
    
    .. math::
        
        \\mathbf{B} = \\begin{pmatrix}
        a_x & a_y & a_z \\\\
        b_x & b_y & b_z \\\\
        c_x & c_y & c_z
        \\end{pmatrix}
    """

    def __init__(
        self,
        box: Union[int, float, Iterable[float], np.ndarray, Box],
        boundary: Optional[Union[Iterable[int], np.ndarray]] = None,
        origin: Optional[Union[Iterable[float], np.ndarray]] = None,
    ) -> None:
        """Initialize a simulation box."""
        if isinstance(box, Box):
            self.__box = box.box.copy()
            self.__origin = box.origin.copy()
            self.__boundary = box.boundary.copy()
            self.__triclinic = box.triclinic
            self.__inverse_box = box.inverse_box.copy()
            self.__volume = float(box.volume)
        else:
            self.__box, self.__origin = self.__get_box_origin(box, origin)
            self.__triclinic = self.__get_triclinic()
            self.__inverse_box = np.linalg.inv(self.box)
            self.__volume = np.linalg.det(self.box)
            self.set_boundary(boundary)

    def __get_origin(
        self, origin: Optional[Union[Iterable[float], np.ndarray]] = None
    ) -> np.ndarray:
        """
        Parse and validate origin input.

        Parameters
        ----------
        origin : Iterable[float] or np.ndarray, optional
            Origin position.

        Returns
        -------
        np.ndarray
            The origin position.

        Raises
        ------
        ValueError
            If origin has invalid shape.
        TypeError
            If origin has invalid type.
        """
        if origin is None:
            origin = np.array([0.0, 0.0, 0.0], np.float64)
        elif isinstance(origin, (list, tuple, np.ndarray)):
            origin = np.array(origin, np.float64)
            if origin.shape != (3,):
                raise ValueError(
                    f"Origin must be a 3-element array, got shape {origin.shape}"
                )
        else:
            raise TypeError(f"Invalid origin type: {type(origin)}")
        return origin

    def __get_box_origin(
        self,
        box: Union[int, float, Iterable[float], np.ndarray],
        origin: Optional[Union[Iterable[float], np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse and validate box input.

        Parameters
        ----------
        box : int, float, Iterable[float], or np.ndarray
            Box specification in various formats.
        origin : Iterable[float], np.ndarray, optional
            Origin information
        Returns
        -------
        Tuple [np.ndarray, np.ndarray]
            The box matrix and origin information.

        Raises
        ------
        ValueError
            If box or origin have invalid shape.
        TypeError
            If box or origin have invalid type.
        """

        if isinstance(box, (int, float)):
            box = np.eye(3, dtype=np.float64) * box
        elif isinstance(box, (list, tuple, np.ndarray)):
            box = np.array(box, np.float64)
            if box.shape == (3,):
                box = np.diag(box)
            elif box.shape == (3, 3):
                pass
            elif box.shape == (4, 3):  # old mdapy format
                origin = np.array(box[-1])
                box = np.array(box[:-1])
            elif box.shape == (3, 4):  # ovito format
                origin = np.array(box[:, -1])
                box = np.array(box[:, :-1])
            else:
                raise ValueError(f"Invalid box shape: {box.shape}")
        else:
            raise TypeError(f"Invalid box type: {type(box)}")

        origin = self.__get_origin(origin)

        return box, origin

    def __get_box(
        self,
        box: Union[int, float, Iterable[float], np.ndarray],
    ) -> np.ndarray:
        if isinstance(box, (int, float)):
            box = np.eye(3, dtype=np.float64) * box
        elif isinstance(box, (list, tuple, np.ndarray)):
            box = np.array(box, np.float64)
            if box.shape == (3,):
                box = np.diag(box)
            elif box.shape == (3, 3):
                pass
            else:
                raise ValueError(f"Invalid box shape: {box.shape}")
        else:
            raise TypeError(f"Invalid box type: {type(box)}")
        return box

    def set_box(self, box: Union[int, float, Iterable[float], np.ndarray]) -> None:
        """
        Set box matrix.

        Parameters
        ----------
        box : int, float, Iterable[float], or 3x3 np.ndarray
            Box specification in various formats.
        """
        self.__box = self.__get_box(box)
        self.__triclinic = self.__get_triclinic()
        self.__inverse_box = np.linalg.inv(self.box)
        self.__volume = np.linalg.det(self.box)

    def __get_boundary(
        self, boundary: Optional[Union[Iterable[int], np.ndarray]]
    ) -> np.ndarray:
        """
        Parse and validate boundary conditions.

        Parameters
        ----------
        boundary : Iterable[int] or np.ndarray, optional
            Boundary condition flags.

        Returns
        -------
        np.ndarray
            Normalized boundary condition array.
        """
        if boundary is None:
            boundary = np.array([1, 1, 1], np.int32)
        elif isinstance(boundary, (list, tuple, np.ndarray)):
            boundary = np.array(boundary, np.int32)
            if boundary.shape != (3,):
                raise ValueError(
                    f"Boundary must be a 3-element array, got shape {boundary.shape}"
                )
            boundary = np.where(boundary != 0, 1, 0)
        else:
            raise TypeError(f"Invalid boundary type: {type(boundary)}")

        return boundary

    def __get_triclinic(self) -> bool:
        """
        Determine if the box is triclinic.

        Returns
        -------
        bool
            True if triclinic, False if orthogonal.
        """
        for i in range(3):
            for j in range(3):
                if i != j and abs(self.__box[i, j]) > 1e-10:
                    return True
        if np.any(np.diag(self.__box) < 0):
            return True
        return False

    @property
    def volume(self) -> float:
        """
        :no-index:

        Get the volume of the simulation box.

        Returns
        -------
        float
            Box volume in cubic units.
        """
        return self.__volume

    @property
    def box(self) -> np.ndarray:
        """
        :no-index:

        Get the box matrix.

        Returns
        -------
        np.ndarray
            3x3 matrix with box vectors as rows.
        """
        return self.__box

    @property
    def inverse_box(self) -> np.ndarray:
        """
        :no-index:

        Get the inverse box matrix.

        Returns
        -------
        np.ndarray
            3x3 inverse box matrix.
        """
        return self.__inverse_box

    @property
    def origin(self) -> np.ndarray:
        """
        :no-index:

        Get the origin position of the box.

        Returns
        -------
        np.ndarray
            3D origin position.
        """
        return self.__origin

    def set_origin(self, origin: Union[Iterable[float], np.ndarray]) -> None:
        """
        Set origin coordinates.

        Parameters
        ----------
        origin : Iterable[float] or np.ndarray
            Origin position.
        """
        self.__origin = self.__get_origin(origin)

    @property
    def boundary(self) -> np.ndarray:
        """
        :no-index:

        Get the boundary condition flags.

        Returns
        -------
        np.ndarray
            Array of boundary flags (1=periodic, 0=fixed).
        """
        return self.__boundary

    def set_boundary(self, boundary: Union[Iterable[int], np.ndarray]) -> None:
        """
        Set boundary conditions.

        Parameters
        ----------
        boundary : Iterable[int] or np.ndarray
            Boundary condition flags (1=periodic, 0=fixed).
        """
        self.__boundary = self.__get_boundary(boundary)

    @property
    def triclinic(self) -> bool:
        """
        :no-index:

        Check if the box is triclinic.

        Returns
        -------
        bool
            True if triclinic, False if orthogonal.
        """
        return self.__triclinic

    def __repr__(self) -> str:
        """
        String representation of the Box object.
        """
        return f"Box information:\n{self.box}\nOrigin: {self.origin}\nTriclinic: {self.triclinic}\nBoundary: {self.boundary}"

    def align_to_lammps_box(self) -> Tuple[Box, np.ndarray]:
        """
        Transform the box to LAMMPS-compatible format.

        Returns
        -------
        Tuple[Box, np.ndarray]
            - Aligned Box object in LAMMPS format
            - 3x3 rotation matrix used for the transformation
        """
        ax = np.linalg.norm(self.box[0])
        bx = self.box[1] @ (self.box[0] / ax)
        by = np.sqrt(np.linalg.norm(self.box[1]) ** 2 - bx**2)
        cx = self.box[2] @ (self.box[0] / ax)
        cy = (self.box[1] @ self.box[2] - bx * cx) / by
        cz = np.sqrt(np.linalg.norm(self.box[2]) ** 2 - cx**2 - cy**2)
        box = np.array([[ax, bx, cx], [0, by, cy], [0, 0, cz]], dtype=np.float64).T
        rotation = np.linalg.solve(self.box, box)
        return Box(box, self.boundary, self.origin), rotation

    def pbc(self, rij: np.ndarray) -> np.ndarray:
        """
        Apply periodic boundary conditions to a displacement vector.

        Parameters
        ----------
        rij : np.ndarray
            3D displacement vector in Cartesian coordinates.

        Returns
        -------
        np.ndarray
            Wrapped displacement vector following minimum image convention.
        """
        rij = rij @ self.inverse_box
        for i in range(3):
            if self.boundary[i] == 1:
                rij[i] -= np.floor(rij[i] + 0.5)
        return rij @ self.box

    def get_thickness(self) -> np.ndarray:
        """
        Calculate the perpendicular thickness along each box dimension.

        Returns
        -------
        np.ndarray
            Array of thicknesses [tx, ty, tz] for each dimension.
        """
        return np.array(
            [
                self.volume / np.linalg.norm(np.cross(self.box[1], self.box[2])),
                self.volume / np.linalg.norm(np.cross(self.box[0], self.box[2])),
                self.volume / np.linalg.norm(np.cross(self.box[0], self.box[1])),
            ],
            dtype=np.float64,
        )

    def check_small_box(self, rc: float) -> np.ndarray:
        """
        Check if the box satisfies the minimum image convention for a given cutoff.

        Parameters
        ----------
        rc : float
            Cutoff radius for interactions.

        Returns
        -------
        np.ndarray
            Integer array [nx, ny, nz] indicating required replications.
        """
        thickness = self.get_thickness()
        repeat = np.ones(3, dtype=np.int32)
        for i in range(3):
            if self.boundary[i] == 1 and thickness[i] < 2 * rc:
                repeat[i] = int(np.ceil(2.0 * rc / thickness[i]))
        return repeat


if __name__ == "__main__":
    box = Box(2)
    print(box)
    a = Box(box)
    print(a)
