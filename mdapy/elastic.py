# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

# Part of code is copied from Elastic (https://github.com/jochym/Elastic).

import numpy as np
import polars as pl
import spglib as spg
from scipy.linalg import lstsq

try:
    from .tool_function import atomic_numbers
except Exception:
    from tool_function import atomic_numbers


class Elastic:

    def __init__(
        self, pos, box, elements_list, type_list, potential, symprec=1e-5
    ) -> None:
        self.pos = pos
        self.box = box[:-1]
        self.inv_box = np.linalg.inv(self.box)
        self.scaled_positions = self.pos @ self.inv_box
        self.elements_list = elements_list
        self.type_list = type_list
        self.type_name = [elements_list[i - 1] for i in self.type_list]
        self.potential = potential
        self.symprec = symprec

    def regular(self, u):
        """Equation matrix generation for the regular (cubic) lattice.
        The order of constants is as follows:

        .. math::
        C_{11}, C_{12}, C_{44}

        Args:
            u (np.ndarray): vector of deformations:

            [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

        Returns:
            np.ndarray: Symmetry defined stress-strain equation matrix
        """

        uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
        return np.array(
            [
                [uxx, uyy + uzz, 0],
                [uyy, uxx + uzz, 0],
                [uzz, uxx + uyy, 0],
                [0, 0, 2 * uyz],
                [0, 0, 2 * uxz],
                [0, 0, 2 * uxy],
            ]
        )

    def tetragonal(self, u):
        """
        Equation matrix generation for the tetragonal lattice.
        The order of constants is as follows:

        .. math::
        C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{66}

        Args:
            u (np.ndarray): vector of deformations:

            [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

        Returns:
            np.ndarray: Symmetry defined stress-strain equation matrix
        """

        uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
        return np.array(
            [
                [uxx, 0, uyy, uzz, 0, 0],
                [uyy, 0, uxx, uzz, 0, 0],
                [0, uzz, 0, uxx + uyy, 0, 0],
                [0, 0, 0, 0, 2 * uxz, 0],
                [0, 0, 0, 0, 2 * uyz, 0],
                [0, 0, 0, 0, 0, 2 * uxy],
            ]
        )

    def orthorombic(self, u):
        """
        Equation matrix generation for the orthorombic lattice.
        The order of constants is as follows:

        .. math::
        C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
        C_{44}, C_{55}, C_{66}

        Args:
            u (np.ndarray): vector of deformations:

            [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

        Returns:
            np.ndarray: Symmetry defined stress-strain equation matrix
        """

        uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
        return np.array(
            [
                [uxx, 0, 0, uyy, uzz, 0, 0, 0, 0],
                [0, uyy, 0, uxx, 0, uzz, 0, 0, 0],
                [0, 0, uzz, 0, uxx, uyy, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2 * uyz, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2 * uxz, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 2 * uxy],
            ]
        )

    def trigonal(self, u):
        """
        The matrix is constructed based on the approach from L&L
        using auxiliary coordinates: :math:`\\xi=x+iy`, :math:`\\eta=x-iy`.
        The components are calculated from free energy using formula
        introduced in :ref:`symmetry` with appropriate coordinate changes.
        The order of constants is as follows:

        .. math::
        C_{11}, C_{33}, C_{12}, C_{13}, C_{44}, C_{14}

        Args:
            u (np.ndarray): vector of deformations:

            [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

        Returns:
            np.ndarray: Symmetry defined stress-strain equation matrix
        """

        # TODO: Not tested yet.
        # TODO: There is still some doubt about the :math:`C_{14}` constant.
        uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
        return np.array(
            [
                [uxx, 0, uyy, uzz, 0, 2 * uxz],
                [uyy, 0, uxx, uzz, 0, -2 * uxz],
                [0, uzz, 0, uxx + uyy, 0, 0],
                [0, 0, 0, 0, 2 * uyz, -4 * uxy],
                [0, 0, 0, 0, 2 * uxz, 2 * (uxx - uyy)],
                [2 * uxy, 0, -2 * uxy, 0, 0, -4 * uyz],
            ]
        )

    def hexagonal(self, u):
        """
        The matrix is constructed based on the approach from L&L
        using auxiliary coordinates: :math:`\\xi=x+iy`, :math:`\\eta=x-iy`.
        The components are calculated from free energy using formula
        introduced in :ref:`symmetry` with appropriate coordinate changes.
        The order of constants is as follows:

        .. math::
        C_{11}, C_{33}, C_{12}, C_{13}, C_{44}

        Args:
            u (np.ndarray): vector of deformations:

            [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

        Returns:
            np.ndarray: Symmetry defined stress-strain equation matrix
        """

        # TODO: Still needs good verification
        uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
        return np.array(
            [
                [uxx, 0, uyy, uzz, 0],
                [uyy, 0, uxx, uzz, 0],
                [0, uzz, 0, uxx + uyy, 0],
                [0, 0, 0, 0, 2 * uyz],
                [0, 0, 0, 0, 2 * uxz],
                [uxy, 0, -uxy, 0, 0],
            ]
        )

    def monoclinic(self, u):
        """Monoclinic group,

        The ordering of constants is:

        .. math::
        C_{11}, C_{22}, C_{33}, C_{12}, C_{13}, C_{23},
        C_{44}, C_{55}, C_{66}, C_{16}, C_{26}, C_{36}, C_{45}

        Args:
            u (np.ndarray): vector of deformations:

            [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

        Returns:
            np.ndarray: Symmetry defined stress-strain equation matrix
        """

        uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
        return np.array(
            [
                [uxx, 0, 0, uyy, uzz, 0, 0, 0, 0, uxy, 0, 0, 0],
                [0, uyy, 0, uxx, 0, uzz, 0, 0, 0, 0, uxy, 0, 0],
                [0, 0, uzz, 0, uxx, uyy, 0, 0, 0, 0, 0, uxy, 0],
                [0, 0, 0, 0, 0, 0, 2 * uyz, 0, 0, 0, 0, 0, uxz],
                [0, 0, 0, 0, 0, 0, 0, 2 * uxz, 0, 0, 0, 0, uyz],
                [0, 0, 0, 0, 0, 0, 0, 0, 2 * uxy, uxx, uyy, uzz, 0],
            ]
        )

    def triclinic(self, u):
        """Triclinic crystals.

        *Note*: This was never tested on the real case. Beware!

        The ordering of constants is:

        .. math::
        C_{11}, C_{22}, C_{33},
        C_{12}, C_{13}, C_{23},
        C_{44}, C_{55}, C_{66},
        C_{16}, C_{26}, C_{36}, C_{46}, C_{56},
        C_{14}, C_{15}, C_{25}, C_{45}

        Args:
            u (np.ndarray): vector of deformations:

            [ :math:`u_{xx}, u_{yy}, u_{zz}, u_{yz}, u_{xz}, u_{xy}` ]

        Returns:
            np.ndarray: Symmetry defined stress-strain equation matrix
        """

        # Based on the monoclinic matrix and not tested on real case.
        # If you have test cases for this symmetry send them to the author.
        uxx, uyy, uzz, uyz, uxz, uxy = u[0], u[1], u[2], u[3], u[4], u[5]
        return np.array(
            [
                [uxx, 0, 0, uyy, uzz, 0, 0, 0, 0, uxy, 0, 0, 0, 0, uyz, uxz, 0, 0],
                [0, uyy, 0, uxx, 0, uzz, 0, 0, 0, 0, uxy, 0, 0, 0, 0, 0, uxz, 0],
                [0, 0, uzz, 0, uxx, uyy, 0, 0, 0, 0, 0, uxy, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 2 * uyz, 0, 0, 0, 0, 0, uxy, 0, uxx, 0, 0, uxz],
                [0, 0, 0, 0, 0, 0, 0, 2 * uxz, 0, 0, 0, 0, 0, uxy, 0, uxx, uyy, uyz],
                [0, 0, 0, 0, 0, 0, 0, 0, 2 * uxy, uxx, uyy, uzz, uyz, uxz, 0, 0, 0, 0],
            ]
        )

    def get_cij_order(self):
        """Give order of of elastic constants for the structure

        Returns:
            tuple: Order of elastic constants as a tuple of strings: C_ij
        """

        orders = {
            1: (
                "C_11",
                "C_22",
                "C_33",
                "C_12",
                "C_13",
                "C_23",
                "C_44",
                "C_55",
                "C_66",
                "C_16",
                "C_26",
                "C_36",
                "C_46",
                "C_56",
                "C_14",
                "C_15",
                "C_25",
                "C_45",
            ),
            2: (
                "C_11",
                "C_22",
                "C_33",
                "C_12",
                "C_13",
                "C_23",
                "C_44",
                "C_55",
                "C_66",
                "C_16",
                "C_26",
                "C_36",
                "C_45",
            ),
            3: ("C_11", "C_22", "C_33", "C_12", "C_13", "C_23", "C_44", "C_55", "C_66"),
            4: ("C_11", "C_33", "C_12", "C_13", "C_44", "C_66"),
            5: ("C_11", "C_33", "C_12", "C_13", "C_44", "C_14"),
            6: ("C_11", "C_33", "C_12", "C_13", "C_44"),
            7: ("C_11", "C_12", "C_44"),
        }
        return orders[self.get_lattice_type()[0]]

    def get_lattice_type(self):
        """Find the symmetry of the crystal using spglib symmetry finder.

        Derive name of the space group and its number extracted from the result.
        Based on the group number identify also the lattice type and the Bravais
        lattice of the crystal. The lattice type numbers are
        (the numbering starts from 1):

        Triclinic (1), Monoclinic (2), Orthorombic (3),
        Tetragonal (4), Trigonal (5), Hexagonal (6), Cubic (7)

        Returns:
            tuple: (lattice type number (1-7), lattice name, space group name, space group number)
        """

        # Table of lattice types and correcponding group numbers dividing
        # the ranges. See get_lattice_type method for precise definition.
        lattice_types = [
            [3, "Triclinic"],
            [16, "Monoclinic"],
            [75, "Orthorombic"],
            [143, "Tetragonal"],
            [168, "Trigonal"],
            [195, "Hexagonal"],
            [231, "Cubic"],
        ]

        cell = (
            self.box,
            self.scaled_positions,
            [atomic_numbers[i] for i in self.type_name],
        )
        dataset = spg.get_symmetry_dataset(cell, symprec=self.symprec)
        sg_name = dataset["international"]
        sg_nr = dataset["number"]

        for n, l in enumerate(lattice_types):
            if sg_nr < l[0]:
                bravais = l[1]
                lattype = n + 1
                break

        return lattype, bravais, sg_name, sg_nr

    def get_cart_deformed_cell(self, axis=0, size=1.0):
        """Return the cell deformed along one of the cartesian directions

        Creates new deformed structure. The deformation is based on the
        base structure and is performed along single axis. The axis is
        specified as follows: 0,1,2 = x,y,z ; sheers: 3,4,5 = yz, xz, xy.
        The size of the deformation is in percent and degrees, respectively.

        Args:
            axis (int): direction of deformation
            size (float): percent size of the deformation

        Returns:
            tuple[np.ndarray, np.ndarray]: deformed box and position
        """
        s = size / 100.0
        L = np.diag(np.ones(3))
        if axis < 3:
            L[axis, axis] += s
        else:
            if axis == 3:
                L[1, 2] += s
                # L[2, 1] += s
            elif axis == 4:
                L[0, 2] += s
                # L[2, 0] += s
            else:
                L[0, 1] += s
                # L[1, 0] += s
        new_box = np.r_[np.dot(self.box, L), np.zeros((1, 3))]
        new_pos = np.dot(self.pos, L)
        return new_box, new_pos

    def get_elementary_deformations(self, n=5, d=2):
        """Generate elementary deformations for elastic tensor calculation.

        The deformations are created based on the symmetry of the crystal and
        are limited to the non-equivalet axes of the crystal.

        :param cryst: Atoms object, basic structure
        :param n: integer, number of deformations per non-equivalent axis
        :param d: float, size of the maximum deformation in percent and degrees

        :returns: list of deformed structures
        """
        # Deformation look-up table
        # Perhaps the number of deformations for trigonal
        # system could be reduced to [0,3] but better safe then sorry
        deform = {
            "Cubic": [[0, 3], self.regular],
            "Hexagonal": [[0, 2, 3, 5], self.hexagonal],
            "Trigonal": [[0, 1, 2, 3, 4, 5], self.trigonal],
            "Tetragonal": [[0, 2, 3, 5], self.tetragonal],
            "Orthorombic": [[0, 1, 2, 3, 4, 5], self.orthorombic],
            "Monoclinic": [[0, 1, 2, 3, 4, 5], self.monoclinic],
            "Triclinic": [[0, 1, 2, 3, 4, 5], self.triclinic],
        }

        lattyp, brav, sg_name, sg_nr = self.get_lattice_type()
        # Decide which deformations should be used
        axis, symm = deform[brav]

        systems = {"pos": [], "box": []}
        for a in axis:
            if a < 3:  # tetragonal deformation
                for dx in np.linspace(-d, d, n):
                    new_box, new_pos = self.get_cart_deformed_cell(axis=a, size=dx)
                    systems["pos"].append(new_pos)
                    systems["box"].append(new_box)
            elif a < 6:  # sheer deformation (skip the zero angle)
                for dx in np.linspace(d / 10.0, d, n):
                    new_box, new_pos = self.get_cart_deformed_cell(axis=a, size=dx)
                    systems["pos"].append(new_pos)
                    systems["box"].append(new_box)
        return systems

    def get_strain(self, new_box):
        """Calculate strain tensor in the Voight notation

        Computes the strain tensor in the Voight notation as a conventional
        6-vector. The calculation is done with respect to the crystal
        geometry passed in refcell parameter.

        :param cryst: deformed structure
        :param refcell: reference, undeformed structure

        :returns: 6-vector of strain tensor in the Voight notation
        """

        du = new_box[:-1] - self.box
        u = np.dot(self.inv_box, du)
        u = (u + u.T) / 2
        return np.array([u[0, 0], u[1, 1], u[2, 2], u[2, 1], u[2, 0], u[1, 0]])

    def get_elastic_tensor(self, n=5, d=2):
        """Calculate elastic tensor of the crystal.

        The elastic tensor is calculated from the stress-strain relation
        and derived by fitting this relation to the set of linear equations
        build from the symmetry of the crystal and strains and stresses
        of the set of elementary deformations of the unit cell.

        It is assumed that the crystal is converged and optimized
        under intended pressure/stress. The geometry and stress on the
        cryst is taken as the reference point. No additional optimization
        will be run. Structures in cryst and systems list must have calculated
        stresses. The function returns tuple of :math:`C_{ij}` elastic tensor,
        raw Birch coefficients :math:`B_{ij}` and fitting results: residuals,
        solution rank, singular values returned by numpy.linalg.lstsq.

        :param cryst: Atoms object, basic structure
        :param systems: list of Atoms object with calculated deformed structures

        :returns: tuple(:math:`C_{ij}` float vector,
                        tuple(:math:`B_{ij}` float vector, residuals, solution rank, singular values))
        """

        # Deformation look-up table
        # Perhaps the number of deformations for trigonal
        # system could be reduced to [0,3] but better safe then sorry
        deform = {
            "Cubic": [[0, 3], self.regular],
            "Hexagonal": [[0, 2, 3, 5], self.hexagonal],
            "Trigonal": [[0, 1, 2, 3, 4, 5], self.trigonal],
            "Tetragonal": [[0, 2, 3, 5], self.tetragonal],
            "Orthorombic": [[0, 1, 2, 3, 4, 5], self.orthorombic],
            "Monoclinic": [[0, 1, 2, 3, 4, 5], self.monoclinic],
            "Triclinic": [[0, 1, 2, 3, 4, 5], self.triclinic],
        }

        lattyp, brav, sg_name, sg_nr = self.get_lattice_type()
        # Decide which deformations should be used
        axis, symm = deform[brav]

        ul = []
        sl = []

        _, _, viri = self.potential.compute(
            self.pos,
            np.r_[self.box, np.zeros((1, 3))],
            self.elements_list,
            self.type_list,
            [1, 1, 1],
        )
        p = (viri.sum(axis=0) / np.abs(np.linalg.det(self.box)))[:3].mean()
        systems = self.get_elementary_deformations(n=n, d=d)
        for box, pos in zip(systems["box"], systems["pos"]):
            ul.append(self.get_strain(box))
            # Remove the ambient pressure from the stress tensor
            _, _, viri = self.potential.compute(
                pos,
                box,
                self.elements_list,
                self.type_list,
                [1, 1, 1],
            )
            vv = viri.sum(axis=0)
            # print(
            #     vv[:3].mean() * 160.2176621 / np.linalg.det(box[:-1]) * 1e4,
            #     np.linalg.det(box[:-1]),
            #     np.inner(box[0], np.cross(box[1], box[2])),
            # )
            if len(vv) == 9:
                shear = (vv[3:6] + vv[6:9]) / 2
            else:
                shear = vv[3:6]
            mm = -np.array(
                [vv[0], vv[1], vv[2], shear[2], shear[1], shear[0]]
            ) / np.abs(np.linalg.det(box[:-1]))
            sl.append(mm - np.array([p, p, p, 0, 0, 0]))
        # print(symm, ul)
        eqm = np.array([symm(u) for u in ul])

        eqm = np.reshape(eqm, (eqm.shape[0] * eqm.shape[1], eqm.shape[2]))
        # print(eqm)
        slm = np.reshape(np.array(sl), (-1,))
        # print(eqm.shape, slm.shape)
        # print(slm)
        Bij = lstsq(eqm, slm)
        # print(Bij[0] / units.GPa)
        # Calculate elastic constants from Birch coeff.
        # TODO: Check the sign of the pressure array in the B <=> C relation
        if symm == self.orthorombic:
            Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p])
        elif symm == self.tetragonal:
            Cij = Bij[0] - np.array([-p, -p, p, p, -p, -p])
        elif symm == self.regular:
            Cij = Bij[0] - np.array([-p, p, -p])
        elif symm == self.trigonal:
            Cij = Bij[0] - np.array([-p, -p, p, p, -p, p])
        elif symm == self.hexagonal:
            Cij = Bij[0] - np.array([-p, -p, p, p, -p])
        elif symm == self.monoclinic:
            # TODO: verify this pressure array
            Cij = Bij[0] - np.array([-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p])
        elif symm == self.triclinic:
            # TODO: verify this pressure array
            Cij = Bij[0] - np.array(
                [-p, -p, -p, p, p, p, -p, -p, -p, p, p, p, p, p, p, p, p, p]
            )

        self.Cij = {}
        for i, j in zip(self.get_cij_order(), Cij):
            self.Cij[i] = j * 160.2176621


if __name__ == "__main__":
    import taichi as ti

    ti.init()
    from lattice_maker import LatticeMaker
    from potential import LammpsPotential, EAM, NEP
    from system import System

    # lat = LatticeMaker(1.42, "GRA", 3, 3, 3)
    lat = LatticeMaker(4.05, "FCC", 1, 1, 1)
    lat.compute()
    # lat.box[2, 2] += 20
    # lat = System(r"D:\Study\Gra-Al\potential_test\elastic\aluminum\elastic\C.poscar")
    # type_list = lat.data["type"].to_numpy()
    potential = NEP(r"D:\Study\Gra-Al\potential_test\elastic\aluminum\elastic\nep.txt")
    # potential = NEP(r"D:\Study\Gra-Al\potential_test\phonon\graphene\nep.txt")
    # potential = LammpsPotential(
    #     """
    # pair_style airebo 3.0
    # pair_coeff * * example/CH.airebo C
    # """,
    #     conversion_factor={"energy": 0, "force": 0, "virial": 1 / 1e4 / 160.214},
    # )
    # potential = LammpsPotential(
    #     """pair_style eam/alloy
    #    pair_coeff * * D:\Package\MyPackage\mdapy\example\Al_DFT.eam.alloy Al""",
    #     conversion_factor={"energy": 0, "force": 0, "virial": 1 / 1e4 / 160.214},
    # )
    # potential = EAM(r"D:\Package\MyPackage\mdapy\example\Al_DFT.eam.alloy")
    elas = Elastic(lat.pos, lat.box, ["Al"], lat.type_list, potential)
    elas.get_elastic_tensor()
    print(elas.Cij)
