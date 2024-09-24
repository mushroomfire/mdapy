# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

from time import time
import numpy as np
import taichi as ti
import polars as pl
import multiprocessing as mt


if __name__ == "__main__":
    from voronoi import _voronoi_analysis
    from lattice_maker import LatticeMaker
    from neighbor import Neighbor
    from load_save_data import SaveFile
    from box import init_box
    from system import System
else:
    import _voronoi_analysis
    from .lattice_maker import LatticeMaker
    from .neighbor import Neighbor
    from .load_save_data import SaveFile
    from .box import init_box
    from .system import System

try:
    from _neigh import (
        build_cell_rec,
    )
except Exception:
    from neigh._neigh import (
        build_cell_rec,
    )


class Cell:
    def __init__(
        self, face_vertices, vertices, volume, cavity_radius, face_areas, pos
    ) -> None:
        self._face_vertices = face_vertices
        self._vertices = vertices
        self._volume = volume
        self._cavity_radius = cavity_radius
        self._face_areas = face_areas
        self.pos = pos

    def face_vertices(self):
        return self._face_vertices

    def face_areas(self):
        return self._face_areas

    def vertices(self):
        return self._vertices

    def volume(self):
        return self._volume

    def cavity_radius(self):
        return self._cavity_radius


class Container(list):
    def __init__(self, pos, box, boundary, num_t) -> None:
        if pos.dtype != np.float64:
            pos = pos.astype(np.float64)
        self.pos = pos
        if box.dtype != np.float64:
            box = box.astype(np.float64)
        if box.shape == (4, 3):
            for i in range(3):
                for j in range(3):
                    if i != j:
                        assert box[i, j] == 0, "Do not support triclinic box."
            self.box = np.zeros((3, 2))
            self.box[:, 0] = box[-1]
            self.box[:, 1] = (
                np.array([box[0, 0], box[1, 1], box[2, 2]]) + self.box[:, 0]
            )
        elif box.shape == (3, 2):
            self.box = box

        self.boundary = np.bool_(boundary)
        self.num_t = num_t
        (
            face_vertices,
            vertices,
            volume,
            cavity_radius,
            face_areas,
        ) = _voronoi_analysis.get_cell_info(
            self.pos, self.box, self.boundary, self.num_t
        )
        for i in range(self.pos.shape[0]):
            self.append(
                Cell(
                    face_vertices[i],
                    vertices[i],
                    volume[i],
                    cavity_radius[i],
                    face_areas[i],
                    self.pos[i],
                )
            )


@ti.data_oriented
class DeleteOverlap:
    def __init__(self, pos, box, rc) -> None:
        self.pos = pos
        self.box, _, _ = init_box(box)
        self.rc = rc
        self.bin_length = rc + 0.1
        self.ncel = ti.Vector(
            [
                max(int(np.floor(np.linalg.norm(box[i]) / self.bin_length)), 3)
                for i in range(3)
            ]
        )
        self.box_length = ti.Vector([np.linalg.norm(box[i]) for i in range(3)])

    @ti.func
    def _pbc_rec(self, rij):
        for m in ti.static(range(3)):
            dx = rij[m]
            x_size = self.box_length[m]
            h_x_size = x_size * 0.5
            if dx > h_x_size:
                dx = dx - x_size
            if dx <= -h_x_size:
                dx = dx + x_size
            rij[m] = dx
        return rij

    @ti.kernel
    def _build_delete_id(
        self,
        pos: ti.types.ndarray(element_dim=1),
        atom_cell_list: ti.types.ndarray(),
        cell_id_list: ti.types.ndarray(),
        delete_id: ti.types.ndarray(),
        box: ti.types.ndarray(element_dim=1),
    ):
        rcsq = self.rc * self.rc

        for i in range(pos.shape[0]):
            icel, jcel, kcel = ti.floor((pos[i] - box[3]) / self.bin_length, dtype=int)
            if icel < 0:
                icel = 0
            elif icel > self.ncel[0] - 1:
                icel = self.ncel[0] - 1
            if jcel < 0:
                jcel = 0
            elif jcel > self.ncel[1] - 1:
                jcel = self.ncel[1] - 1
            if kcel < 0:
                kcel = 0
            elif kcel > self.ncel[2] - 1:
                kcel = self.ncel[2] - 1
            for iicel in range(icel - 1, icel + 2):
                for jjcel in range(jcel - 1, jcel + 2):
                    for kkcel in range(kcel - 1, kcel + 2):
                        j = cell_id_list[
                            iicel % self.ncel[0],
                            jjcel % self.ncel[1],
                            kkcel % self.ncel[2],
                        ]
                        while j > -1:
                            rij = self._pbc_rec(pos[j] - pos[i])
                            rijdis_sq = rij[0] ** 2 + rij[1] ** 2 + rij[2] ** 2
                            if rijdis_sq <= rcsq and j > i:
                                delete_id[j] = 0
                            j = atom_cell_list[j]

    def compute(self):
        atom_cell_list = np.zeros(self.pos.shape[0], dtype=np.int32)
        cell_id_list = np.full(
            (self.ncel[0], self.ncel[1], self.ncel[2]), -1, dtype=np.int32
        )
        build_cell_rec(
            self.pos,
            atom_cell_list,
            cell_id_list,
            np.ascontiguousarray(self.box[-1]),
            np.array([i for i in self.ncel]),
            self.bin_length,
        )
        self.delete_id = np.ones(self.pos.shape[0], int)
        self._build_delete_id(
            self.pos,
            atom_cell_list,
            cell_id_list,
            self.delete_id,
            self.box,
        )


@ti.data_oriented
class CreatePolycrystalline:
    """This class is used to create polycrystalline structure with random grain orientation
    based on the Voronoi diagram. It provides an interesting feature to replace the metallic
    grain boundary into graphene, which can generate a three-dimensional graphene structure.
    The metallic matrix can be FCC, BCC and HCP structure.

    Args:
        box (np.ndarray): (:math:`3, 2`), system box.
        seednumber (int): number of initial seed to generate the Voronoi polygon.
        metal_latttice_constant (float): lattice constant.
        metal_lattice_type (str): metallic matrix structure type, supporting FCC, BCC and HCP.
        randomseed (int, optional): randomseed to generate the random number, this is helpful to reproduce the same structure. If not given, a random seed will be generated.
        metal_overlap_dis (float, optional): minimum distance between metallic particles. If not given, this parameter will be determined by the given metal_lattice_type and metal_latttice_constant.
        add_graphene (bool, optional): whether use graphene as grain boundary. Defaults to False.
        gra_lattice_constant (float, optional): C-C bond length in graphene. Defaults to 1.42.
        metal_gra_overlap_dis (float, optional): minimum distance between metallic particles and graphene. Defaults to 3.1.
        gra_overlap_dis (float, optional): minimum distance between atoms in graphene. Defaults to 1.2.
        seed (np.ndarray, optional): (seednumber, 3) initial position of seed to generate Voronoi polygon. If not given, it will be generated by the given randomseed.
        if_rotation (bool, optional): whether rotate the grain orientation. Defaults to True.
        theta_list (np.ndarray, optional): (seednumber, 3) rotation degree of each grain along x, y, z axis. If not given, it will be generated by the given randomseed.
        face_threshold (float, optional): minimum voronoi polygon face area to add graphene. Defaults to 5.0.
        output_name (str, optional): filename of DUMP file. Defaults to None.
        num_t (int, optional): threads number to generate Voronoi diagram. If not given, use all avilable threads.

    Outputs:
        - **generate an polycrystalline DUMP file with grain ID.**

    Examples:
        >>> import mdapy as mp

        >>> mp.init()

        >>> box = np.array([[0.0, 200.0], [0.0, 200.0], [0.0, 200.0]]) # Generate a box.

        >>> polycry = mp.CreatePolycrystalline(
                      box, 20, 3.615, "FCC", add_graphene=True) # Initilize a Poly class.

        >>> polycry.compute() # Generate a graphene/metal structure with 20 grains.
    """

    def __init__(
        self,
        box,
        seednumber,
        filename=None,
        metal_latttice_constant=None,
        metal_lattice_type=None,
        randomseed=None,
        metal_overlap_dis=2.0,
        add_graphene=False,
        gra_lattice_constant=1.42,
        metal_gra_overlap_dis=3.1,
        gra_overlap_dis=1.2,
        seed=None,
        if_rotation=True,
        theta_list=None,
        face_threshold=5.0,
        output_name=None,
        num_t=None,
    ) -> None:
        box, _, rec = init_box(box)
        if not rec:
            raise "Do not support triclinic box."
        self._real_box = np.zeros((3, 2))
        self._real_box[:, 0] = box[-1]
        self._real_box[:, 1] = (
            np.array([box[0, 0], box[1, 1], box[2, 2]]) + self._real_box[:, 0]
        )

        self._lower = self._real_box[:, 0]
        self.box = np.c_[np.zeros(3), self._real_box[:, 1] - self._real_box[:, 0]]
        self.seednumber = seednumber
        self.filename = filename
        self.metal_latttice_constant = metal_latttice_constant
        self.metal_lattice_type = metal_lattice_type
        if self.filename is None:
            assert self.metal_lattice_type in [
                "FCC",
                "BCC",
                "HCP",
            ], "Only support lattice_type in ['FCC', 'BCC', 'HCP']."
            assert self.metal_latttice_constant is not None

        if randomseed is None:
            self.randomseed = np.random.randint(0, 10000000)
        else:
            self.randomseed = randomseed

        self.metal_overlap_dis = metal_overlap_dis
        self.add_graphene = add_graphene
        self.gra_lattice_constant = gra_lattice_constant
        self.metal_gra_overlap_dis = metal_gra_overlap_dis
        self.gra_overlap_dis = gra_overlap_dis
        if seed is None:
            np.random.seed(self.randomseed)
            self.seed = np.random.rand(self.seednumber, 3) * (
                self.box[:, 1] - self.box[:, 0]
            )
        else:
            self.seed = seed

        self.if_rotation = if_rotation
        if self.if_rotation:
            if theta_list is None:
                np.random.seed(self.randomseed)
                self.theta_list = np.random.rand(self.seednumber, 3) * 360 - 180
            else:
                self.theta_list = theta_list
                self.randomseed = 0  # No random
        else:
            self.theta_list = np.zeros((self.seednumber, 3))
        assert (
            len(self.theta_list) == len(self.seed) == self.seednumber
        ), "shape of theta_lise and seed shoud be equal."
        self.face_threshold = face_threshold
        if output_name is None:
            if self.add_graphene:
                self.output_name = f"GRA-Metal-{self.seednumber}-{self.randomseed}.dump"
            else:
                self.output_name = f"Metal-{self.seednumber}-{self.randomseed}.dump"
        else:
            self.output_name = output_name

        if num_t is None:
            self.num_t = mt.cpu_count()
        else:
            assert num_t >= 1, "num_t should be a positive integer!"
            self.num_t = int(num_t)

    @ti.func
    def _get_plane_equation_coeff(self, p1, p2, p3, coeff):
        """Get plane equation parameters from three points in plane.
        :math:`ax+by+cz+d=0`

        Args:
            p1 (ti.types.ndarray): (3 * 1) point 1 in plane.
            p2 (ti.types.ndarray): (3 * 1) point 2 in plane.
            p3 (ti.types.ndarray): (3 * 1) point 3 in plane.
            coeff (ti.types.ndarray): (4 * 1) zero filled array.

        Returns:
            ti.types.ndarray: (4 * 1) plane equation parameters.
        """
        coeff[0] = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
        coeff[1] = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
        coeff[2] = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        coeff[3] = -(coeff[0] * p1[0] + coeff[1] * p1[1] + coeff[2] * p1[2])
        return coeff

    @ti.func
    def _plane_equation(self, coeff, p):
        # determine point p in plane
        return coeff[0] * p[0] + coeff[1] * p[1] + coeff[2] * p[2] + coeff[3]

    @ti.kernel
    def _get_cell_plane_coeffs(
        self,
        coeffs: ti.types.ndarray(dtype=ti.math.vec4),
        plane_pos: ti.types.ndarray(dtype=ti.math.vec3),
    ):
        # get all plane parameters of a Voronoi cell.
        ti.loop_config(serialize=True)
        for i in range(coeffs.shape[0]):
            coeffs[i] = self._get_plane_equation_coeff(
                plane_pos[i, 0], plane_pos[i, 1], plane_pos[i, 2], coeffs[i]
            )

    def _cell_plane_coeffs(self, cell):
        # get all plane parameters of a Voronoi cell.
        vertices_pos = np.array(cell.vertices())
        face_index = np.array([i[:3] for i in cell.face_vertices()])
        plane_pos = np.array(
            [vertices_pos[face_index[i]] for i in range(face_index.shape[0])]
        )
        coeffs = np.zeros((face_index.shape[0], 4))
        self._get_cell_plane_coeffs(coeffs, plane_pos)
        return coeffs

    @ti.kernel
    def _delete_atoms(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        coeffs: ti.types.ndarray(dtype=ti.math.vec4),
        delete: ti.types.ndarray(),
    ):
        # delete atoms outside the Voronoi cell
        for i in range(pos.shape[0]):
            for j in range(coeffs.shape[0]):
                if self._plane_equation(coeffs[j], pos[i]) < 0:
                    delete[i] = 0
                    break

    def _rotate_pos(self, theta, direction):
        # get rotation matrix along direction.

        radians = np.radians
        cos = np.cos
        sin = np.sin
        theta = radians(theta)
        x, y, z = direction
        rot_mat = np.array(
            [
                [
                    cos(theta) + (1 - cos(theta)) * x**2,
                    (1 - cos(theta)) * x * y - sin(theta) * z,
                    (1 - cos(theta)) * x * z + sin(theta) * y,
                ],
                [
                    (1 - cos(theta)) * y * x + sin(theta) * z,
                    cos(theta) + (1 - cos(theta)) * y**2,
                    (1 - cos(theta)) * y * z - sin(theta) * x,
                ],
                [
                    (1 - cos(theta)) * z * x - sin(theta) * y,
                    (1 - cos(theta)) * z * y + sin(theta) * x,
                    cos(theta) + (1 - cos(theta)) * z**2,
                ],
            ]
        )

        return rot_mat

    def _get_plane_equation_coeff_py(self, p1, p2, p3):
        # get plane equation in python scope.
        coeff = np.zeros(4)
        coeff[0] = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
        coeff[1] = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
        coeff[2] = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        coeff[3] = -(coeff[0] * p1[0] + coeff[1] * p1[1] + coeff[2] * p1[2])
        return coeff

    def _points_in_polygon(self, polygon, pts):
        # check points in polygon plane.

        pts = np.asarray(pts, dtype="float32")
        polygon = np.asarray(polygon, dtype="float32")
        contour2 = np.vstack((polygon[1:], polygon[:1]))
        test_diff = contour2 - polygon
        mask1 = (pts[:, None] == polygon).all(-1).any(-1)
        m1 = (polygon[:, 1] > pts[:, None, 1]) != (contour2[:, 1] > pts[:, None, 1])
        slope = ((pts[:, None, 0] - polygon[:, 0]) * test_diff[:, 1]) - (
            test_diff[:, 0] * (pts[:, None, 1] - polygon[:, 1])
        )
        m2 = slope == 0
        mask2 = (m1 & m2).any(-1)
        m3 = (slope < 0) != (contour2[:, 1] < polygon[:, 1])
        m4 = m1 & m3
        count = np.count_nonzero(m4, axis=-1)
        mask3 = ~(count % 2 == 0)
        mask = mask1 | mask2 | mask3
        return mask

    def _get_pos(self):
        # get the initial position.

        metal_pos = []
        # r_max = np.sqrt(max([cell.max_radius_squared() for cell in cntr]))
        r_max = max([cell.cavity_radius() for cell in self.cntr])
        if self.filename is None:
            lat = LatticeMaker(
                self.metal_latttice_constant, self.metal_lattice_type, 1, 1, 1
            )
            lat.compute()
            box_length = np.array([np.linalg.norm(i) for i in lat.box[:-1]])
            x, y, z = [int(i) for i in np.ceil(r_max / box_length)]
            FCC = LatticeMaker(
                self.metal_latttice_constant, self.metal_lattice_type, x, y, z
            )
            FCC.compute()
        else:
            FCC = System(self.filename)
            box_length = np.array([np.linalg.norm(i) for i in FCC.box[:-1]])
            x, y, z = [int(i) for i in np.ceil(r_max / box_length)]

            FCC.replicate(x, y, z)

        if self.add_graphene:
            gra_pos = []
            x1 = int(np.ceil(r_max / (self.gra_lattice_constant * 3)))
            y1 = int(np.ceil(r_max / (self.gra_lattice_constant * 3**0.5)))
            GRA = LatticeMaker(self.gra_lattice_constant, "GRA", x1, y1, 1)
            GRA.compute()
            gra_vector = np.array([0, 0, 1])
        print("Total grain number:", len(self.cntr))

        for i, cell in enumerate(self.cntr):
            print(f"Generating grain {i}..., volume is {cell.volume()}")
            coeffs = self._cell_plane_coeffs(cell)
            pos = FCC.pos.copy()
            a, b, c = (
                self._rotate_pos(self.theta_list[i, 0], [1, 0, 0]),
                self._rotate_pos(self.theta_list[i, 1], [0, 1, 0]),
                self._rotate_pos(self.theta_list[i, 2], [0, 0, 1]),
            )
            rotate_matrix = a @ b @ c
            pos = np.matmul(pos, rotate_matrix)
            pos = pos - np.mean(pos, axis=0) + cell.pos
            delete = np.ones(pos.shape[0], dtype=int)
            self._delete_atoms(pos, coeffs, delete)
            pos = pos[np.bool_(delete)]
            pos = np.c_[pos, np.ones(pos.shape[0]) * i]
            metal_pos.append(pos)
            if self.add_graphene:
                face_index = cell.face_vertices()
                face_areas = cell.face_areas()
                vertices_pos = np.array(cell.vertices())
                for j in range(coeffs.shape[0]):
                    if face_areas[j] > self.face_threshold:
                        vertices = vertices_pos[face_index[j]]
                        pos = GRA.pos.copy()
                        plane_vector = coeffs[j, :3]
                        plane_vector /= np.linalg.norm(plane_vector)
                        theta = np.degrees(np.arccos(np.dot(gra_vector, plane_vector)))
                        axis = np.cross(gra_vector, plane_vector)
                        direction = axis / (np.linalg.norm(axis) + 1e-6)
                        temp = np.dot(pos, self._rotate_pos(theta, direction))
                        a = self._get_plane_equation_coeff_py(
                            temp[0], temp[1], temp[5]
                        )[:3]
                        a /= np.linalg.norm(a)
                        if not np.isclose(abs(np.dot(a[:3], plane_vector)), 1):
                            theta = 360 - theta
                            temp = np.dot(pos, self._rotate_pos(theta, direction))
                        #     a = self._get_plane_equation_coeff_py(
                        #         temp[0], temp[1], temp[5]
                        #     )[:3]
                        #     a /= np.linalg.norm(a)
                        # assert np.isclose(abs(np.dot(a[:3], plane_vector)), 1), print(
                        #     i, j, np.dot(a[:3], plane_vector)
                        # )
                        pos = temp
                        pos = pos - np.mean(pos, axis=0) + np.mean(vertices, axis=0)
                        vertices = np.r_[vertices, vertices[0].reshape(-1, 3)]
                        # if np.isclose(abs(np.dot(gra_vector, plane_vector)), 0):
                        #     delete = self._points_in_polygon(
                        #         vertices[:, [0, 2]], pos[:, [0, 2]]
                        #     )
                        # else:
                        #     delete = self._points_in_polygon(vertices[:, :2], pos[:, :2])
                        vertices_temp = np.dot(
                            vertices,
                            self._rotate_pos(10, np.array([1, 1, 1]) / np.sqrt(3.0)),
                        )
                        pos_temp = np.dot(
                            pos,
                            self._rotate_pos(10, np.array([1, 1, 1]) / np.sqrt(3.0)),
                        )
                        delete = self._points_in_polygon(vertices_temp, pos_temp)
                        pos = pos[delete]
                        pos = np.c_[pos, np.ones(pos.shape[0]) * i]
                        gra_pos.append(pos)

        metal_pos = np.concatenate(metal_pos)
        if self.add_graphene:
            gra_pos = np.concatenate(gra_pos)
            metal_pos = np.c_[
                np.arange(metal_pos.shape[0]) + 1,
                np.ones(metal_pos.shape[0]),
                metal_pos,
            ]
            gra_pos = np.c_[
                np.arange(gra_pos.shape[0]) + 1 + metal_pos.shape[0],
                np.ones(gra_pos.shape[0]) * 2,
                gra_pos,
            ]
            new_pos = np.r_[metal_pos, gra_pos]
            return new_pos
        else:
            metal_pos = np.c_[
                np.arange(metal_pos.shape[0]) + 1,
                np.ones(metal_pos.shape[0]),
                metal_pos,
            ]
            return metal_pos

    @ti.kernel
    def _wrap_pos(self, pos: ti.types.ndarray(), box: ti.types.ndarray()):
        # wrap position into box.
        boxlength = ti.Vector([box[j, 1] - box[j, 0] for j in range(3)])
        for i in range(pos.shape[0]):
            for j in ti.static(range(3)):
                while pos[i, 2 + j] < box[j, 0]:
                    pos[i, 2 + j] += boxlength[j]
                while pos[i, 2 + j] >= box[j, 1]:
                    pos[i, 2 + j] -= boxlength[j]

    @ti.kernel
    def _find_close(
        self,
        pos: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        atype_list: ti.types.ndarray(),
        grain_id: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        delete_id: ti.types.ndarray(),
        metal_overlap_dis: float,
        gra_overlap_dis: float,
        metal_gra_overlap_dis: float,
        total_overlap_dis: float,
    ):
        # find potential overlap atoms for graphene/metal structure
        for i in range(pos.shape[0]):
            i_type = atype_list[i]
            i_grain = grain_id[i]
            for j in range(neighbor_number[i]):
                j_index = verlet_list[i, j]
                j_type = atype_list[j_index]
                j_grain = grain_id[j_index]
                if i_type == 1:
                    if j_type == 1:
                        if distance_list[i, j] < metal_overlap_dis:
                            if j_index > i:
                                delete_id[j_index] = 0
                else:
                    if j_type == 2:
                        if j_grain > i_grain:
                            if distance_list[i, j] < total_overlap_dis:
                                delete_id[j_index] = 0
                        elif j_grain == i_grain:
                            if distance_list[i, j] < gra_overlap_dis and j_index > i:
                                delete_id[j_index] = 0
                    elif j_type == 1:
                        if distance_list[i, j] < metal_gra_overlap_dis:
                            delete_id[j_index] = 0

    @ti.kernel
    def _find_close_graphene(
        self,
        pos: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        atype_list: ti.types.ndarray(),
        grain_id: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        delete_id: ti.types.ndarray(),
    ):
        # find potential overlap atoms in graphene for graphene/metal structure
        for i in range(pos.shape[0]):
            i_type = atype_list[i]
            i_grain = grain_id[i]
            if i_type == 2:
                n = 0
                for j in range(neighbor_number[i]):
                    j_index = verlet_list[i, j]
                    j_type = atype_list[j_index]
                    j_grain = grain_id[j_index]
                    if j_type == 2:
                        if i_grain > j_grain:
                            n += 1
                if n == neighbor_number[i]:
                    delete_id[i] = 0

    def compute(self, save_dump=True):
        """Do the real polycrystalline structure building."""
        start = time()
        print("Generating voronoi polygon...")
        self.cntr = Container(self.seed, self.box, [1, 1, 1], self.num_t)
        ave_grain_volume = np.mean([cell.volume() for cell in self.cntr])
        new_pos = self._get_pos()
        print("Wraping atoms into box...")
        self._wrap_pos(new_pos, self.box)
        print("Deleting overlap atoms...")
        if self.add_graphene:
            neigh = Neighbor(
                np.ascontiguousarray(new_pos[:, 2:5]),
                self.box,
                rc=self.metal_gra_overlap_dis + 0.1,
                max_neigh=150,
            )
            neigh.compute()
            delete_id = np.ones(new_pos.shape[0], dtype=int)
            self._find_close(
                np.ascontiguousarray(new_pos[:, 2:5]),
                neigh.verlet_list,
                neigh.distance_list,
                new_pos[:, 1].astype(int),
                new_pos[:, -1].astype(int),
                neigh.neighbor_number,
                delete_id,
                self.metal_overlap_dis,
                self.gra_overlap_dis,
                self.metal_gra_overlap_dis,
                1.0,
            )
            new_pos = new_pos[np.bool_(delete_id)]
            neigh = Neighbor(
                np.ascontiguousarray(new_pos[:, 2:5]),
                self.box,
                rc=self.gra_lattice_constant + 0.01,
                max_neigh=20,
            )
            neigh.compute()
            delete_id = np.ones(new_pos.shape[0], dtype=int)
            self._find_close_graphene(
                np.ascontiguousarray(new_pos[:, 2:5]),
                neigh.verlet_list,
                new_pos[:, 1].astype(int),
                new_pos[:, -1].astype(int),
                neigh.neighbor_number,
                delete_id,
            )
            new_pos = new_pos[np.bool_(delete_id)]
            new_pos[:, 0] = np.arange(new_pos.shape[0]) + 1
        else:
            dele = DeleteOverlap(
                np.ascontiguousarray(new_pos[:, 2:5]),
                self.box,
                rc=self.metal_overlap_dis,
            )
            dele.compute()
            new_pos = new_pos[np.bool_(dele.delete_id)]
            new_pos[:, 0] = np.arange(new_pos.shape[0]) + 1

        print(
            f"Total atom numbers: {len(new_pos)}, average grain size: {ave_grain_volume} A^3"
        )
        new_pos[:, 2:5] += self._lower
        new_pos[:, -1] += 1
        self.data = pl.from_numpy(
            new_pos,
            schema={
                "id": pl.Int32,
                "type": pl.Int32,
                "x": pl.Float32,
                "y": pl.Float32,
                "z": pl.Float32,
                "grainid": pl.Int32,
            },
        )
        if save_dump:
            print("Saving atoms into dump file...")
            SaveFile.write_dump(self.output_name, self._real_box, [1, 1, 1], self.data)
        end = time()
        print(f"Time costs: {end-start} s.")


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    ti.init(ti.cpu)
    # polycry = CreatePolycrystalline(
    #     np.array([[0, 200.0], [0, 200.0], [-100, 200.0]]),
    #     20,
    #     4.057,
    #     "FCC",
    #     1,
    #     add_graphene=False,
    # )
    # polycry.compute(save_dump=False)
    # print(polycry.data.head())
    # print(polycry.box)

    a = 4.033
    y_height = 100 * a
    x_height = y_height * 3**0.5
    delta = x_height / 6
    box = np.array([[x_height, 0, 0], [0, y_height, 0], [0, 0, a * 20], [0, 0, 0]])
    y = y_height / 2
    z = a * 10
    seed = np.array(
        [
            [0, 0, z],
            [delta, y, z],
            [2 * delta, 0, z],
            [3 * delta, y, z],
            [4 * delta, 0, z],
            [5 * delta, y, z],
        ]
    )

    seed[:, 0] += 60
    theta_list = np.array(
        [[0, 0, 0], [0, 0, 30], [0, 0, 60], [0, 0, 0], [0, 0, 30], [0, 0, 60]]
    )
    # poly = CreatePolycrystalline(box, len(seed), None, 4.033, 'FCC', metal_overlap_dis=a/2**0.5-0.1, seed=seed, theta_list=theta_list)
    # poly.compute()
    box = np.array([[500, 0, 0], [0, 150, 0], [0, 0, 150], [0, 0, 0]])
    poly = CreatePolycrystalline(
        box,
        10,
        r"C:\Users\herrwu\Desktop\xyz\CubicDiamond.xyz",
        metal_overlap_dis=1.5,
        randomseed=1,
        output_name=r"D:\Study\Diamond\model\Cubic\model.xyz",
    )
    poly.compute(save_dump=False)
    # box = np.array([
    #     [600, 0, 0],
    #     [0, 200, 0],
    #     [0, 0, 200],
    #     [0, 0, 0]
    # ]
    # )
    # poly = CreatePolycrystalline(box, 6, metal_lattice_type='FCC', metal_latttice_constant=3.615, metal_overlap_dis=2.5, randomseed=1, output_name='Cu.dump')
    # poly.compute()
