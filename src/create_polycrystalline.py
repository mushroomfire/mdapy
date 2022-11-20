from .polygon import poly
import numpy as np
import taichi as ti
from .lattice_maker import LatticeMaker
from .neighbor import Neighbor
from time import time


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
    def __init__(self, pos, box, boundary) -> None:
        self.pos = pos
        self.box = box
        self.boundary = np.bool_(boundary)
        self.compute()

    def compute(self):
        volume_radius = np.zeros((self.pos.shape[0], 2))
        face_vertices = (
            np.zeros((self.pos.shape[0], 30, 20), dtype=int) - 1
        )  # 30 faces 20 edges
        face_areas = np.zeros((self.pos.shape[0], 30))
        vertices_pos = np.zeros((self.pos.shape[0], 50, 3))  # 50 vertices
        poly.get_cell_info(
            self.pos,
            self.box,
            self.boundary,
            volume_radius,
            face_vertices,
            vertices_pos,
            face_areas,
        )

        for i in range(self.pos.shape[0]):
            new_face_vertices = []
            for j in face_vertices[i]:
                if j[0] == -1:
                    break
                else:
                    new_face_vertices.append(j[j != -1].tolist())

            self.append(
                Cell(
                    new_face_vertices,
                    vertices_pos[i],
                    volume_radius[i, 0],
                    volume_radius[i, 1],
                    face_areas[i][face_areas[i] > 0],
                    self.pos[i],
                )
            )


@ti.data_oriented
class CreatePolycrystalline:
    def __init__(
        self,
        box,
        seednumber,
        metal_latttice_constant,
        metal_lattice_type,
        randomseed=None,
        metal_overlap_dis=None,
        add_graphene=False,
        gra_lattice_constant=1.42,
        metal_gra_overlap_dis=3.1,
        gra_overlap_dis=1.41,
        seed=None,
        theta_list=None,
        face_threshold=5.0,
        output_name=None,
    ) -> None:
        self.box = box
        self.seednumber = seednumber
        self.metal_latttice_constant = metal_latttice_constant
        self.metal_lattice_type = metal_lattice_type
        assert self.metal_lattice_type in [
            "FCC",
            "BCC",
            "HCP",
        ], "Only support lattice_type in ['FCC', 'BCC', 'HCP']."
        if randomseed is None:
            self.randomseed = np.random.randint(0, 10000000)
        else:
            self.randomseed = randomseed

        if metal_overlap_dis is None:
            if self.metal_lattice_type == "FCC":
                self.metal_overlap_dis = self.metal_latttice_constant / 2**0.5
            elif self.metal_lattice_type == "BCC":
                self.metal_overlap_dis = self.metal_latttice_constant / (0.5 * 3**0.5)
            else:
                self.metal_overlap_dis = 2.0
        else:
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
        if theta_list is None:
            np.random.seed(self.randomseed)
            self.theta_list = np.random.rand(self.seednumber, 3) * 360 - 180
        else:
            self.theta_list = theta_list
        assert (
            len(self.theta_list) == len(self.seed) == self.seednumber
        ), "shape of theta_lise and seed shoud be equal."
        self.face_threshold = face_threshold
        if output_name is None:
            if self.add_graphene:
                self.output_name = f"GRA-Metal-{self.metal_lattice_type}-{self.seednumber}-{self.randomseed}.dump"
            else:
                self.output_name = f"Metal-{self.metal_lattice_type}-{self.seednumber}-{self.randomseed}.dump"
        else:
            self.output_name = output_name

    @ti.func
    def get_plane_equation_coeff(self, p1, p2, p3, coeff):
        coeff[0] = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
        coeff[1] = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
        coeff[2] = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        coeff[3] = -(coeff[0] * p1[0] + coeff[1] * p1[1] + coeff[2] * p1[2])
        return coeff

    @ti.func
    def plane_eqution(self, coeff, p):
        return coeff[0] * p[0] + coeff[1] * p[1] + coeff[2] * p[2] + coeff[3]

    @ti.kernel
    def get_cell_plane_coeffs(
        self,
        coeffs: ti.types.ndarray(element_dim=1),
        plane_pos: ti.types.ndarray(element_dim=1),
    ):
        ti.loop_config(serialize=True)
        for i in range(coeffs.shape[0]):
            coeffs[i] = self.get_plane_equation_coeff(
                plane_pos[i, 0], plane_pos[i, 1], plane_pos[i, 2], coeffs[i]
            )

    def cell_plane_coeffs(self, cell):
        vertices_pos = np.array(cell.vertices())
        face_index = np.array([i[:3] for i in cell.face_vertices()])
        plane_pos = np.array(
            [vertices_pos[face_index[i]] for i in range(face_index.shape[0])]
        )
        coeffs = np.zeros((face_index.shape[0], 4))
        self.get_cell_plane_coeffs(coeffs, plane_pos)
        return coeffs

    @ti.kernel
    def delete_atoms(
        self,
        pos: ti.types.ndarray(element_dim=1),
        coeffs: ti.types.ndarray(element_dim=1),
        delete: ti.types.ndarray(),
    ):
        for i in range(pos.shape[0]):
            for j in range(coeffs.shape[0]):
                if self.plane_eqution(coeffs[j], pos[i]) < 0:
                    delete[i] = 0
                    break

    def rotate_pos(self, theta, direction):

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

    def get_plane_equation_coeff_py(self, p1, p2, p3):
        coeff = np.zeros(4)
        coeff[0] = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
        coeff[1] = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
        coeff[2] = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        coeff[3] = -(coeff[0] * p1[0] + coeff[1] * p1[1] + coeff[2] * p1[2])
        return coeff

    def points_in_polygon(self, polygon, pts):
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

    def get_pos(self):
        metal_pos = []
        # r_max = np.sqrt(max([cell.max_radius_squared() for cell in cntr]))
        r_max = max([cell.cavity_radius() for cell in self.cntr])
        x = y = z = int(np.ceil(r_max / self.metal_latttice_constant))
        FCC = LatticeMaker(
            self.metal_latttice_constant, self.metal_lattice_type, x, y, z
        )
        FCC.compute()
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
            coeffs = self.cell_plane_coeffs(cell)
            pos = FCC.pos.copy()
            rotate_matrix = np.matmul(
                self.rotate_pos(self.theta_list[i, 0], [1, 0, 0]),
                self.rotate_pos(self.theta_list[i, 1], [0, 1, 0]),
                self.rotate_pos(self.theta_list[i, 2], [0, 0, 1]),
            )
            pos = np.matmul(pos, rotate_matrix)
            pos = pos - np.mean(pos, axis=0) + cell.pos
            delete = np.ones(pos.shape[0], dtype=int)
            self.delete_atoms(pos, coeffs, delete)
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
                        temp = np.dot(pos, self.rotate_pos(theta, direction))
                        a = self.get_plane_equation_coeff_py(temp[0], temp[1], temp[5])[
                            :3
                        ]
                        a /= np.linalg.norm(a)
                        if not np.isclose(abs(np.dot(a[:3], plane_vector)), 1):
                            theta = 360 - theta
                            temp = np.dot(pos, self.rotate_pos(theta, direction))
                        #     a = self.get_plane_equation_coeff_py(
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
                        #     delete = self.points_in_polygon(
                        #         vertices[:, [0, 2]], pos[:, [0, 2]]
                        #     )
                        # else:
                        #     delete = self.points_in_polygon(vertices[:, :2], pos[:, :2])
                        # 避免垂直于xy平面
                        vertices_temp = np.dot(vertices, self.rotate_pos(10, [1, 0, 0]))
                        pos_temp = np.dot(pos, self.rotate_pos(10, [1, 0, 0]))
                        delete = self.points_in_polygon(vertices_temp, pos_temp)
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
    def warp_pos(self, pos: ti.types.ndarray(), box: ti.types.ndarray()):
        boxlength = ti.Vector([box[j, 1] - box[j, 0] for j in range(3)])
        for i in range(pos.shape[0]):
            for j in ti.static(range(3)):
                while pos[i, 2 + j] < box[j, 0]:
                    pos[i, 2 + j] += boxlength[j]
                while pos[i, 2 + j] >= box[j, 1]:
                    pos[i, 2 + j] -= boxlength[j]

    @ti.kernel
    def find_close(
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
    def find_close_graphene(
        self,
        pos: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        atype_list: ti.types.ndarray(),
        grain_id: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        delete_id: ti.types.ndarray(),
    ):

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

    @ti.kernel
    def find_close_metal(
        self,
        pos: ti.types.ndarray(),
        verlet_list: ti.types.ndarray(),
        distance_list: ti.types.ndarray(),
        neighbor_number: ti.types.ndarray(),
        delete_id: ti.types.ndarray(),
        metal_overlap_dis: float,
    ):

        for i in range(pos.shape[0]):
            for j in range(neighbor_number[i]):
                if distance_list[i, j] <= metal_overlap_dis and verlet_list[i, j] > i:
                    delete_id[j] = 0

    def write_dump(self, pos):
        with open(self.output_name, "w") as op:
            op.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n")
            op.write(f"{pos.shape[0]}\nITEM: BOX BOUNDS pp pp pp\n")
            op.write(
                f"{self.box[0, 0]} {self.box[0, 1]}\n{self.box[1, 0]} {self.box[1, 1]}\n{self.box[2, 0]} {self.box[2, 1]}\n"
            )
            op.write("ITEM: ATOMS id type x y z grainid\n")
            np.savetxt(op, pos, delimiter=" ", fmt="%d %d %f %f %f %d")

    def compute(self):
        start = time()
        print("Generating voronoi polygon...")
        self.cntr = Container(self.seed, self.box, [1, 1, 1])
        ave_grain_volume = np.mean([cell.volume() for cell in self.cntr])
        # print(ave_grain_volume, self.cntr[0].cavity_radius())
        new_pos = self.get_pos()
        print("Wraping atoms into box...")
        self.warp_pos(new_pos, self.box)
        print("Deleting overlap atoms...")
        if self.add_graphene:
            neigh = Neighbor(
                new_pos[:, 2:5],
                self.box,
                rc=self.metal_gra_overlap_dis + 0.1,
                max_neigh=150,
            )
            neigh.compute()
            delete_id = np.ones(new_pos.shape[0], dtype=int)
            self.find_close(
                new_pos[:, 2:5],
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
                new_pos[:, 2:5],
                self.box,
                rc=self.gra_lattice_constant + 0.01,
                max_neigh=20,
            )
            neigh.compute()
            delete_id = np.ones(new_pos.shape[0], dtype=int)
            self.find_close_graphene(
                new_pos[:, 2:5],
                neigh.verlet_list,
                new_pos[:, 1].astype(int),
                new_pos[:, -1].astype(int),
                neigh.neighbor_number,
                delete_id,
            )
            new_pos = new_pos[np.bool_(delete_id)]
            new_pos[:, 0] = np.arange(new_pos.shape[0]) + 1
        else:
            neigh = Neighbor(
                new_pos[:, 2:5],
                self.box,
                rc=self.metal_gra_overlap_dis + 0.1,
                max_neigh=100,
            )
            neigh.compute()
            delete_id = np.ones(new_pos.shape[0], dtype=int)
            self.find_close_metal(
                new_pos[:, 2:5],
                neigh.verlet_list,
                neigh.distance_list,
                neigh.neighbor_number,
                delete_id,
                self.metal_overlap_dis,
            )
            new_pos = new_pos[np.bool_(delete_id)]
            new_pos[:, 0] = np.arange(new_pos.shape[0]) + 1
        print(
            f"Total atom numbers: {len(new_pos)}, average grain size: {ave_grain_volume} A^3"
        )
        print("Saving atoms into dump file...")
        self.write_dump(new_pos)
        end = time()
        print(f"Time costs: {end-start} s.")
        return ave_grain_volume


if __name__ == "__main__":

    # box = np.array([[0.0, 100], [0.0, 100.0], [0.0, 100.0]])
    # seed = np.random.rand(20, 3) * (box[:, 1] - box[:, 0])
    # boundary = [1, 1, 1]
    # cntr = Container(seed, box, boundary)
    # cell = cntr[0]
    # print("r:", cell.cavity_radius())
    # print("volume:", cell.volume())
    # print("face_vertices:", cell.face_vertices())
    # print("face_areas:", cell.face_areas())
    # print("vertices:", cell.vertices())
    # print(cntr[0])
    # print(len(cntr))
    # for i in cntr:
    #     print(i)
    # print(cntr[0].cavity_radius())
    # print(cntr[:1])
    # print(len(cntr))
    # print(cntr[0].face_vertices())
    # print(cntr[0].vertices())
    ti.init(ti.cpu)
    box = np.array([[0.0, 300.0], [0.0, 300.0], [0.0, 300.0]])
    polycry = CreatePolycrystalline(box, 20, 3.615, "FCC")
    polycry.compute()
    polycry = CreatePolycrystalline(box, 20, 3.615, "FCC", add_graphene=True)
    polycry.compute()
