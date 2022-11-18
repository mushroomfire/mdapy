# @Author: 吴永超
# @Date: 2022-11-18
# 基于voronoi方法生成多晶以及石墨烯晶界
import mdapy as mp
import numpy as np
from tess import Container
import taichi as ti

ti.init(ti.cpu)


@ti.func
def get_plane_equation_coeff(p1, p2, p3, coeff):
    coeff[0] = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
    coeff[1] = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
    coeff[2] = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    coeff[3] = -(coeff[0] * p1[0] + coeff[1] * p1[1] + coeff[2] * p1[2])
    return coeff


@ti.func
def plane_eqution(coeff, p):
    return coeff[0] * p[0] + coeff[1] * p[1] + coeff[2] * p[2] + coeff[3]


@ti.kernel
def get_cell_plane_coeffs(
    coeffs: ti.types.ndarray(element_dim=1), plane_pos: ti.types.ndarray(element_dim=1)
):
    ti.loop_config(serialize=True)
    for i in range(coeffs.shape[0]):
        coeffs[i] = get_plane_equation_coeff(
            plane_pos[i, 0], plane_pos[i, 1], plane_pos[i, 2], coeffs[i]
        )


def cell_plane_coeffs(cell):
    vertices_pos = np.array(cell.vertices())
    face_index = np.array([i[:3] for i in cell.face_vertices()])
    plane_pos = np.array(
        [vertices_pos[face_index[i]] for i in range(face_index.shape[0])]
    )
    coeffs = np.zeros((face_index.shape[0], 4))
    get_cell_plane_coeffs(coeffs, plane_pos)
    return coeffs


@ti.kernel
def delete_atoms(
    pos: ti.types.ndarray(element_dim=1),
    coeffs: ti.types.ndarray(element_dim=1),
    delete: ti.types.ndarray(),
):
    for i in range(pos.shape[0]):
        for j in range(coeffs.shape[0]):
            if plane_eqution(coeffs[j], pos[i]) < 0:
                delete[i] = 0
                break


def rotate_pos(theta, direction):

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


def get_plane_equation_coeff_py(p1, p2, p3):
    coeff = np.zeros(4)
    coeff[0] = (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1])
    coeff[1] = (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2])
    coeff[2] = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    coeff[3] = -(coeff[0] * p1[0] + coeff[1] * p1[1] + coeff[2] * p1[2])
    return coeff


def points_in_polygon(polygon, pts):
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


def get_pos(
    cntr,
    metal_latttice_constant,
    metal_lattice_type,
    gra_lattice_constant,
    randomseed,
    face_threshold=5,
    theta_list=None
):
    assert metal_lattice_type in ["FCC", "BCC"]
    metal_pos = []
    r_max = np.sqrt(max([cell.max_radius_squared() for cell in cntr]))
    x = y = z = int(np.ceil(r_max / metal_latttice_constant))
    FCC = mp.LatticeMaker(metal_latttice_constant, metal_lattice_type, x, y, z)
    FCC.compute()
    gra_pos = []
    x1 = int(np.ceil(r_max / (gra_lattice_constant * 3)))
    y1 = int(np.ceil(r_max / (gra_lattice_constant * 3**0.5)))
    GRA = mp.LatticeMaker(gra_lattice_constant, "GRA", x1, y1, 1)
    GRA.compute()
    gra_vector = np.array([0, 0, 1])
    print("Total grain number:", len(cntr))
    if theta_list is None:
        np.random.seed(randomseed)
        theta_list = np.random.rand(len(cntr), 3)*360-180
    for i, cell in enumerate(cntr):
        print(f"Generating grain {i}..., volume is {cell.volume()}")
        coeffs = cell_plane_coeffs(cell)
        pos = FCC.pos.copy()
        #theta = np.random.randint(0, 180, 3)
        #dirct = np.random.uniform(-100, 100, 3)
        #axis = dirct / np.linalg.norm(dirct)
        #rotate_matrix = rotate_pos(theta, axis)
        rotate_matrix = np.matmul(
            rotate_pos(theta_list[i, 0], [1, 0, 0]),
            rotate_pos(theta_list[i, 1], [0, 1, 0]),
            rotate_pos(theta_list[i, 2], [0, 0, 1]),
        )
        pos = np.matmul(pos, rotate_matrix)
        pos = pos - np.mean(pos, axis=0) + cell.pos
        delete = np.ones(pos.shape[0], dtype=int)
        delete_atoms(pos, coeffs, delete)
        pos = pos[np.bool_(delete)]
        pos = np.c_[pos, np.ones(pos.shape[0]) * i]
        metal_pos.append(pos)
        face_index = cell.face_vertices()
        face_areas = cell.face_areas()
        vertices_pos = np.array(cell.vertices())
        for j in range(coeffs.shape[0]):
            if face_areas[j] > face_threshold:
                vertices = vertices_pos[face_index[j]]
                pos = GRA.pos.copy()
                plane_vector = coeffs[j, :3]
                plane_vector /= np.linalg.norm(plane_vector)
                theta = np.degrees(np.arccos(np.dot(gra_vector, plane_vector)))
                axis = np.cross(gra_vector, plane_vector)
                direction = axis / (np.linalg.norm(axis) + 1e-6)
                temp = np.dot(pos, rotate_pos(theta, direction))
                a = get_plane_equation_coeff_py(temp[0], temp[1], temp[5])[:3]
                a /= np.linalg.norm(a)
                if not np.isclose(abs(np.dot(a[:3], plane_vector)), 1):
                    theta = 360 - theta
                    temp = np.dot(pos, rotate_pos(theta, direction))
                    a = get_plane_equation_coeff_py(temp[0], temp[1], temp[5])[:3]
                    a /= np.linalg.norm(a)
                assert np.isclose(abs(np.dot(a[:3], plane_vector)), 1), print(
                    i, j, np.dot(a[:3], plane_vector)
                )
                pos = temp
                pos = pos - np.mean(pos, axis=0) + np.mean(vertices, axis=0)
                vertices = np.r_[vertices, vertices[0].reshape(-1, 3)]
                if np.isclose(abs(np.dot(gra_vector, plane_vector)), 0):
                    delete = points_in_polygon(vertices[:, [0, 2]], pos[:, [0, 2]])
                else:
                    delete = points_in_polygon(vertices[:, :2], pos[:, :2])
                pos = pos[delete]
                pos = np.c_[pos, np.ones(pos.shape[0]) * i]
                gra_pos.append(pos)

    metal_pos = np.concatenate(metal_pos)
    gra_pos = np.concatenate(gra_pos)
    metal_pos = np.c_[
        np.arange(metal_pos.shape[0]) + 1, np.ones(metal_pos.shape[0]), metal_pos
    ]
    gra_pos = np.c_[
        np.arange(gra_pos.shape[0]) + 1 + metal_pos.shape[0],
        np.ones(gra_pos.shape[0]) * 2,
        gra_pos,
    ]
    new_pos = np.r_[metal_pos, gra_pos]
    return new_pos


def get_pos_metal(cntr, metal_latttice_constant, metal_lattice_type, randomseed, theta_list=None):
    assert metal_lattice_type in ["FCC", "BCC"]
    metal_pos = []
    r_max = np.sqrt(max([cell.max_radius_squared() for cell in cntr]))
    x = y = z = int(np.ceil(r_max / metal_latttice_constant))
    FCC = mp.LatticeMaker(metal_latttice_constant, metal_lattice_type, x, y, z)
    FCC.compute()
    print("Total grain number:", len(cntr))
    if theta_list is None:
        np.random.seed(randomseed)
        theta_list = np.random.rand(len(cntr), 3)*360-180
    for i, cell in enumerate(cntr):
        print(f"Generating grain {i}..., volume is {cell.volume()}")
        coeffs = cell_plane_coeffs(cell)
        pos = FCC.pos.copy()
        #np.random.seed(randomseed * i)
        #theta = np.random.randint(0, 180, 3)
        rotate_matrix = np.matmul(
            rotate_pos(theta_list[i, 0], [1, 0, 0]),
            rotate_pos(theta_list[i, 1], [0, 1, 0]),
            rotate_pos(theta_list[i, 2], [0, 0, 1]),
        )
        #theta = np.random.randint(0, 180)
        #dirct = np.random.uniform(-100, 100, 3)
        #axis = dirct / np.linalg.norm(dirct)
        #rotate_matrix = rotate_pos(theta, axis)
        pos = np.matmul(pos, rotate_matrix)
        pos = pos - np.mean(pos, axis=0) + cell.pos
        delete = np.ones(pos.shape[0], dtype=int)
        delete_atoms(pos, coeffs, delete)
        pos = pos[np.bool_(delete)]
        pos = np.c_[pos, np.ones(pos.shape[0]) * i]
        metal_pos.append(pos)

    metal_pos = np.concatenate(metal_pos)
    metal_pos = np.c_[
        np.arange(metal_pos.shape[0]) + 1, np.ones(metal_pos.shape[0]), metal_pos
    ]
    return metal_pos


@ti.kernel
def warp_pos(pos: ti.types.ndarray(), box: ti.types.ndarray()):
    boxlength = ti.Vector([box[j, 1] - box[j, 0] for j in range(3)])
    for i in range(pos.shape[0]):
        for j in ti.static(range(3)):
            while pos[i, 2 + j] < box[j, 0]:
                pos[i, 2 + j] += boxlength[j]
            while pos[i, 2 + j] >= box[j, 1]:
                pos[i, 2 + j] -= boxlength[j]


@ti.kernel
def find_close(
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
    pos: ti.types.ndarray(),
    verlet_list: ti.types.ndarray(),
    distance_list: ti.types.ndarray(),
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


def write_dump(pos, box, output_name):
    with open(output_name, "w") as op:
        op.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n")
        op.write(f"{pos.shape[0]}\nITEM: BOX BOUNDS pp pp pp\n")
        op.write(
            f"{box[0, 0]} {box[0, 1]}\n{box[1, 0]} {box[1, 1]}\n{box[2, 0]} {box[2, 1]}\n"
        )
        op.write("ITEM: ATOMS id type x y z grainid\n")
        np.savetxt(op, pos, delimiter=" ", fmt="%d %d %f %f %f %d")


def build_graphene_metal_grain_boundary(
    seednumber,
    box,
    randomseed,
    metal_latttice_constant,
    metal_lattice_type,
    gra_lattice_constant,
    metal_overlap_dis,
    gra_overlap_dis,
    metal_gra_overlap_dis,
    total_overlap_dis,
    output_name="GraMetalBoundary.dump",
    seed=None,
    theta_list=None):
    np.random.seed(randomseed)
    if seed is None:
        seed = np.random.rand(seednumber, 3) * (box[:,1]-box[:,0])
        #seed = np.c_[
        #    np.random.rand(seednumber)*(box[0, 1]-box[0, 0]),
        #    np.random.rand(seednumber)*(box[1, 1]-box[1, 0]),
        #    np.random.rand(seednumber)*(box[2, 1]-box[2, 0]),
        #]
    #seed = np.c_[
    #    np.random.uniform(box[0, 0], box[0, 1], seednumber),
    #    np.random.uniform(box[1, 0], box[1, 1], seednumber),
    #    np.random.uniform(box[2, 0], box[2, 1], seednumber),
    #].astype(float)
    #seed = np.c_[
    #    np.random.rand(seednumber)*(box[0, 1]-box[0, 0]),
    #    np.random.rand(seednumber)*(box[1, 1]-box[1, 0]),
    #    np.random.rand(seednumber)*(box[2, 1]-box[2, 0]),
    #]
    cntr = Container(seed, limits=(box[:, 0], box[:, 1]), periodic=np.bool_([1, 1, 1]))
    ave_grain_volume = np.mean([cell.volume() for cell in cntr])
    new_pos = get_pos(
        cntr,
        metal_latttice_constant,
        metal_lattice_type,
        gra_lattice_constant,
        randomseed,
        face_threshold=5,
        theta_list=theta_list
    )
    print("Wraping atoms into box...")
    warp_pos(new_pos, box)
    print("Deleting overlap atoms...")
    neigh = mp.Neighbor(
        new_pos[:, 2:5], box, rc=metal_gra_overlap_dis + 0.1, max_neigh=150
    )
    neigh.compute()
    delete_id = np.ones(new_pos.shape[0], dtype=int)
    find_close(
        new_pos[:, 2:5],
        neigh.verlet_list,
        neigh.distance_list,
        new_pos[:, 1].astype(int),
        new_pos[:, -1].astype(int),
        neigh.neighbor_number,
        delete_id,
        metal_overlap_dis,
        gra_overlap_dis,
        metal_gra_overlap_dis,
        total_overlap_dis,
    )
    new_pos = new_pos[np.bool_(delete_id)]
    neigh = mp.Neighbor(
        new_pos[:, 2:5], box, rc=gra_lattice_constant + 0.01, max_neigh=20
    )
    neigh.compute()
    delete_id = np.ones(new_pos.shape[0], dtype=int)
    find_close_graphene(
        new_pos[:, 2:5],
        neigh.verlet_list,
        neigh.distance_list,
        new_pos[:, 1].astype(int),
        new_pos[:, -1].astype(int),
        neigh.neighbor_number,
        delete_id,
    )
    new_pos = new_pos[np.bool_(delete_id)]
    new_pos[:, 0] = np.arange(new_pos.shape[0]) + 1
    print(
        f"Total atom numbers: {len(new_pos)}, average grain size: {ave_grain_volume} A^3"
    )
    print("Saving atoms into dump file...")
    write_dump(new_pos, box, output_name)
    return ave_grain_volume


def build_metal_grain_boundary(
    seednumber,
    box,
    randomseed,
    metal_latttice_constant,
    metal_lattice_type,
    metal_overlap_dis,
    output_name="MetalBoundary.dump",
    seed=None,
    theta_list=None):
    np.random.seed(randomseed)
    if seed is None:
        seed = np.random.rand(seednumber, 3) * (box[:,1]-box[:,0])
        #seed = np.c_[
        #    np.random.rand(seednumber)*(box[0, 1]-box[0, 0]),
        #    np.random.rand(seednumber)*(box[1, 1]-box[1, 0]),
        #    np.random.rand(seednumber)*(box[2, 1]-box[2, 0]),
        #]
    cntr = Container(seed, limits=(box[:, 0], box[:, 1]), periodic=np.bool_([1, 1, 1]))
    ave_grain_volume = np.mean([cell.volume() for cell in cntr])
    new_pos = get_pos_metal(
        cntr, metal_latttice_constant, metal_lattice_type, randomseed, theta_list=theta_list
    )
    print("Wraping atoms into box...")
    warp_pos(new_pos, box)
    print("Deleting overlap atoms...")
    neigh = mp.Neighbor(
        new_pos[:, 2:5], box, rc=metal_gra_overlap_dis + 0.1, max_neigh=100
    )
    neigh.compute()
    delete_id = np.ones(new_pos.shape[0], dtype=int)
    find_close_metal(
        new_pos[:, 2:5],
        neigh.verlet_list,
        neigh.distance_list,
        neigh.neighbor_number,
        delete_id,
        metal_overlap_dis,
    )
    new_pos = new_pos[np.bool_(delete_id)]
    new_pos[:, 0] = np.arange(new_pos.shape[0]) + 1
    print(
        f"Total atom numbers: {len(new_pos)}, average grain size: {ave_grain_volume} A^3"
    )
    print("Saving atoms into dump file...")
    write_dump(new_pos, box, output_name)
    return ave_grain_volume


if __name__ == "__main__":

    box = np.array([[0.0, 800.0], [0.0, 200.0], [0.0, 200.0]])
    randomseed = 1
    metal_latttice_constant, metal_lattice_type, gra_lattice_constant, = (
        3.615,
        "FCC",
        1.42,
    )
    metal_overlap_dis, gra_overlap_dis, metal_gra_overlap_dis, total_overlap_dis = (
        3.615 / 2**0.5,
        gra_lattice_constant - 0.01,
        3.1,
        1.0,
    )
    # data = np.array([i.split()[1:] for i in open('final_param.txt').readlines()[3:]], dtype=float)
    seed = None # data[:,0:3]
    theta_list = None # data[:,3:]
    #print(theta_list.shape)
    file = open("grain_size.txt", "w")
    for seednumber in [100]:
        output_name_gra_metal = (
            f"GRA-Metal-{metal_lattice_type}-{seednumber}-{randomseed}.dump"
        )
        output_name_metal = f"Metal-{metal_lattice_type}-{seednumber}-{randomseed}.dump"
        grain_size = build_graphene_metal_grain_boundary(
            seednumber,
            box,
            randomseed,
            metal_latttice_constant,
            metal_lattice_type,
            gra_lattice_constant,
            metal_overlap_dis,
            gra_overlap_dis,
            metal_gra_overlap_dis,
            total_overlap_dis,
            output_name_gra_metal,
            seed,
            theta_list
        )
        grain_size = build_metal_grain_boundary(
            seednumber,
            box,
            randomseed,
            metal_latttice_constant,
            metal_lattice_type,
            metal_overlap_dis,
            output_name_metal,
            seed,
            theta_list
        )
        file.write(f"{seednumber} {grain_size}\n")
    file.close()
