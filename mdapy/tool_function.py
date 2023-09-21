import taichi as ti
import numpy as np


def _check_repeat_nearest(pos, box, boundary):
    repeat = [1, 1, 1]
    box_length = [np.linalg.norm(box[i]) for i in range(3)]
    repeat_length = False
    for i in range(3):
        if boundary[i] == 1 and box_length[i] <= 6.0:
            repeat_length = True
    repeat_number = False
    if pos.shape[0] < 50 and sum(boundary) > 0:
        repeat_number = True

    if repeat_length or repeat_number:
        repeat = [1 if boundary[i] == 0 else 3 for i in range(3)]
        while np.product(repeat) * pos.shape[0] < 50:
            for i in range(3):
                if boundary[i] == 1:
                    repeat[i] += 1

        for i in range(3):
            if boundary[i] == 1:
                while repeat[i] * box_length[i] < 6.0:
                    repeat[i] += 1
    return repeat


def _check_repeat_cutoff(box, boundary, rc, factor=2):
    repeat = [1, 1, 1]
    box_length = [np.linalg.norm(box[i]) for i in range(3)]
    for i in range(3):
        if boundary[i] == 1:
            while repeat[i] * box_length[i] <= factor * rc:
                repeat[i] += 1
    return repeat


@ti.kernel
def _wrap_pos(
    pos: ti.types.ndarray(element_dim=1),
    box: ti.types.ndarray(element_dim=1),
    boundary: ti.types.ndarray(),
):
    """This function is used to wrap particle positions into box considering periodic boundarys.

    Args:
        pos (ti.types.ndarray): (Nx3) particle position.

        box (ti.types.ndarray): (4x3) system box.

        boundary (ti.types.ndarray): boundary conditions, 1 is periodic and 0 is free boundary.
    """

    for i in range(pos.shape[0]):
        rij = pos[i] - box[3]
        nz = rij[2] / box[2][2]
        ny = (rij[1] - nz * box[2][1]) / box[1][1]
        nx = (rij[0] - ny * box[1][0] - nz * box[2][0]) / box[0][0]
        n = ti.Vector([nx, ny, nz])
        for j in ti.static(range(3)):
            if boundary[j] == 1:
                while n[j] < 0 or n[j] > 1:
                    if n[j] < 0:
                        n[j] += 1
                        pos[j] += box[j]
                    elif n[j] > 1:
                        n[j] -= 1
                        pos[j] -= box[j]


@ti.kernel
def _partition_select_sort(
    indices: ti.types.ndarray(), keys: ti.types.ndarray(), N: int
):
    """This function sorts N-th minimal value in keys.

    Args:
        indices (ti.types.ndarray): indices.
        keys (ti.types.ndarray): values to be sorted.
        N (int): number of sorted values.
    """
    for i in range(indices.shape[0]):
        for j in range(N):
            minIndex = j
            for k in range(j + 1, indices.shape[1]):
                if keys[i, k] < keys[i, minIndex]:
                    minIndex = k
            if minIndex != j:
                keys[i, minIndex], keys[i, j] = keys[i, j], keys[i, minIndex]
                indices[i, minIndex], indices[i, j] = (
                    indices[i, j],
                    indices[i, minIndex],
                )


@ti.kernel
def _unwrap_pos_with_image_p(
    pos_list: ti.types.ndarray(dtype=ti.math.vec3),
    box: ti.types.ndarray(dtype=ti.math.vec3),
    boundary: ti.types.vector(3, dtype=int),
    image_p: ti.types.ndarray(dtype=ti.math.vec3),
):
    """This function is used to unwrap particle positions
     into box considering periodic boundarys with help of image_p.

    Args:
        pos_list (ti.types.ndarray): (Nframes x Nparticles x 3) particle position.

        box (ti.types.ndarray): (4x3) system box.

        boundary (ti.types.vector): boundary conditions, 1 is periodic and 0 is free boundary.

        image_p (ti.types.ndarray): (Nframes x Nparticles x 3) image_p, such as 1 indicates plus a box distance and -2 means substract two box distances.
    """
    for i, j in pos_list:
        for k in ti.static(range(3)):
            if boundary[k] == 1:
                pos_list[i, j] += image_p[i - 1, j][k] * box[k]


@ti.kernel
def _unwrap_pos_without_image_p(
    pos_list: ti.types.ndarray(dtype=ti.math.vec3),
    box: ti.types.ndarray(dtype=ti.math.vec3),
    boundary: ti.types.vector(3, dtype=int),
    image_p: ti.types.ndarray(dtype=ti.math.vec3),
):
    """This function is used to unwrap particle positions
     into box considering periodic boundarys without help of image_p.

    Args:
        pos_list (ti.types.ndarray): (Nframes x Nparticles x 3) particle position.

        box (ti.types.ndarray): (4x3) system box.

        boundary (ti.types.vector): boundary conditions, 1 is periodic and 0 is free boundary.

        image_p (ti.types.ndarray): (Nframes x Nparticles x 3) fill with 0.
    """

    ti.loop_config(serialize=True)
    for frame in range(1, pos_list.shape[0]):
        for i in range(pos_list.shape[1]):
            for j in ti.static(range(3)):
                if boundary[j] == 1:
                    pos_list[frame, i] += image_p[frame - 1, i][j] * box[j]
            delta = pos_list[frame, i] - pos_list[frame - 1, i]
            nz = delta[2] / box[2][2]
            ny = (delta[1] - nz * box[2][1]) / box[1][1]
            nx = (delta[0] - ny * box[1][0] - nz * box[2][0]) / box[0][0]
            n = ti.Vector([nx, ny, nz])
            for j in ti.static(range(3)):
                if boundary[j] == 1:
                    while n[j] <= -0.5:
                        n[j] += 1
                        image_p[frame, i][j] += 1
                        pos_list[frame, i] += box[j]
                    while n[j] >= 0.5:
                        n[j] -= 1
                        image_p[frame, i][j] -= 1
                        pos_list[frame, i] -= box[j]


def _unwrap_pos(pos_list, box, boundary=[1, 1, 1], image_p=None):
    """This function is used to unwrap particle positions
     into box considering periodic boundarys.

    Args:
        pos_list (np.ndarray): (Nframes x Nparticles x 3) particle position.

        box (np.ndarray): (4x3) system box.

        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].

        image_p (_type_, optional): (Nframes x Nparticles x 3) image_p, such as 1 indicates plus a box distance and -2 means substract two box distances. Defaults to None.
    """
    if image_p is not None:
        boundary = ti.Vector(boundary)
        _unwrap_pos_with_image_p(pos_list, box, boundary, image_p)
    else:
        boundary = ti.Vector(boundary)
        image_p = np.zeros_like(pos_list, dtype=int)
        _unwrap_pos_without_image_p(pos_list, box, boundary, image_p)


def _init_vel(N, T, Mass=1.0):
    Boltzmann_Constant = 8.617385e-5
    np.random.seed(10086)
    x1 = np.random.random(N * 3)
    x2 = np.random.random(N * 3)
    vel = (
        np.sqrt(T * Boltzmann_Constant / Mass)
        * np.sqrt(-2 * np.log(x1))
        * np.cos(2 * np.pi * x2)
    ).reshape(N, 3)
    vel -= vel.mean(axis=0)  # A/ps
    return vel * 100  # m/s
