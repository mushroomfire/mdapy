import taichi as ti
import numpy as np


@ti.kernel
def wrap_pos(
    pos: ti.types.ndarray(), box: ti.types.ndarray(), boundary: ti.types.ndarray()
):
    boxlength = ti.Vector([box[j, 1] - box[j, 0] for j in range(3)])
    for i in range(pos.shape[0]):
        for j in ti.static(range(3)):
            if boundary[j] == 1:
                while pos[i, j] < box[j, 0]:
                    pos[i, j] += boxlength[j]
                while pos[i, j] >= box[j, 1]:
                    pos[i, j] -= boxlength[j]


@ti.kernel
def _unwrap_pos_with_image_p(
    pos_list: ti.types.ndarray(element_dim=1),
    box: ti.types.ndarray(),
    boundary: ti.types.vector(3, dtype=int),
    image_p: ti.types.ndarray(element_dim=1),
):
    boxlength = ti.Vector([box[j, 1] - box[j, 0] for j in range(3)])
    for i, j in pos_list:
        for k in ti.static(range(3)):
            if boundary[k] == 1:
                pos_list[i, j][k] += image_p[i - 1, j][k] * boxlength[k]


@ti.kernel
def _unwrap_pos_without_image_p(
    pos_list: ti.types.ndarray(element_dim=1),
    box: ti.types.ndarray(),
    boundary: ti.types.vector(3, dtype=int),
    image_p: ti.types.ndarray(element_dim=1),
):

    boxlength = ti.Vector([box[j, 1] - box[j, 0] for j in range(3)])
    ti.loop_config(serialize=True)
    for frame in range(1, pos_list.shape[0]):
        for i in range(pos_list.shape[1]):
            for j in ti.static(range(3)):
                if boundary[j] == 1:
                    pos_list[frame, i][j] += image_p[frame - 1, i][j] * boxlength[j]
            delta = pos_list[frame, i] - pos_list[frame - 1, i]
            for j in ti.static(range(3)):
                if boundary[j] == 1:
                    if delta[j] >= boxlength[j] / 2:
                        image_p[frame, i][j] -= 1.0
                        pos_list[frame, i][j] -= boxlength[j]
                    elif delta[j] <= -boxlength[j] / 2:
                        image_p[frame, i][j] += 1.0
                        pos_list[frame, i][j] += boxlength[j]


def unwrap_pos(pos_list, box, boundary=[1, 1, 1], image_p=None):

    if image_p is not None:
        _unwrap_pos_with_image_p(pos_list, box, boundary, image_p)
    else:
        boundary = ti.Vector(boundary)
        image_p = np.zeros_like(pos_list)
        _unwrap_pos_without_image_p(pos_list, box, boundary, image_p)
