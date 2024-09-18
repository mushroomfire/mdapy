# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np
import taichi as ti


@ti.func
def _pbc_rec(rij, boundary, box_length):
    """This func is used to calculate the pair distance in rectangle box."""
    for m in ti.static(range(3)):
        if boundary[m] == 1:
            dx = rij[m]
            x_size = box_length[m]
            h_x_size = x_size * 0.5
            if dx > h_x_size:
                dx = dx - x_size
            if dx <= -h_x_size:
                dx = dx + x_size
            rij[m] = dx
    return rij


@ti.func
def _pbc(
    rij,
    boundary,
    box: ti.types.ndarray(element_dim=1),
    inverse_box: ti.types.ndarray(element_dim=1),
):
    """This func is used to calculate the pair distance in triclinic box."""
    n = rij[0] * inverse_box[0] + rij[1] * inverse_box[1] + rij[2] * inverse_box[2]
    for i in ti.static(range(3)):
        if boundary[i] == 1:
            if n[i] > 0.5:
                n[i] -= 1
            elif n[i] < -0.5:
                n[i] += 1
    return n[0] * box[0] + n[1] * box[1] + n[2] * box[2]


def init_box(box):
    """This function is used to obtain box array.

    - case 1: a float/int number, such as 10, it will generate a rectangle box with box length of 10 A.
    - case 2: a str, such as "10", the result is same with case 1.
    - case 3: a list/tuple/np.ndarray.

    If it includs three elements, such as [10, 20, 30], it will build a rectangle box with box length of x(10), y(20) and z(30). If the input is 2 2-D np.array or nested list.
    We accept the shape of (3, 2), (3, 3) and (4, 3). The shape of (3, 2) indicates the first colume is the origin
    and the second column is the maximum points of box. The shape of (3, 3) indicates the three box vector. The shape
    of (4, 3) indicates the three box vector and the fourth row is the origin position. The defaults origin is [0, 0, 0].
    The final box is a np.ndarray with shape of (4, 3):

    - ax ay az (x axis)
    - bx by bz (y axis)
    - cx cy cz (z axis)
    - ox oy oz (origin)

    Args:
        box (float | str | list | np.ndarray): The input box.

    Returns:
        tuple[np.ndarray, np.ndarray, bool]: box (4, 3), inverse box (3, 3), rec. The rec indicates if the box is reactangle.
    """

    is_a_number = False
    try:
        assert float(box) > 0, "Must be a positive number."
        box = np.eye(3) * float(box)
        box = np.r_[box, np.zeros((1, 3))]
        is_a_number = True
    except Exception:
        pass

    if not is_a_number:
        box = np.array(box, float)

        if box.ndim == 1:
            assert len(box) == 3, "The box length should be 3 for 1-D container."
            box = np.r_[np.diag(box), np.zeros((1, 3))]
        elif box.ndim == 2:
            if box.shape == (3, 2):
                box = np.r_[np.diag(box[:, 1] - box[:, 0]), box[:, 0].reshape(1, -1)]
            elif box.shape == (3, 3):
                box = np.r_[box, np.zeros((1, 3))]
            elif box.shape == (4, 3):
                pass
            else:
                raise "Wrong box shape, support (3, 2), (3, 3) and (4, 3)."
        else:
            raise "Wrong box dimension. support 1 and 2 array dimension."

    inverse_box = np.linalg.inv(box[:-1])

    rec = True
    for i in range(3):
        for j in range(3):
            if i != j and box[i, j] != 0:
                rec = False
                break

    return box, inverse_box, rec


if __name__ == "__main__":
    for a in [
        10,
        "8",
        5.0,
        (3, 2, 1),
        np.array([[-1, 10], [-2, 13.0], [0, 5]]),
        np.array([[10, 0, 0], [0, 11, 0], [0, 0, 15]]),
        np.array([[10, 0, 0], [0, 11, 0], [0, 0, 15], [1, 2, 3]]),
        np.array([[10, 1, 2], [-1, 11, -2], [-3, -4, 15], [1, 2, 3]]),
    ]:
        box, inverse_box, rec = init_box(a)
        print("a is :")
        print(a)
        print("box is :")
        print(box)
        print("is rectangle :")
        print(rec)

    # Test memory
    # import tracemalloc

    # tracemalloc.start()
    # for i in range(1, 100000):
    #     b = Box(i)
    #     a = TestBox(i)
    #     if i % 100 == 0:
    #         current, peak = tracemalloc.get_traced_memory()
    #         print(f"i={i}, Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
