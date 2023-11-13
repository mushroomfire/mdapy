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
                        pos[i] += box[j]
                    elif n[j] > 1:
                        n[j] -= 1
                        pos[i] -= box[j]


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


chemical_symbols = [
    # 0
    "X",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    # 5
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    # 6
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    # 7
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]

atomic_numbers = {symbol: Z for Z, symbol in enumerate(chemical_symbols)}

""" Van der Waals radii in [A] taken from:
A cartography of the van der Waals territories
S. Alvarez, Dalton Trans., 2013, 42, 8617-8636
DOI: 10.1039/C3DT50599E
"""

vdw_radii = np.array(
    [
        np.nan,  # X
        1.2,  # H
        1.43,  # He [larger uncertainty]
        2.12,  # Li
        1.98,  # Be
        1.91,  # B
        1.77,  # C
        1.66,  # N
        1.5,  # O
        1.46,  # F
        1.58,  # Ne [larger uncertainty]
        2.5,  # Na
        2.51,  # Mg
        2.25,  # Al
        2.19,  # Si
        1.9,  # P
        1.89,  # S
        1.82,  # Cl
        1.83,  # Ar
        2.73,  # K
        2.62,  # Ca
        2.58,  # Sc
        2.46,  # Ti
        2.42,  # V
        2.45,  # Cr
        2.45,  # Mn
        2.44,  # Fe
        2.4,  # Co
        2.4,  # Ni
        2.38,  # Cu
        2.39,  # Zn
        2.32,  # Ga
        2.29,  # Ge
        1.88,  # As
        1.82,  # Se
        1.86,  # Br
        2.25,  # Kr
        3.21,  # Rb
        2.84,  # Sr
        2.75,  # Y
        2.52,  # Zr
        2.56,  # Nb
        2.45,  # Mo
        2.44,  # Tc
        2.46,  # Ru
        2.44,  # Rh
        2.15,  # Pd
        2.53,  # Ag
        2.49,  # Cd
        2.43,  # In
        2.42,  # Sn
        2.47,  # Sb
        1.99,  # Te
        2.04,  # I
        2.06,  # Xe
        3.48,  # Cs
        3.03,  # Ba
        2.98,  # La
        2.88,  # Ce
        2.92,  # Pr
        2.95,  # Nd
        np.nan,  # Pm
        2.9,  # Sm
        2.87,  # Eu
        2.83,  # Gd
        2.79,  # Tb
        2.87,  # Dy
        2.81,  # Ho
        2.83,  # Er
        2.79,  # Tm
        2.8,  # Yb
        2.74,  # Lu
        2.63,  # Hf
        2.53,  # Ta
        2.57,  # W
        2.49,  # Re
        2.48,  # Os
        2.41,  # Ir
        2.29,  # Pt
        2.32,  # Au
        2.45,  # Hg
        2.47,  # Tl
        2.6,  # Pb
        2.54,  # Bi
        np.nan,  # Po
        np.nan,  # At
        np.nan,  # Rn
        np.nan,  # Fr
        np.nan,  # Ra
        2.8,  # Ac [larger uncertainty]
        2.93,  # Th
        2.88,  # Pa [larger uncertainty]
        2.71,  # U
        2.82,  # Np
        2.81,  # Pu
        2.83,  # Am
        3.05,  # Cm [larger uncertainty]
        3.4,  # Bk [larger uncertainty]
        3.05,  # Cf [larger uncertainty]
        2.7,  # Es [larger uncertainty]
        np.nan,  # Fm
        np.nan,  # Md
        np.nan,  # No
        np.nan,  # Lr
    ]
)
vdw_radii.flags.writeable = False
