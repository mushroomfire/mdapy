# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

# This file includes some simple and useful function, part of them will be exposed to users.

import taichi as ti
import numpy as np
import os
from datetime import datetime
from functools import wraps
try:
    from box import init_box
except Exception:
    from .box import init_box

import subprocess 



    



   
def run_cmd(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE, 
                            universal_newlines=True, shell=True)
    return result

def timer(function):
    """Decorators function for timing."""

    @wraps(function)
    def timer(*args, **kwargs):
        start = datetime.now()
        result = function(*args, **kwargs)
        end = datetime.now()
        print(f"\nFunction {function.__name__}() is finished. Time costs {end-start}.")
        return result

    return timer


def split_xyz(input_file, outputdir="res", output_file_prefix=None, fix_N=True):
    """This function can be used to split a xyz trajectory into seperate xyz files, which can be read by mp.System.

    Args:
        input_file (str): xyz filename.
        outputdir (str, optional): output folder. Defaults to 'res'.
        output_file_prefix (str, optional): output file prefix. If not given, mdapy will generate one based on the input_file. Defaults to None.
        fix_N (bool, optional): if the whole trajectory has the same atoms, set this to True, it will improve the performance. Defaults to True.
    """
    os.makedirs(outputdir, exist_ok=True)
    if output_file_prefix is None:
        output_file_prefix = os.path.splitext(os.path.basename(input_file))[0]

    has_split = run_cmd('split --help').returncode

    if fix_N and has_split==0:
        with open(input_file, "r") as file:
            line = file.readline()
            Natoms = int(line.strip())
        output_file = os.path.join(outputdir, f"{output_file_prefix}.")
        
        cmd = f"split -l {Natoms+2} {input_file} {output_file} -d -a 5 --additional-suffix .xyz"
        print(f'Start spliting {input_file} by system split command...')
        run_cmd(cmd)
    else:
        print(f'Start spliting {input_file} by mdapy split method...')
        with open(input_file, "r") as file:
            frame = 0
            while True:
                line = file.readline()
                if not line:
                    break
                print(f"\rSaving frame {frame}", end="")
                n = int(line.strip())
                output_file = os.path.join(outputdir, f"{output_file_prefix}.{frame:0>5d}.xyz")
                with open(output_file, "w") as out_file:
                    out_file.write(line)
                    for _ in range(n + 1):
                        out_file.write(file.readline())
                frame += 1
        print('\n')


def split_dump(input_file, outputdir="res", output_file_prefix=None, fix_N=True):
    """This function can be used to split a dump trajectory into seperate dump files, which can be read by mp.System.

    Args:
        input_file (str): xyz filename.
        outputdir (str, optional): output folder. Defaults to 'res'.
        output_file_prefix (str, optional): output file prefix. If not given, mdapy will generate one based on the input_file. Defaults to None.
        fix_N (bool, optional): if the whole trajectory has the same atoms, set this to True, it will improve the performance. Defaults to True.
    """
    os.makedirs(outputdir, exist_ok=True)
    if output_file_prefix is None:
        output_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
    has_split = run_cmd('split --help').returncode
    if fix_N and has_split==0:
        with open(input_file, "r") as file:
            file.readline()
            file.readline()
            file.readline()
            line = file.readline()
            Natoms = int(line.strip())
        output_file = os.path.join(outputdir, f"{output_file_prefix}.")
        
        cmd = f"split -l {Natoms+9} {input_file} {output_file} -d -a 5 --additional-suffix .dump"
        print(f'Start spliting {input_file} by system split command...')
        run_cmd(cmd)
    else:
        print(f'Start spliting {input_file} by mdapy split method...')
        with open(input_file, "r") as file:
            frame = 0
            while True:
                output_file = os.path.join(outputdir, f"{output_file_prefix}.{frame:0>5d}.dump")
                header = []
                line = file.readline()
                if not line:
                    break
                print(f"\rSaving frame {frame}", end="")
                with open(output_file, "w") as out_file:
                    out_file.write(line)
                    header.append(line)
                    for _ in range(8):
                        line = file.readline()
                        if not line:
                            return
                        out_file.write(line)
                        header.append(line)

                n = int(header[3].strip())
                with open(output_file, "a") as out_file:
                    for _ in range(n):
                        out_file.write(file.readline())
                frame += 1
        print('\n')


def _check_repeat_nearest(pos, box, boundary):
    repeat = [1, 1, 1]
    box_length = [np.linalg.norm(box[i]) for i in range(3)]
    repeat_length = False
    for i in range(3):
        if boundary[i] == 1 and box_length[i] <= 10.0:
            repeat_length = True
    repeat_number = False
    if pos.shape[0] < 100 and sum(boundary) > 0:
        repeat_number = True

    if repeat_length or repeat_number:
        repeat = [1 if boundary[i] == 0 else 3 for i in range(3)]
        while np.prod(repeat) * pos.shape[0] < 100:
            for i in range(3):
                if boundary[i] == 1:
                    repeat[i] += 1

        for i in range(3):
            if boundary[i] == 1:
                while repeat[i] * box_length[i] < 10.0:
                    repeat[i] += 1
    return repeat


def _check_repeat_cutoff(box, boundary, rc, factor=4):
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
    inverse_box: ti.types.ndarray(element_dim=1)
):
    """This function is used to wrap particle positions into box considering periodic boundarys.

    Args:
        pos (ti.types.ndarray): (Nx3) particle position.

        box (ti.types.ndarray): (4x3) system box.

        boundary (ti.types.ndarray): boundary conditions, 1 is periodic and 0 is free boundary.
    """

    for i in range(pos.shape[0]):
        rij = pos[i] - box[3]
        n = rij[0] * inverse_box[0] + rij[1] * inverse_box[1] + rij[2] * inverse_box[2]
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
    pos_list: ti.types.ndarray(element_dim=1),
    box_list: ti.types.ndarray(element_dim=1),
    boundary: ti.types.vector(3, dtype=int),
    inverse_box_list: ti.types.ndarray(element_dim=1),
    image_p: ti.types.ndarray(element_dim=1),
    delta_list:ti.types.ndarray(element_dim=1),
):
    """This function is used to unwrap particle positions
     into box considering periodic boundarys without help of image_p.

    Args:
        pos_list (ti.types.ndarray): (Nframes x Nparticles x 3) particle position.

        box_list (ti.types.ndarray): (Nframesx4x3) system box list.

        boundary (ti.types.vector): boundary conditions, 1 is periodic and 0 is free boundary.

        inverse_box (ti.types.ndarray): (Nframesx3x3) inverse box list.

        image_p (ti.types.ndarray): (Nframes x Nparticles x 3) fill with 0.
    """

    Nframe = pos_list.shape[0]
    Natoms = pos_list.shape[1]

    for i, j in ti.ndrange(delta_list.shape[0], delta_list.shape[1]):
        n = delta_list[i, j][0] * inverse_box_list[i, 0] + delta_list[i, j][1] * inverse_box_list[i, 1] + delta_list[i, j][2] * inverse_box_list[i, 2]
    
        for k in ti.static(range(3)):
            if boundary[k] == 1:
                if n[k] <= -0.5:
                    image_p[i, j][k] += 1
                if n[k] >= 0.5:
                    image_p[i, j][k] -= 1

    ti.loop_config(serialize=True)
    for frame in range(1, Nframe):
        for i in range(Natoms):
            pos_list[frame, i] = pos_list[frame-1, i] + delta_list[frame-1, i]
            for j in ti.static(range(3)):
                pos_list[frame, i] += image_p[frame - 1, i][j] * box_list[frame-1, j]


def _unwrap_pos(pos_list, box_list, inverse_box_list, boundary):
    """This function is used to unwrap particle positions
     into box considering periodic boundarys.

    Args:
        pos_list (np.ndarray): (Nframes x Nparticles x 3) particle position.

        box (np.ndarray): (4x3) system box.

        boundary (list, optional): boundary conditions, 1 is periodic and 0 is free boundary. Defaults to [1, 1, 1].

    """

    boundary = ti.Vector(boundary)
    image_p = np.zeros_like(pos_list, np.int32)
    delta_list = pos_list[1:, :, :] - pos_list[:-1, :, :]

    _unwrap_pos_without_image_p(pos_list, box_list, boundary, inverse_box_list, image_p, delta_list)

    # for i in range(12):
    #     print(i, image_p[i][0])


def _init_vel(N, T, Mass=1.0, randomseed=10086):
    Boltzmann_Constant = 8.617385e-5
    np.random.seed(randomseed)
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

# Atomic masses are based on:
#
#   Meija, J., Coplen, T., Berglund, M., et al. (2016). Atomic weights of
#   the elements 2013 (IUPAC Technical Report). Pure and Applied Chemistry,
#   88(3), pp. 265-291. Retrieved 30 Nov. 2016,
#   from doi:10.1515/pac-2015-0305
#
# Standard atomic weights are taken from Table 1: "Standard atomic weights
# 2013", with the uncertainties ignored.
# For hydrogen, helium, boron, carbon, nitrogen, oxygen, magnesium, silicon,
# sulfur, chlorine, bromine and thallium, where the weights are given as a
# range the "conventional" weights are taken from Table 3 and the ranges are
# given in the comments.
# The mass of the most stable isotope (in Table 4) is used for elements
# where there the element has no stable isotopes (to avoid NaNs): Tc, Pm,
# Po, At, Rn, Fr, Ra, Ac, everything after Np
atomic_masses = np.array(
    [
        1.0,  # X
        1.008,  # H [1.00784, 1.00811]
        4.002602,  # He
        6.94,  # Li [6.938, 6.997]
        9.0121831,  # Be
        10.81,  # B [10.806, 10.821]
        12.011,  # C [12.0096, 12.0116]
        14.007,  # N [14.00643, 14.00728]
        15.999,  # O [15.99903, 15.99977]
        18.998403163,  # F
        20.1797,  # Ne
        22.98976928,  # Na
        24.305,  # Mg [24.304, 24.307]
        26.9815385,  # Al
        28.085,  # Si [28.084, 28.086]
        30.973761998,  # P
        32.06,  # S [32.059, 32.076]
        35.45,  # Cl [35.446, 35.457]
        39.948,  # Ar
        39.0983,  # K
        40.078,  # Ca
        44.955908,  # Sc
        47.867,  # Ti
        50.9415,  # V
        51.9961,  # Cr
        54.938044,  # Mn
        55.845,  # Fe
        58.933194,  # Co
        58.6934,  # Ni
        63.546,  # Cu
        65.38,  # Zn
        69.723,  # Ga
        72.630,  # Ge
        74.921595,  # As
        78.971,  # Se
        79.904,  # Br [79.901, 79.907]
        83.798,  # Kr
        85.4678,  # Rb
        87.62,  # Sr
        88.90584,  # Y
        91.224,  # Zr
        92.90637,  # Nb
        95.95,  # Mo
        97.90721,  # 98Tc
        101.07,  # Ru
        102.90550,  # Rh
        106.42,  # Pd
        107.8682,  # Ag
        112.414,  # Cd
        114.818,  # In
        118.710,  # Sn
        121.760,  # Sb
        127.60,  # Te
        126.90447,  # I
        131.293,  # Xe
        132.90545196,  # Cs
        137.327,  # Ba
        138.90547,  # La
        140.116,  # Ce
        140.90766,  # Pr
        144.242,  # Nd
        144.91276,  # 145Pm
        150.36,  # Sm
        151.964,  # Eu
        157.25,  # Gd
        158.92535,  # Tb
        162.500,  # Dy
        164.93033,  # Ho
        167.259,  # Er
        168.93422,  # Tm
        173.054,  # Yb
        174.9668,  # Lu
        178.49,  # Hf
        180.94788,  # Ta
        183.84,  # W
        186.207,  # Re
        190.23,  # Os
        192.217,  # Ir
        195.084,  # Pt
        196.966569,  # Au
        200.592,  # Hg
        204.38,  # Tl [204.382, 204.385]
        207.2,  # Pb
        208.98040,  # Bi
        208.98243,  # 209Po
        209.98715,  # 210At
        222.01758,  # 222Rn
        223.01974,  # 223Fr
        226.02541,  # 226Ra
        227.02775,  # 227Ac
        232.0377,  # Th
        231.03588,  # Pa
        238.02891,  # U
        237.04817,  # 237Np
        244.06421,  # 244Pu
        243.06138,  # 243Am
        247.07035,  # 247Cm
        247.07031,  # 247Bk
        251.07959,  # 251Cf
        252.0830,  # 252Es
        257.09511,  # 257Fm
        258.09843,  # 258Md
        259.1010,  # 259No
        262.110,  # 262Lr
        267.122,  # 267Rf
        268.126,  # 268Db
        271.134,  # 271Sg
        270.133,  # 270Bh
        269.1338,  # 269Hs
        278.156,  # 278Mt
        281.165,  # 281Ds
        281.166,  # 281Rg
        285.177,  # 285Cn
        286.182,  # 286Nh
        289.190,  # 289Fl
        289.194,  # 289Mc
        293.204,  # 293Lv
        293.208,  # 293Ts
        294.214,  # 294Og
    ]
)

atomic_masses.flags.writeable = False


if __name__ == '__main__':
    input_file = 'train.xyz'
    print(os.path.splitext(os.path.basename(input_file)))
    timer(split_xyz)(input_file)

    

