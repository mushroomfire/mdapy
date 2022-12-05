# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import mdapy as mp
import numpy as np

mp.init("cpu")

box = np.array([[0, 800.0], [0, 200.0], [0, 200.0]])  # create a box
seednumber = 20  # create 20 seeds to generate the voronoi polygon
metal_lattice_constant = 3.615  # lattice constant of metallic matrix
metal_lattice_type = "FCC"  # lattice type of metallic matrix
randomseed = 1  # control the crystalline orientations per grains
add_graphene = True  # use graphen as grain boundary
poly = mp.CreatePolycrystalline(
    box,
    seednumber,
    metal_lattice_constant,
    metal_lattice_type,
    randomseed=randomseed,
    add_graphene=add_graphene,
    gra_overlap_dis=1.2,
)
poly.compute()  # generate a polycrystalline with graphene boundary
