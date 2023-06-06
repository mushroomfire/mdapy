# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import mdapy as mp
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

mp.init("cpu")
print("Test Polygon class...")
box = np.array([[-100.0, 100.0], [0, 200.0], [0, 200.0]])  # create a box
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
poly.compute()

system = mp.System("GRA-Metal-FCC-20-1.dump")
print(f"Atom number: {system.N}.")
print("Test Neighbor class...")
system.build_neighbor(rc=5.0)
print("Test Cluster class...")
system.cal_cluster_analysis(rc=4.0)
print("Test Voronoi class...")
system.cal_voronoi_volume()
print("Test RDF class...")
system.cal_pair_distribution(rc=5.0)
print("RDF is:")
print(system.PairDistribution.g_total[:10])
print("Test PTM class...")
system.cal_polyhedral_template_matching()
print("system data is:")
print(system.data.head())
print("All c++ classes are tested passed.")
os.remove("GRA-Metal-FCC-20-1.dump")
