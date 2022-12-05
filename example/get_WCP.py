# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import mdapy as mp

mp.init(arch="cpu")

system = mp.System("CoCuFeNiPd-4M.data")
system.cal_warren_cowley_parameter()  # calculate WCP parameter
fig, ax = system.WarrenCowleyParameter.plot(
    elements_list=["Co", "Cu", "Fe", "Ni", "Pd"]
)  # plot WCP matrix
fig.savefig("WCP.png", dpi=300, bbox_inches="tight", transparent=True)
