# Copyright (c) 2022, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import mdapy as mp

mp.init("cpu")

dump_list = [
    f"melt.{i}.dump" for i in range(100)
]  # obtain all the dump filenames in a list
MS = mp.MultiSystem(dump_list)  # read all the dump file to generate a MultiSystem class
MS.cal_mean_squared_displacement()  # calculate the mean squared displacement
fig, ax = MS.MSD.plot()  # one can plot the MSD per frame
fig.savefig("MSD.png", dpi=300, bbox_inches="tight", transparent=True)
MS.cal_lindemann_parameter()  # calculate the lindemann index
fig, ax = MS.Lindemann.plot()  # one can plot lindemann index per frame
fig.savefig("Lindemann_index.png", dpi=300, bbox_inches="tight", transparent=True)
MS.write_dumps()  # save results to a serials of dump files
