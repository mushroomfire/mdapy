import mdapy as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt

plt.switch_backend("tkagg")
from mdapy import pltset, cm2inch

from ovito.data import DataCollection, Particles, SimulationCell
from ovito.modifiers import (
    PolyhedralTemplateMatchingModifier,
    IdentifyFCCPlanarFaultsModifier,
)
from ovito.pipeline import Pipeline, StaticSource


def mdapy_sftb(structure_type, verlet_list):
    sftb = mp.IdentifySFTBinFCC(structure_type, verlet_list)
    sftb.compute()


def sftb_average_time(ave_num=3):
    time_list = []
    print("*" * 30)

    for num in [10, 15, 20, 25, 30, 35, 40, 45, 50]:  # 5, 25, 45, 65, 85, 105
        FCC = mp.LatticeMaker(3.615, "HCP", num, 50, 50)
        FCC.compute()
        pos = FCC.pos
        box = FCC.box
        N = FCC.N
        print(f"Build {N} atoms...")
        mdapy_t_cpu, ovito_t, mdapy_t_serial, mdapy_t_gpu = 0.0, 0.0, 0.0, 0.0

        ptm = mp.PolyhedralTemplateMatching(pos, box, return_verlet=True)
        ptm.compute()
        structure_type = np.array(ptm.output[:, 0], int)
        verlet_list = np.ascontiguousarray(
            ptm.ptm_indices[structure_type == 2][:, 1:13]
        )

        mp.init(arch="cpu")
        mdapy_sftb(structure_type, verlet_list)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu parallel...")
            start = time()
            mdapy_sftb(structure_type, verlet_list)
            end = time()
            mdapy_t_cpu += end - start

        mp.init(arch="cpu", cpu_max_num_threads=1)
        mdapy_sftb(structure_type, verlet_list)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu serial...")
            start = time()
            mdapy_sftb(structure_type, verlet_list)
            end = time()
            mdapy_t_serial += end - start

        mp.init(arch="gpu", device_memory_GB=6.0)
        mdapy_sftb(structure_type, verlet_list)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy gpu...")
            start = time()
            mdapy_sftb(structure_type, verlet_list)
            end = time()
            mdapy_t_serial += end - start

        particles = Particles()
        data = DataCollection()
        data.objects.append(particles)
        particles.create_property("Position", data=pos)
        cell = SimulationCell(pbc=(True, True, True))
        cell[...] = [
            [box[0, 1], 0, 0, box[0, 0]],
            [0, box[1, 1], 0, box[1, 0]],
            [0, 0, box[2, 1], box[2, 0]],
        ]
        data.objects.append(cell)
        pipeline = Pipeline(source=StaticSource(data=data))
        pipeline.modifiers.append(
            PolyhedralTemplateMatchingModifier(
                output_orientation=True, output_interatomic_distance=True
            )
        )
        pipeline.compute()
        for turn in range(ave_num):
            print(f"Running {turn} turn in ovito...")

            start = time()
            pipeline.modifiers.append(IdentifyFCCPlanarFaultsModifier())
            pipeline.compute()
            end = time()
            ovito_t += end - start

        time_list.append(
            [
                N,
                mdapy_t_cpu / ave_num,
                mdapy_t_gpu / ave_num,
                mdapy_t_serial / ave_num,
                ovito_t / ave_num,
            ]
        )
        print("*" * 30)
    time_list = np.array(time_list)

    return time_list


def plot(time_list, title=None, savefig=True):
    pltset(fontkind="Times New Roman")
    colorlist = [i["color"] for i in list(plt.rcParams["axes.prop_cycle"])]
    fig = plt.figure(figsize=(cm2inch(10), cm2inch(8)), dpi=150)
    plt.subplots_adjust(left=0.16, bottom=0.165, top=0.92, right=0.95)
    N_max = time_list[-1, 0]
    exp_max = int(np.log10(N_max))
    x, y = time_list[:, 0] / 10**exp_max, time_list[:, 1]

    plt.plot(x, y, "o-", label=f"mdapy-cpu-parallel")

    y = time_list[:, 2]
    plt.plot(x, y, "o-", label=f"mdapy-gpu")

    y = time_list[:, 3]
    plt.plot(x, y, "o-", label=f"mdapy-cpu-serial")

    y = time_list[:, 4]
    plt.plot(x, y, "o-", label=f"ovito")

    if title is not None:
        plt.title(title, fontsize=12)
    plt.legend()
    plt.xlabel("Number of atoms ($\mathregular{10^%d}$)" % exp_max)
    plt.ylabel("Time (s)")
    if savefig:
        plt.savefig("SFTB.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == "__main__":
    mp.init()
    time_list = sftb_average_time(ave_num=3)
    np.savetxt("time_list.txt", time_list)
    time_list = np.loadtxt("time_list.txt")
    plot(time_list, "Identify SFs and TBs")
