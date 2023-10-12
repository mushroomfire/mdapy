import mdapy as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mdapy import pltset, cm2inch

import MDAnalysis
import MDAnalysis.analysis.msd
import gsd

import mdtraj
import pyfftw
import os


Nframe = 1000


def get_u_mdanalysis(N, pos_list):
    def create_frame(i, N, pos_list):
        frame = gsd.hoomd.Frame()
        frame.configuration.step = i
        frame.particles.N = N
        frame.particles.position = pos_list[i]
        return frame

    f = gsd.hoomd.open(name="MSD.gsd", mode="wb")
    f.extend(create_frame(i, N, pos_list) for i in range(Nframe))
    f.close()

    topology = MDAnalysis.core.topology.Topology(n_atoms=N)
    u = MDAnalysis.Universe(topology, "MSD.gsd", dt=1.0)
    return u


def msd_average_time(ave_num=3):
    time_list = []
    print("*" * 30)

    for N in [100, 500, 1000, 2500, 5000, 7500, 10000]:
        pos_list = np.cumsum(
            np.random.choice([-1.0, 1.0], size=(Nframe, N, 3)), axis=0
        ) * np.sqrt(
            2
        )  # Generate a random walk data

        print(f"Create {N} atoms...")
        mdapy_t_cpu, mdapy_single_t, mdanalysis_t, mdtraj_t = 0.0, 0.0, 0.0, 0.0

        pyfftw.config.NUM_THREADS = 48
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu parallel...")

            start = time()
            mp_msd = mp.MeanSquaredDisplacement(pos_list)
            mp_msd.compute()
            end = time()
            mdapy_t_cpu += end - start

        pyfftw.config.NUM_THREADS = 1
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu serial...")
            start = time()
            mp_msd = mp.MeanSquaredDisplacement(pos_list)
            mp_msd.compute()
            end = time()
            mdapy_single_t += end - start

        u = get_u_mdanalysis(N, pos_list)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdanalysis...")
            start = time()
            MDAnalysis.analysis.msd.EinsteinMSD(
                u, select="all", msd_type="xyz", fft=True
            ).run()
            end = time()
            mdanalysis_t += end - start
        os.system("rm MSD.gsd")

        for turn in range(ave_num):
            print(f"Running {turn} turn in mdtraj...")
            mdt = mdtraj.Trajectory(pos_list, topology=None)
            start = time()
            mdtraj.rmsd(mdt, mdt) ** 2
            end = time()
            mdtraj_t += end - start

        time_list.append(
            [
                N,
                mdapy_t_cpu / ave_num,
                mdapy_single_t / ave_num,
                mdanalysis_t / ave_num,
                mdtraj_t / ave_num,
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
    plt.plot(x, y, "o-", label=f"mdapy-cpu-serial")

    y = time_list[:, 3]
    plt.plot(x, y, "o-", label=f"mdanalysis")

    y = time_list[:, 4]
    plt.plot(x, y, "o-", label=f"mdtraj")

    if title is not None:
        plt.title(title, fontsize=12)
    plt.legend()
    plt.xlabel("Number of atoms ($\mathregular{10^%d}$)" % exp_max)
    plt.ylabel("Time (s)")
    if savefig:
        plt.savefig("results.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == "__main__":
    # mp.init()
    # time_list = msd_average_time(ave_num=3)
    # np.savetxt('time_list.txt', time_list)
    time_list = np.loadtxt("time_list.txt")
    plot(time_list, "MSD for 1000 frames")
