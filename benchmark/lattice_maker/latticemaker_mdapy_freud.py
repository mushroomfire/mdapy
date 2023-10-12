import mdapy as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mdapy import pltset, cm2inch
import freud


def mdapy_maker(num):
    FCC = mp.LatticeMaker(3.615, "FCC", num, 100, 100)
    FCC.compute()


def lattice_average_time(ave_num=3):
    time_list = []
    print("*" * 30)

    for num in [5, 25, 45, 65, 85, 105, 125]:
        mdapy_t_cpu, mdapy_t_gpu, mdapy_t_serial, freud_t = 0.0, 0.0, 0.0, 0.0

        mp.init(arch="gpu", device_memory_GB=6.5)
        mdapy_maker(num)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy gpu...")

            start = time()
            mdapy_maker(num)
            end = time()
            mdapy_t_gpu += end - start

        mp.init(arch="cpu")
        mdapy_maker(num)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu parallel...")

            start = time()
            mdapy_maker(num)
            end = time()
            mdapy_t_cpu += end - start

        mp.init(arch="cpu", cpu_max_num_threads=1)
        mdapy_maker(num)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu serial...")

            start = time()
            mdapy_maker(num)
            end = time()
            mdapy_t_serial += end - start

        for turn in range(ave_num):
            print(f"Running {turn} turn in freud...")

            start = time()
            n_repeats = (num, 100, 100)
            uc = freud.data.UnitCell.fcc()
            box, points = uc.generate_system(n_repeats, scale=3.615)
            end = time()
            freud_t += end - start

        time_list.append(
            [
                num * 40000,
                mdapy_t_cpu / ave_num,
                mdapy_t_gpu / ave_num,
                mdapy_t_serial / ave_num,
                freud_t / ave_num,
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

    plt.plot(x, y, "o-", label=f"mdapy-cpu-parallel,")

    y = time_list[:, 2]
    plt.plot(x, y, "o-", label=f"mdapy-gpu")

    y = time_list[:, 3]
    plt.plot(x, y, "o-", label=f"mdapy-cpu-serial")

    y = time_list[:, 4]
    plt.plot(x, y, "o-", label=f"freud")

    if title is not None:
        plt.title(title, fontsize=12)
    plt.legend()
    plt.xlabel("Number of atoms ($\mathregular{10^%d}$)" % exp_max)
    plt.ylabel("Time (s)")
    if savefig:
        plt.savefig("lattice.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == "__main__":
    time_list = lattice_average_time(ave_num=3)
    np.savetxt("time_list.txt", time_list)
    time_list = np.loadtxt("time_list.txt")
    plot(time_list, "Build lattice structure")
