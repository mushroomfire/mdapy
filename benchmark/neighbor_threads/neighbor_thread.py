import mdapy as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mdapy import pltset, cm2inch


def mdapy_neigh(pos, box, rc=5.0, max_neigh=50):
    neigh = mp.Neighbor(pos=pos, box=box, rc=rc, max_neigh=max_neigh)
    neigh.compute()


def test_neighbor_thread_time(ave_num=3, rc=5.0):
    num_list = [25, 50, 75, 100, 125]
    thread_list = [1, 2, 4, 8, 12, 24]
    time_list_cpu = []
    time_list_gpu = []
    for num in num_list:
        FCC = mp.LatticeMaker(3.615, "FCC", num, 100, 100)
        FCC.compute()
        pos = FCC.pos
        box = FCC.box
        print(f"Build {FCC.N} atoms...")
        print("*" * 30)
        for thread in thread_list:
            mp.init(cpu_max_num_threads=thread)
            mdapy_neigh(pos, box, rc=rc, max_neigh=50)
            mdapy_t_cpu = 0.0
            for turn in range(ave_num):
                print(f"Running {turn} turn in mdapy with {thread} threads...")
                start = time()
                mdapy_neigh(pos, box, rc=rc, max_neigh=50)
                end = time()
                mdapy_t_cpu += end - start

            time_list_cpu.append([num, thread, mdapy_t_cpu / ave_num])
            print("*" * 30)

        mp.init(arch="gpu", device_memory_GB=6.0)
        mdapy_neigh(pos, box, rc=rc, max_neigh=50)
        mdapy_t_gpu = 0.0
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy with GPU...")
            start = time()
            mdapy_neigh(pos, box, rc=rc, max_neigh=50)
            end = time()
            mdapy_t_gpu += end - start

        time_list_gpu.append([num, mdapy_t_gpu / ave_num])
        print("*" * 30)

    return np.array(time_list_cpu), np.array(time_list_gpu)


def plot(time_list_cpu, time_list_gpu, title=None, savefig=True):
    pltset(fontkind="Times New Roman")
    fig = plt.figure(figsize=(cm2inch(10), cm2inch(8)), dpi=150)
    plt.subplots_adjust(left=0.16, bottom=0.165, top=0.92, right=0.95)
    for i, j in enumerate([25, 50, 75, 100, 125]):
        data = time_list_cpu[time_list_cpu[:, 0] == j]
        colorlist = [i["color"] for i in list(plt.rcParams["axes.prop_cycle"])]
        plt.plot(data[:, 1], data[:, 2], "o-", c=colorlist[i])

        plt.plot(60, time_list_gpu[i, 1], "s", c=colorlist[i])
        plt.text(
            0.14,
            data[0, 2],
            r"$\mathregular{%d \times 10^6}$ atoms" % (j * 100 * 100 * 4 / 1000000),
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.text(18, 20, "CPU")
    plt.text(44, 20, "GPU")

    plt.xlim(0.1, 100)
    plt.ylim(0.7, 30)

    plt.fill_between([0.1, 36], 30, 0.7, alpha=0.2)
    plt.fill_between([36, 100], 30, 0.7, alpha=0.2)

    plt.xlabel("Number of threads")
    plt.ylabel("Time (s)")
    if title is not None:
        plt.title(title, fontsize=12)
    if savefig:
        plt.savefig(
            "neighbor_thread.png", dpi=300, bbox_inches="tight", transparent=True
        )
    plt.show()


if __name__ == "__main__":
    # mp.init()
    # time_list_cpu, time_list_gpu = test_neighbor_thread_time(ave_num=3)
    # np.savetxt('time_list_cpu.txt', time_list_cpu)
    # np.savetxt('time_list_gpu.txt', time_list_gpu)
    time_list_cpu = np.loadtxt("time_list_cpu.txt")
    time_list_gpu = np.loadtxt("time_list_gpu.txt")
    plot(time_list_cpu, time_list_gpu, "Build neighbor")
