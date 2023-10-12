import mdapy as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mdapy import pltset, cm2inch
import freud


def mdapy_bond_order(pos, box, qlist=[4, 6, 8]):
    bond = mp.SteinhardtBondOrientation(pos, box, qlist=qlist)
    bond.compute()


def order_average_time(ave_num=3):
    time_list = []
    print("*" * 30)

    for num in [5, 25, 45, 65, 85, 105, 125]:
        FCC = mp.LatticeMaker(3.615, "FCC", num, 100, 100)
        FCC.compute()
        pos, box, N = FCC.pos, FCC.box, FCC.N
        print(f"Build {N} atoms...")
        mdapy_t_cpu, mdapy_t_gpu, freud_t = 0.0, 0.0, 0.0

        mp.init(arch="gpu", device_memory_GB=6.5)
        mdapy_bond_order(pos, box)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy gpu...")

            start = time()
            mdapy_bond_order(pos, box)
            end = time()
            mdapy_t_gpu += end - start

        mp.init(arch="cpu")
        mdapy_bond_order(pos, box)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu...")

            start = time()
            mdapy_bond_order(pos, box)
            end = time()
            mdapy_t_cpu += end - start

        for turn in range(ave_num):
            print(f"Running {turn} turn in freud...")

            f_box = freud.box.Box(Lx=box[0, 0], Ly=box[1, 1], Lz=box[2, 2])

            shift_pos = (
                pos
                - np.min(pos, axis=0)
                - np.array([np.linalg.norm(box[i]) for i in range(3)]) / 2
            )
            start = time()
            ql = freud.order.Steinhardt(l=[4, 6, 8])
            ql.compute((f_box, shift_pos), {"num_neighbors": 12, "exclude_ii": True})
            end = time()
            freud_t += end - start

        time_list.append(
            [N, mdapy_t_cpu / ave_num, mdapy_t_gpu / ave_num, freud_t / ave_num]
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

    popt = np.polyfit(x, y, 1)
    plt.plot(x, np.poly1d(popt)(x), c=colorlist[0])
    plt.plot(x, y, "o", label=f"mdapy-cpu, k={popt[0]:.1f}")

    y = time_list[:, 2]
    popt = np.polyfit(x, y, 1)
    plt.plot(x, np.poly1d(popt)(x), c=colorlist[1])
    plt.plot(x, y, "o", label=f"mdapy-gpu, k={popt[0]:.1f}")

    y = time_list[:, 3]
    popt = np.polyfit(x, y, 1)
    plt.plot(x, np.poly1d(popt)(x), c=colorlist[2])
    plt.plot(x, y, "o", label=f"freud, k={popt[0]:.1f}")

    if title is not None:
        plt.title(title, fontsize=12)
    plt.legend()
    plt.xlabel("Number of atoms ($\mathregular{10^%d}$)" % exp_max)
    plt.ylabel("Time (s)")
    if savefig:
        plt.savefig("order.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == "__main__":
    mp.init()
    time_list = order_average_time(ave_num=3)
    np.savetxt("time_list.txt", time_list)
    time_list = np.loadtxt("time_list.txt")
    plot(time_list, "Order parameter")
