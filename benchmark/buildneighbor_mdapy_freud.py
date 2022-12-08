import mdapy as mp
import freud
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mdapy import pltset, cm2inch, timer


@timer
def test_neighbor_average_time(ave_num=3, kind="cpu", check=False, cal_freud=True):
    assert kind in ["cpu", "gpu"]
    time_list = []
    print("*" * 30)
    for num in [5, 25, 45, 65, 85, 105, 125]:
        FCC = mp.LatticeMaker(3.615, "FCC", num, 100, 100)
        FCC.compute()
        print(f"Build {FCC.N} atoms...")
        freud_t, mdapy_t = 0.0, 0.0
        for turn in range(ave_num):
            if cal_freud:
                print(f"Running {turn} turn in freud...")
                box = freud.box.Box(
                    Lx=FCC.box[0, 1], Ly=FCC.box[1, 1], Lz=FCC.box[2, 1]
                )
                shift_pos = FCC.pos - np.min(FCC.pos, axis=0) - (FCC.box[:, 1]) / 2
                start = time()
                aq = freud.locality.AABBQuery(box, shift_pos)
                nlist = aq.query(
                    shift_pos, {"r_max": 5.0, "exclude_ii": True}
                ).toNeighborList()
                end = time()
                freud_t += end - start

            print(f"Running {turn} turn in mdapy...")
            start = time()
            neigh = mp.Neighbor(FCC.pos, FCC.box, 5.0, max_neigh=50)
            neigh.compute()
            end = time()
            mdapy_t += end - start
            if check:
                print(f"Checking results of {turn} turn...")
                assert (nlist.neighbor_counts == neigh.neighbor_number).all()
        time_list.append([FCC.N, freud_t / ave_num, mdapy_t / ave_num])
        print("*" * 30)
    time_list = np.array(time_list)
    if kind == "cpu":
        np.savetxt(
            "time_list_cpu_neighbor.txt",
            time_list,
            delimiter=" ",
            header="N freud mdapy",
        )
    elif kind == "gpu":
        np.savetxt(
            "time_list_gpu_neighbor.txt",
            time_list,
            delimiter=" ",
            header="N freud mdapy",
        )
    return time_list


def plot(time_list, title=None, kind="cpu", save_fig=True):

    assert kind in ["cpu", "gpu", "cpu-gpu"]
    if kind in ["cpu", "gpu"]:
        assert time_list.shape[1] == 3
    else:
        assert time_list.shape[1] == 4
    pltset()
    colorlist = [i["color"] for i in list(plt.rcParams["axes.prop_cycle"])]
    fig = plt.figure(figsize=(cm2inch(10), cm2inch(8)), dpi=150)
    plt.subplots_adjust(left=0.16, bottom=0.165, top=0.92, right=0.95)
    N_max = time_list[-1, 0]
    exp_max = int(np.log10(N_max))
    x, y = time_list[:, 0] / 10**exp_max, time_list[:, 1]

    popt = np.polyfit(x, y, 1)
    plt.plot(x, np.poly1d(popt)(x), c=colorlist[0])
    plt.plot(x, y, "o", label=f"freud, k={popt[0]:.1f}")

    if kind == "cpu-gpu":
        y1 = time_list[:, 2]
        popt = np.polyfit(x, y1, 1)
        plt.plot(x, np.poly1d(popt)(x), c=colorlist[1])
        plt.plot(x, y1, "o", label=f"mdapy-cpu, k={popt[0]:.1f}")

        y2 = time_list[:, 3]
        popt = np.polyfit(x, y2, 1)
        plt.plot(x, np.poly1d(popt)(x), c=colorlist[2])
        plt.plot(x, y2, "o", label=f"mdapy-gpu, k={popt[0]:.1f}")
    else:
        y1 = time_list[:, 2]
        popt = np.polyfit(x, y1, 1)
        plt.plot(x, np.poly1d(popt)(x), c=colorlist[1])
        plt.plot(x, y1, "o", label=f"mdapy-{kind}, k={popt[0]:.1f}")

    if title is not None:
        plt.title(title, fontsize=12)
    plt.legend()
    plt.xlabel("Number of atoms ($\mathregular{10^%d}$)" % exp_max)
    plt.ylabel("Time (s)")
    if save_fig:
        plt.savefig(
            "buildneighbor_mdapy_freud.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    plt.show()


if __name__ == "__main__":

    mp.init("gpu", device_memory_GB=6.0)
    time_list_gpu = test_neighbor_average_time(
        3, kind="gpu", cal_freud=True, check=False
    )
    mp.init("cpu")
    time_list_cpu = test_neighbor_average_time(3, kind="cpu", cal_freud=False)
    time_list_cpu = np.loadtxt("time_list_cpu_neighbor.txt")
    time_list_gpu = np.loadtxt("time_list_gpu_neighbor.txt")
    time_list = np.zeros((time_list_gpu.shape[0], 4))
    time_list[:, 0] = time_list_cpu[:, 0]
    time_list[:, 1] = time_list_gpu[:, 1]
    time_list[:, 2] = time_list_cpu[:, 2]
    time_list[:, 3] = time_list_gpu[:, 2]
    plot(time_list, title="Build neighbor", kind="cpu-gpu", save_fig=True)
