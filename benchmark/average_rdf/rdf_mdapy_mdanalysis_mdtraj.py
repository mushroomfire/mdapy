import mdapy as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mdapy import pltset, cm2inch

import MDAnalysis
import MDAnalysis.analysis.rdf
import gsd

import mdtraj

trajectory_filename = "rdf_benchmark.gsd"
mm = gsd.hoomd.open(trajectory_filename)
topology = MDAnalysis.core.topology.Topology(n_atoms=mm[0].particles.position.shape[0])
u = MDAnalysis.Universe(topology, trajectory_filename, dt=1.0)
mdt = mdtraj.load(trajectory_filename)


def average_rdf_mdapy(rc, nbin=75, mm=mm):
    g = []
    for i in range(len(mm)):
        pos = mm[i].particles.position
        box = mm[i].configuration.box.reshape(2, 3)[::-1].T
        box -= box[0, 1] / 2
        rho = pos.shape[0] / np.product(box[:, 1] - box[:, 0])
        max_neigh = int(rho * 4 / 3 * np.pi * rc**3) + 80
        neigh = mp.Neighbor(pos=pos, box=box, rc=rc, max_neigh=max_neigh)
        neigh.compute()
        mrdf = mp.PairDistribution(
            neigh.rc,
            nbin,
            rho,
            neigh.verlet_list,
            neigh.distance_list,
            neigh.neighbor_number,
        )
        mrdf.compute()
        g.append(mrdf.g_total)
    g = np.sum(np.array(g), axis=0) / len(mm)
    return g


def rdf_average_time(ave_num=3):
    time_list = []
    print("*" * 30)

    for rc in range(1, 8):
        print(f"rc is {rc}...")
        mdapy_t_cpu, mdapy_t_gpu, mdapy_single_t, mdanalysis_t, mdtraj_t = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        mp.init(arch="gpu", device_memory_GB=6.5)
        average_rdf_mdapy(rc)  # Avoid the JIT time
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy gpu...")

            start = time()
            average_rdf_mdapy(rc)
            end = time()
            mdapy_t_gpu += end - start

        mp.init(arch="cpu")
        average_rdf_mdapy(rc)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu parallel...")

            start = time()
            average_rdf_mdapy(rc)
            end = time()
            mdapy_t_cpu += end - start

        mp.init(arch="cpu", cpu_max_num_threads=1)
        average_rdf_mdapy(rc)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu serial...")
            start = time()
            average_rdf_mdapy(rc)
            end = time()
            mdapy_single_t += end - start

        for turn in range(ave_num):
            print(f"Running {turn} turn in mdanalysis...")
            start = time()
            rdf = MDAnalysis.analysis.rdf.InterRDF(
                g1=u.atoms, g2=u.atoms, nbins=75, range=(0.0001, rc)
            ).run()
            end = time()
            mdanalysis_t += end - start

        for turn in range(ave_num):
            print(f"Running {turn} turn in mdtraj...")
            start = time()
            pair = mdt.top.select_pairs("name A", "name A")
            mdtraj.compute_rdf(mdt, pair, (0.0, rc), n_bins=75)
            end = time()
            mdtraj_t += end - start

        time_list.append(
            [
                rc,
                mdapy_t_cpu / ave_num,
                mdapy_t_gpu / ave_num,
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

    x, y = time_list[:, 0], time_list[:, 1]
    plt.plot(x, y, "o-", label=f"mdapy-cpu-parallel")

    y = time_list[:, 2]
    plt.plot(x, y, "o-", label=f"mdapy-gpu")

    y = time_list[:, 3]
    plt.plot(x, y, "o-", label=f"mdapy-cpu-serial")

    y = time_list[:, 4]
    plt.plot(x, y, "o-", label=f"mdanalysis")

    y = time_list[:, 5]
    plt.plot(x, y, "o-", label=f"mdtraj")

    plt.ylim(-3, 60)

    if title is not None:
        plt.title(title, fontsize=12)
    plt.legend()
    plt.xlabel("$\mathregular{r_c}$ ($\mathregular{\AA}$)")
    plt.ylabel("Time (s)")
    if savefig:
        plt.savefig("results.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == "__main__":
    mp.init()
    time_list = rdf_average_time(ave_num=3)
    np.savetxt("time_list.txt", time_list)
    time_list = np.loadtxt("time_list.txt")
    plot(time_list, "RDF for 5 frames, 15625 atoms")
