import mdapy as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt

plt.switch_backend("tkagg")
from mdapy import pltset, cm2inch

from ovito.data import CutoffNeighborFinder, DataCollection, Particles, SimulationCell
from ovito.pipeline import Pipeline, StaticSource


def modify(
    frame: int, data: DataCollection, cutoff=5.0, sigma=0.2, use_local_density=False
):
    # Validate input parameters:
    assert cutoff > 0.0
    assert sigma > 0.0 and sigma < cutoff

    # Show message in OVITO's status bar:
    yield "Calculating local entropy"

    # Overall particle density:
    global_rho = data.particles.count / data.cell.volume

    # Initialize neighbor finder:
    finder = CutoffNeighborFinder(cutoff, data)

    # Create output array for local entropy values
    local_entropy = np.empty(data.particles.count)

    # Number of bins used for integration:
    nbins = int(cutoff / sigma) + 1

    # Table of r values at which the integrand will be computed:
    r = np.linspace(0.0, cutoff, num=nbins)
    rsq = r**2

    # Precompute normalization factor of g_m(r) function:
    prefactor = rsq * (4 * np.pi * global_rho * np.sqrt(2 * np.pi * sigma**2))
    prefactor[0] = prefactor[1]  # Avoid division by zero at r=0.

    # Iterate over input particles:
    for particle_index in range(data.particles.count):
        yield particle_index / data.particles.count

        # Get distances r_ij of neighbors within the cutoff range.
        r_ij = finder.neighbor_distances(particle_index)

        # Compute differences (r - r_ji) for all {r} and all {r_ij} as a matrix.
        r_diff = np.expand_dims(r, 0) - np.expand_dims(r_ij, 1)

        # Compute g_m(r):
        g_m = np.sum(np.exp(-(r_diff**2) / (2.0 * sigma**2)), axis=0) / prefactor

        # Estimate local atomic density by counting the number of neighbors within the
        # spherical cutoff region:
        if use_local_density:
            local_volume = 4 / 3 * np.pi * cutoff**3
            rho = len(r_ij) / local_volume
            g_m *= global_rho / rho
        else:
            rho = global_rho

        # Compute integrand:
        integrand = np.where(g_m >= 1e-10, (g_m * np.log(g_m) - g_m + 1.0) * rsq, rsq)

        # Integrate from 0 to cutoff distance:
        local_entropy[particle_index] = -2.0 * np.pi * rho * np.trapz(integrand, r)

    # Output the computed per-particle entropy values to the data pipeline.
    data.particles_.create_property("Entropy", data=local_entropy)


def mdapy_entropy(pos, box):
    neigh = mp.Neighbor(pos, box, 5.0, max_neigh=50)
    neigh.compute()
    vol = np.product(box[:, 1])
    entropy = mp.AtomicEntropy(
        vol, neigh.verlet_list, neigh.distance_list, 5.0, sigma=0.2
    )
    entropy.compute()


def entropy_average_time(ave_num=3):
    time_list = []
    print("*" * 30)
    for num in [10, 15, 20, 25, 30, 35, 40, 45, 50]:
        FCC = mp.LatticeMaker(3.615, "FCC", num, 50, 50)
        FCC.compute()
        pos, box, N = FCC.pos, FCC.box, FCC.N
        print(f"Build {N} atoms...")
        mdapy_t_cpu, mdapy_t_gpu, mdapy_t_serial, ovito_t = 0.0, 0.0, 0.0, 0.0
        for turn in range(ave_num):
            print(f"Running {turn} turn in ovito...")

            particles = Particles()
            data = DataCollection()
            data.objects.append(particles)
            particles.create_property("Position", data=pos)
            cell = SimulationCell(pbc=(True, True, True))
            cell[...] = np.c_[box[:-1], box[-1]]
            data.objects.append(cell)

            pipeline = Pipeline(source=StaticSource(data=data))
            start = time()
            pipeline.modifiers.append(modify)
            pipeline.compute()
            end = time()
            ovito_t += end - start

        mp.init(arch="cpu")
        mdapy_entropy(pos, box)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu parallel...")

            start = time()
            mdapy_entropy(pos, box)
            end = time()
            mdapy_t_cpu += end - start

        mp.init(arch="gpu", device_memory_GB=6.0)
        mdapy_entropy(pos, box)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy gpu...")

            start = time()
            mdapy_entropy(pos, box)
            end = time()
            mdapy_t_gpu += end - start

        mp.init(arch="cpu", cpu_max_num_threads=1)
        mdapy_entropy(pos, box)
        for turn in range(ave_num):
            print(f"Running {turn} turn in mdapy cpu serial...")

            start = time()
            mdapy_entropy(pos, box)
            end = time()
            mdapy_t_serial += end - start

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
        plt.savefig("entropy.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()


if __name__ == "__main__":
    mp.init()
    time_list = entropy_average_time(ave_num=3)
    np.savetxt("time_list.txt", time_list)
    time_list = np.loadtxt("time_list.txt")
    plot(time_list, "Calculate entropy")
