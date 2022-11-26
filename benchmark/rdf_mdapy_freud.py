import mdapy as mp
import freud
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mdapy.plot.pltset import pltset, cm2inch
from mdapy.tools.timer import timer

def plot_rdf(pair, rdf, r_max):
    fig = plt.figure(figsize=(cm2inch(10), cm2inch(7)), dpi=150)
    plt.subplots_adjust(bottom=0.22, top=0.95)

    plt.plot(pair.r, pair.g_total, lw=1., label = 'mdapy')
    plt.plot(rdf.bin_centers, rdf.rdf, '--', lw=1., label = 'freud')

    plt.legend()

    plt.xlim(0, r_max)

    plt.xlabel('r ($\mathregular{\AA}$)')
    plt.ylabel('g(r)')
    plt.show()

@timer
def test_rdf_average_time(ave_num=3, bins=200, r_max = 5., check_freud=False, check_plot=False):
    time_list = []
    print('*'*30)
    for num in [5, 10, 15, 20, 25, 30, 50, 70, 100, 150, 200, 250]:
        FCC = mp.LatticeMaker(3.615, 'FCC', num, 100, 100)
        FCC.compute()
        print(f'Build {FCC.N} atoms...')
        freud_t, mdapy_t = 0., 0.
        for turn in range(ave_num):
            print(f'Running {turn} turn in freud...')
            box = freud.box.Box(Lx=FCC.box[0, 1], Ly=FCC.box[1, 1], Lz=FCC.box[2, 1])
            shift_pos = FCC.pos-np.min(FCC.pos, axis=0) - (FCC.box[:,1])/2
            start = time()
            rdf = freud.density.RDF(bins, r_max)
            rdf.compute(system=(box, shift_pos), reset=False)
            end = time()
            freud_t += (end-start)
            
            print(f'Running {turn} turn in mdapy...')
            start = time()
            neigh = mp.Neighbor(FCC.pos, FCC.box, r_max, max_neigh=50)
            neigh.compute()
            rho = FCC.N/np.product(FCC.box[:,1])
            pair = mp.PairDistribution(r_max, bins, rho, neigh.verlet_list, neigh.distance_list)
            pair.compute()
            end = time()
            mdapy_t += (end-start)
            if check_freud:
                print(f'Checking results of {turn} turn...')
                assert np.isclose(pair.r, rdf.bin_centers).all() & np.isclose(pair.g_total, rdf.rdf).all()
            if check_plot:
                plot_rdf(pair, rdf, r_max)
            
        time_list.append([FCC.N, freud_t/ave_num, mdapy_t/ave_num])
        print('*'*30)
    time_list = np.array(time_list)
    np.savetxt('time_list_cpu_rdf.txt', time_list, delimiter=' ', header='N freud mdapy')
    return time_list
    
def plot(time_list, kind = 'cpu', save_fig=True):

    assert kind in ['cpu', 'gpu', 'cpu-gpu']
    if kind in ['cpu', 'gpu']:
        assert time_list.shape[1] == 3
    else:
        assert time_list.shape[1] == 4
    pltset()
    colorlist = [i['color'] for i in list(plt.rcParams['axes.prop_cycle'])]
    fig = plt.figure(figsize=(cm2inch(10), cm2inch(8)), dpi=150)
    plt.subplots_adjust(left=0.16, bottom=0.165, top=0.95, right=0.95)
    N_max = time_list[-1, 0]
    exp_max = int(np.log10(N_max))
    x, y = time_list[:, 0]/10**exp_max, time_list[:, 1]
    
    popt = np.polyfit(x, y, 1)
    plt.plot(x, np.poly1d(popt)(x), c=colorlist[0])
    plt.plot(x, y, 'o', label = f'freud, k={popt[0]:.1f}')

    if kind == 'cpu-gpu':
        y1 = time_list[:, 2]
        popt = np.polyfit(x, y1, 1)
        plt.plot(x, np.poly1d(popt)(x), c=colorlist[1])
        plt.plot(x, y1, 'o', label = f'mdapy-cpu, k={popt[0]:.1f}')
        
        y2 = time_list[:, 3]
        popt = np.polyfit(x, y2, 1)
        plt.plot(x, np.poly1d(popt)(x), c=colorlist[2])
        plt.plot(x, y2, 'o', label = f'mdapy-gpu, k={popt[0]:.1f}')
    else:
        y1 = time_list[:, 2]
        popt = np.polyfit(x, y1, 1)
        plt.plot(x, np.poly1d(popt)(x), c=colorlist[1])
        plt.plot(x, y1, 'o', label = f'mdapy-cpu, k={popt[0]:.1f}')

    plt.legend()
    plt.xlabel('Number of atoms ($\mathregular{10^%d}$)' % exp_max)
    plt.ylabel('Time (s)')
    if save_fig:
        plt.savefig('rdf_mdapy_freud.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()

if __name__ == '__main__':
    mp.init('cpu')
    time_list = test_rdf_average_time(ave_num=3, bins=200, r_max = 5., check_freud=True, check_plot=False)
    plot(time_list, kind = 'cpu', save_fig=True)