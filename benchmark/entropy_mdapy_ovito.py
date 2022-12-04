import mdapy as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mdapy.plot.pltset import pltset, cm2inch
from mdapy.tools.timer import timer

from ovito.data import CutoffNeighborFinder, DataCollection, Particles, SimulationCell
from ovito.io import import_file
from ovito.pipeline import Pipeline, StaticSource

def modify(frame: int, data: DataCollection, cutoff = 5.0, sigma = 0.2, use_local_density = False):
    # Validate input parameters:
    assert(cutoff > 0.0)
    assert(sigma > 0.0 and sigma < cutoff)

    # Show message in OVITO's status bar:
    yield 'Calculating local entropy'

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
    prefactor[0] = prefactor[1] # Avoid division by zero at r=0.

    # Iterate over input particles:
    for particle_index in range(data.particles.count):
        yield particle_index / data.particles.count

        # Get distances r_ij of neighbors within the cutoff range.
        r_ij = finder.neighbor_distances(particle_index)

        # Compute differences (r - r_ji) for all {r} and all {r_ij} as a matrix.
        r_diff = np.expand_dims(r, 0) - np.expand_dims(r_ij, 1)
        
        # Compute g_m(r):
        g_m = np.sum(np.exp(-r_diff**2 / (2.0*sigma**2)), axis=0) / prefactor

        # Estimate local atomic density by counting the number of neighbors within the 
        # spherical cutoff region:
        if use_local_density:
            local_volume = 4/3 * np.pi * cutoff**3
            rho = len(r_ij) / local_volume
            g_m *= global_rho / rho
        else:
            rho = global_rho

        # Compute integrand:
        integrand = np.where(g_m >= 1e-10, (g_m * np.log(g_m) - g_m + 1.0) * rsq, rsq)
        
        # Integrate from 0 to cutoff distance:
        local_entropy[particle_index] = -2.0 * np.pi * rho * np.trapz(integrand, r)

    # Output the computed per-particle entropy values to the data pipeline.
    data.particles_.create_property('Entropy', data=local_entropy)
    
@timer
def test_entropy_average_time(ave_num=3, cutoff = 5., sigma=0.2, check_ovito=False):
    time_list = []
    print('*'*30)
    for num in [10, 15, 30, 35, 40, 45, 50]:
        FCC = mp.LatticeMaker(3.615, 'FCC', num, 50, 50)
        FCC.compute()
        print(f'Build {FCC.N} atoms...')
        ovito_t, mdapy_t = 0., 0.
        for turn in range(ave_num):
            print(f'Running {turn} turn in ovito...')
            
            particles = Particles()
            data = DataCollection()
            data.objects.append(particles)
            particles.create_property('Position', data=FCC.pos)
            # particles.create_property('Particle Type', data=np.ones(FCC.N, dtype=int))
            cell = SimulationCell(pbc = (True, True, True))
            cell[...] = [[FCC.box[0, 1],0,0,FCC.box[0, 0]],
                         [0,FCC.box[1, 1],0,FCC.box[1, 0]],
                         [0,0,FCC.box[2, 1],FCC.box[2, 0]]]
            data.objects.append(cell)
            pipeline = Pipeline(source = StaticSource(data = data))
            
            start = time()
            pipeline.modifiers.append(modify)
            data = pipeline.compute()
            end = time()
            ovito_t += (end-start)
            
            print(f'Running {turn} turn in mdapy...')
            start = time()
            neigh = mp.Neighbor(FCC.pos, FCC.box, cutoff, max_neigh=50)
            neigh.compute()
            vol = np.product(FCC.box[:,1])
            entropy = mp.AtomicEntropy(vol, neigh.distance_list, cutoff, sigma=sigma)
            entropy.compute()
            end = time()
            mdapy_t += (end-start)
            if check_ovito:
                print(f'Checking results of {turn} turn...')
                assert np.allclose(entropy.entropy, data.particles['Entropy'][...])
            
        time_list.append([FCC.N, ovito_t/ave_num, mdapy_t/ave_num])
        print('*'*30)
    time_list = np.array(time_list)
    np.savetxt('time_list_cpu_entropy.txt', time_list, delimiter=' ', header='N ovito mdapy')
    return time_list
    
def plot(time_list, title=None, kind = 'cpu', save_fig=True):

    assert kind in ['cpu', 'gpu', 'cpu-gpu']
    if kind in ['cpu', 'gpu']:
        assert time_list.shape[1] == 3
    else:
        assert time_list.shape[1] == 4
    pltset()
    colorlist = [i['color'] for i in list(plt.rcParams['axes.prop_cycle'])]
    fig = plt.figure(figsize=(cm2inch(10), cm2inch(8)), dpi=150)
    plt.subplots_adjust(left=0.16, bottom=0.165, top=0.92, right=0.95)
    N_max = time_list[-1, 0]
    exp_max = int(np.log10(N_max))
    x, y = time_list[:, 0]/10**exp_max, time_list[:, 1]
    
    popt = np.polyfit(x, y, 1)
    plt.plot(x, np.poly1d(popt)(x), c=colorlist[0])
    plt.plot(x, y, 'o', label = f'ovito, k={popt[0]:.1f}')

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
        plt.plot(x, y1, 'o', label = f'mdapy, k={popt[0]:.1f}')
    if title is not None:
        plt.title(title, fontsize=12)
    plt.legend()
    plt.xlabel('Number of atoms ($\mathregular{10^%d}$)' % exp_max)
    plt.ylabel('Time (s)')
    if save_fig:
        plt.savefig('entropy_mdapy_ovito.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    
if __name__ == '__main__':
    mp.init('cpu')
    import matplotlib
    matplotlib.use('Agg')
    #time_list = test_entropy_average_time(ave_num=3, check_ovito=True)
    time_list = np.loadtxt('time_list_cpu_entropy.txt')
    plot(time_list, title='Calculate atomic entropy', kind = 'cpu', save_fig=True)