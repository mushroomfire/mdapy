import mdapy as mp
import numpy as np
from time import time
import matplotlib.pyplot as plt
from mdapy.plot.pltset import pltset, cm2inch
from mdapy.tools.timer import timer

from ovito.data import CutoffNeighborFinder, DataCollection, Particles, SimulationCell
from ovito.io import import_file
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import CentroSymmetryModifier

    
@timer
def test_csp_average_time(ave_num=3, check_ovito=False):
    time_list = []
    print('*'*30)
    for num in [5, 10, 15, 20, 25, 30, 50, 70, 100, 150, 200, 250]:
        FCC = mp.LatticeMaker(3.615, 'FCC', num, 100, 100)
        FCC.compute()
        print(f'Build {FCC.N} atoms...')
        ovito_t, mdapy_t = 0., 0.
        for turn in range(ave_num):
            print(f'Running {turn} turn in ovito...')
            particles = Particles()
            data = DataCollection()
            data.objects.append(particles)
            particles.create_property('Position', data=FCC.pos)
            cell = SimulationCell(pbc = (True, True, True))
            cell[...] = [[FCC.box[0, 1],0,0,FCC.box[0, 0]],
                         [0,FCC.box[1, 1],0,FCC.box[1, 0]],
                         [0,0,FCC.box[2, 1],FCC.box[2, 0]]]
            data.objects.append(cell)
            pipeline = Pipeline(source = StaticSource(data = data))

            start = time()
            pipeline.modifiers.append(CentroSymmetryModifier())
            data = pipeline.compute()
            end = time()
            ovito_t += (end-start)
            
            print(f'Running {turn} turn in mdapy...')
            start = time()
            system = mp.System(box=FCC.box, pos=FCC.pos)
            system.cal_centro_symmetry_parameter()
            end = time()
            mdapy_t += (end-start)
            if check_ovito:
                print(f'Checking results of {turn} turn...')
                assert np.allclose(system.data['csp'].values, np.array(data.particles['Centrosymmetry'][...]))
            
        time_list.append([FCC.N, ovito_t/ave_num, mdapy_t/ave_num])
        print('*'*30)
    time_list = np.array(time_list)
    np.savetxt('time_list_cpu_csp.txt', time_list, delimiter=' ', header='N ovito mdapy')
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
        plt.savefig('csp_mdapy_ovito.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.show()
    
if __name__ == '__main__':
    mp.init('cpu')
    import matplotlib
    matplotlib.use('Agg')
    time_list = test_entropy_average_time(ave_num=3, check_ovito=True)
    #time_list = np.loadtxt('time_list_cpu_csp.txt')
    plot(time_list, title='Calculate CSP', kind = 'cpu', save_fig=True)