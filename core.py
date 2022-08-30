import numpy as np
import sys

from pyAnalysis.compute_neighbor.neigh_cell_p import build_cell, build_verlet_list_cell
from pyAnalysis.compute_temperature.temperature_p import get_temp
from pyAnalysis.compute_entropy.entropy_p import compute_entropy_p
from pyAnalysis.compute_wcp.wcp import get_WCP_f, get_WCP_py
from pyAnalysis.read_dump.dump import read_data 
from pyAnalysis.write_dump.write import write_dump
from pyAnalysis.screen_output.custom_print import print_color, Color
from pyAnalysis.timer.timer import timer

@timer
def build_system(filename, amass, rc, units, max_neigh=100, num_threads=1):
    system = read_data(filename, amass, rc, units, max_neigh, num_threads)
    print_color(f'The system has {system.N} atoms.')
    return system

@timer
def build_neigh(system):
    system.rsquare = system.rc**2
    system.bin_l = system.rc+1.0
    system.ncel = np.floor(system.box_l/system.bin_l).astype(int)
    system.cell_id_list = np.zeros(system.ncel[0]*system.ncel[1]*system.ncel[2], dtype=int).reshape(system.ncel[0], system.ncel[1], system.ncel[2])
    system.atom_cell_list = np.zeros(system.N, dtype = int)
    system.verlet_list = np.zeros((system.N, system.max_neigh), dtype=int)-1
    try:
        system.atom_cell_list, system.cell_id_list = build_cell(system.bin_l, system.box, 
                                                system.pos, system.atom_cell_list,
                                                system.cell_id_list, system.num_threads)

        system.verlet_list = build_verlet_list_cell(system.pos, system.atom_cell_list,
                                                    system.cell_id_list, system.verlet_list,
                                                    system.bin_l, system.rsquare, system.boundary,
                                                    system.box, system.num_threads)
        system.neigh = True 
    except:
        print('Failed. Maybe one can increase the max_neigh and try again!')
        sys.exit()

@timer
def get_temperature(system):
    if not system.neigh:
        build_neigh(system)
    if system.units == 'metal':
        factor = 100.0
    elif system.units == 'real':
        factor = 100000.0
    T = np.zeros(system.N)
    T = get_temp(system.verlet_list, system.vel, T, 
                        system.mass, system.num_threads, factor)
    system.data['temp'] = T 
    system.head[-1] = system.head[-1].strip() + ' temp'

@timer
def get_entropy(system):
    if not system.neigh:
        build_neigh(system)
    sigma = 0.2
    use_local_density = 0
    nbins = int(system.rc / sigma) + 1
    local_entropy = compute_entropy_p(system.pos, system.vol, system.verlet_list, 
                                      system.box, system.boundary, system.rc, 
                                      sigma, nbins, use_local_density, system.num_threads)
    system.data['entropy'] = local_entropy
    system.head[-1] = system.head[-1].strip() + ' entropy'


if __name__ == '__main__':

    filename_list = [r'./example/test.dump', r'./example/test1.dump']
    amass = [58.933, 63.546, 55.847, 58.693, 106.42]
    rc = 5.0
    num_threads = 4
    max_neigh = 110
    units = 'metal'

    for filename in filename_list:
        print_color(f'Reading {filename}...')
        system = build_system(filename, amass, rc, units, max_neigh, num_threads)
        print_color('Calculating temperature...')
        get_temperature(system)
        print_color('Calculating entropy...')
        get_entropy(system)
        #print_color('Calculating solid parameter...')
        #get_solid(system)
        print_color('Saving dump file...')
        write_dump(system)
        print_color('')
