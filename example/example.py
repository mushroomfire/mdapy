from pyAnalysis import core 
from datetime import datetime


start = datetime.now()
filename_list = [r'./test.dump', r'./test1.dump']
amass = [58.933, 63.546, 55.847, 58.693, 106.42]
rc = 5.0
num_threads = 4
max_neigh = 110
units = 'metal'

for filename in filename_list:
    core.print_color(f'Reading {filename}...')
    system = core.build_system(filename, amass, rc, units, max_neigh, num_threads)
    core.print_color('Calculating temperature...')
    core.get_temperature(system)
    core.print_color('Calculating entropy...')
    core.get_entropy(system)
    #core.print_color('Calculating solid parameter...')
    #core.get_solid(system)
    core.print_color('Saving dump file...')
    core.write_dump(system)
    core.print_color('')
end = datetime.now()
core.print_color(f'All done! Time costs {end-start}.')

