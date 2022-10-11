import mdapy as mp
import numpy as np
from time import time

#mp.init("cpu", kernel_profiler=True)
mp.init("cpu")
print("mdapy version:", mp.__version__)
print("Building system...")
system = mp.System("cna.dump")
start = time()
print("Building neighbor...")
system.build_neighbor(rc=6.875, max_neigh=150)
print("Calculating csp...")
system.cal_centro_symmetry_parameter(12)
print("Calculating entropy...")
system.cal_atomic_entropy()
print("Calculating atomic temperature...")
system.cal_atomic_temperature(amass=np.array([58.933, 58.693, 55.847, 26.982, 63.546]))
end = time()
print("Time costs: ", end - start, "s.")
# print("Save to dump file...")
# system.write_dump()

#mp.profiler.print_kernel_profiler_info()
