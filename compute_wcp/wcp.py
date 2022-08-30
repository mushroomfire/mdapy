from pyAnalysis.compute_wcp.wcp_s import get_wcp_real
from pyAnalysis.timer.timer import timer
import numpy as np 
import numba as nb


@timer
def get_WCP_f(system):
    type_list = system.type_list.astype(int)
    verlet_list = system.verlet_list
    data = system.data[['id', 'type']].values.astype(int)
    N_type = len(type_list)
    Zmn = np.zeros((N_type, N_type))
    wcp = get_wcp_real(type_list, verlet_list, data, Zmn)
    return np.round(wcp, 2)
    
    
@nb.jit(nopython=True)
def get_WCP_real(type_list, verlet_list, data, N):
    
    Zmn = np.zeros((len(type_list), len(type_list)))
    Zm = np.zeros((len(type_list), len(type_list)))
    for atomtype1 in type_list:

        mtype = np.where(data[:, 1]==atomtype1)[0]
        for i in mtype:
            m_neigh = verlet_list[i][verlet_list[i]>-1]
            for atomtype2 in type_list:
                Zm[atomtype1-1, atomtype2-1] += len(m_neigh)
            for j in m_neigh:
                Zmn[atomtype1-1, int(data[j, 1]-1)] += 1
    Alpha_n = np.array([len(data[data[:, 1]==i])/N for i in type_list])
    return 1 - Zmn/(Alpha_n*Zm)

@timer
def get_WCP_py(system):
    type_list = system.type_list
    verlet_list = system.verlet_list
    data = system.data[['id', 'type']].values
    N = system.N
    wcp = get_WCP_real(type_list, verlet_list, data, N)
    return np.round(wcp, 2)