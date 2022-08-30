import numpy as np 
import numba as nb
#from numba import prange

@nb.jit(nopython=True) 
def pbc(Rij, L, boundary):
    for i in range(3):
        if boundary[i] == 1:
            Rij[i] -= L[i]*np.round(Rij[i]/L[i])
    return Rij

@nb.jit(nopython=True) 
def compute_entropy(data, vol, verlet_list, box_l, boundary, cutoff, r, sigma=0.2, use_local_density=False):
    
    N = data.shape[0]
    global_rho = N / vol
    pos = data[:, 2:5]
    local_entropy = np.zeros(N)
    nbins = int(cutoff / sigma) + 1
    rsq = r**2
    prefactor = rsq * (4 * np.pi * global_rho * np.sqrt(2 * np.pi * sigma**2))
    prefactor[0] = prefactor[1] 

    for i in range(N):
        r_ij = np.zeros(verlet_list.shape[1])
        for nei in range(verlet_list.shape[1]):
            j = verlet_list[i, nei]
            if j > -1 and j != i:
                Rij = pos[i,:] - pos[j,:]
                Rij = pbc(Rij, box_l, boundary)
                r_ij[nei] = np.sqrt(np.sum(np.square(Rij)))
        #print(i, pos[i, :], r_ij, len(r_ij))
        r_ij = r_ij[r_ij>0]
        
        r_diff = np.expand_dims(r, 0) - np.expand_dims(r_ij, 1)
        g_m = np.sum(np.exp(-r_diff**2 / (2.0*sigma**2)), axis=0) / prefactor

        if use_local_density:
            local_volume = 4/3 * np.pi * cutoff**3
            rho = len(r_ij) / local_volume
            g_m *= global_rho / rho
        else:
            rho = global_rho

        integrand = np.where(g_m >= 1e-10, (g_m * np.log(g_m) - g_m + 1.0) * rsq, rsq)

        local_entropy[i] = -2.0 * np.pi * rho * np.trapz(integrand, r)
    return local_entropy