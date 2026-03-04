import numpy as np
from mdapy import NEP
from mdapy import XYZTrajectory


def test_qnep():
    for mode in ["mode1", "mode2"]:
        path = rf"input_files/qnep/{mode}/"
        nep = NEP(f"{path}/nep.txt")
        traj = XYZTrajectory("input_files/qnep/train.xyz")

        e_m = []
        f_m = []
        v_m = []
        charge_m = []
        bec_m = []
        for system in traj:
            nep = nep 
            nep.calculate(system.data, system.box)
            e_m.append(nep.results["energies"].mean())
            f_m.append(nep.results["forces"])
            v_m.append(nep.results["virials"].mean(axis=0)[[0, 4, 8, 1, 5, 6]])
            charge_m.append(nep.results["charges"])
            bec_m.append(nep.results["bec"])
            

        e_m = np.array(e_m)
        f_m = np.concatenate(f_m)
        v_m = np.array(v_m)
        charge_m = np.concatenate(charge_m).flatten()
        bec_m = np.concatenate(bec_m)

        e_g = np.loadtxt(f"{path}/energy_train.out")[:, 0]
        f_g = np.loadtxt(f"{path}/force_train.out")[:, :3]
        v_g = np.loadtxt(f"{path}/virial_train.out")[:, :6]
        charge_g = np.loadtxt(f"{path}/charge_train.out")
        bec_g = np.loadtxt(f"{path}/bec_train.out")[:, :9]
        N = 384
        for i in range(len(traj)):
            charge_g[i * N : (i + 1) * N] -= charge_g[i * N : (i + 1) * N].mean()

        atol = 1e-4
        assert np.allclose(e_m, e_g, atol=atol), f"{mode} energy is wrong."
        assert np.allclose(f_m, f_g, atol=atol), f"{mode} force is wrong."
        assert np.allclose(v_m, v_g, atol=atol), f"{mode} virial is wrong."
        assert np.allclose(bec_m, bec_g, atol=atol), f"{mode} bec is wrong."
        assert np.allclose(charge_m, charge_g, atol=atol), f"{mode} charge is wrong."
