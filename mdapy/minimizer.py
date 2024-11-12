# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np

try:
    from potential import BasePotential
    from box import init_box

except Exception:
    from .potential import BasePotential
    from .box import init_box


class Minimizer:
    """This function use the fast inertial relaxation engine (FIRE) method to minimize the system, including optimizing position and box.
    More details can be found in paper:

    Guénolé, Julien, et al. "Assessment and optimization of the fast inertial relaxation engine (fire) for energy minimization in atomistic simulations and its implementation in lammps." Computational Materials Science 175 (2020): 109584.

    Args:
        pos (np.ndarray): atom position.
        box (np.ndarray): system box.
        boundary (list): boundary, such as [1, 1, 1].
        potential (BasePotential): mdapy potential.
        elements_list (list): elemental name, such as ['Al', 'C'].
        type_list (np.ndarray): atom type list.
        mini_type (str, optional): only support 'FIRE' now. Defaults to "FIRE".
        fmax (float, optional): maximum force per atom to consider as converged. Defaults to 1e-5.
        max_itre (int, optional): maximum iteration times. Defaults to 200.
        volume_change (bool, optional): whether change the box to optimize the pressure. Defaults to False.
        hydrostatic_strain (bool, optional): sonstrain the cell by only allowing hydrostatic deformation. Defaults to False.
    """

    def __init__(
        self,
        pos,
        box,
        boundary,
        potential,
        elements_list,
        type_list,
        mini_type="FIRE",
        fmax=1e-5,
        max_itre=200,
        volume_change=False,
        hydrostatic_strain=False,
    ) -> None:
        self.pos = pos
        self.box, _, _ = init_box(box)
        self.boundary = [int(i) for i in boundary]
        assert len(self.boundary) == 3
        assert isinstance(potential, BasePotential)
        self.potential = potential
        self.elements_list = elements_list
        self.type_list = type_list
        assert mini_type == "FIRE", "Only support FIRE now."
        self.mini_type = mini_type
        self.fmax = fmax
        self.max_itre = max_itre

        self.volume_change = volume_change
        self.hydrostatic_strain = hydrostatic_strain

        self.orig_box = self.box.copy()

    def _deform_grad(self, box):
        return np.linalg.solve(self.orig_box[:-1], box).T

    def _get_volume(self, box):
        return np.inner(box[0], np.cross(box[1], box[2]))

    def compute(self):
        self.pe_itre = []
        f_inc = 1.1
        f_dec = 0.5
        alpha_start = 0.25
        alpha = alpha_start
        f_alpha = 0.99
        dt = 1.0
        dt_max = 10 * dt
        dt_min = 0.02 * dt
        N_min = 20
        m = 5.0
        N_neg = 0

        if self.max_itre > 10:
            base = self.max_itre // 10
        else:
            base = 1

        if not self.volume_change:
            vel = np.zeros_like(self.pos)
            print("Energy minimization started.")
            for step in range(self.max_itre):
                pe_atom, force, _ = self.potential.compute(
                    self.pos,
                    self.box,
                    self.elements_list,
                    self.type_list,
                    self.boundary,
                )
                pe = pe_atom.sum()
                self.pe_itre.append(pe)
                fmax = np.abs(force).max()
                if step % base == 0 or fmax < self.fmax:
                    print(f"Step {step}: Pe = {pe:.6f} eV, f_max = {fmax:.6f} eV/A.")
                    if fmax < self.fmax:
                        break

                P = (vel * force).sum()

                if P > 0:
                    if N_neg > N_min:
                        dt = min(dt * f_inc, dt_max)
                        alpha *= f_alpha
                    N_neg += 1
                else:
                    dt = max(dt * f_dec, dt_min)
                    alpha = alpha_start
                    self.pos += vel * (-0.5 * dt)
                    vel.fill(0.0)
                    N_neg = 0

                # implicit Euler integration
                F_modulus = np.linalg.norm(force)
                v_modulus = np.linalg.norm(vel)

                vel = vel + force * dt / m
                vel = vel * (1 - alpha) + force * (alpha * v_modulus) / F_modulus

                self.pos += vel * dt

            print("Energy minimization finished.")

        else:
            N = self.pos.shape[0]
            vel = np.zeros((N + 3, 3), dtype=self.pos.dtype)
            print("Energy minimization with volume change started.")
            for step in range(self.max_itre):
                pe_atom, force_atoms, virial = self.potential.compute(
                    self.pos,
                    self.box,
                    self.elements_list,
                    self.type_list,
                    self.boundary,
                )
                pe = pe_atom.sum()
                self.pe_itre.append(pe)

                volume = self._get_volume(self.box)
                virial_9 = np.zeros((3, 3))
                if virial.shape[1] == 9:  # xx, yy, zz, xy, xz, yz, yx, zx, zy.
                    virial = virial.sum(axis=0)
                    virial_9[0, 0] = virial[0]
                    virial_9[1, 1] = virial[1]
                    virial_9[2, 2] = virial[2]
                    virial_9[0, 1] = virial[3]
                    virial_9[0, 2] = virial[4]
                    virial_9[1, 2] = virial[5]
                    virial_9[1, 0] = virial[6]
                    virial_9[2, 0] = virial[7]
                    virial_9[2, 1] = virial[8]
                else:
                    assert virial.shape[1] == 6  # xx, yy, zz, xy, xz, yz
                    virial = virial.sum(axis=0)
                    virial_9[0, 0] = virial[0]
                    virial_9[1, 1] = virial[1]
                    virial_9[2, 2] = virial[2]
                    virial_9[0, 1] = virial[3]
                    virial_9[0, 2] = virial[4]
                    virial_9[1, 2] = virial[5]
                    virial_9[1, 0] = virial[3]
                    virial_9[2, 0] = virial[4]
                    virial_9[2, 1] = virial[5]

                cur_deform_grad = self._deform_grad(self.box[:-1])

                force_atoms = force_atoms @ cur_deform_grad
                virial_9_deform = np.linalg.solve(cur_deform_grad, virial_9.T).T

                if self.hydrostatic_strain:
                    vtr = virial_9_deform.trace()
                    virial_9_deform = np.diag([vtr / 3.0] * 3)

                force = np.zeros_like(vel)
                force[:N] = force_atoms
                force[N:] = virial_9_deform / N
                fmax = np.abs(force).max()
                if step % base == 0 or fmax < self.fmax:
                    pressure = (
                        (virial_9[0, 0] + virial_9[1, 1] + virial_9[2, 2]) / 3 / volume
                    )
                    print(
                        f"Step {step}: Pe = {pe:.6f} eV, f_max = {fmax:.6f} eV/A, pressure = {pressure * 160.2176621:.6f} GPa."
                    )

                    if fmax < self.fmax:
                        break

                P = (vel * force).sum()

                if P > 0:
                    if N_neg > N_min:
                        dt = min(dt * f_inc, dt_max)
                        alpha *= f_alpha
                    N_neg += 1
                else:
                    dt = max(dt * f_dec, dt_min)
                    alpha = alpha_start
                    self.pos += vel[:N] * (-0.5 * dt)

                    vel.fill(0.0)
                    N_neg = 0

                # implicit Euler integration
                F_modulus = np.linalg.norm(force)
                v_modulus = np.linalg.norm(vel)

                vel = vel + force * dt / m
                vel = vel * (1 - alpha) + force * (alpha * v_modulus) / F_modulus

                self.pos += vel[:N] * dt
                self.box[:-1] += self.box[:-1] @ (vel[N:] / N).T

            print("Energy minimization with volume change finished.")


if __name__ == "__main__":
    import taichi as ti
    from potential import NEP
    from lattice_maker import LatticeMaker

    ti.init()

    nep = NEP(r"D:\Study\Gra-Al\potential_test\validating\graphene\itre_45\nep.txt")

    element_name, lattice_constant, lattice_type, potential = (
        "C",
        1.42,
        "GRA",
        nep,
    )
    x, y, z = 5, 5, 1
    fmax = 1e-5
    max_itre = 200
    lat = LatticeMaker(lattice_constant, lattice_type, x, y, z)
    lat.compute()
    lat.box[2, 2] += 20
    # pe_atom, _, _ = potential.compute(
    #     lat.pos, lat.box, [element_name], lat.type_list, [1, 1, 1]
    # )
    # E_bulk = pe_atom.sum()
    # print(f"Bulk energy: {E_bulk:.2f} eV.")

    # mini_defect = Minimizer(
    #     np.ascontiguousarray(lat.pos[1:]),
    #     lat.box,
    #     [1, 1, 1],
    #     potential,
    #     [element_name],
    #     np.ascontiguousarray(lat.type_list[1:]),
    #     fmax=fmax,
    #     max_itre=max_itre,
    # )
    # mini_defect.compute()
    # E_defect = mini_defect.pe_itre[-1]
    # print(f"Defect energy: {E_defect:.2f} eV.")

    # E_v = E_defect - E_bulk * (lat.N - 1) / lat.N
    # print(f"Vacancy formation energy: {E_v:.2f} eV.")

    mini = Minimizer(
        lat.pos,
        lat.box,
        [1, 1, 1],
        potential,
        [element_name],
        lat.type_list,
        fmax=fmax,
        max_itre=max_itre,
        volume_change=False,
        hydrostatic_strain=True,
    )
    mini.compute()
    print(mini.box)

    # e, f, v = potential.compute(mini.pos, mini.box, [element_name], lat.type_list)
    # vol = np.inner(mini.box[0], np.cross(mini.box[1], mini.box[2]))
    # print(e.sum(), f.max(), v.sum(axis=0) / vol * 160.21)
