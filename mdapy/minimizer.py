# Copyright (c) 2022-2024, mushroomfire in Beijing Institute of Technology
# This file is from the mdapy project, released under the BSD 3-Clause License.

import numpy as np

try:
    from potential import BasePotential
    from box import init_box
    from lattice_maker import LatticeMaker
except Exception:
    from .potential import BasePotential
    from .box import init_box
    from .lattice_maker import LatticeMaker


class Minimizer:
    """This function use the fast inertial relaxation engine (FIRE) method to minimize the system with fixed box.
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
        fmax (float, optional): maximum force per atom to consider as converged. Defaults to 0.05.
        max_itre (int, optional): maximum iteration times. Defaults to 100.
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
        fmax=0.05,
        max_itre=100,
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

        vel = np.zeros_like(self.pos)

        print("Energy minimization started.")

        for step in range(self.max_itre):
            pe_atom, force, _ = self.potential.compute(
                self.pos, self.box, self.elements_list, self.type_list, self.boundary
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


if __name__ == "__main__":
    import taichi as ti
    from potential import NEP

    ti.init()

    nep = NEP(r"D:\Study\Gra-Al\potential_test\validating\graphene\itre_45\nep.txt")

    element_name, lattice_constant, lattice_type, potential = "Al", 4.033, "FCC", nep
    x, y, z = 5, 5, 5
    fmax = 0.05
    max_itre = 10
    lat = LatticeMaker(lattice_constant, lattice_type, x, y, z)
    lat.compute()

    pe_atom, _, _ = potential.compute(
        lat.pos, lat.box, [element_name], lat.type_list, [1, 1, 1]
    )
    E_bulk = pe_atom.sum()
    print(f"Bulk energy: {E_bulk:.2f} eV.")

    mini_defect = Minimizer(
        np.ascontiguousarray(lat.pos[1:]),
        lat.box,
        [1, 1, 1],
        potential,
        [element_name],
        np.ascontiguousarray(lat.type_list[1:]),
        fmax=fmax,
        max_itre=max_itre,
    )
    mini_defect.compute()
    E_defect = mini_defect.pe_itre[-1]
    print(f"Defect energy: {E_defect:.2f} eV.")

    E_v = E_defect - E_bulk * (lat.N - 1) / lat.N
    print(f"Vacancy formation energy: {E_v:.2f} eV.")
