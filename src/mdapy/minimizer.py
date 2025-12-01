# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.

"""
This module implements the **Fast Inertial Relaxation Engine (FIRE2)** algorithm
for atomic structure optimization. It is a direct translation and adaptation of
the FIRE2 method from **ASE (Atomic Simulation Environment)**.

References
----------
- J. Guénolé, W.G. Nöhring, A. Vaid, F. Houllé, Z. Xie, A. Prakash, E. Bitzek,
  *Assessment and optimization of the fast inertial relaxation engine (FIRE) for energy
  minimization in atomistic simulations and its implementation in LAMMPS*,
  Computational Materials Science 175 (2020) 109584.
  https://doi.org/10.1016/j.commatsci.2020.109584

- S. Echeverri Restrepo, P. Andric,
  *ABC-FIRE: Accelerated Bias-Corrected Fast Inertial Relaxation Engine*,
  Computational Materials Science 218 (2023) 111978.
  https://doi.org/10.1016/j.commatsci.2022.111978

When ``optimize_cell=True``, this implementation behaves equivalently to
``ase.optimize.FIRE2`` combined with ``ase.constraints.UnitCellFilter``,
allowing simultaneous optimization of both atomic positions and cell shape.

"""

from mdapy.system import System
import numpy as np
import polars as pl


def _voigt_6_to_full_3x3_stress(voigt):
    """Convert 6-component Voigt stress vector to full 3x3 symmetric stress tensor."""
    xx, yy, zz, yz, xz, xy = voigt
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


class FIRE:
    """
    Implementation of the FIRE2 (and optional ABC-FIRE) energy minimization algorithm.

    This optimizer relaxes atomic structures using a velocity-Verlet–like
    integration with adaptive time-stepping, as proposed in the FIRE method.
    It can optionally optimize both the **atomic positions** and **simulation cell**
    (similar to ASE’s ``UnitCellFilter``).

    Parameters
    ----------
    system : mdapy.system.System
        The system to be relaxed. It must provide ``get_force()``, ``get_energy()``,
        and ``get_virials()`` methods, and an associated ``Box`` object.

    dt : float, optional
        Initial time step for the integration (default: 0.1).

    maxstep : float, optional
        Maximum allowed atomic displacement per step (default: 0.2 Å).

    dtmax : float, optional
        Maximum time step allowed during optimization (default: 1.0).

    dtmin : float, optional
        Minimum time step (default: 2e-3).

    Nmin : int, optional
        Minimum number of consecutive “good” steps (dot(v, f) > 0) before
        increasing the time step (default: 20).

    finc : float, optional
        Factor by which the time step is increased after successful steps (default: 1.1).

    fdec : float, optional
        Factor by which the time step is decreased after bad steps (default: 0.5).

    astart : float, optional
        Initial value of the velocity-force mixing coefficient ``α`` (default: 0.25).

    fa : float, optional
        Multiplicative decay factor for ``α`` after successful steps (default: 0.99).

    use_abc : bool, optional
        Whether to use the **ABC-FIRE** algorithm (default: False).

    optimize_cell : bool, optional
        Whether to optimize the simulation cell along with atomic positions
        (default: False). Equivalent to wrapping ASE’s ``UnitCellFilter``.

    mask : array_like of shape (3,3) or (6,), optional
        Tensor mask controlling which components of strain are relaxed.
        ``1`` means free, ``0`` means fixed. Default is fully relaxed.

    cell_factor : float, optional
        Scale factor applied to cell degrees of freedom. Equivalent to ASE’s
        ``cell_factor``. Default is the number of atoms.

    hydrostatic_strain : bool, optional
        If True, only isotropic (hydrostatic) deformation is allowed (default: False).

    constant_volume : bool, optional
        If True, constrain relaxation to approximately constant volume
        (default: False).

    scalar_pressure : float, optional
        External scalar pressure (in eV/Å³) added to the enthalpy term (default: 0.0).

    Notes
    -----
    - When ``optimize_cell=True``, the deformation gradient is treated as
      additional three “virtual atoms”, following the formalism from
      Tadmor *et al.*, Phys. Rev. B 59, 235 (1999).
    - This class modifies both atomic positions and the simulation box
      to minimize the potential energy (or enthalpy if pressure is applied).

    Examples
    --------
    >>> from mdapy.minimizer import FIRE
    >>> from mdapy import build_crystal
    >>> from mdapy.nep import NEP
    >>> system = build_crystal("Al", "fcc", 4.05)
    >>> system.calc = NEP("Al.txt")  # A NEP model
    >>> fire = FIRE(system, optimize_cell=True)
    >>> fire.run(steps=1000, fmax=1e-4)
    """

    def __init__(
        self,
        system: System,
        dt: float = 0.1,
        maxstep: float = 0.2,
        dtmax: float = 1.0,
        dtmin: float = 2e-3,
        Nmin: int = 20,
        finc: float = 1.1,
        fdec: float = 0.5,
        astart: float = 0.25,
        fa: float = 0.99,
        use_abc: bool = False,
        optimize_cell: bool = False,
        mask=None,
        cell_factor=None,
        hydrostatic_strain: bool = False,
        constant_volume: bool = False,
        scalar_pressure: float = 0.0,
    ):
        self.system = system
        self.dt = dt
        self.Nsteps = 0
        self.maxstep = maxstep
        self.dtmax = dtmax
        self.dtmin = dtmin
        self.Nmin = Nmin
        self.finc = finc
        self.fdec = fdec
        self.astart = astart
        self.fa = fa
        self.a = astart
        self.use_abc = use_abc
        self.optimize_cell = optimize_cell
        self.scalar_pressure = scalar_pressure
        self.hydrostatic_strain = hydrostatic_strain
        self.constant_volume = constant_volume

        self.N = self.system.N
        self.ndof = self.N if not optimize_cell else self.N + 3

        if optimize_cell:
            self.orig_box = self.system.box.box.copy()
            if cell_factor is None:
                cell_factor = float(self.N)
            self.cell_factor = cell_factor
            if mask is None:
                mask = np.ones((3, 3))
            elif len(mask) == 6:
                mask = _voigt_6_to_full_3x3_stress(mask)
            self.mask = mask
        else:
            self.orig_box = None
            self.cell_factor = None
            self.mask = None

    def get_forces(self) -> np.ndarray:
        """
        Compute the combined atomic and (optionally) cell forces.

        Returns
        -------
        np.ndarray
            Force array of shape (N, 3) when ``optimize_cell=False``,
            or (N+3, 3) when ``optimize_cell=True``.
            The last three rows correspond to the cell forces derived from the virial stress tensor.
        """
        atoms_forces = self.system.get_force()
        if self.optimize_cell:
            volume = self.system.box.volume
            virial = self.system.get_virials().sum(axis=0).reshape(3, 3) + (
                -np.diag([self.scalar_pressure] * 3) * volume
            )

            cur_deform_grad = np.linalg.solve(self.orig_box, self.system.box.box).T
            atoms_forces = atoms_forces @ cur_deform_grad
            virial = np.linalg.solve(cur_deform_grad, virial.T).T

            if self.hydrostatic_strain:
                vtr = virial.trace()
                virial = np.diag([vtr / 3.0] * 3)

            if (self.mask != 1.0).any():
                virial *= self.mask

            if self.constant_volume:
                vtr = virial.trace()
                np.fill_diagonal(virial, np.diag(virial) - vtr / 3.0)

            cell_forces = virial / self.cell_factor
            return np.vstack((atoms_forces, cell_forces))
        else:
            return atoms_forces

    def update_data_box(self, extended_dr: np.ndarray):
        """
        Update atomic positions and simulation cell based on the provided displacement.

        Parameters
        ----------
        extended_dr : np.ndarray
            Displacement array of shape (N, 3) if not optimizing the cell,
            or (N+3, 3) when ``optimize_cell=True``.
        """
        if self.optimize_cell:
            cur_positions = self.system.data.select("x", "y", "z").to_numpy()  # (N, 3)
            cur_deform_grad = np.linalg.solve(self.orig_box, self.system.box.box).T
            cur_unstrained = np.linalg.solve(cur_deform_grad, cur_positions.T).T

            dr_atoms = extended_dr[: self.N]
            dr_cell = extended_dr[self.N :]

            new_unstrained = cur_unstrained + dr_atoms
            new_deform_grad = cur_deform_grad + (dr_cell / self.cell_factor)

            deform = (new_deform_grad - np.eye(3)).T * self.mask
            new_box = self.orig_box @ (np.eye(3) + deform)
            self.system.update_box(new_box)

            new_positions = new_unstrained @ (np.eye(3) + deform)
            self.system.update_data(
                self.system.data.with_columns(
                    pl.lit(new_positions[:, 0]).alias("x"),
                    pl.lit(new_positions[:, 1]).alias("y"),
                    pl.lit(new_positions[:, 2]).alias("z"),
                ),
                True,
            )
        else:
            self.system.update_data(
                self.system.data.with_columns(
                    pl.col("x") + extended_dr[:, 0],
                    pl.col("y") + extended_dr[:, 1],
                    pl.col("z") + extended_dr[:, 2],
                ),
                True,
            )

    def run(self, steps: int, fmax=1e-4, show_process=True):
        """
        Run the FIRE relaxation process.

        Parameters
        ----------
        steps : int
            Maximum number of optimization steps.

        fmax : float, optional
            Convergence criterion for maximum force (default: 1e-4 eV/Å).

        show_process : bool, optional
            Whether to print per-step progress (default: True).

        Notes
        -----
        This function adaptively updates velocity, time step, and the
        mixing parameter ``α`` according to the FIRE2 algorithm.
        If ``use_abc=True``, it applies the ABC-FIRE bias correction
        and per-direction displacement capping.

        """
        self.v = None
        if show_process:
            print(f"{'Step':>6} {'Energy':>15} {'fmax':>15} {'pressure':>15}")
        for step in range(steps):
            extended_f = self.get_forces()

            cfmax = np.sqrt((extended_f**2).sum(axis=1).max())
            if show_process:
                if self.optimize_cell:
                    energy = (
                        self.system.get_energy()
                        + self.scalar_pressure * self.system.box.volume
                    )
                else:
                    energy = self.system.get_energy()
                stress = self.system.get_stress()
                press = -stress[:3].mean()
                print(f"{step:6d} {energy:15.6f} {cfmax:15.6f} {press:15.6f}")
            if cfmax < fmax:
                return

            if self.v is None:
                self.v = np.zeros((self.ndof, 3))
            else:
                vf = np.vdot(extended_f, self.v)
                if vf > 0.0:
                    self.Nsteps += 1
                    if self.Nsteps > self.Nmin:
                        self.dt = min(self.dt * self.finc, self.dtmax)
                        self.a *= self.fa
                else:
                    self.Nsteps = 0
                    self.dt = max(self.dt * self.fdec, self.dtmin)
                    self.a = self.astart
                    dr = -0.5 * self.dt * self.v
                    self.update_data_box(dr)
                    extended_f = self.get_forces()
                    self.v *= 0.0

            self.v += self.dt * extended_f

            if self.use_abc:
                self.a = max(self.a, 1e-10)
                abc_multiplier = 1.0 / (1.0 - (1.0 - self.a) ** (self.Nsteps + 1))
                v_mix = (1.0 - self.a) * self.v + self.a * extended_f / np.sqrt(
                    np.vdot(extended_f, extended_f)
                ) * np.sqrt(np.vdot(self.v, self.v))
                self.v = abc_multiplier * v_mix

                if np.all(self.v):
                    v_tmp = []
                    for car_dir in range(3):
                        abs_v_dir = np.abs(self.v[:, car_dir])
                        cap = np.where(
                            abs_v_dir * self.dt > self.maxstep,
                            (self.maxstep / self.dt) * (self.v[:, car_dir] / abs_v_dir),
                            self.v[:, car_dir],
                        )
                        v_tmp.append(cap)
                    self.v = np.array(v_tmp).T
            else:
                self.v = (1.0 - self.a) * self.v + self.a * extended_f / np.sqrt(
                    np.vdot(extended_f, extended_f)
                ) * np.sqrt(np.vdot(self.v, self.v))

            dr = self.dt * self.v
            if not self.use_abc:
                normdr = np.sqrt(np.vdot(dr, dr))
                if normdr > self.maxstep:
                    dr = self.maxstep * dr / normdr
            self.update_data_box(dr)

        self.system.calc.results = {}


if __name__ == "__main__":
    pass
