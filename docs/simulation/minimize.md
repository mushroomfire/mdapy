# Energy minimization

`mdapy.simulation.minimize(system, calculator, *, max_force=1e-3, max_steps=1000,
max_step=0.1, timestep=0.001, cell=None, pressure=0.0, max_stress=1e-2,
save_every=None, callback=None)` minimizes the energy over the positions, and
optionally the box, with FIRE 2.0
([Guénolé et al. (2020)](../reference/references.md#guenole2020)), the fast inertial relaxation
engine of [Bitzek et al. (2006)](../reference/references.md#bitzek2006), as LAMMPS
`min_style fire` does with its default settings.

<!-- skip-test -->
```python
from mdapy.simulation import minimize

result = minimize(system, potential)  # the positions in a fixed box
result = minimize(system, potential, cell="full")  # and the box
result.system.box, result.stress  # the relaxed box and its stress in GPa
result = minimize(system, potential, cell="isotropic", pressure=10.0)
```

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `system` | `System` | | | the starting configuration, with a `mass` or an `element` column; a `movable` column, (N, 3) bool as VASP's selective dynamics, keeps each atom still where it is False |
| `calculator` | `Calculator` | | | computes energies and forces, and the virial when the box relaxes |
| `max_force` | float | 1e-3 | eV/Å | converged when every force is below it |
| `max_steps` | int | 1000 | | largest number of steps |
| `max_step` | float | 0.1 | Å | an atom moving at its current velocity goes at most this far along x, y or z in one step (LAMMPS `dmax`) |
| `timestep` | float | 0.001 | ps | first time step of the dynamics (the LAMMPS `timestep` in metal units) |
| `cell` | str | `None` | | how the box may change, in the table below |
| `pressure` | float | 0.0 | GPa | external hydrostatic pressure when the box relaxes |
| `max_stress` | float | 1e-2 | GPa | when the box relaxes, converged when every relaxed component of the stress is this close to $-p$ (−`pressure` on the diagonal, 0 off it) |
| `save_every` | int | `None` | | keeps the configuration every this many steps, with the first and the last, in `frames` |
| `callback` | callable | `None` | | called after each step as `callback(step, system, energy, max_force)`; returning True stops the minimization |

`cell` chooses how the box changes along its periodic directions; the atoms
move with it, and the lattice vectors along directions that are not periodic
stay as they are. At a `pressure` $p$ the minimization minimizes the enthalpy
$E + pV$, so the stress of the relaxed box is $-p$ on the diagonal.

| `cell` | The box | Relaxed stress components |
|---|---|---|
| `None` | does not change | none |
| `"isotropic"` | is scaled by one factor | the mean of the diagonal |
| `"axes"` | is scaled along each Cartesian axis by its own factor | the diagonal |
| `"full"` | takes any strain without rotation | all |

It returns a `MinimizeResult`:

| Attribute | Type | Meaning |
|---|---|---|
| `system` | `System` | the last configuration, with `forces` and `atomic_energies` columns and `energy`, `virial` (eV) and `stress` (GPa) in `info` |
| `converged` | bool | whether every force fell below `max_force` and, when the box relaxes, every relaxed stress component came within `max_stress` of $-p$ |
| `steps` | int | number of steps taken |
| `energy` | float | energy of the last configuration in eV |
| `max_force` | float | largest force of the last configuration in eV/Å |
| `stress` | `numpy.ndarray` or `None` | (3, 3) stress of the last configuration in GPa, positive under tension; `None` when the calculator does not compute the virial |
| `frames` | tuple of `System` or `None` | with `save_every`, the kept configurations, with the columns and `info` of `system` and the step as `timestep` in `info`; `None` otherwise |

`virial` and `stress` are in `info`, and `stress` is set, whenever the
calculator computes the virial, also when the box does not change.

The steps on the positions are those of LAMMPS: the same time step control,
mixing and half step back after an uphill step, with masses from the `mass`
column or, when there is none, the standard atomic weights of the elements.
Masses change the path, not the minimum; LAMMPS takes them from the data file
or, for EAM, from the potential file. A box relaxation reaches the minimum of
LAMMPS `fix box/relax` (which LAMMPS runs with `min_style cg`, not FIRE) by
another path; LAMMPS keeps its box in restricted triclinic form, so the two
boxes agree up to a rotation.

The kept frames show the minimization on its way, with the box and the stress
of each step, and can be written as a trajectory:

<!-- skip-test -->
```python
result = minimize(system, potential, cell="full", save_every=10)
mdapy.write("relaxation.xyz", result.frames)
```

`minimize` raises `InvalidParameter` for a non-positive `max_force`,
`max_step`, `timestep`, `max_stress` or `save_every`, a negative
`max_steps`, an unknown `cell`, or a calculator that does not compute
energies and forces, and the virial when the box relaxes, and
[`MissingProperty`](../reference/api.md#mdapy.errors.MissingProperty) for a
system with neither masses nor elements.

!!! quote "Please cite"
    [Guénolé et al. (2020)](../reference/references.md#guenole2020) and [Bitzek et al. (2006)](../reference/references.md#bitzek2006).
