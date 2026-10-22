# Simulation

`mdapy.simulation` runs simulations with any
[calculator](calculators.md): energy minimization, of the positions and
optionally of the box, and elastic constants. Molecular dynamics is left to
LAMMPS, GPUMD or ASE.

<!-- skip-test -->
```python
import mdapy
from mdapy.calculators import EAM
from mdapy.simulation import elastic_constants, minimize

potential = EAM("Cu.eam.alloy")
crystal = mdapy.read("Cu.data")
relaxed = minimize(crystal, potential, cell="full", max_stress=1e-4)
relaxed.converged, relaxed.energy, relaxed.stress
elastic_constants(relaxed.system, potential).constants  # (6, 6) GPa
```

## Energy minimization

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
| `system` | `System` | | | the starting configuration, with a `mass` or an `element` column |
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

## Elastic constants

`mdapy.simulation.elastic_constants(system, calculator, *, strain=1e-3,
relax=True, max_force=1e-4, max_steps=5000)` gives the elastic constants of a
periodic crystal from its stress under small strains.

<!-- skip-test -->
```python
from mdapy.simulation import elastic_constants, minimize

relaxed = minimize(crystal, potential, cell="full", max_stress=1e-4).system
result = elastic_constants(relaxed, potential)
result.constants  # (6, 6) GPa
result.bulk_modulus(), result.shear_modulus("voigt")
```

!!! warning "Relax the box first"
    The constants are those of the box as it is given. Relax it to zero
    stress first, as above, or the constants belong to a crystal under
    stress.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `system` | `System` | | | the crystal, periodic in every direction, with a `mass` or an `element` column when its positions relax |
| `calculator` | `Calculator` | | | computes the virial, and energies and forces when the positions relax |
| `strain` | float | 1e-3 | | the magnitude $\delta$ of the strains |
| `relax` | bool | `True` | | whether the positions relax in each strained box (relaxed-ion constants) or stay put (clamped-ion constants) |
| `max_force` | float | 1e-4 | eV/Å | convergence of each relaxation, as in `minimize` |
| `max_steps` | int | 5000 | | largest number of steps of each relaxation |

### Definition

The box and the atoms are strained by $\pm\delta$ in each of the six
components $k$ of the strain in Voigt order xx, yy, zz, yz, xz, xy, the shear
components as engineering strains (half on each off-diagonal entry), and the
constants are central differences of the stress $\sigma$ (positive under
tension) in the same order:

$$
C_{jk} = \frac{\sigma_j(+\delta\, \mathbf e_k) - \sigma_j(-\delta\, \mathbf e_k)}{2\delta}.
$$

With $S = C^{-1}$, the moduli of a polycrystal are the Voigt and Reuss bounds
and their mean, the Hill average
([Hill (1952)](../reference/references.md#hill1952)):

$$
\begin{aligned}
K_\mathrm V &= \tfrac19\left[C_{11} + C_{22} + C_{33} + 2(C_{12} + C_{13} + C_{23})\right], &
K_\mathrm R &= \left[S_{11} + S_{22} + S_{33} + 2(S_{12} + S_{13} + S_{23})\right]^{-1}, \\
G_\mathrm V &= \tfrac1{15}\left[C_{11} + C_{22} + C_{33} - (C_{12} + C_{13} + C_{23}) + 3(C_{44} + C_{55} + C_{66})\right], &
G_\mathrm R &= 15\left[4(S_{11} + S_{22} + S_{33}) - 4(S_{12} + S_{13} + S_{23}) + 3(S_{44} + S_{55} + S_{66})\right]^{-1}, \\
K_\mathrm H &= \tfrac12 (K_\mathrm V + K_\mathrm R), &
G_\mathrm H &= \tfrac12 (G_\mathrm V + G_\mathrm R).
\end{aligned}
$$

### Results

It returns an `ElasticResult`:

| Attribute or method | Type | Meaning |
|---|---|---|
| `constants` | `numpy.ndarray` | (6, 6) elastic constants $C$ in GPa, in Voigt order; read-only |
| `bulk_modulus(average="hill")` | float | the bulk modulus in GPa: the `"voigt"`, `"reuss"` or `"hill"` average |
| `shear_modulus(average="hill")` | float | the shear modulus in GPa, with the same averages |

`elastic_constants` raises `InvalidParameter` for a box that is not periodic
in every direction, a non-positive `strain`, a calculator without the
properties it needs, or a relaxation that does not converge within
`max_steps`; `bulk_modulus` and `shear_modulus` raise it for an unknown
`average`.
