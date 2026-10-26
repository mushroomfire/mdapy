# Quasi-harmonic approximation

`mdapy.lattice_dynamics.quasi_harmonic(system, calculator, temperatures, *,
volumes=None, pressures=None, pressure=0.0, supercell=None, mesh=20,
displacement=0.01, max_force=1e-6, max_stress=1e-4, max_steps=5000)` gives the thermal expansion, bulk modulus,
heat capacity and Gibbs energy of a crystal at finite temperature from its
[phonons](phonons.md) at several volumes.

<!-- skip-test -->
```python
import numpy as np

from mdapy.build import build_crystal
from mdapy.lattice_dynamics import primitive_cell, quasi_harmonic
from mdapy.simulation import minimize

crystal = minimize(
    build_crystal("fcc", 3.615, elements="Cu"),
    potential,
    cell="full",
    max_stress=1e-4,
).system
result = quasi_harmonic(primitive_cell(crystal), potential, np.arange(0, 1001, 50))
result.volume  # Å³ per atom at each temperature
result.thermal_expansion / 3  # linear thermal expansion coefficient, 1/K
result.bulk_modulus  # GPa
result.heat_capacity  # C_P, eV/K per atom
```

At each volume the box keeps its volume while its shape and the positions
relax, so that a crystal whose shape changes with the volume, such as hcp
with its c/a, takes its best shape at each; the phonons then give the free
energy of the vibrations. The volumes must hold the equilibrium volume at
every temperature: the default, from 0.97 to 1.09 times the volume of
`system`, reaches the melting point of most metals. Instead of volumes,
`pressures` relax the box and the positions under each pressure. The cells
of the grid are kept, for [elastic constants at finite
temperature](thermal-elastic.md).

The quasi-harmonic approximation leaves out the anharmonic vibrations
beyond the change of the frequencies with the volume, so it holds well
below the melting point; near it, molecular dynamics is needed. The bulk
modulus comes from the curvature of a fit through the volumes; with the
default nine volumes it is good to about 1 %.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `system` | `System` | | | the cell of the crystal, periodic in every direction, with an `element` or a `mass` column; best relaxed and primitive |
| `calculator` | `Calculator` | | | computes energies and forces |
| `temperatures` | array_like | | K | the temperatures, not negative |
| `volumes` | array_like | 9 from 0.97 to 1.09 | | the volumes of the phonons, as ratios to the volume of `system`; five or more |
| `pressures` | array_like | | GPa | instead of volumes, the pressures under which the box and the positions relax; five or more |
| `pressure` | float | 0 | GPa | the pressure $P$ |
| `supercell` | int or 3 ints | | | repeats of the cell for the phonons, as in [`phonons`](phonons.md) |
| `mesh` | int or 3 ints | 20 | | wave vectors along each reciprocal lattice vector for the thermal properties |
| `displacement` | float | 0.01 | Å | displacement of the atoms for the force constants |
| `max_force` | float | 1e-6 | eV/Å | convergence of the relaxation at each volume |
| `max_stress` | float | 1e-4 | GPa | convergence of the relaxation of the box |
| `max_steps` | int | 5000 | | largest number of steps of each relaxation |
| `symmetry` | bool or None | `None` | | use the symmetry operations of the crystal, found by [spglib](../analysis/space-group.md), in the phonons, as in [`phonons`](phonons.md): `True`, `False`, or by default whenever spglib is installed |

## Results

It returns a `QuasiHarmonicResult`, all per atom; arrays are read-only.

| Attribute | Type | Unit | Meaning |
|---|---|---|---|
| `temperatures` | (T,) array | K | the temperatures |
| `pressure` | float | GPa | the pressure |
| `volume` | (T,) array | Å³ | the equilibrium volume $V(T)$ |
| `gibbs_energy` | (T,) array | eV | the Gibbs energy $G(T, P)$ |
| `entropy` | (T,) array | eV/K | the entropy $S$ |
| `bulk_modulus` | (T,) array | GPa | the isothermal bulk modulus $B_T$ |
| `thermal_expansion` | (T,) array | 1/K | the volumetric thermal expansion coefficient $\alpha$; a third of it is the linear one of a cubic crystal |
| `heat_capacity` | (T,) array | eV/K | the heat capacity at constant pressure $C_P$ |
| `volumes` | (V,) array | Å³ | the volumes of the phonons |
| `energies` | (V,) array | eV | the static energy at each volume |
| `free_energies` | (V, T) array | eV | the free energy of the vibrations at each volume and temperature |
| `cells` | tuple of `System` | | the relaxed cell at each volume |

## Errors

`quasi_harmonic` raises `InvalidParameter` for a negative temperature,
both volumes and pressures, fewer than five of them, a relaxation that does
not converge within `max_steps`, imaginary frequencies at a volume, and an
equilibrium volume outside the volumes, as well as for the errors of
[`phonons`](phonons.md).

## Definition

At each temperature $T$, the free energy

$$
F(V, T) = E(V) + F_\mathrm{vib}(V, T) + PV,
$$

with $E$ the static energy and $F_\mathrm{vib}$ the harmonic free energy
of the vibrations at volume $V$, is fit by the third-order Birch–Murnaghan
form, a cubic polynomial in $V^{-2/3}$, by least squares. Its minimum is the
equilibrium volume $V(T)$ and the Gibbs energy $G = F(V(T), T)$, its
curvature there the bulk modulus $B_T = V\,\partial^2F/\partial V^2$. The
entropy $S_\mathrm{vib}$ and the heat capacity $C_V$ of the vibrations,
fit alike, give at $V(T)$

$$
S = S_\mathrm{vib}, \qquad
\alpha = \frac{1}{B_T}\left(\frac{\partial S_\mathrm{vib}}{\partial V}\right)_T, \qquad
C_P = C_V + T V B_T \alpha^2.
$$
