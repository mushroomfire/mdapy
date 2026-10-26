# Elastic constants from molecular dynamics

`mdapy.lattice_dynamics.md_elastic_constants(system, potential, temperature,
*, pressure=0.0, strain=0.01, timestep=0.001, equilibration=10000,
production=20000, every=10, blocks=5, barostat="tri", seed=1)` gives the
isothermal elastic constants of a crystal at a temperature from the mean
stress of [molecular dynamics](dynamics.md) in LAMMPS under small strains,
with their standard errors. It takes in the anharmonic vibrations that the
[quasi-harmonic approximation](thermal-elastic.md) leaves out, at the cost
of thirteen runs.

<!-- skip-test -->
```python
from mdapy.calculators import LAMMPS
from mdapy.lattice_dynamics import md_elastic_constants
from mdapy.transform import replicate

potential = LAMMPS(
    "pair_style eam/alloy\npair_coeff * * Cu_mishin1.eam.alloy Cu", elements=["Cu"]
)
crystal = replicate(cell, (6, 6, 6))  # large enough for small fluctuations
result = md_elastic_constants(crystal, potential, 600.0, timestep=0.002)
result.constants[0, 0], result.errors[0, 0]  # C11 and its standard error, GPa
result.bulk_modulus(), result.shear_modulus()
```

The crystal first runs at the temperature and pressure (Nosé–Hoover,
`fix npt`): `equilibration` steps, then `production` steps over which its box
is averaged every `every` steps. Its last atoms, put in the mean box, are the
reference. For each of the six components of the strain in Voigt order,
strains of $+\delta$ and $-\delta$ of the reference (shears as engineering
strains, half on each off-diagonal entry, the atoms and their velocities
moved with the box) each run `equilibration` steps at constant volume and
temperature (`fix nvt`), then `production` steps over which the stress is
averaged:

$$
C_{jk} = \frac{\bar\sigma_j(+\delta\,\mathbf e_k) - \bar\sigma_j(-\delta\,\mathbf e_k)}{2\delta},
$$

with $\sigma$ positive under tension. The standard error of each mean comes
from the means of `blocks` blocks of the run, and the two errors of a
column add in quadrature. The constants are those at constant temperature;
under a pressure, the stress–strain coefficients.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `system` | `System` | | | the crystal, periodic in every direction, with an `element` column |
| `potential` | `LAMMPS` | | | the interaction, as a [LAMMPS calculator](lammps.md) |
| `temperature` | float | | K | the temperature, positive |
| `pressure` | float | 0 | GPa | the pressure |
| `strain` | float | 0.01 | | the magnitude $\delta$ of the strains |
| `timestep` | float | 0.001 | ps | the time step |
| `equilibration` | int | 10000 | | steps before each average |
| `production` | int | 20000 | | steps of each average |
| `every` | int | 10 | | steps between the values averaged |
| `blocks` | int | 5 | | blocks of each average for its standard error |
| `barostat` | str | `"tri"` | | how the box finds its mean shape: `"iso"`, `"aniso"` or `"tri"` |
| `seed` | int | 1 | | seed of the starting velocities |

## Results

An `ElasticResult`, as of [`elastic_constants`](elastic-constants.md), with
`errors`, the (6, 6) standard errors in GPa.

## Errors

`md_elastic_constants` raises `InvalidParameter` for a box that is not
periodic, a temperature, strain or numbers of steps that are not positive,
fewer than two blocks or fewer samples than blocks, and the errors of
[`MolecularDynamics`](dynamics.md).
