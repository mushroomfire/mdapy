# Elastic constants from molecular dynamics

`mdapy.lattice_dynamics.md_elastic_constants(system, potential, temperature,
*, pressure=0.0, strain=0.01, timestep=0.001, equilibration=10000,
production=20000, every=10, blocks=5, barostat="tri", seed=1,
method="strain")` gives the isothermal elastic constants of a crystal at a
temperature from [molecular dynamics](dynamics.md) in LAMMPS, with their
standard errors: from the mean stress under small strains, at the cost of
thirteen runs, or, with `method="fluctuation"`, from the fluctuations of the
stress in one run, [below](#from-the-fluctuations-of-the-stress). Both take
in the anharmonic vibrations that the [quasi-harmonic
approximation](thermal-elastic.md) leaves out.

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
temperature, with a Langevin thermostat (`fix langevin` and `fix nve`), then
`production` steps over which the stress is averaged. The two runs of a
component share the noise of the thermostat, which their difference cancels
in part. A Nosé–Hoover thermostat, being deterministic, does not bring all
the vibrations of a crystal back to the temperature after a strain: with it
C11 of silicon with the Tersoff potential at 300 K came out 2 % too high.

$$
C_{jk} = \frac{\bar\sigma_j(+\delta\,\mathbf e_k) - \bar\sigma_j(-\delta\,\mathbf e_k)}{2\delta},
$$

with $\sigma$ positive under tension. The standard error of each mean comes
from the means of `blocks` blocks of the run, and the two errors of a
column add in quadrature. Each entry off the diagonal is the mean of its
two estimates, $C_{jk}$ and $C_{kj}$, so that the constants are symmetric.
The constants are those at constant temperature; under a pressure, the
stress–strain coefficients.

A strain of 1 % takes the crystal beyond its linear response where the
thermal noise is small: at 20 K C11 of copper came out 0.2 % above its limit
of small strains. At low temperatures a strain of 0.2 %, with longer runs,
gives that limit.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `system` | `System` | | | the crystal, periodic in every direction, or a [sheet](sheets.md#elastic-constants-from-molecular-dynamics), with an `element` column |
| `potential` | `LAMMPS` | | | the interaction, as a [LAMMPS calculator](lammps.md) |
| `temperature` | float64 | | K | the temperature, positive |
| `pressure` | float64 | 0 | GPa | the pressure |
| `strain` | float64 | 0.01 | | the magnitude $\delta$ of the strains |
| `timestep` | float64 | 0.001 | ps | the time step |
| `equilibration` | int32 | 10000 | | steps before each average |
| `production` | int32 | 20000 | | steps of each average |
| `every` | int32 | 10 | | steps between the values averaged |
| `blocks` | int32 | 5 | | blocks of each average for its standard error |
| `barostat` | str | `"tri"` | | how the box finds its mean shape: `"iso"`, `"aniso"` or `"tri"` |
| `seed` | int32 | 1 | | seed of the starting velocities and of the thermostat of the strained runs |
| `method` | str | `"strain"` | | `"strain"`, the mean stress under strains, or `"fluctuation"`, the fluctuations of the stress |

## From the fluctuations of the stress

With `method="fluctuation"` the crystal runs once more after the mean box:
`equilibration` steps and then `production` steps in the reference box, at
the temperature with a Langevin thermostat. The constants are

$$
C_{ijkl} = \langle C^{\mathrm B}_{ijkl} \rangle
- \frac{V}{k_B T} \left( \langle \sigma_{ij} \sigma_{kl} \rangle
- \langle \sigma_{ij} \rangle \langle \sigma_{kl} \rangle \right)
+ \frac{(N - 1) k_B T}{V} \left( \delta_{ik} \delta_{jl} + \delta_{il} \delta_{jk} \right),
$$

the Born term, the second derivatives of the energy with the strain (LAMMPS
`compute born/matrix numdiff`, strains of $10^{-5}$, every ten records),
then the fluctuations of the virial stress $\sigma$ (every `every` steps),
then the momenta of the atoms but their center of mass. Under a mean stress
they are made stress-strain coefficients, as the strains give them (Wallace):
under a pressure $P$, $C_{11} - P$, $C_{12} + P$ and $C_{44} - P$. The errors
come from `blocks` blocks of the run; `strain` is unused.

<!-- skip-test -->
```python
fluctuating = md_elastic_constants(
    crystal, potential, 600.0, timestep=0.002, production=300000, method="fluctuation"
)
```

The two methods agree within their errors, for copper (EAM of Mishin) from 20
to 900 K and from 256 to 2048 atoms, for copper under 10 GPa, and for C11 and
C12 of silicon (Tersoff) at 10 and 300 K. What they cost differs:

| | Strain | Fluctuation |
|---|---|---|
| runs | 13 | 2 |
| error at equal steps, copper at 300 K | 1 | 1/1.5 to 1/6 for C11 and C12, 1/4 to 1/25 for C44 |
| error with the size | falls as $1/\sqrt N$ | does not change: 256 atoms do |
| constants of strong internal relaxation, as C44 of silicon (92 GPa unrelaxed, 10 relaxed) | ±0.1 GPa | ±1.5 GPa: the fluctuations must cancel most of the Born term |
| at low temperature | 1 % strain beyond the linear response (+0.2 % on C11 of copper at 20 K) | exact in the limit of small strains |
| liquids and glasses | the box cannot keep a liquid | the bulk modulus, and no shear |
| cost of one record | a step | a step, and some 15 steps for the Born term every ten records |

Take the fluctuations for metals and other crystals whose atoms do not relax
much within their cell, for small cells, at low temperatures, and for
liquids; take the strains for crystals whose constants relax strongly, as
the shear of the diamond structures, and for large cells. With the same
number of steps the fluctuations take one run where the strains take twelve:
`production=300000` costs about what the defaults of the strains cost.

## Results

An `ElasticResult`, as of [`elastic_constants`](elastic-constants.md), with
`errors`, the (6, 6) standard errors in GPa.

## Errors

`md_elastic_constants` raises `InvalidParameter` for a box that is neither
periodic nor that of a sheet, a temperature, strain or numbers of steps that are not positive,
fewer than two blocks or fewer samples than blocks, a `method` other than
`"strain"` and `"fluctuation"`, and the errors of
[`MolecularDynamics`](dynamics.md). The fluctuations need a LAMMPS with the
package EXTRA-COMPUTE, for `compute born/matrix`.
