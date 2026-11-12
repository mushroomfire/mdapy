# Sheets

[Elastic constants](elastic-constants.md), [phonons](phonons.md), the
[quasi-harmonic approximation](quasi-harmonic.md), [elastic constants from
molecular dynamics](md-elastic.md) and [molecular dynamics](dynamics.md) of
two-dimensional materials such as graphene. A box that is periodic along two
vectors in the x-y plane and not along a third along z makes a system a sheet;
nothing else changes in the calls.

<!-- skip-test -->
```python
import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import LAMMPS
from mdapy.lattice_dynamics import elastic_constants, phonons
from mdapy.simulation import minimize

potential = LAMMPS("pair_style lcbop\npair_coeff * * C.lcbop C", elements=["C"])
crystal = build_crystal("graphene", 2.46, c=20.0, elements="C")
sheet = crystal.with_box(mdapy.Box(crystal.box.matrix, pbc=(True, True, False)))
sheet = minimize(sheet, potential, cell="full", max_stress=1e-3).system

elastic = elastic_constants(sheet, potential)
elastic.constants  # (3, 3) N/m: 331, 73 and 129
elastic.young_modulus(), elastic.poisson_ratio()  # 315 N/m, 0.22

result = phonons(sheet, potential)
bands = result.band_structure("hexagonal")  # Γ-M-K-Γ
dos = result.dos(100)  # the mesh is 100 x 100 x 1
thermal = result.thermal([100, 300, 1000], mesh=100)  # F, E, S, C_V per atom
```

The phonons of graphene, drawn with the same calls as those of a crystal:

<!-- skip-test -->
```python
from mdapy.visualization import plot

with plot.style():
    fig, (left, right) = plot.figure(
        "single", ncols=2, sharey=True, width_ratios=[3, 1]
    )
    plot.band_structure(bands, ax=left)
    plot.phonon_dos(dos, ax=right)
    plot.save(fig, "images/sheets", formats=("png",))
```

![Phonons of graphene](images/sheets.png)

*Graphene with the LCBOP potential of LAMMPS. The lowest branch near Γ, the
vibration across the sheet, is quadratic; the two above it are linear.*

The height of the box does not matter and the atoms need no vacuum around
them: a sheet has no images along z. A calculator that always takes the box as
periodic, such as [VASP](vasp.md), needs a box high enough that the images of
the sheet do not meet.

## Elastic constants

`elastic_constants` strains the sheet along xx, yy and xy only and returns a
`SheetElasticResult`, per unit of the area of the sheet, so that no thickness
has to be chosen. The parameters are those of
[`elastic_constants`](elastic-constants.md).

| Attribute or method | Type | Meaning |
|---|---|---|
| `constants` | (3, 3) float64 | elastic constants $C$ in N/m, in the order xx, yy, xy; read-only |
| `young_modulus(axis="x")` | float64 | the Young's modulus in N/m along `"x"` or `"y"` |
| `poisson_ratio(axis="x")` | float64 | the contraction across a stretch along `"x"` or `"y"`, over the stretch |
| `shear_modulus()` | float64 | the shear modulus in N/m |
| `area_modulus()` | float64 | the area modulus in N/m: the tension per unit of relative change of the area |

With $S = C^{-1}$ these are

$$
E_x = \frac{1}{S_{11}}, \quad E_y = \frac{1}{S_{22}}, \quad
\nu_x = -\frac{S_{12}}{S_{11}}, \quad \nu_y = -\frac{S_{12}}{S_{22}}, \quad
G = \frac{1}{S_{33}}, \quad K = \frac{1}{S_{11} + S_{22} + 2 S_{12}}.
$$

## Phonons

`phonons` takes the parameters and returns the `Phonons` of a crystal, with
these differences:

- `supercell` repeats the cell along its plane only: one integer means both
  vectors, three integers end in 1, and by default it is the supercell nearest
  a square at least twice the cutoff of the calculator wide;
- a `mesh` of one integer in `dos` and `thermal` means the plane, and three
  integers end in 1. The vibrations across the sheet make the thermal
  properties converge slowly with the mesh, so a mesh of 100 is a good start;
- the paths `"auto"` and `"hexagonal"` of `band_structure` keep the sections
  in the plane: Γ-M-K-Γ for a hexagonal sheet.

`primitive_cell` takes a sheet and gives the primitive cell of its plane, in a
box of the same height.

## Quasi-harmonic approximation

`quasi_harmonic` computes the phonons of a sheet at several areas, the
height of its box kept, and returns a `SheetQuasiHarmonicResult`:

<!-- skip-test -->
```python
import numpy as np
from mdapy.lattice_dynamics import quasi_harmonic

qha = quasi_harmonic(
    sheet, potential, [0, 100, 300], volumes=np.linspace(1.0, 1.025, 6), mesh=100
)
qha.area  # Å² per atom: 2.641, 2.639, 2.630, the relaxed sheet 2.619
qha.thermal_expansion  # 1/K: 0, -1.1e-5, -2.3e-5, as the sheet shrinks
qha.area_modulus  # N/m: 191, 178, 134
```

`volumes` gives the areas, as ratios to the area of `sheet`, and
`pressures` and `pressure` are in the plane, in N/m: a tension is a negative
pressure. The default areas are 9 from 1 to 1.04, as a sheet compressed in
its plane bends: its vibrations across the plane, quadratic in the wave
vector, have imaginary frequencies under any compression the mesh can
resolve, and `quasi_harmonic` then raises `InvalidParameter`. A sheet that
shrinks on heating, as graphene, can therefore be followed only while its
equilibrium area stays above that of the relaxed sheet: with LCBOP up to some
400 K, its zero-point vibrations stretching it by 0.8 %. A tension, through
`pressure`, keeps it stretched. The long-range part of LCBOP reaches its
cutoff between the default areas, so the example takes fewer.

| Attribute | Type | Meaning |
|---|---|---|
| `temperatures` | (T,) float64 | the temperatures in K |
| `pressure` | float64 | the pressure in the plane in N/m |
| `area` | (T,) float64 | the equilibrium area per atom in Å² |
| `gibbs_energy` | (T,) float64 | the Gibbs energy per atom in eV: static, vibrational and the pressure times the area |
| `entropy` | (T,) float64 | the entropy per atom in eV/K |
| `area_modulus` | (T,) float64 | the isothermal area modulus in N/m |
| `thermal_expansion` | (T,) float64 | the thermal expansion of the area in 1/K |
| `heat_capacity` | (T,) float64 | the heat capacity at constant pressure per atom in eV/K |
| `areas`, `energies` | (A,) float64 | the areas per atom of the phonons in Å² and the static energy at each in eV |
| `free_energies` | (A, T) float64 | the vibrational free energy per atom at each area and temperature in eV |
| `cells` | tuple of `System` | the relaxed cell at each area |

The free energy is fitted, at each temperature, by a cubic polynomial in
$1/A$, the form of the third-order Birch–Murnaghan equation in two dimensions;
the area modulus is $K = A\, \partial^2 G / \partial A^2$, the thermal
expansion $\alpha = (\partial S/\partial A)/K$ and the heat capacity
$C_\sigma = C_A + T A K \alpha^2$.

## Elastic constants from molecular dynamics

`md_elastic_constants` takes a sheet and returns its `SheetElasticResult` in
N/m, with the standard errors in `errors`: the mean shape of the sheet under
`pressure` in its plane, then the mean tensions under strains along xx, yy
and xy. `pressure` is in GPa over the height of the box, as in molecular
dynamics below, and the barostats change the sheet in its plane only.

<!-- skip-test -->
```python
from mdapy.lattice_dynamics import md_elastic_constants
from mdapy.transform import replicate

hot = md_elastic_constants(
    replicate(sheet, (8, 8, 1)), potential, 300.0, timestep=0.0005
)
hot.constants, hot.errors  # (3, 3) N/m
```

## In gigapascals

A sheet has no thickness of its own, so its constants are in N/m. With a thickness
$h$ chosen by convention, such as the 3.35 Å between the layers of graphite, the
constants of a material of that thickness are $C[\mathrm{GPa}] = 10\, C[\mathrm{N/m}]
/ h[\text{Å}]$: 331 N/m is 988 GPa.

## Molecular dynamics

[`MolecularDynamics`](dynamics.md) runs a sheet as it is. Its barostats change the
sheet in its plane and keep the height of the box: alike along its two vectors
for `"iso"` and `"xy"`, along each for `"aniso"`, and in any way in the plane,
with the angle between them, for `"tri"`:

<!-- skip-test -->
```python
from mdapy.simulation import MolecularDynamics

with MolecularDynamics(sheet, potential) as md:
    md.velocities(600.0, seed=1)
    md.run(20000, ensemble="npt", temperature=300.0, pressure=0.0, barostat="xy")
```

LAMMPS divides the forces by the volume of the box, with its vacuum. A tension of
$\sigma$ in N/m is therefore the pressure $-10\, \sigma / h$ GPa, with the height of the
box $h$ in Å; at zero pressure the height does not matter.

## Errors

`elastic_constants`, `md_elastic_constants`, `quasi_harmonic`, `phonons` and
`primitive_cell` raise `InvalidParameter` for a box that is not periodic in every
direction and is not that of a sheet, and `MolecularDynamics.run` for a barostat
of such a box.
`phonons` raises it for a supercell that leaves the plane, `dos` and `thermal`
for a mesh with more than one wave vector along z. The methods of
`SheetElasticResult` raise it for an axis other than `"x"` or `"y"` and for
constants that cannot be inverted.

`quasi_harmonic` raises it at an area where the sheet has imaginary frequencies,
as when it is compressed, and where its equilibrium area is outside the areas.
The moduli of `ElasticResult`, for a sheet in a periodic box with vacuum, raise
`InvalidParameter`, as its constants cannot be inverted.
