# Elastic constants of graphene

A sheet of graphene is stiff in its plane and soft across it. Its vibrations
across the plane, whose frequency is quadratic in the wave vector, ripple it at
any temperature, and the ripples soften it in its plane. This page computes the
elastic constants of graphene with a [NEP](../user-guide/potentials/nep.md) model of
carbon trained with TorchNEP on the GAP-2017 data, that of the
[graphene polycrystal](graphene-polycrystal.md): static, from the quasi-harmonic
approximation, and from molecular dynamics from 100 to 1000 K. It then shows how
they depend on the size of the sheet. See
[Sheets](../user-guide/properties/sheets.md),
[Quasi-harmonic approximation](../user-guide/properties/quasi-harmonic.md) and
[Elastic constants from molecular dynamics](../user-guide/properties/md-elastic.md)
for the functions themselves.

!!! note "Cite the model and the data"

    The model comes from [Wu et al. (2026), TorchNEP](../reference/references.md#wu2026torchnep),
    trained on the data of [Deringer and Csányi (2017)](../reference/references.md#deringer2017).
    Cite both when you use it.

Download the model [`nep.txt`](carbon/nep.txt) into the working directory.

## At 0 K

<!-- skip-test -->
```python
import numpy as np
import mdapy
from mdapy.build import build_crystal
from mdapy.calculators import NEP
from mdapy.properties.elastic import elastic_constants
from mdapy.simulation import minimize

potential = NEP("nep.txt")
crystal = build_crystal("graphene", 2.46, c=20.0, elements="C")
sheet = crystal.with_box(mdapy.Box(crystal.box.matrix, pbc=(True, True, False)))
sheet = minimize(sheet, potential, cell="full", max_stress=1e-3).system


def hexagonal(c):
    """C11, averaged over xx and yy, C12 and C66 of a sheet, in N/m."""
    return np.mean(np.diag(c)[:2]), c[0, 1], c[2, 2]


static = elastic_constants(sheet, potential)
print("C11 %.1f, C12 %.1f, C66 %.1f N/m" % hexagonal(static.constants))
print(f"area modulus {static.area_modulus():.1f} N/m")
```

```text
C11 359.2, C12 51.2, C66 154.0 N/m
area modulus 205.2 N/m
```

The area modulus is the tension per unit of relative change of the area:
$(C_{11} + C_{12})/2$ for a hexagonal sheet.

## The quasi-harmonic approximation

[`quasi_harmonic`](../user-guide/properties/quasi-harmonic.md) finds the area
of the sheet at each temperature from the free energy of its vibrations, and
the area modulus there. It takes a second:

<!-- skip-test -->
```python
from mdapy.properties.thermodynamics import quasi_harmonic

temperatures = np.arange(0, 951, 25.0)
qha = quasi_harmonic(sheet, potential, temperatures, mesh=100)
for t, area, modulus in zip(temperatures[::8], qha.area[::8], qha.area_modulus[::8]):
    print(f"{t:3.0f} K: area {area:.4f} Å² per atom, area modulus {modulus:.1f} N/m")
```

```text
  0 K: area 2.5940 Å² per atom, area modulus 194.2 N/m
200 K: area 2.5919 Å² per atom, area modulus 187.1 N/m
400 K: area 2.5887 Å² per atom, area modulus 177.7 N/m
600 K: area 2.5856 Å² per atom, area modulus 166.7 N/m
800 K: area 2.5827 Å² per atom, area modulus 154.5 N/m
```

The vibrations at zero point stretch the sheet by 0.5 % from its static area,
2.581 Å² per atom, which softens it to 194 N/m at 0 K. On heating, the
vibrations across the sheet shrink it and soften it. Near 950 K its area
reaches the static one, under which these vibrations have imaginary
frequencies, and the approximation ends there, as
[Sheets](../user-guide/properties/sheets.md#quasi-harmonic-approximation)
explains.

## From molecular dynamics

[`md_elastic_constants`](../user-guide/properties/md-elastic.md) takes all the
anharmonic vibrations. For a sheet it uses by default the fluctuations of the
stress at the mean area of the sheet. Here it runs on 16 × 16 cells, 512 atoms,
with a time step of 0.5 fs: 500 ps at zero pressure for the mean area, then
500 ps at that area for the fluctuations, at each temperature. The model runs on a
[GPU](../user-guide/simulation/gpu.md) in double precision, as the Born term takes
differences of the virial under strains of 10⁻⁵:

<!-- skip-test -->
```python
from mdapy.properties.elastic import md_elastic_constants
from mdapy.transform import replicate

gpu = NEP("nep.txt", device="cuda", precision="double")
big = replicate(sheet, (16, 16, 1))
hot = [100, 200, 300, 400, 600, 800, 1000]
md = [
    md_elastic_constants(
        big, gpu, t, timestep=0.0005, equilibration=20000, production=1000000
    )
    for t in hot
]
for t, result in zip(hot, md):
    print(
        "%4d K: C11 %.1f, C12 %.1f, C66 %.1f N/m" % (t, *hexagonal(result.constants)),
        f"area modulus {result.area_modulus():.1f} N/m",
    )
```

```text
 100 K: C11 349.4, C12 46.0, C66 151.4 N/m area modulus 197.7 N/m
 200 K: C11 341.0, C12 41.8, C66 148.3 N/m area modulus 191.4 N/m
 300 K: C11 332.2, C12 36.7, C66 146.4 N/m area modulus 184.5 N/m
 400 K: C11 326.1, C12 34.4, C66 144.1 N/m area modulus 180.3 N/m
 600 K: C11 309.2, C12 27.0, C66 138.9 N/m area modulus 168.0 N/m
 800 K: C11 303.4, C12 25.6, C66 137.2 N/m area modulus 164.5 N/m
1000 K: C11 298.4, C12 26.0, C66 134.2 N/m area modulus 162.2 N/m
```

Each temperature took about an hour on one V100. The standard errors are 1 to
3 N/m for C11, 0.6 to 2 N/m for C12 and 0.2 to 1.2 N/m for C66: the vibrations
across the sheet change slowly, and the fluctuations need long runs.

## Why not strains

`method="strain"` takes the mean stress under strains of +1 % and −1 %. A
strain that compresses a sheet wrinkles it: the sheet bends out of its plane
rather than shorten, and resists less than under a stretch. On the same 512
atoms at 100 K, 1 % of stretch along x raises the tension by 3.4 N/m, and 1 %
of compression lowers it by 2.4 N/m. The heights of the atoms spread by 0.10 Å
under the stretch, 0.13 Å at rest and 0.48 Å under the compression. The strains
average the two responses: at 100 K they give C11 298 N/m along x and 247 N/m
along y, which the symmetry of graphene makes equal, against 349 N/m for both
from the fluctuations. `md_elastic_constants` therefore takes the fluctuations
for a sheet unless `method="strain"` is given.

## Molecular dynamics and the quasi-harmonic approximation

On 16 × 16 cells the two agree within 5 % from 100 to 800 K, in the figure below,
when the approximation takes a mesh of as many cells as the sheet: molecular dynamics
is the stiffer below some 250 K, as its atoms are classical, without the vibrations at
zero point that soften the sheet in the approximation, and the softer above, as it
takes the anharmonic vibrations that the approximation leaves out.

These grow with the size of the sheet. The vibrations across the sheet are soft:
their frequency falls as the square of the wave vector, so the longest of them, as
long as the sheet, move the most, and they ripple the sheet. Stretching a rippled
sheet first flattens its ripples, which costs little. The sheet is therefore softer
in its plane than a flat one, and the more so the larger it is, as longer ripples fit
in it. The quasi-harmonic approximation takes the vibrations as harmonic at each
area: it sees that they stiffen as the sheet is stretched, which softens the sheet as
it heats, but not that ripples of different wavelengths change one another.
Molecular dynamics takes both. To compare them on the same vibrations, give the
approximation a `mesh` of as many cells as the sheet of molecular dynamics, here at
300 K:

<!-- skip-test -->
```python
cells = [8, 16, 24, 32]
by_size = [
    md_elastic_constants(
        replicate(sheet, (n, n, 1)), gpu, 300.0, timestep=0.0005,
        equilibration=20000, production=1000000,
    )
    for n in cells
]  # fmt: skip
by_mesh = [
    quasi_harmonic(sheet, potential, [300.0], mesh=n).area_modulus[0] for n in cells
]
```

| Sheet | Atoms | Area modulus, molecular dynamics | Area modulus, quasi-harmonic |
|---|---:|---:|---:|
| 8 × 8 | 128 | 195.5 ± 0.1 N/m | 190.3 N/m |
| 16 × 16 | 512 | 184.5 ± 1.2 N/m | 186.5 N/m |
| 24 × 24 | 1152 | 167.6 ± 2.1 N/m | 184.8 N/m |
| 32 × 32 | 2048 | 158.6 ± 4.8 N/m | 184.0 N/m |

On 8 × 8 cells molecular dynamics is 3 % the stiffer, and 4 % at 100 K, 201.4 ± 0.1
N/m against 193.5 N/m: the vibrations at zero point of the approximation. On larger
sheets molecular dynamics goes on softening, by 19 % from 8 × 8 to 32 × 32 cells,
while the approximation levels off, at 183 N/m for a mesh of 100. A crystal has no
such vibrations, and there the two agree within 5 % up to 1000 K, as for
[CrCoNi](crconi.md#from-molecular-dynamics). The constants of a free sheet from
molecular dynamics are those of its size.

<!-- skip-test -->
```python
from mdapy.visualization import plot

strained = [
    md_elastic_constants(
        big, gpu, t, timestep=0.0005, equilibration=20000, production=200000,
        method="strain",
    )
    for t in hot
]  # fmt: skip
small = quasi_harmonic(sheet, potential, temperatures, mesh=16)


def spread(result):
    """The standard error of the area modulus, the errors taken as independent."""
    e = result.errors
    return np.sqrt(e[0, 0] ** 2 + e[1, 1] ** 2 + 4 * e[0, 1] ** 2) / 4


with plot.style():
    fig, (left, right) = plot.figure("double", ncols=2)
    left.plot(temperatures, qha.area_modulus, label="quasi-harmonic, mesh 100")
    left.plot(temperatures, small.area_modulus, "--", label="quasi-harmonic, mesh 16")
    left.errorbar(hot, [r.area_modulus() for r in md], [spread(r) for r in md],
                  fmt="o", capsize=2, label="MD, fluctuations")  # fmt: skip
    left.plot(hot, [r.area_modulus() for r in strained], "x", color="0.5",
              label="MD, strains")  # fmt: skip
    left.plot(0, static.area_modulus(), "s", color="0.2", label="static")
    left.set_xlabel("temperature (K)")
    left.set_ylabel("area modulus (N/m)")
    left.legend()
    right.plot(cells, by_mesh, "--", marker="^", color="C1", label="quasi-harmonic")
    right.errorbar(cells, [r.area_modulus() for r in by_size],
                   [spread(r) for r in by_size], fmt="o", color="C2", capsize=2,
                   label="MD, fluctuations")  # fmt: skip
    right.set_xticks(cells, [f"{n} × {n}" for n in cells])
    right.set_xlabel("cells of the sheet, and of the mesh")
    right.set_ylabel("area modulus at 300 K (N/m)")
    right.legend()
    plot.save(fig, "images/graphene-elastic", formats=("png",))
```

![The area modulus of graphene against the temperature and the size of the sheet](images/graphene-elastic.png)

*The area modulus of graphene with the NEP model. Left, against the temperature:
the lines are the quasi-harmonic approximation, on a mesh of 100 cells and on one
of 16, as many as the sheet of molecular dynamics; the points are molecular
dynamics on 16 × 16 cells, by the fluctuations of the stress, with their
standard errors, and, in gray, by strains, which the wrinkling of the
compressed sheet makes too soft. Right, at 300 K against the size of the sheet:
the two agree on the smaller sheets and part as it grows.*

## What to keep in mind

- **Molecular dynamics is classical.** The quasi-harmonic approximation
  includes the vibrations at zero point, which stretch the sheet and soften it
  by 5 % at 0 K. Below some 250 K molecular dynamics is therefore the stiffer.
- **The quasi-harmonic approximation ends near 950 K** for this model, where the
  sheet shrinks to its static area. A tension, through `pressure`, keeps it
  stretched.
- **Fluctuations need long runs on a sheet.** A million steps of 0.5 fs give
  errors of 1 to 3 N/m on 512 atoms, and the errors grow with the size of the
  sheet.
- **The size is part of the result.** Give the number of cells with the
  constants of a free sheet.
