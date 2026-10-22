# Calculators

A calculator computes the energy, forces and virial of a system. mdapy
defines one interface, `mdapy.calculators.Calculator`, which its own
potentials implement and which other packages can implement to work with
everything in mdapy that needs energies and forces.

<!-- skip-test -->
```python
import mdapy
from mdapy.calculators import EAM

potential = EAM("CuNi.eam.alloy")
system = mdapy.read("alloy.xyz")
result = potential.compute(system, ("energy", "forces", "virial"))
result.energy  # eV
result.forces  # (N, 3) eV/Å
result.stress(system)  # (3, 3) GPa
system = result.attach(system)  # forces as a column, energy and virial in info
```

Minimizations and elastic constants with a calculator are on the page
[Simulation](simulation.md).

## The calculator interface

Every calculator has these members:

| Name | Type | Meaning |
|---|---|---|
| `cutoff` | float | interaction cutoff in Å |
| `elements` | tuple of str or `None` | the elements the calculator handles, `None` for any |
| `implemented` | frozenset of str | the properties `compute` can return |
| `compute(system, properties=("energy", "forces"), *, neighbors=None)` | `CalculatorResult` | at least the requested properties |

`properties` is any of `"energy"`, `"atomic_energies"`, `"forces"`,
`"virial"` and `"atomic_virials"` (`mdapy.calculators.PROPERTIES`); a
calculator may compute more than requested. `neighbors` is a
[`NeighborList`](neighbor.md) of the same system built with a cutoff of at
least `calculator.cutoff`, so that several calculators and analyses share
one search; it is built when not given.

`compute` raises
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter) for
an unknown or unimplemented property, and for a neighbor list built for
another system, with a smaller cutoff, or of nearest neighbors;
[`MissingProperty`](../reference/api.md#mdapy.errors.MissingProperty) when
the system lacks a column the calculator needs.

## Results

A `CalculatorResult` holds:

| Attribute | Shape | Unit | Meaning |
|---|---|---|---|
| `energy` | float | eV | total energy |
| `atomic_energies` | (N,) | eV | they sum to `energy` |
| `forces` | (N, 3) | eV/Å | |
| `virial` | (3, 3) | eV | positive under compression |
| `atomic_virials` | (N, 3, 3) | eV | they sum to `virial` |

Values that were not computed are `None`; the arrays are read-only.

| Method | Returns |
|---|---|
| `stress(system)` | (3, 3) stress $\sigma = -W/V$ in GPa, positive under tension; `InvalidParameter` without a virial |
| `attach(system)` | the system with `forces`, `atomic_energies` and `atomic_virials` as columns and `energy` and `virial` in `info` |

The virial is the response of the energy to a homogeneous strain
$\varepsilon$ of the box and the positions,

$$
W_{ab} = -\frac{\partial E}{\partial \varepsilon_{ab}} = \sum_i x_{i,a} F_{i,b},
$$

the second form holding for systems without periodic directions.

## Writing a calculator

A calculator subclasses `Calculator`, sets `cutoff`, `elements` and
`implemented`, and defines `compute`, which must accept `neighbors=`.

`mdapy.calculators.testing.check_calculator(calculator, system)` checks an
implementation on a small system and raises `AssertionError` naming the first
failed check and its largest deviation. It checks that atomic energies and
atomic virials sum to the energy and the virial; that forces and the virial
agree with finite differences of the energy; that translation, rotation and
reordering of the atoms change nothing beyond rotating and reordering
vectors; that a cell doubled along **a** has twice the energy; and that a
neighbor list given with a larger cutoff changes nothing.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `calculator` | `Calculator` | | | the calculator to check |
| `system` | `System` | | | a few dozen atoms the calculator handles |
| `displacement` | float | 1e-4 | Å | step of the finite differences, also used as strain |
| `rtol`, `atol` | float | 1e-6 | | relative and absolute tolerances of the comparisons |
| `atoms` | int | 3 | | atoms whose forces are checked by finite differences |
| `seed` | int | 0 | | seed of the random rotation, translation and choice of atoms |

## EAM

`EAM(path)` reads an embedded atom method potential in the `eam/alloy`
(setfl) format of LAMMPS and computes every property.

| Attribute | Type | Meaning |
|---|---|---|
| `path` | str or `None` | the potential file, `None` for a built potential |
| `elements` | tuple of str or `None` | the elements of the potential, in the order of the file; `None` for an average-atom potential |
| `masses` | (n,) float64 | their masses in amu; the mass of the average atom for an average-atom potential |
| `cutoff` | float | cutoff radius in Å |

| Method | Returns |
|---|---|
| `compute(system, properties, *, neighbors=None, progress=None, cancel=None)` | as every calculator, with the `progress=` and `cancel=` of [Parallelism](parallelism.md#progress-and-cancellation) |
| `write(path)` | writes the potential as an `eam/alloy` file, which LAMMPS reads and which reads back to the same tables |
| `EAM.zhou(elements, *, nr=2000, nrho=2000)` | the potential of Zhou, Johnson and Wadley for the elements |
| `average(concentrations)` | the average-atom potential of an alloy of the elements |

The atoms need an `element` column; elements the potential does not have
raise `InvalidParameter`, and a malformed file raises
[`FileFormatError`](../reference/api.md#mdapy.errors.FileFormatError) with
its line. A file may name species that are not elements, such as the `A` of
an average atom; they are kept when the potential is written, but no atom
of a system is one of them.

The energy of atom $i$ is

$$
E_i = F_{s_i}\Big(\sum_j \rho_{s_j}(r_{ij})\Big) + \frac{1}{2}\sum_j \phi_{s_i s_j}(r_{ij}),
$$

with the embedding energy $F$, the electron density $\rho$ and the pair
potential $\phi$ of the species $s$ of the atoms, interpolated from the
tables of the file as LAMMPS `pair_style eam/alloy` does. Each pair's virial
is shared equally by its two atoms.

!!! quote "Please cite"
    [Daw and Baskes (1984)](../reference/references.md#daw1984), and the paper of the potential.

### Potentials of Zhou, Johnson and Wadley

<!-- skip-test -->
```python
potential = EAM.zhou(["Cu", "Ni", "Al"])
potential.write("CuNiAl.eam.alloy")
```

`EAM.zhou` builds the potential for any of Cu, Ag, Au, Ni, Pd, Pt, Al, Pb,
Fe, Mo, Ta, W, Mg, Co, Ti, Zr and Cr, with the tables that the database tool
of LAMMPS (`tools/eam_database`) writes for them: the parameters of
[Zhou et al. (2004)](../reference/references.md#zhou2004), those of
[Lin et al. (2008)](../reference/references.md#lin2008) for Cr, and their rule for the pair
potential between two elements,

$$
\phi_{ab}(r) = \frac{1}{2}\left(\frac{\rho_b(r)}{\rho_a(r)}\phi_{aa}(r)
+ \frac{\rho_a(r)}{\rho_b(r)}\phi_{bb}(r)\right).
$$

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `elements` | sequence of str | | | the elements, in the order of the file |
| `nr` | int | 2000 | | points of the tables in distance, up to the cutoff $\sqrt{10}/2$ times the largest fcc lattice constant $\sqrt 2\, r_e$ |
| `nrho` | int | 2000 | | points of the embedding tables in density |

The parameters are fitted to the pure metals; the pair potentials between
elements come from the rule above, not from a fit to alloys. An unknown or
repeated element raises `InvalidParameter`.

!!! quote "Please cite"
    [Zhou et al. (2004)](../reference/references.md#zhou2004); with Cr, also [Lin et al. (2008)](../reference/references.md#lin2008).

### Average atom

<!-- skip-test -->
```python
average = potential.average({"Cu": 0.4, "Ni": 0.4, "Al": 0.2})
average.compute(system)  # every atom is the average atom
average.write("CuNiAl.average.eam.alloy")  # Cu, Ni, Al and the average atom A
```

`average(concentrations)` returns the average-atom potential of
[Varvenne et al. (2016)](../reference/references.md#varvenne2016) for a random alloy with the
fractions $c_a$ of the elements:

$$
F_A = \sum_a c_a F_a, \qquad \rho_A = \sum_a c_a \rho_a, \qquad
\phi_{Ab} = \sum_a c_a \phi_{ab}, \qquad
\phi_{AA} = \sum_{a,b} c_a c_b \phi_{ab}.
$$

`concentrations` is a mapping from element to fraction, or the fractions in
the order of `elements`; elements left out have none, and the fractions sum
to 1. In mdapy every atom of a system is the average atom, whatever its
element. The written file holds the elements of the potential and the
average atom, named `A`, last, so that LAMMPS can mix them, for example
with `pair_coeff * * CuNiAl.average.eam.alloy A Cu` for a Cu solute in the
average alloy. Invalid fractions raise `InvalidParameter`.

!!! quote "Please cite"
    [Varvenne et al. (2016)](../reference/references.md#varvenne2016), and the paper of the potential averaged.

## NEP

`NEP(path)` reads a neuroevolution potential of version 4 in the `nep.txt`
format of GPUMD, without (`nep4`) or with (`nep4_zbl`) a ZBL repulsion, and
computes every property.

<!-- skip-test -->
```python
from mdapy.calculators import NEP

potential = NEP("nep.txt")
result = potential.compute(system, ("energy", "forces", "virial"))
descriptors = potential.descriptors(system)  # (N, dim)
```

| Attribute | Type | Meaning |
|---|---|---|
| `path` | str | the potential file |
| `elements` | tuple of str | the elements of the potential, in the order of the file |
| `cutoff` | float | the largest radial or angular cutoff in Å |

| Method | Returns |
|---|---|
| `compute(system, properties, *, neighbors=None, progress=None, cancel=None)` | as every calculator, with the `progress=` and `cancel=` of [Parallelism](parallelism.md#progress-and-cancellation) |
| `descriptors(system, *, neighbors=None, progress=None, cancel=None)` | (N, dim) descriptors of each atom, scaled as the network sees them: the radial ones, then the angular ones by $l$ and $n$, then the four-body and five-body ones by $n$ |

The atoms need an `element` column; elements the potential does not have
raise `InvalidParameter`. Other models of GPUMD (NEP3, NEP5, charge, dipole
and polarizability models) and malformed files raise
[`FileFormatError`](../reference/api.md#mdapy.errors.FileFormatError).

The model is that of GPUMD ([Fan et al. (2022)](../reference/references.md#fan2022)), version 4
([Song et al. (2024)](../reference/references.md#song2024)): radial descriptors from Chebyshev functions of
the distance, angular ones from real spherical harmonics up to
$l_\text{max}$, with optional four-body and five-body invariants, each scaled
by its factor in the file and fed to one hidden layer of $\tanh$ neurons per
element, which gives the energy $E_i$ of atom $i$. The cutoff of a pair is
the mean of the cutoffs of its two elements. With ZBL, each pair within the
angular cutoff adds the screened Coulomb repulsion of
[Ziegler et al. (1985)](../reference/references.md#ziegler1985),
$V(r) = \frac{k_e Z_i Z_j}{r}\,\phi(r/a)\, f_s(r)$, half to each atom; the
switch $f_s$ is 1 below an inner cutoff and falls as a cosine to 0 at an
outer cutoff, as the `zbl` line of the file chooses:

| `zbl` line | screening $\phi$ | inner cutoff | outer cutoff |
|---|---|---|---|
| `zbl r1 r2` | universal ZBL | r1 | r2 |
| `zbl r1 r2 f` | universal ZBL | 0 | the smaller of r2 and f times the sum of the two covalent radii GPUMD uses |
| `zbl 0 0` | from a table at the end of the file, per pair of elements | from the table | from the table |

As in GPUMD, the derivative of $E_i$ by the vector to neighbor $j$ adds to
the atomic virial of atom $j$, so atomic virials are those of GPUMD; EAM
shares each pair's virial equally between its two atoms, as LAMMPS does. Both
sum to the same virial.

!!! quote "Please cite"
    [Fan et al. (2022)](../reference/references.md#fan2022) and [Song et al. (2024)](../reference/references.md#song2024); with ZBL, also [Ziegler et al. (1985)](../reference/references.md#ziegler1985); and the paper of the potential.

## LAMMPS

<!-- skip-test -->
```python
from mdapy.calculators import LAMMPS

calculator = LAMMPS(
    "pair_style eam/alloy\npair_coeff * * CuNi.eam.alloy Cu Ni",
    elements=["Cu", "Ni"],
)
calculator.compute(system, ("energy", "forces", "virial"))
calculator.close()
```

`LAMMPS(pair, elements, *, commands="", cmdargs=(), cutoff=None)` makes any
interaction that LAMMPS computes a calculator, through the LAMMPS Python
module, installed as the
[LAMMPS documentation](https://docs.lammps.org/Python_install.html)
describes.

!!! warning "Metal units only"
    LAMMPS runs in metal units, those of mdapy: Å, eV, ps and amu. The
    commands in `pair` must define the interaction for metal units;
    potentials written for other units, such as real or si, give wrong
    results.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `pair` | str | | | the LAMMPS commands of the interaction, such as `pair_style` and `pair_coeff`, for metal units |
| `elements` | sequence of str | | | the element of each LAMMPS atom type, from type 1 |
| `commands` | str | `""` | | LAMMPS commands run before `pair`, such as `newton on` |
| `cmdargs` | sequence of str | `()` | | command-line arguments of LAMMPS, such as `["-k", "on", "t", "4", "-sf", "kk"]` for Kokkos with four threads |
| `cutoff` | float | `None` | Å | the largest cutoff of the interaction; asked from LAMMPS when not given, which LAMMPS older than 30 Sep 2026 cannot tell |

| Attribute or method | Meaning |
|---|---|
| `elements` | the element of each atom type |
| `cutoff` | the largest cutoff of the interaction in Å |
| `compute(system, properties, *, neighbors=None)` | every property; LAMMPS searches its own neighbors, so `neighbors` is not used |
| `close()` | closes the LAMMPS instance; the next computation starts a new one |

The calculator keeps one LAMMPS instance while the atoms and their elements
stay the same, as during [`minimize`](simulation.md#energy-minimization), also when the box changes but not its
periodic directions; it starts a new one for other atoms, elements or
periodic directions. Any box works: LAMMPS receives it rotated to its restricted
triclinic form, and forces and virials are rotated back. Along non-periodic
directions LAMMPS has fixed boundaries around the atoms. With Kokkos, LAMMPS
is shut down when Python exits.

The constructor raises `InvalidParameter` when LAMMPS rejects the commands,
with the message of LAMMPS, or cannot tell the cutoff that is not given, and
`ImportError` without the LAMMPS Python module; `compute` raises
`InvalidParameter` for atoms of an element that no atom type has.

!!! quote "Please cite"
    [Thompson et al. (2022)](../reference/references.md#thompson2022), and the paper of the potential.
