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

relaxed = mdapy.minimize(system, potential, max_force=1e-4)
relaxed.converged, relaxed.energy
```

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

### Potentials of Zhou, Johnson and Wadley

<!-- skip-test -->
```python
potential = EAM.zhou(["Cu", "Ni", "Al"])
potential.write("CuNiAl.eam.alloy")
```

`EAM.zhou` builds the potential for any of Cu, Ag, Au, Ni, Pd, Pt, Al, Pb,
Fe, Mo, Ta, W, Mg, Co, Ti, Zr and Cr, with the tables that the database tool
of LAMMPS (`tools/eam_database`) writes for them: the parameters of Zhou et
al. (2004), those of Lin et al. (2008) for Cr, and their rule for the pair
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

### Average atom

<!-- skip-test -->
```python
average = potential.average({"Cu": 0.4, "Ni": 0.4, "Al": 0.2})
average.compute(system)  # every atom is the average atom
average.write("CuNiAl.average.eam.alloy")  # Cu, Ni, Al and the average atom A
```

`average(concentrations)` returns the average-atom potential of Varvenne et
al. (2016) for a random alloy with the fractions $c_a$ of the elements:

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

## LAMMPS

<!-- skip-test -->
```python
from mdapy.calculators import LAMMPS

calculator = LAMMPS(
    "pair_style eam/alloy\npair_coeff * * CuNi.eam.alloy Cu Ni",
    elements=["Cu", "Ni"],
)
calculator.compute(system, ("energy", "forces", "virial"))
mdapy.minimize(system, calculator)
calculator.close()
```

`LAMMPS(pair, elements, *, commands="", cmdargs=(), cutoff=None)` makes any
interaction that LAMMPS computes a calculator, through the LAMMPS Python
module, installed as the
[LAMMPS documentation](https://docs.lammps.org/Python_install.html)
describes.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `pair` | str | | | the LAMMPS commands of the interaction, such as `pair_style` and `pair_coeff`, in metal units |
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

The calculator keeps one LAMMPS instance while only the positions change, as
during `minimize`, and starts a new one for another box, other atoms or
elements. Any box works: LAMMPS receives it rotated to its restricted
triclinic form, and forces and virials are rotated back. Along non-periodic
directions LAMMPS has fixed boundaries around the atoms. With Kokkos, LAMMPS
is shut down when Python exits.

The constructor raises `InvalidParameter` when LAMMPS rejects the commands,
with the message of LAMMPS, or cannot tell the cutoff that is not given, and
`ImportError` without the LAMMPS Python module; `compute` raises `InvalidParameter` for atoms of an element that no
atom type has.

## Energy minimization

`mdapy.minimize(system, calculator, *, max_force=1e-3, max_steps=1000,
max_step=0.1, timestep=0.001, save_every=None, callback=None)` minimizes the energy over the
positions, in a fixed box, with FIRE 2.0, as LAMMPS `min_style fire` does
with its default settings.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `system` | `System` | | | the starting configuration, with a `mass` or an `element` column |
| `calculator` | `Calculator` | | | computes energies and forces |
| `max_force` | float | 1e-3 | eV/Å | converged when every force is below it |
| `max_steps` | int | 1000 | | largest number of steps |
| `max_step` | float | 0.1 | Å | an atom moving at its current velocity goes at most this far along x, y or z in one step (LAMMPS `dmax`) |
| `timestep` | float | 0.001 | ps | first time step of the dynamics (the LAMMPS `timestep` in metal units) |
| `save_every` | int | `None` | | keeps the configuration every this many steps, with the first and the last, in `frames` |
| `callback` | callable | `None` | | called after each step as `callback(step, system, energy, max_force)`; returning True stops the minimization |

It returns a `MinimizeResult`:

| Attribute | Type | Meaning |
|---|---|---|
| `system` | `System` | the last configuration, with `forces` and `atomic_energies` columns and `energy` in `info` |
| `converged` | bool | whether every force fell below `max_force` |
| `steps` | int | number of steps taken |
| `energy` | float | energy of the last configuration in eV |
| `max_force` | float | largest force of the last configuration in eV/Å |
| `frames` | tuple of `System` or `None` | with `save_every`, the kept configurations, with `forces` and `atomic_energies` columns and `energy` and the step as `timestep` in `info`; `None` otherwise |

The steps are those of LAMMPS: the same time step control, mixing and half
step back after an uphill step, with masses from the `mass` column or, when
there is none, the standard atomic weights of the elements. Masses change
the path, not the minimum; LAMMPS takes them from the data file or, for EAM,
from the potential file.

The kept frames show the minimization on its way and can be written as a
trajectory:

<!-- skip-test -->
```python
result = mdapy.minimize(system, potential, save_every=10)
mdapy.write("relaxation.xyz", result.frames)
```

`minimize` raises `InvalidParameter` for a non-positive `max_force`,
`max_step`, `timestep` or `save_every`, a negative `max_steps`, or a
calculator that does not compute both energies and forces, and
[`MissingProperty`](../reference/api.md#mdapy.errors.MissingProperty) for a
system with neither masses nor elements.

## References

- M. S. Daw and M. I. Baskes, Embedded-atom method: derivation and
  application to impurities, surfaces, and other defects in metals,
  *Phys. Rev. B* **29**, 6443 (1984). doi:10.1103/PhysRevB.29.6443
- LAMMPS `pair_style eam`: <https://docs.lammps.org/pair_eam.html>
- X. W. Zhou, R. A. Johnson and H. N. G. Wadley, Misfit-energy-increasing
  dislocations in vapor-deposited CoFe/NiFe multilayers, *Phys. Rev. B*
  **69**, 144113 (2004). doi:10.1103/PhysRevB.69.144113
- Z. Lin, R. A. Johnson and L. V. Zhigilei, Computational study of the
  generation of crystal defects in a bcc metal target irradiated by short
  laser pulses, *Phys. Rev. B* **77**, 214108 (2008).
  doi:10.1103/PhysRevB.77.214108
- C. Varvenne, A. Luque, W. G. Nöhring and W. A. Curtin, Average-atom
  interatomic potential for random alloys, *Phys. Rev. B* **93**, 104201
  (2016). doi:10.1103/PhysRevB.93.104201
- E. Bitzek, P. Koskinen, F. Gähler, M. Moseler and P. Gumbsch, Structural
  relaxation made simple, *Phys. Rev. Lett.* **97**, 170201 (2006).
  doi:10.1103/PhysRevLett.97.170201
- J. Guénolé, W. G. Nöhring, A. Vaid, F. Houllé, Z. Xie, A. Prakash and
  E. Bitzek, Assessment and optimization of the fast inertial relaxation
  engine (FIRE) for energy minimization in atomistic simulations and its
  implementation in LAMMPS, *Comput. Mater. Sci.* **175**, 109584 (2020).
  doi:10.1016/j.commatsci.2020.109584
- LAMMPS `min_style fire`: <https://docs.lammps.org/min_style.html>
