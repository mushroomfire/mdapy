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

mdapy's calculators are [EAM](eam.md), [NEP](nep.md) and [LAMMPS](lammps.md),
which makes any interaction of LAMMPS a calculator. Any calculator serves
[energy minimization](minimize.md) and [elastic constants](elastic-constants.md).

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
[`NeighborList`](../getting-started/neighbor.md) of the same system built with a cutoff of at
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
