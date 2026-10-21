# Calculators

A calculator computes the energy, forces and virial of a system. mdapy
defines one interface, `mdapy.calculators.Calculator`, which its own
potentials implement and which other packages can implement to work with
everything in mdapy that needs energies and forces. `EAM` is the first
calculator of mdapy.

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

## The interface

| Member | Meaning |
|---|---|
| `cutoff` | interaction cutoff in Å |
| `elements` | the elements the calculator handles, or None for any |
| `implemented` | the properties `compute` can return |
| `compute(system, properties=("energy", "forces"), *, neighbors=None)` | returns a `CalculatorResult` with at least the requested properties |

A `CalculatorResult` holds:

| Attribute | Shape | Unit | Meaning |
|---|---|---|---|
| `energy` | | eV | total energy |
| `atomic_energies` | (N,) | eV | they sum to `energy` |
| `forces` | (N, 3) | eV/Å | |
| `virial` | (3, 3) | eV | positive under compression |
| `atomic_virials` | (N, 3, 3) | eV | they sum to `virial` |

with `stress(system)`, the stress $\sigma = -W/V$ in GPa (positive under
tension), and `attach(system)`, which returns the system with the per-atom
values as columns and the energy and virial in `info`. Values that were not
computed are None; the arrays are read-only.

The virial is defined by the response of the energy to a homogeneous strain
$\varepsilon$ of the box and the positions,

$$
W_{ab} = -\frac{\partial E}{\partial \varepsilon_{ab}} = \sum_i x_{i,a} F_{i,b},
$$

the second form holding for systems without periodic directions.

A neighbor list passed as `neighbors=` must have been built for the same
system with at least the calculator's cutoff; it lets several calculators or
analyses share one search.

## Checking a calculator

`mdapy.calculators.testing.check_calculator(calculator, system)` tests any
implementation and raises `AssertionError` naming the first failed check and
its largest deviation:

- atomic energies and atomic virials sum to the energy and the virial;
- forces agree with central finite differences of the energy, and the virial
  with finite differences under strain;
- translation, rotation and reordering of the atoms change nothing beyond
  rotating and reordering vectors;
- a cell doubled along **a** has twice the energy;
- a neighbor list given with a larger cutoff changes nothing.

Its tolerances, finite-difference step and the number of atoms checked are
parameters.

## EAM

`EAM(path)` reads a potential in the `eam/alloy` (setfl) format of LAMMPS
and computes every property. Each function is tabulated and interpolated
exactly as LAMMPS `pair_style eam` does: a cubic spline through the
tabulated values whose slopes come from fourth-order finite differences, and
an embedding energy extended linearly beyond the largest tabulated density.
The energy of atom $i$ is

$$
E_i = F_{s_i}\Big(\sum_j \rho_{s_j}(r_{ij})\Big) + \frac{1}{2}\sum_j \phi_{s_i s_j}(r_{ij}),
$$

and each pair's virial is shared equally by its two atoms. The atoms need an
`element` column; elements the file does not have raise
[`InvalidParameter`](../reference/api.md#mdapy.errors.InvalidParameter).

## Validation

The tests in `tests/calculators/test_eam.py`, on a synthetic two-element
potential written by the tests, check that:

- energies, forces and atomic virials agree with an independent NumPy
  implementation of the same interpolation and sums over periodic images to
  10<sup>−11</sup>, for periodic, partly periodic and non-periodic systems
  and for a box smaller than the cutoff;
- `check_calculator` passes;
- results are the same for any number of threads;
- malformed files raise `FileFormatError` with the line, and the reading
  follows LAMMPS (values continuing on further lines, comments).

`tests/calculators/test_eam_lammps.py` runs where the LAMMPS Python module
is installed and compares with LAMMPS stable_30Sep2026 `pair_style eam/alloy`:
the energy agrees to 10<sup>−12</sup>, atomic energies to 10<sup>−11</sup>,
and forces, the virial and atomic virials to 10<sup>−9</sup> (with the
pressure converted by LAMMPS's own constant for metal units).

## References

- M. S. Daw and M. I. Baskes, Embedded-atom method: derivation and
  application to impurities, surfaces, and other defects in metals,
  *Phys. Rev. B* **29**, 6443 (1984). doi:10.1103/PhysRevB.29.6443
- LAMMPS `pair_style eam`: <https://docs.lammps.org/pair_eam.html>
