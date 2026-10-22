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

## Energy minimization

`mdapy.minimize(system, calculator, *, max_force=1e-3, max_steps=1000,
max_step=0.2, callback=None)` minimizes the energy over the positions in a
fixed box with FIRE, the fast inertial relaxation engine. It returns a
`MinimizeResult` with the last `system` (with `forces` and atomic energies as
columns and `energy` in `info`), whether it `converged` (every force below
`max_force`, in eV/Å), the number of `steps`, the `energy` and the largest
force. No atom moves more than `max_step` Å in one step. `callback(step,
system, energy, max_force)` is called after each step and stops the
minimization by returning True.

FIRE follows Bitzek et al. with the parameters ASE uses: a starting time step
of 0.1, a largest time step of 1.0, five steps before the time step grows by
10 %, halving it after an uphill step, and a mixing parameter starting at 0.1
and shrinking by 1 % per step. Unlike ASE, the step limit applies to each
atom rather than to the displacement of all atoms together.

Each step hands the calculator its neighbor list (`neighbors=`), which a
calculator must accept. The neighbors are searched 1 Å beyond the
calculator's cutoff, and the following steps take their pairs from that
search until an atom has moved 0.5 Å; the list is exactly the one a new
search would give, so the steps are the same, and a minimization with EAM
is about twice as fast.

## Validation

The tests in `tests/calculators/test_minimize.py` check that a harmonic dimer
reaches its bond length to 10<sup>−8</sup> Å; that a rattled fcc crystal
relaxes back to the perfect lattice, its energy and its twelve equal
neighbor distances; that results are the same for any number of threads;
that reusing a neighbor search gives the same steps, to the last bit, as a
calculator that searches at every step, with fewer than half as many
searches; and the step limit, the callback and the errors.

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
pressure converted by LAMMPS's own constant for metal units). Relaxing a
vacancy in a perturbed alloy, `minimize` and LAMMPS `min_style fire` reach the
same energy to 10<sup>−8</sup> eV and the same positions to 10<sup>−6</sup> Å,
up to a rigid translation along which each run drifts.

## References

- M. S. Daw and M. I. Baskes, Embedded-atom method: derivation and
  application to impurities, surfaces, and other defects in metals,
  *Phys. Rev. B* **29**, 6443 (1984). doi:10.1103/PhysRevB.29.6443
- LAMMPS `pair_style eam`: <https://docs.lammps.org/pair_eam.html>
- E. Bitzek, P. Koskinen, F. Gähler, M. Moseler and P. Gumbsch, Structural
  relaxation made simple, *Phys. Rev. Lett.* **97**, 170201 (2006).
  doi:10.1103/PhysRevLett.97.170201
