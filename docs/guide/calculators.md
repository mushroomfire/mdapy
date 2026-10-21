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

## NEP

`NEP(path)` reads a neuroevolution potential of version 4 in the `nep.txt`
format of GPUMD, without (`nep4`) or with (`nep4_zbl`) a ZBL repulsion, and
computes every property. `NEP.descriptors(system)` returns the descriptors of
each atom, scaled as the network sees them. Other models of GPUMD (NEP3,
NEP5, charge, dipole and polarizability models) raise
[`FileFormatError`](../reference/api.md#mdapy.errors.FileFormatError).

<!-- skip-test -->
```python
from mdapy.calculators import NEP

potential = NEP("nep.txt")
result = potential.compute(system, ("energy", "forces", "virial"))
descriptors = potential.descriptors(system)  # (N, dim)
```

The radial descriptors of atom $i$ sum functions of the distances to its
neighbors $j$ within the pair's radial cutoff $r_c$,

$$
q^i_n = \sum_j g_n(r_{ij}), \qquad
g_n(r) = \sum_{k=0}^{K} c^{t_i t_j}_{nk} f_k(r), \qquad
f_k(r) = \tfrac{1}{2}\big[T_k\big(2(r/r_c - 1)^2 - 1\big) + 1\big] f_c(r),
$$

with Chebyshev polynomials $T_k$, the cutoff function
$f_c(r) = \tfrac{1}{2}[1 + \cos(\pi r / r_c)]$ below $r_c$, and coefficients
$c$ for each ordered pair of elements. The cutoff of a pair is the mean of the
two elements' cutoffs. The angular descriptors use other functions $g_n$, with
their own coefficients and the angular cutoff, and real orthonormal spherical
harmonics $Y_{lm}$:

$$
A^i_{nlm} = \sum_j g_n(r_{ij})\, Y_{lm}(\hat{\mathbf r}_{ij}), \qquad
q^i_{nl} = \sum_m \big(A^i_{nlm}\big)^2
= \frac{2l + 1}{4\pi} \sum_{j,k} g_n(r_{ij})\, g_n(r_{ik})\, P_l(\cos\theta_{jik})
$$

for $l = 1, \ldots, l_\text{max}$. The four-body descriptors couple three
$l = 2$ terms and the five-body descriptors four $l = 1$ terms, written here
with complex harmonics and Wigner 3j symbols:

$$
q^{i,4}_n = \sum_{m_1 m_2 m_3}
\begin{pmatrix} 2 & 2 & 2 \\ m_1 & m_2 & m_3 \end{pmatrix}
A^i_{n2m_1} A^i_{n2m_2} A^i_{n2m_3}, \qquad
q^{i,5}_n = \sum_{L=0}^{2} \sum_{M, m_k} (-1)^{L+M}
\begin{pmatrix} 1 & 1 & L \\ m_1 & m_2 & -M \end{pmatrix}
\begin{pmatrix} L & 1 & 1 \\ M & m_3 & m_4 \end{pmatrix}
A^i_{n1m_1} A^i_{n1m_2} A^i_{n1m_3} A^i_{n1m_4} = \tfrac{7}{15} \big(q^i_{n1}\big)^2.
$$

The descriptors, each multiplied by its scale factor $s_d$ from the file, are
the input of one hidden layer of $\tanh$ neurons with weights of the atom's
element:

$$
E_i = \sum_k w^{(1)}_k \tanh\Big(\sum_d w^{(0)}_{kd}\, s_d\, q^i_d - b^{(0)}_k\Big) - b^{(1)}.
$$

With ZBL, each pair within the angular cutoff adds the screened Coulomb
repulsion $V(r) = \frac{k_e Z_i Z_j}{r}\,\phi(r/a)\, f_s(r)$, half to each
atom, with $a = 0.46848\,\text{Å} / (Z_i^{0.23} + Z_j^{0.23})$ (as in GPUMD,
$1/0.46848$ is rounded to 2.134563), $k_e = 14.399645$ eV Å and
$\phi(x) = \sum_{k=1}^{4} a_k e^{-b_k x}$. The switch $f_s$ is 1 below an
inner cutoff and falls as a cosine to 0 at an outer cutoff. The `zbl` line of
the file chooses among three forms:

| `zbl` line | $a_k$, $b_k$ | inner cutoff | outer cutoff |
|---|---|---|---|
| `zbl r1 r2` | universal ZBL | r1 | r2 |
| `zbl r1 r2 f` | universal ZBL | 0 | the smaller of r2 and f times the sum of the two covalent radii GPUMD uses |
| `zbl 0 0` | from a table at the end of the file, per pair of elements | from the table | from the table |

The derivative of $E_i$ by the vector to neighbor $j$ adds to the atomic
virial of atom $j$, as in GPUMD and NEP_CPU, so atomic virials agree with
theirs. (EAM shares each pair's virial equally instead; both sum to the same
virial.)

## Energy minimization

`mdapy.minimize(system, calculator, *, max_force=1e-3, max_steps=1000,
max_step=0.2, cell=None, pressure=0.0, max_stress=1e-2, callback=None)`
minimizes the energy over the positions, and optionally the box, with FIRE,
the fast inertial relaxation engine. It returns a `MinimizeResult` with the
last `system` (with `forces` and atomic energies as columns and `energy` in
`info`), whether it `converged`, the number of `steps`, the `energy`, the
largest force and, when the box relaxed, the `stress` in GPa. No atom moves
more than `max_step` Å in one step. `callback(step, system, energy,
max_force)` is called after each step and stops the minimization by returning
True.

<!-- skip-test -->
```python
import mdapy

result = mdapy.minimize(system, potential, cell="full", max_stress=1e-3)
result.system.box  # the relaxed box
result = mdapy.minimize(system, potential, cell="isotropic", pressure=10.0)
```

`cell` chooses how the box changes along its periodic directions; directions
that are not periodic keep their lattice vector.

| `cell` | The box |
|---|---|
| `None` | does not change |
| `"isotropic"` | is scaled by one factor |
| `"axes"` | is scaled along each Cartesian axis by its own factor |
| `"full"` | takes any homogeneous strain |

With a relaxing box, the atoms move with it: the variables are the positions
$\mathbf u_i$ in the starting box and the deformation gradient $F$ of the
box, with $\mathbf x_i = \mathbf o + F \mathbf u_i$. Their generalized forces
are $F^T \mathbf f_i$ and $(W - pV I)\,F^{-T}$, the virial less the external
pressure $p$ times the volume, projected onto the strains `cell` allows; $F$
is scaled by the number of atoms so that its steps are comparable to those of
the atoms. The minimization converges when every force is below `max_force`
and every relaxed component of $\sigma + pI$ is below `max_stress` (GPa); at
a pressure it minimizes the enthalpy $E + pV$.

FIRE follows Bitzek et al. with the parameters ASE uses: a starting time step
of 0.1, a largest time step of 1.0, five steps before the time step grows by
10 %, halving it after an uphill step, and a mixing parameter starting at 0.1
and shrinking by 1 % per step. Unlike ASE, the step limit applies to each
atom rather than to the displacement of all atoms together.

## Validation

The tests in `tests/calculators/test_minimize.py` check that a harmonic dimer
reaches its bond length to 10<sup>−8</sup> Å; that a rattled fcc crystal
relaxes back to the perfect lattice, its energy and its twelve equal
neighbor distances; that results are the same for any number of threads;
and the step limit, the callback and the errors. For the box, they check that
a stretched, a tetragonal and a sheared fcc box relax back to cubic at the
minimum of $E(a)$ (by finite differences of the energy) with every stress
component below 10<sup>−5</sup> GPa; that at 5 GPa the stress is −5 GPa and
equals $-\mathrm dE/\mathrm dV$ by finite differences; and that a slab
relaxes only along its periodic directions.

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

The tests in `tests/calculators/test_nep.py`, on synthetic potentials with
random parameters written by the tests, check that:

- descriptors and atomic energies agree with an independent NumPy
  implementation to 10<sup>−12</sup>, for each form of ZBL, shared and
  per-element cutoffs, periodic, partly periodic and non-periodic systems, a
  box smaller than the cutoff, and $l_\text{max}$ from 1 to 8 with and without
  four- and five-body terms. The NumPy implementation computes the three-body
  terms as the double sums of Legendre polynomials and the four- and five-body
  terms with Wigner 3j symbols, a different route from mdapy's;
- `check_calculator` passes;
- results are the same for any number of threads;
- malformed files raise `FileFormatError` with the line.

`tests/calculators/test_nep_reference_data.py` runs where the test data of
torchnep are found and compares with results of GPUMD and NEP_CPU computed for
those potentials. Against NEP_CPU, in double precision, for 14 potentials of
one to four elements with every form of ZBL, the energies, forces, virials and
atomic virials agree to 10<sup>−13</sup> of their largest value. Against GPUMD,
which computes in single precision and prints six digits, energies agree to
10<sup>−5</sup> eV/atom and descriptors to 10<sup>−5</sup>.

## References

- M. S. Daw and M. I. Baskes, Embedded-atom method: derivation and
  application to impurities, surfaces, and other defects in metals,
  *Phys. Rev. B* **29**, 6443 (1984). doi:10.1103/PhysRevB.29.6443
- LAMMPS `pair_style eam`: <https://docs.lammps.org/pair_eam.html>
- Z. Fan et al., GPUMD: A package for constructing accurate machine-learned
  potentials and performing highly efficient atomistic simulations,
  *J. Chem. Phys.* **157**, 114801 (2022). doi:10.1063/5.0106617
- K. Song et al., General-purpose machine-learned potential for 16 elemental
  metals and their alloys, *Nat. Commun.* **15**, 10208 (2024).
  doi:10.1038/s41467-024-54554-x
- J. F. Ziegler, J. P. Biersack and U. Littmark, *The Stopping and Range of
  Ions in Solids* (Pergamon, 1985).
- E. Bitzek, P. Koskinen, F. Gähler, M. Moseler and P. Gumbsch, Structural
  relaxation made simple, *Phys. Rev. Lett.* **97**, 170201 (2006).
  doi:10.1103/PhysRevLett.97.170201
