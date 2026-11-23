# Plane-wave DFT

!!! warning "Under development"
    MDAPY's density functional theory is new. It takes norm-conserving,
    ultrasoft and PAW pseudopotentials, runs on one NVIDIA GPU per
    configuration, and its speed is still being worked on. Its results are those of Quantum ESPRESSO on the same
    input, as [below](#how-the-results-compare-with-quantum-espresso).

<!-- skip-test -->
```python
import json
from pathlib import Path

from mdapy.calculators import DFT
from mdapy.io import read, write

dft = DFT(
    600.0,  # eV, the cutoff of the plane waves
    pseudopotentials="pseudo-dojo/nc-sr-04_pbe_standard",  # Al.upf, Ni.upf, ...
    kspacing=0.2,  # 1/Å
    smearing="marzari-vanderbilt",
    sigma=0.1,  # eV
    magmoms={"Ni": 0.6},
)
frames = []
for path in sorted(Path("structures").glob("*.xyz")):
    system = read(path)
    frames.append(dft.compute(system, ("energy", "forces", "virial")).attach(system))
write("train.xyz", frames)  # energy, virial and forces of each frame
# the settings that make these labels comparable with others
Path("train.json").write_text(json.dumps(dft.last["provenance"], indent=1))
```

`DFT(encut, *, ecutrho=None, pseudopotentials=None, kpoints=None, kspacing=0.25,
smearing="gaussian", sigma=0.1, magmoms=None, functional=None, tolerance=1e-6,
max_iterations=100, mixing=0.7, magnetic_mixing=1.6, kerker=1.5, bands=None, seed=0,
gpu=None, restart=False, symmetry=None, comm=None)` computes the
energies, forces and virials of periodic cells by Kohn–Sham density functional
theory in plane waves, on an NVIDIA GPU, as a [calculator](calculators.md) of
MDAPY. It suits the labels of training sets: configurations of up to a few
hundred atoms, one at a time. It needs the package `mdapy-dft-cuda12`, built with
CUDA 12 for GPUs from the V100 on.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `encut` | float64 | | eV | the kinetic energy cutoff of the plane waves, as VASP's ENCUT |
| `ecutrho` | float64 | 4 `encut`, 8 with ultrasoft or PAW pseudopotentials | eV | the cutoff of the density, as QE's ecutrho |
| `pseudopotentials` | mapping or path | `$MDAPY_PSEUDO_DIR` | | UPF files by element, or a folder with `{element}.upf` or one `{element}_*.upf` of each element |
| `kpoints` | (3,) int32 | `None` | | a Gamma-centered mesh of k-points |
| `kspacing` | float64 | 0.25 | 1/Å | without `kpoints`, the largest spacing of the mesh, 2π included, as VASP's KSPACING |
| `smearing` | str | `"gaussian"` | | `"gaussian"`, `"fermi-dirac"`, `"methfessel-paxton"` (first order), `"marzari-vanderbilt"`, or `"fixed"` for insulators |
| `sigma` | float64 | 0.1 | eV | the width of the smearing |
| `magmoms` | mapping or (N,) float64 | `None` | Bohr magneton | the first magnetic moments, by element or one per atom; with them the spins are collinear |
| `functional` | str | that of the pseudopotentials, as Quantum ESPRESSO takes it | | `"pbe"`, `"pbe_pw"`, `"pbesol"`, `"pbesol_pw"`, `"lda_pw"` or `"lda_pw_mod"` |
| `tolerance` | float64 | 1e-6 | eV | the self-consistent field stops when the energy changes by less and the Hartree energy of the change of the density is below it |
| `max_iterations` | int32 | 100 | | the steps of the self-consistent field before it fails |
| `mixing` | float64 | 0.7 | | the share of the residual of the density taken at each step |
| `magnetic_mixing` | float64 | 1.6 | | with spin, the share of the residual of the magnetization taken at each step, as VASP's AMIX_MAG |
| `kerker` | float64 | 1.5 | 1/Å | the wave vector of Kerker's preconditioner of the mixing, 0 for none |
| `bands` | int32 | `None` | | bands per k-point and spin; by default four or 10 % more than the occupied ones for fixed occupations, else the larger of 1.2 N/2 and N/2 + 4 for N electrons |
| `symmetry` | bool | `None` | | the symmetry of the crystal, found by [spglib](https://spglib.readthedocs.io): the irreducible k-points alone, the density, the occupations of the projectors, the forces and the stress made symmetric; by default when spglib is installed |
| `seed` | int32 | 0 | | seed of the random first bands |
| `gpu` | int32 | `None` | | the GPU of this process, by its index among those it sees; by default the first |
| `restart` | bool | `False` | | start each computation from the bands and the density of the last, as the steps of a relaxation or of molecular dynamics may; labels of a training set should start cold |
| `comm` | mpi4py communicator | `None` | | processes that compute each configuration together, one GPU each, sharing its k-points and spins; every process calls `compute` with the same system and gets the whole result; without `gpu` each takes the GPU of its rank on its machine |

| Attribute or method | Meaning |
|---|---|
| `cutoff` | 0: the calculator needs no neighbor list |
| `compute(system, properties, *, neighbors=None)` | the free energy in eV, the forces in eV/Å and, when asked, the virial in eV; `neighbors` is not used |
| `last` | of the last computation: `iterations`, `restarted`, `fermi` (eV), `magnetization` and `absolute_magnetization` (Bohr magnetons, with spin), `net_force` (eV/Å), `kpoints`, `grid`, `provenance` |

The system must be periodic along a, b and c and have an `element` column. The
energy is the free energy of the smearing, VASP's TOTEN. `provenance` holds
what makes the labels of a set comparable: the SHA-256 of each pseudopotential,
the cutoff, the mesh, the smearing and its width, the functional, the spins, the
grid and the bands. Labels of a set should share them.

## Many GPUs

A configuration runs on one GPU. Many configurations, as the labels of a
training set, share the GPUs of a machine as processes of their own, one GPU
each, started by `mpirun`:

<!-- skip-test -->
```python
from mpi4py import MPI

from mdapy.calculators import DFT
from mdapy.io import read, write

rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
dft = DFT(600.0, kspacing=0.2, gpu=rank % 4)  # 4 GPUs on each machine
mine = [read(f"structures/{i}.xyz") for i in range(rank, 1000, size)]
labelled = [dft.compute(s, ("energy", "forces", "virial")).attach(s) for s in mine]
write(f"train-{rank}.xyz", labelled)
```

One configuration of many k-points or of two spins can also run on several GPUs:
with `comm`, the processes share its pairs of spin and k-point, each GPU holding
the bands of its own, and every process gets the whole result. This pays when a
configuration has at least as many such pairs as GPUs.

<!-- skip-test -->
```python
from mpi4py import MPI

from mdapy.calculators import DFT
from mdapy.io import read

dft = DFT(600.0, kpoints=(4, 4, 4), magmoms={"Fe": 2.5}, comm=MPI.COMM_WORLD)
result = dft.compute(read("fe54.xyz"), ("energy", "forces", "virial"))
if MPI.COMM_WORLD.Get_rank() == 0:
    print(result.energy)
```

## What it computes

- **Plane waves** of kinetic energy up to `encut` at each k-point of the mesh;
  k-points related by time reversal count once. The bands live on the FFT grid
  of four times the cutoff, the density and the potentials on that of
  `ecutrho`, with 2, 3, 5 and 7 as factors. When the mesh is Γ alone, as for
  large cells, the bands are real: they keep half of the plane waves, the
  other half being their complex conjugates, with real Fourier transforms and
  real products of the bands, for the same results in less time and half the
  memory.
- **Pseudopotentials:** in UPF 2, norm-conserving, ultrasoft or PAW, with
  their projectors, the core charge of the nonlinear core correction and,
  ultrasoft and PAW, their augmentation charges by angular momentum
  (`q_with_l`, as the files of a recent `ld1.x`). The tested ones are
  PseudoDojo's norm-conserving tables and the ultrasoft and PAW files of
  pslibrary 1.0.0 made by `ld1.x`.
- **Ultrasoft pseudopotentials** solve H ψ = ε S ψ with the overlap
  S = 1 + Σ |β⟩ q ⟨β|; the strengths of the projectors of each atom follow the
  potential at every step, and the density adds the augmentation charges.
- **PAW** adds to that the one-center terms of each atom, on its radial mesh:
  the all-electron and pseudo densities of the occupations of its projectors,
  their Hartree and exchange-correlation energies, and the strengths these
  add to the projectors. The occupations are mixed along with the density, as
  Quantum ESPRESSO mixes them. The energy is Quantum ESPRESSO's total energy,
  without the energy of the frozen core.
- **Exchange and correlation:** LDA (Perdew–Wang), PBE and PBEsol, with and
  without spin. `"pbe_pw"` is PBE on Perdew and Wang's correlation with the
  constants of their paper, as Quantum ESPRESSO's PBE, and the default for PBE
  pseudopotentials. `"pbe"` uses those of libxc; the two differ by about 1e-6 of
  the correlation energy, 0.01 to 0.25 meV per atom. `"pbesol_pw"` and `"pbesol"`
  are PBEsol likewise; a PBEsol run may take PBE
  pseudopotentials, as Quantum ESPRESSO's `input_dft` does, but PBEsol ones
  are better.
- **The self-consistent field:** the bands by block Davidson, from the
  pseudo-orbitals of the free atoms at the start, the density mixed
  by Pulay's method in the metric of Kresse and Furthmüller with Kerker's
  preconditioner. A run that does not converge raises. With `restart`, a run
  starts from the bands of the last, plane wave by plane wave, and from its
  density with the free atoms moved from where they were to where they are;
  a run of other k-points, bands, spins or a number of atoms starts cold.
- **Symmetry:** the operations of the crystal that keep its atoms, their
  kinds and first moments, found by spglib to 1e-5 bohr. The k-points of the
  mesh reduce to the irreducible ones, k equivalent to R^T k and to −k. The
  output density is the average of its images on the sphere of reciprocal
  space, so the fractional translations need not fit the FFT grid; the
  occupations of the projectors and the one-center strengths of PAW are averaged
  with the rotations of the real harmonics, and the forces and the stress over
  the operations. A configuration without symmetry, as most of a training set,
  runs on the mesh with time reversal alone.
- **Forces** with Quantum ESPRESSO's correction for a density not quite
  self-consistent. Their mean, which only the FFT grid leaves, is taken out
  and kept in `last["net_force"]`.
- **The virial** is the derivative of the energy under a strain that keeps the
  Miller indices of the plane waves. It lacks the Pulay stress of an
  incomplete basis: test the cutoff, as with VASP, by the change of the stress.

The constructor raises `InvalidParameter` for a cutoff, width, tolerance or
mesh that is not positive, an unknown smearing, or a folder that does not
exist. `compute` raises `InvalidParameter` for a box that is not periodic
along a, b and c, an element without a pseudopotential, or a moment for each
atom of the wrong number; `MissingProperty` without an `element` column;
and [`CalculationFailed`](../../reference/api/utilities.md#mdapy.errors.CalculationFailed)
when the self-consistent field does not converge, with its last energies.

## How the results compare with Quantum ESPRESSO

Its [speed against Quantum ESPRESSO](../../performance/dft.md) on GPUs and processors is on
its own page. The [verification](dft-verification.md) adds the equations of state of 72 elements against
all-electron codes.

Quantum ESPRESSO 7.6 (`pw.x`) runs on the same cells, PseudoDojo files, cutoff,
k-points, FFT grid and smearing, with `pbe_pw`. The tests run it each time. The
differences:

| | Silicon, diamond, aluminium (four smearings) | bcc iron, fcc nickel, antiferromagnetic chromium, with spin |
|---|---|---|
| total energy and each of its terms | below 1e-7 hartree per atom | below 1e-7 hartree per atom once the rounded constant below is allowed for |
| eigenvalues of the occupied bands, Fermi level | below 1e-7 hartree | below 1e-6 hartree |
| forces | below 1e-7 hartree/bohr | |
| stress, in sheared cells | 0.003 kbar | 0.02 kbar |
| magnetization | | below 1e-4 Bohr magneton |

With the ultrasoft files of pslibrary 1.0.0, on one grid and on two (`ecutrho`
4 and 8 times `encut`), silicon and aluminium agree in every term of the
energy, the occupied eigenvalues and the Fermi level to 1e-7 hartree and in
the forces to 1e-7 hartree/bohr; bcc iron with spin in the energy to 1e-7
hartree per atom once the rounded constant is allowed for, and in the
magnetization to 1e-6 Bohr magneton. The stress of sheared silicon agrees to
0.01 kbar, that of sheared iron with spin to 0.03 kbar, and both with
differences of strained cells at a fixed basis to 1e-7 hartree.

With the PAW files of pslibrary 1.0.0 (`ecutrho` 8 times `encut`),
aluminium, silicon of two displaced atoms and bcc iron with spin agree in the
energy to 1e-8 hartree per atom (iron once the rounded constant is allowed
for), in the occupied eigenvalues to 1e-7 hartree, in the forces to 1e-7
hartree/bohr, in the stress to 0.01 kbar and, iron, in the magnetization to
1e-6 Bohr magneton.

What differs is Quantum ESPRESSO's own:

- Its Ewald sum is converged to about 1e-7 Ry.
- Its spin-polarized PBE correlation takes γ = 0.031091, rounded, where its
  paramagnetic one takes (1 − ln 2)/π². This moves the energy of iron by 1e-6
  hartree, and with PAW by 4e-6 more through the one-center densities. A NumPy
  evaluation of the difference on MDAPY's densities accounts for it to 1e-8.
- With FFTW its grids take the factors 2, 3 and 5 only; the comparisons use its
  grid.

The forces also agree with differences of the energy of displaced atoms, to 1e-7
hartree/bohr. The stress agrees with differences of strained cells whose plane
waves keep their Miller indices, to 5e-8 hartree.

!!! quote "Please cite"
    The pseudopotentials ([van Setten et al. 2018](../../reference/references.md#vansetten2018)
    for PseudoDojo) and the functional ([Perdew and Wang 1992](../../reference/references.md#perdew1992),
    [Perdew et al. 1996](../../reference/references.md#perdew1996), and for PBEsol
    [Perdew et al. 2008](../../reference/references.md#perdew2008)); the methods:
    [Davidson (1975)](../../reference/references.md#davidson1975),
    [Kresse and Furthmüller (1996)](../../reference/references.md#kresse1996),
    [Kerker (1981)](../../reference/references.md#kerker1981),
    [White and Bird (1994)](../../reference/references.md#white1994),
    [Nielsen and Martin (1985)](../../reference/references.md#nielsen1985),
    [Methfessel and Paxton (1989)](../../reference/references.md#methfessel1989),
    [Marzari et al. (1999)](../../reference/references.md#marzari1999).
    The comparisons are with Quantum ESPRESSO
    ([Giannozzi et al. 2017](../../reference/references.md#giannozzi2017)).
