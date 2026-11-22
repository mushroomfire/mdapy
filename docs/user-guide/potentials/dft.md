# Plane-wave DFT

!!! warning "Under development"
    MDAPY's density functional theory is new. It takes norm-conserving
    pseudopotentials only, runs on one NVIDIA GPU, and its speed has not been
    worked on yet. Its results are those of Quantum ESPRESSO on the same
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

`DFT(encut, *, pseudopotentials=None, kpoints=None, kspacing=0.25,
smearing="gaussian", sigma=0.1, magmoms=None, functional=None, tolerance=1e-6,
max_iterations=100, mixing=0.7, magnetic_mixing=1.6, kerker=1.5, bands=None, seed=0,
gpu=None, restart=False)` computes the
energies, forces and virials of periodic cells by Kohn–Sham density functional
theory in plane waves, on an NVIDIA GPU, as a [calculator](calculators.md) of
MDAPY. It suits the labels of training sets: configurations of up to a few
hundred atoms, one at a time. It needs the package `mdapy-dft-cuda12`, built with
CUDA 12 for GPUs from the V100 on.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `encut` | float64 | | eV | the kinetic energy cutoff of the plane waves, as VASP's ENCUT; the density takes four times it |
| `pseudopotentials` | mapping or path | `$MDAPY_PSEUDO_DIR` | | UPF files by element, or a folder with `{element}.upf` or one `{element}_*.upf` of each element |
| `kpoints` | (3,) int32 | `None` | | a Gamma-centered mesh of k-points |
| `kspacing` | float64 | 0.25 | 1/Å | without `kpoints`, the largest spacing of the mesh, 2π included, as VASP's KSPACING |
| `smearing` | str | `"gaussian"` | | `"gaussian"`, `"fermi-dirac"`, `"methfessel-paxton"` (first order), `"marzari-vanderbilt"`, or `"fixed"` for insulators |
| `sigma` | float64 | 0.1 | eV | the width of the smearing |
| `magmoms` | mapping or (N,) float64 | `None` | Bohr magneton | the first magnetic moments, by element or one per atom; with them the spins are collinear |
| `functional` | str | that of the pseudopotentials | | `"pbe"`, `"pbe_pw"`, `"lda_pw"` or `"lda_pw_mod"` |
| `tolerance` | float64 | 1e-6 | eV | the self-consistent field stops when the energy changes by less and the Hartree energy of the change of the density is below it |
| `max_iterations` | int32 | 100 | | the steps of the self-consistent field before it fails |
| `mixing` | float64 | 0.7 | | the share of the residual of the density taken at each step |
| `magnetic_mixing` | float64 | 1.6 | | with spin, the share of the residual of the magnetization taken at each step, as VASP's AMIX_MAG |
| `kerker` | float64 | 1.5 | 1/Å | the wave vector of Kerker's preconditioner of the mixing, 0 for none |
| `bands` | int32 | `None` | | bands per k-point and spin; by default four or 10 % more than the occupied ones for fixed occupations, else the larger of 1.2 N/2 and N/2 + 4 for N electrons |
| `seed` | int32 | 0 | | seed of the random first bands |
| `gpu` | int32 | `None` | | the GPU of this process, by its index among those it sees; by default the first |
| `restart` | bool | `False` | | start each computation from the bands and the density of the last, as the steps of a relaxation or of molecular dynamics may; labels of a training set should start cold |

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

## What it computes

- **Plane waves** of kinetic energy up to `encut` at each k-point of the mesh;
  k-points related by time reversal count once. The density lives on the FFT grid
  of four times the cutoff, with 2, 3, 5 and 7 as factors.
- **Pseudopotentials:** norm-conserving, in UPF 2, with their projectors and the
  core charge of the nonlinear core correction. PseudoDojo's tables are the
  tested ones. Ultrasoft and PAW files are refused.
- **Exchange and correlation:** LDA (Perdew–Wang) and PBE, with and without
  spin. `"pbe_pw"` is PBE on Perdew and Wang's correlation with the constants
  of their paper, as Quantum ESPRESSO's PBE. `"pbe"` uses those of libxc; the two
  differ by about 1e-6 of the correlation energy.
- **The self-consistent field:** the bands by block Davidson, the density mixed
  by Pulay's method in the metric of Kresse and Furthmüller with Kerker's
  preconditioner. A run that does not converge raises. With `restart`, a run
  starts from the bands of the last, plane wave by plane wave, and from its
  density with the free atoms moved from where they were to where they are;
  a run of other k-points, bands, spins or a number of atoms starts cold.
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

What differs is Quantum ESPRESSO's own:

- Its Ewald sum is converged to about 1e-7 Ry.
- Its spin-polarized PBE correlation takes γ = 0.031091, rounded, where its
  paramagnetic one takes (1 − ln 2)/π². This moves the energy of iron by 1e-6
  hartree. A NumPy evaluation of the difference on MDAPY's density accounts for
  it to 1e-8.
- With FFTW its grids take the factors 2, 3 and 5 only; the comparisons use its
  grid.

The forces also agree with differences of the energy of displaced atoms, to 1e-7
hartree/bohr. The stress agrees with differences of strained cells whose plane
waves keep their Miller indices, to 5e-8 hartree.

!!! quote "Please cite"
    The pseudopotentials ([van Setten et al. 2018](../../reference/references.md#vansetten2018)
    for PseudoDojo) and the functional ([Perdew and Wang 1992](../../reference/references.md#perdew1992),
    [Perdew et al. 1996](../../reference/references.md#perdew1996)); the methods:
    [Davidson (1975)](../../reference/references.md#davidson1975),
    [Kresse and Furthmüller (1996)](../../reference/references.md#kresse1996),
    [Kerker (1981)](../../reference/references.md#kerker1981),
    [White and Bird (1994)](../../reference/references.md#white1994),
    [Nielsen and Martin (1985)](../../reference/references.md#nielsen1985),
    [Methfessel and Paxton (1989)](../../reference/references.md#methfessel1989),
    [Marzari et al. (1999)](../../reference/references.md#marzari1999).
    The comparisons are with Quantum ESPRESSO
    ([Giannozzi et al. 2017](../../reference/references.md#giannozzi2017)).
