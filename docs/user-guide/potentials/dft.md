# Plane-wave DFT

!!! warning "Under development"
    MDAPY's density functional theory is new. It takes norm-conserving,
    ultrasoft and PAW pseudopotentials, runs on one NVIDIA GPU (or AMD GPU, built
    from source) per configuration, and its speed is still being worked on. Its results are those of Quantum ESPRESSO on the same
    input, as [below](#how-the-results-compare-with-quantum-espresso).

<!-- skip-test -->
```python
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
write("train.xyz", frames)  # the energy, virial, forces and parameters of each frame
```

The bands of a crystal at any wave vectors and its density of states follow from its
self-consistent field, as the [electronic structure](../properties/electronic-structure.md).

`DFT(encut, *, ecutrho=None, pseudopotentials=None, kpoints=None, kspacing=0.25,
smearing="gaussian", sigma=0.1, magmoms=None, hubbard=None, functional=None, tolerance=1e-6,
max_iterations=100, mixing=None, magnetic_mixing=1.0, kerker=1.5, preconditioner="kerker",
bands=None, seed=0, gpu=None, restart=False, symmetry=None, precision="double",
kernels="mdapy", comm=None, log=None, start=None)` computes the
energies, forces and virials of periodic cells by Kohn–Sham density functional
theory in plane waves, on an NVIDIA GPU, as a [calculator](calculators.md) of
MDAPY. It suits the labels of training sets: configurations of up to a few
hundred atoms, one at a time. It runs in the [GPU part](../simulation/gpu.md#installing-the-gpu-part)
of MDAPY, the package `mdapy-cuda12` for GPUs from the V100 on, on MDAPY's own kernels:

```bash
pip install "mdapy[cuda12]"
```

To compare them with NVIDIA's cuBLAS, cuFFT and cuSOLVER (`kernels="vendor"`), the extra
`dft` installs those from NVIDIA's packages, `pip install "mdapy[cuda12,dft]"` (see
[NVIDIA's libraries for the DFT](../../getting-started/installation.md#nvidias-libraries-for-the-dft)).

For an AMD GPU the same sources are compiled with HIP of [ROCm](https://rocm.docs.amd.com)
6 into the package `mdapy-rocm6`, in a clone of MDAPY:

```bash
pip install ./rocm6 -Ccmake.define.CMAKE_HIP_ARCHITECTURES=gfx90a
```

`gfx90a` names the MI200 series. The build needs HIP, hipBLAS, hipFFT and hipSOLVER of
ROCm, the run the HIP runtime, as `module load rocm` gives it on a cluster, and
`kernels="vendor"` the libraries of ROCm too, which it loads from the folders of
`LD_LIBRARY_PATH`, `$ROCM_PATH/lib` or `/opt/rocm/lib`. There the calculations of several
threads take turns on the GPU instead of running at once.
Without the GPU part, or the libraries of the vendor for `kernels="vendor"`, `DFT` raises
an error that says what to install.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `encut` | float64 | | eV | the kinetic energy cutoff of the plane waves, as VASP's ENCUT |
| `ecutrho` | float64 | 4 `encut`, 8 with ultrasoft or PAW pseudopotentials | eV | the cutoff of the density, as QE's ecutrho |
| `pseudopotentials` | mapping or path | `$MDAPY_PSEUDO_DIR` | | UPF files by element, or a folder with `{element}.upf` or one `{element}_*.upf` of each element |
| `kpoints` | (3,) int32 | `None` | | a Gamma-centered mesh of k-points |
| `kspacing` | float64 | 0.25 | 1/Å | without `kpoints`, the largest spacing of the mesh, 2π included, as VASP's KSPACING |
| `smearing` | str | `"gaussian"` | | `"gaussian"`, `"fermi-dirac"`, `"methfessel-paxton"` (first order), `"marzari-vanderbilt"`, or `"fixed"` for insulators |
| `sigma` | float64 | 0.1 | eV | the width of the smearing |
| `magmoms` | mapping or (N,) float64 | `None` | Bohr magneton | the first magnetic moments, by element or one per atom; with them the spins are collinear, and the magnetization stays theirs for the first steps, while the density settles ([below](#the-first-moments-and-the-magnetic-state)) |
| `hubbard` | mapping of str to (str, float64) | `None` | eV | DFT+U: for an element the label of one of its pseudo-orbitals, such as `"3d"`, and U |
| `functional` | str | that of the pseudopotentials, as Quantum ESPRESSO takes it | | `"pbe"`, `"pbe_pw"`, `"pbesol"`, `"pbesol_pw"`, `"lda_pw"` or `"lda_pw_mod"` |
| `tolerance` | float64 | 1e-6 | eV | the self-consistent field stops when the energy changes by less and the Hartree energy of the change of the density is below it |
| `max_iterations` | int32 | 100 | | the steps of the self-consistent field before it fails |
| `mixing` | float64 | None | | the share of the residual of the density taken at each step; None for 0.7 without spin, as Quantum ESPRESSO, and 0.4 with it |
| `magnetic_mixing` | float64 | 1.0 | | with spin, the share of the residual of the magnetization taken at each step, as VASP's AMIX_MAG |
| `kerker` | float64 | 1.5 | 1/Å | the wave vector of Kerker's preconditioner of the mixing, 0 for none |
| `preconditioner` | str | `"kerker"` | `"kerker"`, `"ldos"` | of the mixing of the total density: `"ldos"` from the local density of states at the Fermi level of each step ([Herbst and Levitt 2021](../../reference/references.md#herbst2021)), which screens the metal and not the vacuum or the insulator of a cell: slabs, surfaces, and magnetic alloys whose moments Kerker's lets wander or fall |
| `bands` | int32 | `None` | | bands per k-point and spin; by default four or 10 % more than the occupied ones for fixed occupations, else the larger of 1.2 N/2 and N/2 + 4 for N electrons |
| `symmetry` | bool | `None` | | the symmetry of the crystal, found by [spglib](https://spglib.readthedocs.io): the irreducible k-points alone, the density, the occupations of the projectors, the forces and the stress made symmetric; by default when spglib is installed |
| `seed` | int32 | 0 | | seed of the random first bands |
| `gpu` | int32 | `None` | | the GPU of this process, by its index among those it sees; by default the first |
| `restart` | bool or str | `False` | `True`, `"atomic"`, `"second_order"` | start each computation from the bands and the density of the last, as the steps of a relaxation or of molecular dynamics may; `"second_order"` extrapolates the density from the last three computations ([Arias et al. 1992](../../reference/references.md#arias1992)), fewer iterations along molecular dynamics; labels of a training set should start cold |
| `moment_radius` | float64 | `None` | Å | with spin, the radius of the spheres within which the magnetic moment of each atom is integrated; by default half the shortest distance between two atoms of the cell, their periodic images included, the same for all atoms, so that the spheres do not overlap |
| `precision` | str | `"double"` | `"double"`, `"mixed"`, `"auto"` | `"mixed"`: the first steps of the self-consistent field in single precision, the last in double, for GPUs of little double precision (most cards for games and workstations); the label is that of double precision, `last["single_steps"]` the steps in single precision. `"auto"`: mixed on those GPUs, double on V100, A100, H100 and GH200. Mixed precision holds about a tenth more of the GPU's memory at its peak: a cell that only just fits in double runs with `"double"` |
| `kernels` | str | `"mdapy"` | `"mdapy"`, `"vendor"` | Whose kernels run on the GPU. `"mdapy"`: those of MDAPY, the fast Fourier transforms, the dense linear algebra (the products of matrices and the triangular solves), the Hermitian eigensolver and the Cholesky factorization, which is all the DFT takes from the libraries of the vendor, which it does not load; only sizes of a grid given that are not products of 2, 3, 5 and 7 go to the FFT of the vendor. `"vendor"`: the GPU libraries of the vendor throughout (cuFFT, cuBLAS and cuSOLVER on NVIDIA GPUs, of the extra `dft`; hipFFT, hipBLAS and hipSOLVER on AMD GPUs). The results agree to rounding; `last["provenance"]` keeps which one ran |
| `comm` | mpi4py communicator | `None` | | processes that compute each configuration together, one GPU each, sharing its k-points and spins; every process calls `compute` with the same system and gets the whole result; without `gpu` each takes the GPU of its rank on its machine |
| `log` | path or stream | `None` | | where every step of the self-consistent field of each computation is written, with the parameters it took ([below](#the-log-and-the-parameters-of-each-computation)) |
| `start` | path | `None` | | a ground state that `save` wrote, which the first computation starts from ([below](#the-ground-state-in-a-file)) |

| Attribute or method | Meaning |
|---|---|
| `cutoff` | 0: the calculator needs no neighbor list |
| `compute(system, properties, *, neighbors=None)` | the free energy in eV, the forces in eV/Å and, when asked, the virial in eV; `neighbors` is not used |
| `last` | of the last computation: `iterations`, `restarted`, `fermi` (eV), `magnetization` and `absolute_magnetization` (Bohr magnetons, with spin), `magnetic_moments` ((N,) float64 Bohr magnetons, with spin: the magnetization density, spin up minus spin down, integrated within a sphere of `moment_radius` around each atom, in the order of the atoms) and `moment_radius` (Å, the radius used), `held_steps` (with spin, the first steps whose magnetization stayed that of `magmoms`) and `moments_lost` (with spin, whether the moments it started with are gone, [below](#the-first-moments-and-the-magnetic-state)), `net_force` (eV/Å), `kpoints`, `grid`, `gpu_memory` (the [memory of the GPU](#memory-of-the-gpu) in MiB), `provenance`, `calculation` (its number among the computations of the calculator) and `parameters` (every parameter it took, [below](#the-log-and-the-parameters-of-each-computation)) |
| `log`, `start` | the log and the ground state given |
| `save(path)` | writes the ground state of the last computation ([below](#the-ground-state-in-a-file)) |

The system must be periodic along a, b and c and have an `element` column. The
energy is the free energy of the smearing, VASP's TOTEN. `provenance` holds
what makes the labels of a set comparable: the SHA-256 of each pseudopotential,
the cutoff, the mesh, the smearing and its width, the functional, the spins, the
grid and the bands. Labels of a set should share them. It also holds the
`kernels`, which changes the results by rounding alone. With MDAPY's kernels
the same computation on the same GPU gives the same numbers to the last bit at
each run; the libraries of the vendor do not promise it when the k-points are
solved side by side.

The magnetic moments are those of the smooth density on the grid, with the
augmentation charges of ultrasoft and PAW pseudopotentials but not the one-center
corrections of PAW, within spheres that do not overlap: they depend on the radius, and
their sum is less than `magnetization`, which counts the whole cell. Compare them across
labels of a set only with the same `moment_radius`.

## The log and the parameters of each computation

<!-- skip-test -->
```python
from mdapy.calculators import DFT
from mdapy.io import read, write

dft = DFT(300.0, kpoints=(2, 2, 2), sigma=0.2, log="dft.log")
system = read("al4.xyz")
write("labels.xyz", dft.compute(system, ("energy", "forces", "virial")).attach(system))
```

```text
# MDAPY 2.2.0, DFT, 2026-11-25 13:15:35
# engine: MDAPY on NVIDIA RTX A2000 12GB, compute capability 8.6
# machine: 12th Gen Intel(R) Core(TM) i5-12600K, 16 logical CPUs, 16 usable, 31.0 GB of memory
#
# calculation 1: 4 atoms, Al 4; mesh 2 2 2, 6 k-points, 8 operations, 10 bands, grid 24 24 24
# parameters: {"mdapy": "2.2.0", "arguments": {"encut": 300.0, "ecutrho": 1200.0, "kpoints": [2, 2, 2], "kspacing": 0.25, "smearing": "gaussian", "sigma": 0.2, "magmoms": null, "hubbard": {}, "functional": "pbe_pw", "tolerance": 1e-06, "max_iterations": 100, "mixing": 0.7, "magnetic_mixing": 1.0, "kerker": 1.5, "preconditioner": "kerker", "bands": 10, "seed": 0, "restart": false, "symmetry": true, "moment_radius": null, "precision": "double", "kernels": "mdapy"}, "pseudopotentials": {"Al": {"file": "Al_std.upf", "sha256": "b02eaa07c5d98f5eeae1bec4155854a7a94ea3635ca6b42e914f3f7ebd6912fb"}}, "internal": {"offset": [0.0, 0.0, 0.0], "minimum": 0.0, "history": 8, "metric_weight": 20.0, "davidson_iterations": 100, "davidson_basis": 2, "correct_forces": true, "magnetic_kerker": 0.0, "atomic_start": true, "kpoint_threads": 0, "first_tolerance": 0.0, "conditioning": 1000000.0, "output_energies": false, "adaptive_damping": false, "first_held": 0, "hold": 0.27211386245988, "hold_steps": 12}, "environment": {}, "gpu": "NVIDIA RTX A2000 12GB", "processes": 1}
# calculation  step         free_energy     change   residual  magnetization  absolute      fermi  davidson      wall
            1     1       -250.09668526        nan   4.48e-01         0.0000    0.0000     8.0558        39      0.11
            1     2       -250.00546055   9.12e-02   8.03e-02         0.0000    0.0000     8.0607        30      0.16
            1     3       -249.97892058   2.65e-02   8.11e-05         0.0000    0.0000     8.0699        22      0.21
            1     4       -249.97899110  -7.05e-05   4.58e-07         0.0000    0.0000     8.0696        38      0.25
            1     5       -249.97899131  -2.10e-07   2.63e-08         0.0000    0.0000     8.0696        36      0.29
# calculation 1: converged in 5 steps, 0.86 s; energy -249.978991 eV, largest force 0.184590 eV/Å, pressure -4.4583 GPa
```

`log` takes a path, or a stream such as `sys.stdout`. The first computation of a
calculator writes the file anew, and the next ones add to it. The heading names MDAPY,
the GPU and the machine. Each computation then writes:

- a line with its atoms, the mesh and its irreducible k-points, the operations of the
  symmetry, the bands and the FFT grid;
- its parameters as one line of JSON, when they differ from the last ones written;
- a row for every step of the self-consistent field, the steps in single precision of
  `precision="mixed"` marked where they begin and end, and with spin the steps whose
  magnetization stays that of `magmoms`;
- its end: the steps, the seconds, the energy, the largest force and, with the virial,
  the pressure; or, if it stopped, where and why.

Lines that are not rows start with `#`, so that `numpy.loadtxt("dft.log")` reads the rows
of every computation as one table; its first column tells them apart.

| Column | Unit | Meaning |
|---|---|---|
| `calculation` | | the number of the computation among those of the calculator, `last["calculation"]` |
| `step` | | the step of the self-consistent field, from 1 |
| `free_energy` | eV | the free energy of the step, of its input density (Harris–Foulkes); that of the last step is the label |
| `change` | eV | from the step before, `nan` for the first |
| `residual` | eV | the Hartree energy of the change of the density, which `tolerance` bounds |
| `magnetization` | Bohr magneton | the moment of the cell, 0 without spin |
| `absolute` | Bohr magneton | the integral of the absolute magnetization over the cell, which also counts moments that cancel; 0 without spin |
| `fermi` | eV | the Fermi level |
| `davidson` | | the iterations of the eigensolver, summed over the k-points and spins |
| `wall` | s | the seconds since the self-consistent field began |

The log costs the run nothing it does not already have. The numbers of a step are those
the self-consistent field keeps of each step, already on the processor, so
nothing more comes from the GPU. Without a log, no step is reported. Under MPI the first
process writes; threads that share a GPU give each calculator its own file.

### What each label keeps

Each computation keeps every parameter it took in `last["parameters"]`, its defaults and
`"auto"` resolved: the mixing, `ecutrho`, the mesh, the bands, the symmetry, the precision
and the kernels, as they ran. Its result keeps them in `info["dft"]`, which `attach` puts
into the frame and `write` into each frame of an extended XYZ file. A label so says how it
was computed, with or without a log. With spin the result also keeps `magnetization` and
`absolute_magnetization` in its `info`, so that the magnetic states of the labels of a set
can be compared from the file. A later version of MDAPY, whose defaults may change,
computes a label again from its record:

<!-- skip-test -->
```python
from mdapy.calculators import DFT
from mdapy.io import read

frame = read("labels.xyz")
record = frame.info["dft"]
again = DFT(pseudopotentials="pseudo-dojo", **record["arguments"])
print(again.compute(frame).energy, frame.info["energy"])
```

| Key | Meaning |
|---|---|
| `mdapy` | the version of MDAPY |
| `arguments` | the arguments of `DFT` that give the same computation: those of the table above but `pseudopotentials`, `gpu`, `comm`, `log` and `start` |
| `pseudopotentials` | of each element, the name of its file and its SHA-256: another file of the same name is another pseudopotential |
| `internal` | the settings of the self-consistent field that `DFT` does not take, such as the 8 steps of the history of Pulay's mixing, by their names inside, in eV and Å as the rest |
| `environment` | the variables `MDAPY_DFT_*` that were set |
| `gpu`, `processes` | the GPU, and the processes of `comm` |

The record gives the same computation, not the same digits: another GPU, or other
kernels, add in another order, and the results agree to the tolerance.

### The ground state in a file

`save` writes the ground state of the last computation, and `start` takes it up again:
the first computation then starts from its density, and from the occupations of the
projectors of PAW or DFT+U, in place of those of the free atoms, as to compute the same
structure with another tolerance or more steps of the self-consistent field. Its first
step solves the bands tightly in that potential, and it converges in a few steps: 4 atoms
of aluminium in 3 instead of 7, PAW silicon in 4 instead of 10.

<!-- skip-test -->
```python
from mdapy.calculators import DFT
from mdapy.io import read

system = read("al32.xyz")
dft = DFT(500.0, kspacing=0.2)
dft.compute(system)
dft.save("al32.npz")

tighter = DFT(500.0, kspacing=0.2, tolerance=1e-9, start="al32.npz")
result = tighter.compute(system)  # in a few steps
```

The file is NumPy's `.npz`, which `numpy.load(path)` reads without pickles. The file
is written whole or not at all, by the first process under MPI. It holds:

| Array | Shape | Unit | Meaning |
|---|---|---|---|
| `format`, `version` | | | `"MDAPY DFT ground state"` and 1, the version of the file |
| `parameters` | | | the record of the computation, as `last["parameters"]`, as JSON |
| `elements` | (N,) str | | the atoms |
| `cell`, `positions` | (3, 3), (N, 3) float64 | Å | the rows of the cell, and the atoms as given |
| `energy`, `fermi` | float64 | eV | the free energy and the Fermi level |
| `forces` | (N, 3) float64 | eV/Å | the forces |
| `density` | (spins, n1, n2, n3) float64 | electrons/Å³ | on the FFT grid of the density, with the augmentation charges; with spin, up then down |
| `kpoints`, `weights` | (K, 3), (K,) float64 | | the irreducible k-points in fractions of the reciprocal vectors, and their weights, adding to 1 |
| `eigenvalues` | (spins, K, bands) float64 | eV | the bands |
| `occupations` | (spins, K, bands) float64 | | of each band, from 0 to 1 (Methfessel–Paxton and Marzari–Vanderbilt a little beyond); a band holds 2 electrons without spin, 1 with |
| `becsum` | (spins, pairs) float64 | | with PAW or DFT+U, the occupations of the projectors |
| `magnetic_moments` | (N,) float64 | Bohr magneton | with spin, as `last["magnetic_moments"]` |

`spins` is 1 without spin and 2 with. The wave functions are not kept: the density
takes a computation back to its ground state in a few steps.

A computation started from a file must be of its elements, in the same order, its cell
and atoms to 1e-6 Å (an atom moved by a vector of the cell is the same), and its
pseudopotentials by their SHA-256; and, as the density is on its grid, of its cutoffs,
symmetry and spin. `InvalidParameter` says which differs. `save` before a computation
raises `InvalidParameter`, as does a file that is not a ground state or one of a later
version.

### In molecular dynamics

The DFT's log and that of the dynamics go to two files:

<!-- skip-test -->
```python
from mdapy.calculators import DFT
from mdapy.io import read
from mdapy.simulation import MolecularDynamics

dft = DFT(500.0, kspacing=0.3, restart="second_order", log="dft.log")
with MolecularDynamics(read("al32.xyz"), dft, timestep=0.002, log="md.log") as md:
    md.velocities(300, seed=1)
    md.run(1000, ensemble="nvt", temperature=300, thermo=10)
```

`md.log` holds a row every 10 steps of the dynamics; `dft.log` holds every step of the
self-consistent field of each of them. The same file for both raises
`InvalidParameter`. A thousand steps of about ten steps of the self-consistent field each
make some 10 000 rows, about 1.5 MB.

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

A small cell leaves most of a GPU idle. Its k-points are solved side by side on threads
of the processor by themselves, and several configurations share one GPU from threads of
one process, each with a calculator of its own:

<!-- skip-test -->
```python
from concurrent.futures import ThreadPoolExecutor

from mdapy.calculators import DFT
from mdapy.io import read_trajectory, write

structures = list(read_trajectory("structures.xyz"))


def label(part):
    dft = DFT(600.0, kspacing=0.2)
    return [dft.compute(s, ("energy", "forces", "virial")).attach(s) for s in part]


with ThreadPoolExecutor(4) as pool:
    parts = list(pool.map(label, [structures[i::4] for i in range(4)]))
write("train.xyz", [s for part in parts for s in part])
```

Four to eight threads fill a GPU with cells of a few dozen atoms; a cell of hundreds of
atoms fills it alone. Configurations side by side hold the sum of their
[memory](#memory-of-the-gpu).

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

## Memory of the GPU

A configuration holds on the GPU the bands of each of its k-points and spins, n plane
waves by the bands in complex numbers of 16 bytes, with n about $\Omega\,(2E)^{3/2}/6\pi^2$
for a cell of Ω bohr³ and `encut` E in hartree (half of that at Γ alone, where the bands
are real); they take most of its memory when it has many k-points. Besides them it
holds the projectors of the atoms, the basis of the eigensolver (twice the bands), the
boxes of the Fourier transforms (256 MiB at most) and the densities and potentials on
the grid. The CUDA context takes 0.3 to 0.6 GiB more, and the libraries of the vendor
(`kernels="vendor"`) about 0.4 GiB for each host thread that solves k-points on an H100
or GH200 (0.05 GiB on a V100): small cells of many k-points run eight. Measured on a
GH200 with the libraries of the vendor in double precision, for cells perturbed as those
of a training set, so that time reversal alone reduces their k-points:

| Cell | Cutoff, kspacing | k-points × spins | Bands | In use (GiB) | On the GPU (GiB) |
|---|---|---|---:|---:|---:|
| Al4, fcc | 600 eV, 0.15 1/Å | 666 × 1 | 10 | 0.4 | 4.4 |
| Al32 | 600 eV, 0.15 1/Å | 112 × 1 | 58 | 2.4 | 3.5 |
| Al108 | 600 eV, Γ | 1 × 1 | 195 | 2.6 | 3.6 |
| Al108 | 600 eV, 0.15 1/Å | 36 × 1 | 195 | 10.3 | 11.4 |
| Fe16, bcc, 3 μB | 900 eV, 0.2 1/Å | 112 × 2 | 154 | 6.4 | 7.8 |
| Fe54, bcc, 3 μB | 900 eV, 0.2 1/Å | 36 × 2 | 519 | 25.0 | 26.8 |

"In use" is the peak of the arrays of the DFT, "on the GPU" that of the device, with the
context and the libraries. Mixed precision holds 5 to 10 % more. Configurations side by
side hold the sum of theirs: eight Al32 at once hold 19.2 GiB and fit a V100 of 32 GB,
sixteen do not. A run keeps what it uses in a memory pool and gives back to the device,
at its end, what the pool holds beyond the use of its steps: a process holds the memory
of its last runs, not the most it ever took. `last["gpu_memory"]` holds the largest at
the end of the steps, in MiB: `in_use` by the arrays of the DFT and `reserved` by its
pool, of the whole process, the configurations of its threads together, and `device`
in use on the GPU, by the CUDA context, the libraries and other processes too. See
[the measurements](../../performance/dft.md#memory).

## What it computes

- **Plane waves** of kinetic energy up to `encut` at each k-point of the mesh;
  k-points related by time reversal count once. The bands live on the FFT grid
  of four times the cutoff, the density and the potentials on that of
  `ecutrho`, chosen as Quantum ESPRESSO chooses them: along each vector at
  least 2 m + 1 points, m the largest Miller index of the sphere, with 2, 3
  and 5 as the only factors, and with the symmetry a multiple of the
  denominators of its fractional translations (4 in diamond). When the mesh is Γ alone, as for
  large cells, the bands are real: they keep half of the plane waves, the
  other half being their complex conjugates, with real Fourier transforms and
  real products of the bands, for the same results in less time and half the
  memory.
- **Pseudopotentials:** in UPF 2, norm-conserving, ultrasoft or PAW, with
  their projectors, the core charge of the nonlinear core correction and,
  ultrasoft and PAW, their augmentation charges by angular momentum
  (`q_with_l`, as the files of a recent `ld1.x` or of atompaw); and in UPF 1,
  norm-conserving or ultrasoft, such as GBRV's, whose augmentation charges are
  pseudized inside `rinner` as Quantum ESPRESSO pseudizes them. Every file of
  SSSP 1.3 is read, and its folder can be given as `pseudopotentials`. The
  tested ones are PseudoDojo's norm-conserving tables, the ultrasoft and PAW
  files of pslibrary 1.0.0, GBRV's ultrasoft tungsten and atompaw's lanthanum.
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
- **DFT+U** of [Dudarev et al. (1998)](../../reference/references.md#dudarev1998):
  E_U = Σ (U/2) Tr[n(1 − n)] over the Hubbard atoms and spins, n the
  occupations of the pseudo-orbital named in `hubbard` on Quantum ESPRESSO's
  atomic projectors, the pseudo-orbitals as they are, neither orthogonalized
  nor normalized ([Cococcioni and de Gironcoli 2005](../../reference/references.md#cococcioni2005)).
  The occupations start as Quantum ESPRESSO starts them, the majority spin
  filled first, and are mixed with the density; the forces and the stress
  include them. U belongs to its projectors: values fitted with those of
  other codes, such as the spheres of VASP's PAW or the orthogonalized
  orbitals of Quantum ESPRESSO's linear response, do not carry over.
  Norm-conserving pseudopotentials only, for now.
- **The self-consistent field:** the bands by block Davidson, from the
  pseudo-orbitals of the free atoms at the start, the density mixed
  by Pulay's method in the metric of Kresse and Furthmüller with Kerker's
  preconditioner. A run that does not converge raises. With `restart`, a run
  starts from the bands of the last, plane wave by plane wave, and from its
  density with the free atoms moved from where they were to where they are
  (`restart=True`, Quantum ESPRESSO's `pot_extrapolation='atomic'`); with
  `restart="second_order"` the density less its free atoms is extrapolated
  from the last three runs, d(t+1) = d(t) + α (d(t) − d(t−1)) + β (d(t−1) −
  d(t−2)), α and β fitted to the displacements of the atoms (one step: the
  atomic; two: α = 1, β = 0). A run of other k-points, bands, spins or a number of atoms starts cold.
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
atom of the wrong number; `ValueError` for a Hubbard orbital that is not in
the file or a Hubbard element with an ultrasoft or PAW pseudopotential; `MissingProperty` without an `element` column;
and [`CalculationFailed`](../../reference/api/utilities.md#mdapy.errors.CalculationFailed)
when the self-consistent field does not converge, with its last energies.

## When the self-consistent field does not converge

The defaults suit bulk cells. Cells of metal and vacuum, surfaces and vacancies converge
better with the preconditioner of the local density of states, which screens the metal and
not the vacuum:

<!-- skip-test -->
```python
from mdapy.calculators import DFT
from mdapy.errors import CalculationFailed

dft = DFT(600.0, preconditioner="ldos")
try:
    result = dft.compute(slab, ("energy", "forces", "virial"))
except CalculationFailed:  # slower still: a smaller share of the residual, more steps
    dft = DFT(600.0, preconditioner="ldos", mixing=0.2, max_iterations=200)
    result = dft.compute(slab, ("energy", "forces", "virial"))
print(dft.last["iterations"])
```

## The first moments and the magnetic state

A magnet has more than one self-consistent state: moments of other sizes and signs, and
always the state without moments. `magmoms` chooses among them, by the signs and the sizes
of the moments the atoms start with. Give each atom about the moment it is expected to
carry, with its sign:

<!-- skip-test -->
```python
from mdapy.calculators import DFT

dft = DFT(600.0, magmoms={"Cr": -1.0, "Co": 1.5, "Ni": 0.6})
result = dft.compute(cell, ("energy", "forces", "virial"))
print(dft.last["magnetization"], dft.last["absolute_magnetization"])
print(dft.last["magnetic_moments"], dft.last["held_steps"])
```

For the first steps the magnetization stays that of `magmoms`, and the total density alone
is mixed: until the Hartree energy of the residual of the total density is below 0.27 eV
per atom, 12 steps at most. `last["held_steps"]` counts these steps, and the log marks
where they begin and end. In the potential of superposed free atoms, which the first
step has, the bands do not carry the moments they are given, however tightly they are
solved. In a cell of Co2CrNi started as above, with 3.6 Bohr magnetons on Co and Ni and −1
on Cr, the first step returns 0.6 and −0.5, the second 2.2 and −0.1, and the steps after,
as the density settles, 2.0 and −1.2: near the 2.1 and −2.2 of the ferrimagnet the cell
converges to. A magnetization mixed from the first steps has lost its start before the
density is right, and the run ends where those steps left it, in a state of smaller
moments or none, tens of meV per atom too high. A computation from a ground state
(`restart`, `start`) holds nothing: its magnetization is that of its state.

What tells a wrong state:

- A computation whose moments are gone at its end warns, and `last["moments_lost"]` is
  `True`: the absolute magnetization is below 0.05 Bohr magnetons for each atom that
  started with a moment. That is the state without spin, the ground state of a cell that
  is not magnetic and otherwise a stationary state above it.
- `last["magnetic_moments"]` gives the moment of each atom, and the log the magnetization
  and the absolute magnetization of every step. With each label, `info["magnetization"]`
  and `info["absolute_magnetization"]` go into the file: compare them across the labels of
  a set, and start the cells of a set alike.
- Two patterns of signs are two states, whatever the sizes. Where the order is not known,
  as for atoms of chromium, manganese or fcc iron that neighbour each other, compute the
  patterns and keep the lowest:

<!-- skip-test -->
```python
from mdapy.calculators import DFT

patterns = {
    "ferrimagnetic": {"Cr": -1.0, "Co": 1.5, "Ni": 0.6},
    "ferromagnetic": {"Cr": 1.0, "Co": 1.5, "Ni": 0.6},
}
states = {}
for name, magmoms in patterns.items():
    dft = DFT(600.0, magmoms=magmoms)
    states[name] = (dft.compute(cell).energy, dft.last["magnetic_moments"])
lowest = min(states, key=lambda name: states[name][0])
```

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

With DFT+U against Quantum ESPRESSO's `HUBBARD (atomic)`, antiferromagnetic NiO
of four displaced atoms with U = 6 eV on the 3d of nickel, and two displaced
atoms of copper without spin with U = 4 eV, agree in the energy to 1e-8 hartree
per atom (NiO once the rounded constant is allowed for), in the Hubbard energy
to the 8 digits Quantum ESPRESSO prints, in the traces of the occupations to
1e-5, in the forces to 1e-6 hartree/bohr, in the stress to 0.003 kbar, in the
occupied eigenvalues to 1e-6 hartree and, NiO, in the magnetization to 1e-6
Bohr magneton.

What differs is Quantum ESPRESSO's own:

- Its Ewald sum is converged to about 1e-7 Ry.
- Its spin-polarized PBE correlation takes γ = 0.031091, rounded, where its
  paramagnetic one takes (1 − ln 2)/π². This moves the energy of iron by 1e-6
  hartree, and with PAW by 4e-6 more through the one-center densities. A NumPy
  evaluation of the difference on MDAPY's densities accounts for it to 1e-8.

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
