<!-- needs: mdapy._cuda -->

# On the GPU

The potentials of MDAPY compute on an NVIDIA GPU with CUDA, and
[molecular dynamics](dynamics.md) with them runs there whole, for systems of
hundreds of thousands to millions of atoms: `device="cuda"` puts a potential on the
GPU, and `MolecularDynamics` runs on the GPU with a potential that is there.

```python
from mdapy.build import build_crystal
from mdapy.calculators import EAM
from mdapy.simulation import MolecularDynamics

potential = EAM.zhou(["Cu"], device="cuda")  # mixed precision
crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(40, 40, 40))  # 256 000 atoms

with MolecularDynamics(crystal, potential, timestep=0.002) as md:
    md.velocities(600, seed=1)
    md.run(5000, ensemble="npt", temperature=300, pressure=0.0, thermo=100)
    hot = md.system
```

The calls are those of the processor, and so are the results: `thermo`, `frames`,
`iterate` and `system` give what they give there. The GPU keeps the atoms, their
neighbor list and the steps of the integrator; each step copies back only the few
numbers the thermostat and the barostat need, and the atoms come back when a frame or
the system is asked for.

The total energy of copper at 600 K at constant energy, in mixed and in double
precision:

```python
import numpy as np
from mdapy.visualization import plot

energies = {}
for precision in ("mixed", "double"):
    copper = EAM.zhou(["Cu"], device="cuda", precision=precision)
    with MolecularDynamics(crystal, copper, timestep=0.002) as md:
        md.velocities(600, seed=1)
        md.run(5000, thermo=50)
        thermo = md.thermo
    energies[precision] = (thermo["time"], thermo["total_energy"] / len(crystal))

with plot.style():
    fig, ax = plot.figure("single")
    for precision, (time, energy) in energies.items():
        ax.plot(time, 1e6 * (energy - energy[1]), label=precision)
    ax.set(xlabel="Time (ps)", ylabel="Total energy − at 0.1 ps (µeV/atom)")
    ax.legend()
    plot.save(fig, "images/gpu-energy", formats=("png",))
```

![Total energy at constant energy in mixed and double precision](images/gpu-energy.png)

*The total energy of 256 000 copper atoms at constant energy from 600 K, from its
value at 0.1 ps. In the first picosecond, alike in both precisions, it is the error of
the time step of 2 fs, as the atoms leave the perfect crystal, where no force acts.
After it, double precision keeps the energy to 0.2 µeV per atom, and mixed precision
lets it drift by about 0.6 µeV per atom per ps; a thermostat takes that out.*

## Building the GPU part

The wheels of MDAPY on PyPI hold no GPU part. MDAPY built from its source with the
CUDA toolkit, 12 or newer, holds it, the module `mdapy._cuda`:

```bash
pip install mdapy --no-binary mdapy -Ccmake.define.MDAPY_CUDA=ON
```

It is compiled for the GPU of the machine; `-Ccmake.define.CMAKE_CUDA_ARCHITECTURES="80;86;90"`
compiles it for others, such as those of a cluster. The CUDA runtime is linked into
the module, so running it needs only the driver of the GPU.

## Potentials

| Calculator | On the GPU |
|---|---|
| [`EAM`](eam.md), `EAM.zhou` and their average atoms | `eam/alloy`, `eam/fs` and funcfl files of `eam` |
| [`LennardJones`](pair-potentials.md), [`Morse`](pair-potentials.md) | every pair of elements, with `shift` and the mixing rules |

| Parameter | Type | Default | Meaning |
|---|---|---|---|
| `device` | str | `"cpu"` | `"cpu"`, or `"cuda"` to compute on the GPU |
| `precision` | str | `"mixed"` on the GPU | `"mixed"`: the distance and the terms of each pair in single precision, the positions, the velocities and every sum in double, as the GPU package of LAMMPS in mixed precision; or `"double"` throughout. The processor computes in double precision only |

`compute` of a potential on the GPU copies the atoms there, builds a
[neighbor list](../../basics/neighbor.md) of its own, with a skin of 1 Å that the
next system of as many atoms uses again while it fits, and copies the results back;
a neighbor list given to it is not used. Every calculator has `device` and `precision`.

## Precision

| | Mixed | Double |
|---|---|---|
| Forces, atom by atom | within about 1e-5 of the largest force | those of the processor to rounding, 1e-14 |
| Energies | within about 1e-5 of the energy of each atom | to rounding |
| Total virial | within about 1e-4 of its largest component | to rounding |
| Molecular dynamics | the same runs as long as rounding allows | the runs of the processor, step by step, in every ensemble |

In double precision a consumer GPU computes much slower than in single, a part of
its speed in single precision. Langevin dynamics draws its random forces on the GPU:
a seed gives the same run there every time, but not the run of the processor with
that seed.

## Molecular dynamics

`MolecularDynamics` with a potential on the GPU runs every ensemble, `"nve"`, `"nvt"`,
`"npt"` and `"nph"`, with every barostat and both thermostats, as
[MDAPY's own dynamics](dynamics.md#in-mdapy-with-any-calculator) does on the
processor. Along every periodic direction the box must be thicker than twice the
cutoff of the potential and the skin, 1 Å, at every step, also as a barostat changes
it; smaller boxes, small systems and a `movable` column are for the processor.

## Errors

A potential with `device="cuda"` raises `InvalidParameter` when MDAPY was built
without its GPU part, when there is no GPU, for a `device` or a `precision` not in
the tables above, and for `"mixed"` on the processor; `compute` and
`MolecularDynamics` raise it for a box too thin along a periodic direction, and
`MolecularDynamics` for a `movable` column. The dynamics raises `CalculationFailed`
when the positions, the energy or the forces stop being finite.

See also: [Molecular dynamics on the GPU](../../performance/gpu.md), how fast it is.
