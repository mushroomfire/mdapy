# Molecular dynamics

`mdapy.simulation.MolecularDynamics(system, potential, *, timestep=0.001)`
runs molecular dynamics of a system in LAMMPS, with the interaction of a
[LAMMPS calculator](lammps.md), and gives back the thermodynamic state, the
atoms along the way and the system at the end.

<!-- skip-test -->
```python
from mdapy.build import build_crystal
from mdapy.calculators import LAMMPS
from mdapy.simulation import MolecularDynamics

potential = LAMMPS(
    "pair_style eam/alloy\npair_coeff * * Cu_mishin1.eam.alloy Cu", elements=["Cu"]
)
crystal = build_crystal("fcc", 3.615, elements="Cu", repeat=(8, 8, 8))

with MolecularDynamics(crystal, potential, timestep=0.002) as md:
    md.velocities(600, seed=1)
    md.run(10000, ensemble="npt", temperature=300, pressure=0.0, thermo=100)
    md.run(5000, ensemble="nvt", temperature=300, frames=500)
    md.thermo  # polars DataFrame: step, time, temperature, energies, ...
    md.frames  # the atoms every 500 steps of the second run
    hot = md.system  # the atoms at the end, with velocities and images
```

LAMMPS runs in metal units, those of mdapy, through its Python module,
installed as the [LAMMPS documentation](https://docs.lammps.org/Python_install.html)
describes; `potential` gives the commands of the interaction, the element
of each atom type and the command-line arguments of LAMMPS, such as those
of Kokkos. The box may have any orientation and need not be periodic:
LAMMPS gets it rotated into its own form, its non-periodic directions
shrink-wrapped, and everything comes back in the orientation of `system`.
The velocities of `system`, if it has them, are where the dynamics starts.

| Name | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `system` | `System` | | | the atoms, with an `element` column, and `velocities` in Å/ps when it has them |
| `potential` | `LAMMPS` | | | the interaction, as a LAMMPS calculator |
| `timestep` | float | 0.001 | ps | the time step |

## Running

| Method or attribute | Meaning |
|---|---|
| `velocities(temperature, *, seed)` | random velocities of a temperature in K, Gaussian, with no total momentum; the same for the same seed and atoms, whatever their positions |
| `run(steps, *, ensemble="nve", temperature=None, pressure=None, barostat="iso", thermostat_damping=None, barostat_damping=None, thermo=100, frames=None, progress=None, cancel=None)` | runs `steps` steps |
| `command(commands)` | runs any LAMMPS commands, one or several lines |
| `lammps` | the LAMMPS instance itself |
| `close()` | closes the instance; leaving a `with` block closes it too |

| Parameter of `run` | Type | Default | Unit | Meaning |
|---|---|---|---|---|
| `ensemble` | str | `"nve"` | | `"nve"`, or `"nvt"` with a Nosé–Hoover thermostat, or `"npt"` with a thermostat and a barostat (LAMMPS `fix nve`, `nvt`, `npt`) |
| `temperature` | float or (float, float) | | K | the temperature of the thermostat, or its values at the start and end of the run; for `"nvt"` and `"npt"` |
| `pressure` | float or (float, float) | | GPa | the pressure of the barostat, or its values at the start and end; for `"npt"` |
| `barostat` | str | `"iso"` | | the box changes alike in every direction (`"iso"`), along each axis (`"aniso"`) or in any way (`"tri"`) |
| `thermostat_damping` | float | 100 time steps | ps | the damping time of the thermostat |
| `barostat_damping` | float | 1000 time steps | ps | the damping time of the barostat |
| `thermo` | int | 100 | | records the thermodynamic state every this many steps, and at the start and the end |
| `frames` | int | `None` | | keeps the atoms every this many steps |
| `progress` | callable | `None` | | called as `progress(done, total)` with the steps run |
| `cancel` | `CancelToken` | `None` | | stops the run, which raises `Cancelled` |

A run is a LAMMPS `run` in pieces between the steps at which something is
recorded; the pieces run as one run would, a temperature or pressure
changing from its start to its end value across all of them.

## Results

| Attribute | Type | Meaning |
|---|---|---|
| `system` | `System` | the atoms now, with `velocities` and `image` columns, the box, and `timestep` and `time` in `info`; the other columns of `system`, but forces and energies, which are out of date |
| `thermo` | polars `DataFrame` | the thermodynamic state of all the runs so far, one row per record |
| `frames` | tuple of `System` | the atoms kept in all the runs so far, as `system` gives them |
| `step` | int | the time step number of LAMMPS |

| Column of `thermo` | Type | Unit | Meaning |
|---|---|---|---|
| `step` | int64 | | time step number |
| `time` | f64 | ps | time |
| `temperature` | f64 | K | temperature |
| `potential_energy`, `kinetic_energy`, `total_energy` | f64 | eV | energies of all the atoms |
| `pressure` | f64 | GPa | pressure |
| `volume` | f64 | Å³ | volume of the box |
| `stress` | 6 × f64 | GPa | stress in Voigt order xx, yy, zz, yz, xz, xy, positive under tension, in the axes of `system` |

## Errors

`MolecularDynamics` raises `InvalidParameter` for a potential that is not a
LAMMPS calculator, a time step that is not positive, elements without an
atom type, and commands that LAMMPS rejects; `MissingProperty` for a system
without elements; `ImportError` without the LAMMPS Python module.
`velocities` raises `InvalidParameter` for a negative temperature or a seed
that is not positive; `run` for a negative number of steps, an unknown
ensemble or barostat, a thermostat without a temperature, a barostat
without a pressure, a barostat other than `"iso"` in a box that is not
periodic, and `thermo` or `frames` that are not positive; `command` for
commands LAMMPS rejects; and `lammps` after `close`.
