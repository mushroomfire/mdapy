# Cheat sheet

The lessons and the most used parts of the user guide on one page. Everything here
assumes:

<!-- skip-test -->
```python
import numpy as np
import mdapy as mp
```

and the imports in [the table at the end](#where-each-thing-is-imported-from). The
sections after [Show and save it](#show-and-save-it) go beyond the lessons; each links the
pages that tell more.

## Get a system

| I want | Write |
|---|---|
| a file | `system = read("file.data")` |
| a crystal | `build_crystal("fcc", 3.615, elements="Cu", repeat=(8, 8, 8))` |
| a polycrystal | `build_polycrystal(crystal, mp.Box(100.0), 8, seed=1)` |
| an alloy | `random_alloy(crystal, {"Cu": 0.5, "Ni": 0.5}, seed=1)` |
| vacancies | `add_vacancies(crystal, 10, seed=1)` |
| more copies | `replicate(crystal, (2, 2, 1))` |
| a crystal in a box along x, y and z | `orthogonal_cell(crystal)` |
| atoms of my own | `mp.System(positions, mp.Box(20.0), elements="Cu")` |
| to know what I have | `print(system)` |

## Read it

| I want | Write |
|---|---|
| a column | `system["positions"]`, `system["element"]` |
| a component | `system["positions.z"]` |
| an expression | `system["positions.x**2 + positions.y**2"]` |
| the number of atoms | `len(system)` |
| the box | `system.box.matrix`, `system.box.volume`, `system.box.pbc` |
| global values | `system.info["timestep"]` |
| anything else | `system.table`, a polars DataFrame |

## Change it

A system never changes: each of these returns a new one.

| I want | Write |
|---|---|
| a new or changed column | `system.with_columns(charge=values)` |
| the columns of an analysis | `system.with_columns(result)`, named after it, as `cna_structure` |
| the columns of two runs of one analysis | `system.with_columns(near=first, far=second)` |
| to move atoms | `system.with_columns(positions=new_positions)` |
| another box | `system.with_box(matrix, scale_positions=True)` |
| global values | `system.with_info(energy=-3.54)` |
| some atoms, by condition | `system.filter("positions.z > 10", element="Cu")` |
| some atoms, by index | `system.gather(indices)` |
| the atoms in another order | `system.sort("id")` |
| to drop columns | `system.drop("velocities")` |

## Ask it

Every analysis has the same three steps:

<!-- skip-test -->
```python
result = Analysis(parameters).compute(system)
system = system.with_columns(result)  # per-atom values, as columns
```

| I want | Write |
|---|---|
| the structure of each atom | `CommonNeighborAnalysis()`, `PolyhedralTemplateMatching()` |
| their numbers | `result.counts()` |
| the number of neighbors | `CoordinationNumber(cutoff=3.0)` |
| the radial distribution | `RadialDistributionFunction(cutoff=8.0)`, then `result.r`, `result.g` |
| the dislocations | `DislocationAnalysis("fcc")`, then `result.lines`, `result.density` |
| them in OVITO | `result.write("lines.ca")`, and `read_dislocations("lines.ca")` back |
| one list for several analyses | `nl = mp.NeighborList.build(system, 5.0)`, then `compute(system, neighbors=nl)` |

## Compute energies

| I want | Write |
|---|---|
| a potential | `EAM.zhou(["Cu"])` or `EAM("Cu.eam.alloy")` |
| a machine-learned potential | `NEP("nep.txt")` |
| a potential of LAMMPS | `LAMMPS("pair_style eam/alloy\npair_coeff * * Cu.eam.alloy Cu", elements=["Cu"])` |
| energy and forces | `potential.compute(system, ("energy", "forces"))` |
| to relax the atoms | `minimize(system, potential)`, then `result.system` |
| to relax the box as well | `minimize(system, potential, cell="full")` |

## Follow a trajectory

| I want | Write |
|---|---|
| to open a file | `with read_trajectory("dump.*.lammpstrj") as trajectory:` |
| a frame, or some frames | `trajectory[0]`, `trajectory[::10]` |
| the frames through a function | `trajectory.map(function)` |
| one number per frame | `TimeSeries(function).compute(trajectory)` |
| to write them | `write("out.xyz", trajectory.map(function))` |

## Show and save it

| I want | Write |
|---|---|
| a file | `write("out.xyz", system)` |
| a picture | `render(system, "picture.png")` |
| colors by a column | `render(system, "picture.png", color="coordination")` |
| colors by structure | `render(system, "picture.png", color=result)` |
| what the colors mean | `render(system, "picture.png", color=result, legend=True)` |
| some atoms only | `render(system, "picture.png", atoms=system["positions.x"] < 20)` |
| on the graphics card | `render(system, "picture.png", device="gpu")` |
| a figure | `with plot.style(): fig, ax = plot.figure("single")` |
| to save a figure | `plot.save(fig, "figure")` |

## Simulate

Molecular dynamics runs with any calculator, in MDAPY itself, or in LAMMPS with a
`LAMMPS` calculator; the methods written with `lammps` below still need a `LAMMPS`
calculator. See
[Molecular dynamics](../../user-guide/simulation/running.md),
[Defect energies](../../user-guide/properties/defect-energies.md),
[Melting point](../../user-guide/properties/melting-point.md) and
[Free energy](../../user-guide/properties/free-energy.md).

| I want | Write |
|---|---|
| molecular dynamics | `with MolecularDynamics(system, potential, timestep=0.002) as md:` |
| to start it at a temperature | `md.velocities(600, seed=1)` |
| to run it | `md.run(10000, ensemble="npt", temperature=300, pressure=0.0, save_every=500)` |
| what it gives | `md.thermo`, `md.frames`, `md.system` |
| the energy of a vacancy | `vacancy_formation_energy(crystal, potential).energy` |
| of a surface | `surface_energy(crystal, potential).energy` |
| the melting point | `melting_point(crystal, lammps, 1000.0).temperature` |
| free energies | `free_energy(crystal, lammps, temperatures)` |

## Lattice dynamics

Of a relaxed crystal, `relaxed = minimize(crystal, potential, cell="full").system`, and its
primitive cell, `cell = primitive_cell(relaxed)`. See
[Elastic constants](../../user-guide/properties/elastic-constants.md),
[Electronic structure](../../user-guide/properties/electronic-structure.md),
[Phonons](../../user-guide/properties/phonons.md),
[Force constants](../../user-guide/properties/force-constants.md),
[Quasi-harmonic approximation](../../user-guide/properties/quasi-harmonic.md) and
[Thermal conductivity](../../user-guide/properties/thermal-conductivity.md).

| I want | Write |
|---|---|
| elastic constants | `result = elastic_constants(relaxed, potential)`, then `result.constants` in GPa |
| the moduli of a polycrystal | `result.bulk_modulus()`, `result.shear_modulus()`, `result.young_modulus()` |
| a modulus along a direction | `result.young_modulus([1, 1, 1])`, `result.poisson_ratio([1, 0, 0], [0, 1, 0])` |
| it along every direction | `plot.elastic_surface(result)` |
| elastic constants at a temperature | `md_elastic_constants(relaxed, lammps, 300.0)` |
| the bands of the electrons | `structure = electronic_structure(cell, dft)`, then `plot.electronic_bands(structure.band_structure("fcc"))` |
| their density of states | `plot.electronic_dos(structure.dos())`, states per eV of the cell |
| phonons | `result = phonons(cell, potential)`, then `result.frequencies([0.5, 0.5, 0.5])` in THz |
| their bands and density of states | `plot.band_structure(result.band_structure("fcc"))`, `plot.phonon_dos(result.dos(20))` |
| their free energy | `result.thermal([100, 300, 600])` |
| force constants of DFT or any calculator | `displaced_supercells(cell, supercell=4)`, their forces, then `force_constants(cell, frames)` |
| those of phonopy | `read_force_constants("FORCE_CONSTANTS", cell, 4)`, and `constants.write("FORCE_CONSTANTS")` |
| the thermal expansion | `quasi_harmonic(cell, potential, temperatures)`, then `result.volume` |
| the thermal conductivity | `thermal_conductivity(constants, temperatures, mesh=11)`, then `result.kappa` |
| its third order force constants | `force_constants(cell, frames, cutoff_third=4.5)` of `displaced_supercells(cell, supercell=4, method="random", count=2)` |
| it from molecular dynamics | `green_kubo(system, lammps, 300.0)`, then `result.kappa` |

## Fit data

See [Interpolation](../../user-guide/fitting/interpolation.md),
[Curve fitting](../../user-guide/fitting/curve-fitting.md) and
[Equations of state](../../user-guide/fitting/equation-of-state.md).

| I want | Write |
|---|---|
| a smooth curve through points | `curve = Spline(x, y)`, then `curve(new_x)`, and its slope `curve(new_x, 1)` |
| a smooth surface through a grid | `GridSpline(x, y, z)` |
| a polynomial | `line = fit_polynomial(x, y, 2)`, then `line.coefficients`, `line(new_x)` |
| one of two variables | `fit_surface(x, y, z, 2)` |
| a peak, a decay, an Arrhenius law | `fit = fit_curve(gaussian, x, y)`, then `fit["center"]`, `fit.errors` |
| a function of my own | `fit_curve(lambda x, a, b: a * np.exp(-x / b), x, y, guess=[1.0, 10.0])`, then `fit.parameters` |
| an equation of state | `eos = fit_eos(volumes, energies)`, then `eos.v0`, `eos.b0`, `eos.pressure(11.0)` |

## Train a potential

See [Perturbed structures](../../user-guide/mlip/perturbation.md),
[Checking frames](../../user-guide/mlip/checks.md),
[Errors of a model](../../user-guide/mlip/errors.md) and
[Choosing frames](../../user-guide/mlip/selection.md).

| I want | Write |
|---|---|
| shaken and strained copies | `frames = perturb(crystal, 50, seed=1)` |
| to check frames | `check_frames(frames)` |
| the errors of a model | `p = predict(frames, NEP("nep.txt"))`, then `p.rmse()`, `p.worst(10)` |
| its parity plot | `plot.parity(p.forces[:, 0], p.forces[:, 1], unit="eV/Å")` |
| the frames most unlike each other | `farthest_point_sampling(structure_descriptors(frames, nep), 100)` |

## Where each thing is imported from

| Module | Holds |
|---|---|
| `mdapy` | `System`, `Box`, `Topology`, `NeighborList`, `set_num_threads` |
| `mdapy.io` | `read`, `write`, `read_trajectory` |
| `mdapy.build` | `build_crystal`, `build_polycrystal`, `random_alloy`, `add_vacancies` and others |
| `mdapy.transform` | `replicate`, `orthogonal_cell`, `wrap`, `combine`, `slice` and others |
| `mdapy.identify` | the structure analyses |
| `mdapy.analysis` | every other analysis, `DislocationAnalysis` and `read_dislocations` among them |
| `mdapy.calculators` | `EAM`, `NEP`, `LAMMPS`, `VASP` |
| `mdapy.simulation` | `MolecularDynamics`, `minimize`, `thermal_velocities`; the controls in `mdapy.simulation.controls` |
| `mdapy.properties.defects` | `vacancy_formation_energy`, `surface_energy`, `stacking_fault_energy`, `stacking_fault_free_energy` |
| `mdapy.properties.elastic` | `elastic_constants`, `thermal_elastic_constants`, `md_elastic_constants` |
| `mdapy.properties.electronic` | `electronic_structure` |
| `mdapy.properties.phonons` | `phonons`, `force_constants`, `thermal_phonons`, `bond_stiffness` and others |
| `mdapy.properties.transport` | `thermal_conductivity`, `green_kubo`, `hnemd`, `nemd` |
| `mdapy.properties.thermodynamics` | `quasi_harmonic`, `free_energy`, `melting_point`, phase diagrams |
| `mdapy.fitting` | `Spline`, `GridSpline`, `fit_polynomial`, `fit_curve`, `fit_eos` and others |
| `mdapy.mlip` | `perturb`, `check_frames`, `predict`, `farthest_point_sampling`, descriptors |
| `mdapy.visualization` | `render`, `plot`, `write_movie` |
| `mdapy.interop` | `to_ase`, `from_ase`, `to_ovito`, `from_ovito` and others |
