# From MDAPY 1.x

MDAPY 2.0 is written anew, and code written for 1.x does not run with it unchanged. This
page gives the 2.0 way of each common 1.x call. MDAPY 1.x stays on PyPI for code that
needs it, `pip install "mdapy<2"` (see [Compatibility](../about/compatibility.md)).

## What changed

- **A system never changes.** Each change returns a new system, such as
  `system.with_columns(...)` or `system.with_box(...)`. The 1.x `system.cal_*` methods,
  which wrote columns into `system.data`, are now analysis classes:
  `result = Analysis(parameters).compute(system)`, then
  `system = system.with_columns(result)` adds the result's columns.
- **Positions are one column**, `positions`, of (N, 3) float64, and so are `velocities`
  and `forces`; 1.x had `x`, `y`, `z`, `vx`, ... `system.table` is the polars DataFrame
  of the atoms (1.x `system.data`).
- **Everything has a module**: `mdapy.io`, `mdapy.build`, `mdapy.transform`,
  `mdapy.identify` (structure analyses), `mdapy.analysis`, `mdapy.calculators`,
  `mdapy.simulation`, `mdapy.lattice_dynamics`, `mdapy.mlip`, `mdapy.visualization`
  and `mdapy.interop`. The top level holds only the foundations: `System`, `Box`,
  `Topology`, `NeighborList` and the threads.
- **Neighbor lists are objects**, built once and given to the analyses that share them,
  `compute(system, neighbors=nl)`, not a state of the system.
- **Potentials are given to what uses them**, as `minimize(system, potential)`, not
  attached as `system.calc`.
- **Results have named attributes**, and figures are drawn by
  [`mdapy.visualization.plot`](../user-guide/visualization/plots.md), not by `plot()`
  methods of the results.

The units are LAMMPS's metal units, Å, eV and ps, as in 1.x; stress and pressure are in GPa
(see [Units](../reference/units.md)).

## One script in both

In MDAPY 1.x:

<!-- skip-test -->
```python
import mdapy as mp

system = mp.build_crystal("Cu", "fcc", 3.615, nx=10, ny=10, nz=10)
system.cal_common_neighbor_analysis()
system.cal_centro_symmetry_parameter(12)
rdf = system.cal_radial_distribution_function(rc=5.0, nbin=100)
rdf.plot()
system.write_dump("copper.dump")
```

In MDAPY 2.0:

```python
from mdapy.analysis import RadialDistributionFunction
from mdapy.build import build_crystal
from mdapy.identify import CentroSymmetryParameter, CommonNeighborAnalysis
from mdapy.io import write
from mdapy.visualization import plot

system = build_crystal("fcc", 3.615, elements="Cu", repeat=(10, 10, 10))
system = system.with_columns(
    CommonNeighborAnalysis().compute(system),  # the column cna_structure
    CentroSymmetryParameter(12).compute(system),  # the column csp
)
rdf = RadialDistributionFunction(5.0, bins=100).compute(system)
with plot.style():
    fig, ax = plot.figure()
    plot.rdf(rdf, ax=ax)
write("copper.dump", system)
```

## Systems and files

| MDAPY 1.x | MDAPY 2.0 |
|---|---|
| `mp.System("file.dump")` | `read("file.dump")`, from `mdapy.io`; the format from the name, or given: `read(path, "lammps-dump")` |
| `mp.System(pos=pos, box=10)` | `mp.System(pos, mp.Box(10.0))` |
| `mp.System(data=df, box=box)` | `mp.System(table=df, box=box)`, the table with a `positions` column: `df.with_columns(positions=pl.concat_arr("x", "y", "z"))` |
| `mp.System(ase_atom=atoms)`, `mp.System(ovito_atom=data)` | `from_ase(atoms)`, `from_ovito(data)`, from `mdapy.interop` |
| `system.to_ase()`, `system.to_ovito()` | `to_ase(system)`, `to_ovito(system)` |
| `system.write_dump("a.dump")`, `write_data`, `write_xyz`, `write_poscar`, `write_mp` | `write("a.dump", system)`, the format from the name: `a.data`, `a.xyz`, `POSCAR`, `a.mdapy` |
| `compress=True`, `mp.compress_file` | a name that ends in `.gz`, such as `write("a.dump.gz", system)` |
| `write_data(..., element_list=["Cu", "Ni"])` | `write("a.data", system, elements=["Cu", "Ni"])` |
| `XYZTrajectory(file)`, `Trajectory(file)` | `read_trajectory(file)`, frames read when used, by index or slice |
| `trajectory.save(file)` | `write(file, frames)` |
| `system.data` | `system.table` |
| `system.N` | `len(system)` |
| `system.box.box`, `system.box.boundary` | `system.box.matrix`, `system.box.pbc` |
| `system.global_info` | `system.info` |
| `system.get_positions()`, `get_velocities()` | `system["positions"]`, `system["velocities"]` |
| `system.set_element(...)` | `system.with_columns(element=...)` |
| `system.update_data(df)` | `system.with_table(df)` |
| `system.update_box(box, scale_pos=True)` | `system.with_box(box, scale_positions=True)` |
| `system.wrap_pos()` | `wrap(system)`, from `mdapy.transform` |
| `system.replicate(nx, ny, nz)` | `replicate(system, (nx, ny, nz))` |
| `mp.unwrap_trajectory(trajectory)` | `unwrap(frame)` of each frame, by its `image` column; analyses of trajectories follow the atoms themselves (`unwrap=True`) |
| `system.delete_overlap(rc)` | `system.filter(~find_overlapping(system, rc))`, from `mdapy.neighbor` |

See [Systems](../basics/system.md), [Changing a system](../basics/changing.md) and
[Reading and writing files](../basics/io.md).

## Neighbors

| MDAPY 1.x | MDAPY 2.0 |
|---|---|
| `system.build_neighbor(rc, max_neigh)` | `nl = mp.NeighborList.build(system, rc)`, then `compute(system, neighbors=nl)`; [no `max_neigh`](#what-20-does-otherwise) |
| `system.build_nearest_neighbor(k)` | `mp.NeighborList.build(system, num_neighbors=k)` |
| `system.build_voronoi_neighbor()` | `mp.NeighborList.build(system, voronoi=True)` |
| `system.create_bonds(rc)` | `create_bonds(system, rc)`, from `mdapy.transform`: the bonds in the topology |

See [Neighbor lists](../basics/neighbor.md).

## Structure analyses

From `mdapy.identify`; `system.with_columns(result)` adds the columns.

| MDAPY 1.x | MDAPY 2.0 | Column |
|---|---|---|
| `cal_common_neighbor_analysis(rc=None)` | `CommonNeighborAnalysis(cutoff=None)` | `cna_structure` |
| `cal_polyhedral_template_matching(structure)` | `PolyhedralTemplateMatching(structures=("fcc", "hcp", "bcc"))` | `ptm_structure`, `ptm_rmsd`, ... |
| `cal_ackland_jones_analysis()` | `AcklandJonesAnalysis()` | `ackland_jones_structure` |
| `cal_identify_diamond_structure()` | `DiamondStructure()` | `diamond_structure` |
| `cal_centro_symmetry_parameter(N)` | `CentroSymmetryParameter(N)` | `csp` |
| `cal_common_neighbor_parameter(rc)` | `CommonNeighborParameter(rc)` | `cnp` |
| `cal_chill_plus(cutoff)` | `ChillPlus(cutoff)` | `chill_plus_structure` |
| `identify_fcc_planar_faults=True` | `PlanarFaults()` | `planar_fault` |

The structures keep the numbers of 1.x, which are those of OVITO: 0 other, 1 fcc, 2 hcp,
3 bcc, 4 ico for the common neighbor analysis, Ackland-Jones and the first five of
polyhedral template matching. `result.names` gives the names, `result.counts()` the
number of atoms of each.

## Other analyses

From `mdapy.analysis`.

| MDAPY 1.x | MDAPY 2.0 |
|---|---|
| `cal_steinhardt_bond_orientation(llist, nnn=12)` | `SteinhardtParameters(degrees, num_neighbors=12)`; `cutoff=`, `voronoi=True`, `solid=True` for solid-like atoms |
| `cal_structure_entropy(rc, sigma)` | `EntropyFingerprint(cutoff, sigma)` |
| `cal_cluster_analysis(rc)` | `ClusterAnalysis(cutoff)`, or `cutoffs=` for each pair of elements |
| `cal_voronoi_volume()` | `VoronoiAnalysis()` |
| `cal_atomic_temperature(rc)` | `AtomicTemperature(cutoff)` |
| `cal_chemical_species()` | `ChemicalSpecies()` |
| `cal_radial_distribution_function(rc, nbin)` | `RadialDistributionFunction(cutoff, bins)`: `result.r`, `result.g`, `result.pair(a, b)` |
| `cal_structure_factor(k_min, k_max, nbins)` | `StructureFactor(q_max, bins)`: `result.q`, `result.s` |
| `cal_bond_analysis(rc, nbin)` | `BondLengthDistribution()` of the bonds of the topology, `BondAngleDistribution(cutoff, bins)` |
| `cal_angular_distribution_function(rc_dict, nbin)` | `BondAngleDistribution(cutoffs=..., bins=...)`, `result.triplet(a, b, c)` |
| `cal_warren_cowley_parameter(rc)` | `WarrenCowley(cutoff)` |
| `WignerSeitzAnalysis(ref).compute(current)` | `WignerSeitz(reference).compute(system)` |
| `AtomicStrain(rc, ref).compute(current)` | `AtomicStrain(reference, cutoff).compute(system)` |
| `MeanSquaredDisplacement(pos_list).compute()` | `MeanSquaredDisplacement().compute(frames)` |
| `LindemannParameter(pos_list).compute()` | `LindemannIndex().compute(frames)` |
| `VoidAnalysis(system, rc).compute()` | `SurfaceMesh(radius, regions=True).compute(system)` |
| `SpatialBinning(data, box, direction, bin_width).compute(name, operation)` | `SpatialBinning(column, bins, axes=..., reduction=...)` |

See [the analyses](../user-guide/analysis/overview.md).

## Building structures

From `mdapy.build`.

| MDAPY 1.x | MDAPY 2.0 |
|---|---|
| `mp.build_crystal("Cu", "fcc", 3.615, nx=3, ny=3, nz=3)` | `build_crystal("fcc", 3.615, elements="Cu", repeat=(3, 3, 3))` |
| `miller1=..., miller2=..., miller3=...` | `orientation=(x, y, z)` |
| `mp.build_hea(element_list, element_ratio, ...)` | `random_alloy(build_crystal(...), {"Cu": 0.5, "Ni": 0.5}, seed=1)` |
| `mp.SQS(system, cutoffs).compute()` | `sqs_alloy(system, fractions)` |
| `mp.CreatePolycrystal(unit, box, seed_number).compute()` | `build_polycrystal(unit, mp.Box(box), grains, seed=1)`; `add_graphene` for graphene at the boundaries |
| `build_partial_dislocation_fcc(...)` | [`insert_dislocation`](../user-guide/build/dislocations.md), `dislocation_cylinder` and the other dislocation builders |

## Potentials and simulations

| MDAPY 1.x | MDAPY 2.0 |
|---|---|
| `mp.NEP("nep.txt")`, `mp.EAM("Cu.eam.alloy")` | `NEP("nep.txt")`, `EAM("Cu.eam.alloy")`, from `mdapy.calculators` |
| `mp.EAMGenerator(elements)` | `EAM.zhou(elements)`, `.write(path)` |
| `mp.EAMAverage(filename, concentration)` | `EAM(filename).average(fractions)` |
| `LammpsPotential(pair_parameter, element_list)` | `LAMMPS(pair, elements)` |
| `NEP4ASE("nep.txt")` | `to_ase_calculator(NEP("nep.txt"))`, from `mdapy.interop` |
| `system.calc = calc; system.get_energy()`, `get_force()`, `get_virials()` | `calc.compute(system, ("energy", "forces", "virial"))`: `result.energy`, `result.forces`, `result.virial` |
| `mp.FIRE(system).run(steps, fmax)` | `minimize(system, calc, max_force=fmax, max_steps=steps)`, from `mdapy.simulation`; `cell="full"` relaxes the box too |
| `LammpsRunner(...).run_md(...)` | `with MolecularDynamics(system, potential) as md: md.run(steps, ensemble="nvt", temperature=300)` |
| `get_sfe_fcc(name, a, calc)` | `stacking_fault_energy(crystal, calc, direction)` |
| `mp.get_elastic_constant(system, calc)` | `elastic_constants(system, calc)`, from `mdapy.lattice_dynamics`: `result.constants` |
| `MDElastic(...)`, `QHAElastic(...)` | `md_elastic_constants(...)`, `quasi_harmonic(...)` and `thermal_elastic_constants(...)` |
| `Phonon(path, unitcell=system, repeat=...)` | `phonons(system, calc, supercell=...)`: `.band_structure(path)`, `.dos()`, `.thermal(temperatures)` |
| `potential_tool.PCA`, `fps_sample` | `principal_components`, `farthest_point_sampling`, from `mdapy.mlip` |
| `outcar2xyz(...)` | `write("train.xyz", read_trajectory("OUTCAR", "outcar"))` |

See [Calculators](../user-guide/simulation/calculators.md),
[Minimization](../user-guide/simulation/minimize.md) and
[Molecular dynamics](../user-guide/simulation/dynamics.md).

## Pictures and figures

| MDAPY 1.x | MDAPY 2.0 |
|---|---|
| `TachyonRender(...).render_system(system, output_figure="a.png")` | `render(system, "a.png")`, from `mdapy.visualization`; `device="gpu"` on the graphics card |
| `colors=..., radii=...` | `color=` (a column, colors or a structure result), `radius=` |
| `View(system)` in a notebook | `show_image(render(system))` |
| `mp.set_figure()`, `mp.save_figure(fig, name)` | `with plot.style(): fig, ax = plot.figure()`, `plot.save(fig, name)` |
| `rdf.plot()` and the other `plot()` methods | `plot.rdf(rdf)`, `plot.structure_factor`, `plot.msd`, `plot.warren_cowley`, `plot.band_structure` and others |

See [Rendering](../user-guide/visualization/rendering.md) and
[Plots](../user-guide/visualization/plots.md).

## What 2.0 does otherwise

**No `max_neigh`.** A neighbor list holds any number of neighbors for each atom, as one
list of pairs. Nothing has to be guessed before the search, no atom loses neighbors to a
limit set too low, a dense region and a gas share one list, and the memory follows the
pairs there are. The same holds for every analysis, which takes no such limit either.

**`split_xyz`**: a trajectory is read frame by frame, and each frame written to a file of
its own:

```python
from mdapy.io import read_trajectory

write("frames.xyz", [system, system])  # two frames in one file
with read_trajectory("frames.xyz") as trajectory:
    for k, frame in enumerate(trajectory):
        write(f"frame_{k}.xyz", frame)
```

**`average_by_neighbor`**: a neighbor list gives it in a few lines of NumPy, here the
mean of `csp` over each atom and its neighbors:

```python
import numpy as np
import mdapy as mp

nl = mp.NeighborList.build(system, 3.0)
csp = system["csp"]
total = csp + np.bincount(nl.i, weights=csp[nl.j], minlength=len(system))
system = system.with_columns(csp_average=total / (nl.counts + 1))
```

**The notebook viewer `View`** is left out to keep MDAPY lean: `render` makes the
picture and `show_image(render(system))` shows it in a notebook. A graphical
application with fuller features is developed separately.

**Coming in a later 2.x release**: `orthogonal_cell`, `set_pka`, `Spline` and
`BondStiffness`. They will be additions, so code written for 2.0 keeps working (see
[Compatibility](../about/compatibility.md)).

**Threads**: `MDAPY_NUM_THREADS` sets the number of threads as in 1.x, and
`mp.set_num_threads(n)` changes it while running (see
[Parallelism](../basics/parallelism.md)).
