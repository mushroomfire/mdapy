# Release notes

## 2.0.0 (in development)

- `plot.style(*, font_size=None, line_width=None, colors=None, settings=None)` changes
  the size of the text, the width of the lines and the colors they take in turn without
  matplotlib's names, and takes any other setting of matplotlib by its name;
  [Figures with MDAPY](../examples/plotting.md) shows how.
- A fix of one's own, added with `MolecularDynamics.command`, works in `run` as in one
  run of LAMMPS: the setup of `run` now knows where the run ends, and a `fix deform`
  stretched nothing before, as fixes that plan the whole run at their setup planned
  one of 0 steps. [Molecular dynamics through LAMMPS](../examples/molecular-dynamics.md)
  stretches copper in this way.
- The documentation has versions: `latest`, the newest release, which readers see by
  default, each release by its minor version, and `dev`, the code of master. A menu at
  the top switches between them; links without a version, as before, go to `latest`.
- Extended XYZ files with both `force` and `forces`, as when one program adds its own
  to a file of another, are read: a property named as its column, here `forces`,
  gives it, and `force` keeps its name; they gave an error before. The same holds
  for `charge` beside `initial_charges`.
- Pictures of large systems seen from far away have no rings: the intersections of the
  rays with atoms and bonds keep their digits however far the camera is, and the rays of
  shadows and occlusion leave a surface by more the farther it is from the camera. The
  rings showed on the GPU above some hundred Å from the eye, more faintly on the
  processor.

## 2.0.0rc1

The release candidate of MDAPY 2.0: its API is that of 2.0.0, and a test keeps it.
Until 2.0.0 only fixes follow.

- [Compatibility](compatibility.md) says what 2.x promises from 2.0.0 on: code that works
  keeps working, the API changes only with 3.0, a replaced name warns before it goes, and
  a result changes only to fix a bug, named here.
- Built packages for Linux on ARM (aarch64), tested like the others.
- Python 3.15 is tested and supported.
- The documentation is published from the tag of each release, so that it describes the
  version `pip` installs.
- `CITATION.cff` gives GitHub's "Cite this repository" the paper to cite.
- [Papers citing MDAPY](citations.md), by year.
- [From MDAPY 1.x](../getting-started/from-1x.md) gives the 2.0 way of each common 1.x
  call.
- The API of 2.0.0 is fixed, and a test keeps it: a name that goes away or a parameter
  that changes fails it, and so does an addition these notes do not name. Before that, one
  name for each concept and options given by name only. From 2.0.0b1:
    - `cutoff` for every cutoff radius: `RadialDistributionFunction(cutoff)` and
      `StructureFactor(..., cutoff=)` (were `r_max`), `SOAP(species, cutoff)` (was
      `r_cut`); `SOAP.reach` is the reach of its neighbors (was `SOAP.cutoff`).
    - `BondLengthDistribution(cutoff=None, bins=100)` in the order of
      `BondAngleDistribution`; `BondLengthResult.bond_lengths` (was `lengths`).
    - `sizes` of `GrainResult` and `CrystalClusters` (were `size`), as of the other
      results; `TimeAverageResult.num_frames` and `thermal_phonons(..., num_frames=)`
      (were `frames`); `MolecularDynamics.run(..., save_every=)` (was `frames`), as
      `minimize`.
    - `atom_colors(..., colormap=)` (was `colormap_name`), `atom_radii(system, kind=)`
      (was `radius`), `GrainResult.atom_colors(*, ungrouped=)` (was `none`).
    - `melting_point(..., num_atoms=)` and `check_calculator(..., num_atoms=)` (were
      `atoms`), `principal_components(points, count=)` (was `n`),
      `read_gpumd(directory)` (was `folder`), `EAM.average(fractions)` (was
      `concentrations`), as `random_alloy`; `SpatialBinning(column, bins)` (was
      `property`), as `Histogram`.
    - Given by name only: `scale_positions` of `System.with_box`, `seed` of
      `random_rotations`, `partial` and `core` of `Dislocation`, the options of
      `color_coding` and of `SOAP`, `atoms` of `CoordinationPolyhedra.compute` and of
      `trajectory_lines`, `by` of `Predictions.worst`, `tolerance` and `candidates` of
      `element_from_mass`, `name` of `freeze`, `mesh` of `Phonons.thermal`, `formats` of
      `plot.save`.
    - One path for each name: `mdapy.NeighborList` (no longer in `mdapy.neighbor`);
      `mdapy.core` holds `LammpsBox` and `Result` only, the others are at the top level.
- Plain XYZ files read: a comment line without `=` is free text, kept as
  `info["comment"]`, and each atom line then holds an element symbol and x, y, z.
- Every NaN is written as `nan` in text files; some systems wrote `-nan(ind)` or
  `nan(snan)`, which no reader takes back.
- An atom count too large for 64 bits in a damaged header gives a FileFormatError.
- Damaged files of every format are tested to give an error of MDAPY, and the text
  kernels are fuzzed in CI.

## 2.0.0b1

The beta of MDAPY 2.0: every feature planned for 2.0 is in, and until 2.0.0 only
documentation and bug fixes follow.

- `render` puts its legend in a column of its own at the left of the picture and frames
  the scene beside it, so that the legend never covers the atoms or the box. See
  [Rendering](../user-guide/visualization/rendering.md).
- `render(line_color=...)` takes a color for each line, such as one for each atom whose
  path `trajectory_lines` gives.
- The ninth of `LABEL_COLORS` is brown: it was the orange of the first.
- Tests run with the lowest NumPy and polars that MDAPY accepts, and every week with their
  newest releases, the nightly wheels of NumPy and the pre-releases of polars.
- `render` and `Scene.render` no longer take `aperture` and `focal_distance`: the depth of
  field they gave only blurred a picture, and was easily set wrong.

Documentation:

- The README shows the examples as a gallery of pictures, each linked to its page.
- [Polyhedral template matching](../user-guide/analysis/ptm.md) shows every structure and
  every chemical ordering it knows, the RMSD and orientations;
  [Voronoi analysis](../user-guide/analysis/voronoi.md) draws cells as voro++ does;
  [Space group](../user-guide/analysis/space-group.md) explains the point group with its
  axes and stereograms; [Slip systems](../user-guide/build/slip-systems.md) has the
  pyramidal systems of hcp; [Frozen values](../user-guide/analysis/freeze.md) follows the
  atoms of a crystal into its melt.
- The pictures of stacking faults at finite temperature, of phonons at finite temperature
  and of phase diagrams come with the code that draws them, and every picture with a
  legend has it beside the scene.
- [The CrCoNi example](../examples/crconi.md) compares the elastic constants of molecular
  dynamics, run with `pair_style nep/kk` on a GPU, with the quasi-harmonic ones and the
  moduli of both with measurements.
- [Special quasirandom structures](../user-guide/build/sqs.md) plot their Warren–Cowley
  parameters shell by shell against those of a random alloy.
- The [performance](../performance/comparison.md) pages show for each case how many times
  faster MDAPY is at every size, the largest gains on top. The analyses are measured again
  with 2.0.0b1, Steinhardt parameters over Voronoi faces now at 2 million atoms too; the
  dislocation analysis, still experimental, is left out.
- [Reporting a problem](reporting.md) says which form of an issue to choose.

## 2.0.0a2

- `find_overlapping` and `build_polycrystal` delete as few atoms as they can: the atoms that
  overlap the fewest are kept first. `find_overlapping` used to delete the later atom of every
  pair, also when the earlier one was deleted too. See
  [Overlapping atoms](../basics/overlapping.md).
- `build_polycrystal` keeps molecules whole when the unit has a `molecule` column: a molecule
  belongs to the grain of its center, those too close go whole, and the topology is kept. It
  also keeps the other columns of the unit, such as charges. `find_overlapping` has an option
  `molecules`. See [A molecular polycrystal](../examples/molecular-polycrystal.md).
- `plot.warren_cowley` draws the Warren–Cowley parameters as a matrix with its values written
  in, as MDAPY 1.x did. See [Warren–Cowley parameters](../user-guide/analysis/warren-cowley.md).
- `build_polycrystal` places only the atoms of each grain, on all threads, and gives the same
  atoms in the same order as before: 2 million atoms of copper in 20 grains, with `min_distance`
  0.01 Å, take 0.017 s instead of 1.9 s on 10 threads, and the process peaks at 182 MB instead
  of 1043 MB. See [Building structures](../performance/builders.md).
- `build_crystal` runs on all threads and gives the same atoms in the same order: nearly 10
  million atoms of copper take 0.017 s instead of 0.25 s on 10 threads, and the build takes
  300 MB instead of 750 MB.
- `Box.wrap` runs on all threads.
- Gzipped files are written on all threads, as one stream of blocks that every program reads,
  no larger than before: a LAMMPS dump of 2 million atoms takes 0.7 s instead of 4.4 s on 10
  threads. Use `.gz` to compress: the other formats compress on one thread. See
  [Compressed files](../basics/io.md#compressed-files).
- LAMMPS data files are read and written about as fast as dumps: their atoms are no longer
  sorted to check their IDs, and what depends on the type is found once for each type. On 40
  threads, 10 million atoms take 0.15 s instead of 0.62 s to read and 0.35 s instead of 0.52 s
  to write.
- Text files are written and read faster, with less memory and the same text and values: on
  10 threads, a LAMMPS dump of 10 million atoms and 5 columns takes 0.11 s instead of 0.30 s to
  write and 0.12 s instead of 0.37 s to read. A file on disk takes the text straight from the
  threads that format it, while they format the next part, and the lines of a file are found on
  all threads.

Documentation:

- Examples of [a gradient polycrystal](../examples/gradient-polycrystals.md) and of
  [a polycrystal of graphene](../examples/graphene-polycrystal.md).
- Issue forms for a bug, a feature request, performance, the documentation and a question, a
  template for pull requests, and the rules for them in
  [Contributing](contributing.md): an issue first, and a contribution is licensed under the
  Mozilla Public License 2.0.
- [Installation](../getting-started/installation.md#an-old-linux) says what an old Linux needs,
  and `show_versions` prints the glibc in use on Linux.

Continuous integration:

- A change of the documentation alone runs only ruff and the examples of the documentation,
  and does not stop a run of the tests of the code. A new workflow builds the documentation
  and publishes it to GitHub Pages.
- The coverage runs in parallel, and its report goes to Codecov.
- Direct3D 12 and Vulkan run their tests of pictures on drivers on the processor, asked for
  with `MDAPY_SOFTWARE_GPU`, as the runners of Windows and Linux have no GPU.
- The benchmarks predict at least the memory of the positions for more atoms.

## 2.0.0a1

MDAPY 2.0 is a complete rewrite and is not compatible with MDAPY 1.x. The name
is written MDAPY, with `mdapy` for the package, the imports and the commands,
and it has a new logo.

The top level of `mdapy` holds the foundations: `System`, `Box`, `Topology`,
`NeighborList`, the threads, the versions and the modules. Everything else comes from its
module, as `mdapy.io.read` or `from mdapy.build import build_crystal`.

Columns of results are named by their method and options, as `cna_structure`, `ptm_rmsd`
or `q6_average_voronoi`, so that several analyses stay side by side;
`with_columns(near=result)` gives a prefix, and two values for a column in one call are an
error.

### Structures

- `mdapy.System`: one atomic configuration with a polars table of per-atom
  properties, a box, global information and an optional topology. Systems
  are immutable: `with_*` methods return new systems that share the columns
  they do not change and check only the ones they do; arrays given to a
  system are copied, its table is handed out as a copy, it takes no new
  attributes, and `copy.deepcopy` returns the system itself; systems, boxes,
  topologies and triangle meshes pickle, for other processes. Every column,
  the positions too, is read by name and changed with `with_columns`; the
  positions are an (N, 3) array column read without copying,
  `system["positions"]`. A summary of a few lines
  shows the atoms, the box, the columns, the global information and the
  topology. `select`, `drop` and `rename` choose, remove and rename
  columns as polars does, and `with_info({})` clears the global
  information. See [Systems](../basics/system.md) and
  [Changing a system](../basics/changing.md).
- `System.filter`, `System.gather` and `System.sort` keep atoms by a
  condition, take them by index and order them, as polars does with rows.
  Conditions are Python expressions over the columns, such as
  `"positions.x**2 + positions.y**2 < 25 and element == 'Cu'"`, with the
  components of array columns by name or index, comparisons, logic,
  membership, arithmetic and common functions; keywords, masks and polars
  expressions work too. `System.where` gives the atoms of the same
  conditions as a boolean mask. `system["positions.z"]` gives the values of a
  column, a component or such an expression as a read-only NumPy array,
  without a copy for columns and components. See
  [Selecting and ordering atoms](../basics/selecting.md).
- `mdapy.Topology`: bonds, angles, dihedrals and impropers with types and
  labels, kept consistent when atoms are selected or reordered. See
  [Topology](../basics/topology.md).
- `mdapy.Box`: immutable simulation box with lattice vectors, origin and
  periodicity; fractional coordinates, wrapping with image flags, exact
  minimum images, and conversion to LAMMPS restricted triclinic form. See
  [Simulation box](../basics/box.md).

### Reading and writing

- `mdapy.io.read` and `mdapy.io.write` read one frame and write one or more
  frames, finding the format from the file name. Files compressed with gzip,
  bzip2, xz or zstd (Python 3.14) are read and written transparently.
  `mdapy.io.formats` and `mdapy.io.detect_format` list and identify formats. See
  [Reading and writing files](../basics/io.md).
- Extended XYZ with the conventions of ASE and GPUMD: velocities from `vel`
  (Å/fs) or ASE `momenta`, stress in GPa, any per-atom column and any global
  value, written so that they read back unchanged. See
  [Extended XYZ](../reference/formats/extxyz.md).
- LAMMPS data files of the atom styles atomic, charge, bond, angle, molecular
  and full, following LAMMPS stable_30Sep2026: type labels, masses with
  element guessing, bonds to impropers as the topology, force-field
  coefficients kept unchanged, image flags, velocities, metal and real units,
  and restricted and general triclinic boxes. See
  [LAMMPS data](../reference/formats/lammps-data.md).
- LAMMPS text dumps of the atom and custom styles, with any number of frames:
  units and time items, orthogonal, triclinic and general triclinic boxes,
  scaled and unwrapped coordinates, elements and type labels, and every other
  column, array columns such as `c_stress[1]` to `c_stress[6]` kept together.
  See [LAMMPS dump](../reference/formats/lammps-dump.md).
- VASP POSCAR and CONTCAR: scaling factors, POTCAR labels, selective
  dynamics as the `movable` column, Direct and Cartesian coordinates, and the
  velocity block of a CONTCAR. See [VASP POSCAR](../reference/formats/poscar.md).
- The MDAPY file format `.mdapy`: a ZIP archive of Parquet tables that keeps
  every column with its type, the box, the information and the topology of
  each frame, a table of frames with typed columns of the information, and
  recovery of files whose writing stopped early. See
  [MDAPY files](../reference/formats/mdapy.md).
- `mdapy.io.read_trajectory` opens the frames of any file lazily: frames are
  found and read when used, slices share the open file, and iteration starts
  at once; also the frames of several files, given as a pattern such as
  `dump.*.lammpstrj` or a list, each in the format of its name, and frames that
  move the atoms of a
  topology read from a data file, matched by id, with its bonds, charges
  and molecules, as OVITO's load trajectory; `Trajectory.map` gives the
  frames through a function, each computed when it is used, for the
  analyses of trajectories. See [Trajectories](../basics/trajectory.md). `Trajectory.file_of` gives the file that holds a frame.
- Reading CIF files, crystals expanded by their symmetry operations as
  pymatgen, ASE and OVITO read them, and PDB files, with atom names,
  residues, models as frames and the bonds of `CONECT`; VASP OUTCAR
  files, one frame per ionic step with the forces, the free energy TOTEN,
  the virial and whether the step converged, as ASE reads them. See
  [CIF](../reference/formats/cif.md), [PDB](../reference/formats/pdb.md) and
  [VASP OUTCAR](../reference/formats/outcar.md).
- Damaged files give a `FileFormatError` in every format: bytes that are not
  text, a `.mdapy` file cut or changed anywhere, a file without atoms that read
  as an empty system before.

### Neighbor search

- `mdapy.NeighborList.build(system, cutoff)` and
  `NeighborList.build(system, num_neighbors=k)`: every pair within a cutoff,
  or the k nearest neighbors of each atom, in any box, with periodic images
  enumerated so that small boxes work, in CSR form with int8 image shifts
  (15 bytes per pair); `within` and `sorted` derive lists without a new
  search. Distances are computed by one formula in both, nearest neighbors are
  ordered exactly with ties by atom and image, and results are the same for
  any number of threads. A k-d tree finds nearest neighbors in clusters and
  slabs as fast as in crystals. A list serves every system with the same
  positions in the same box (`fits`), so it outlives `with_columns`. A system
  whose atoms fill a tiny part of the space they span, which would take hours,
  is refused with an error.
  See [Neighbor lists](../basics/neighbor.md).
- `mdapy.neighbor.expand_selection`: a selection grown by the atoms within a
  cutoff, the nearest neighbors or the bonded atoms of the selected ones,
  step by step, as OVITO's expand selection. See
  [Expanding a selection](../basics/expand-selection.md).
- `mdapy.neighbor.find_overlapping`: the atoms closer than a distance to
  another, the later atom of each such pair so that deleting them leaves
  none that close, or all of them, as OVITO's find overlapping particles.
  See [Overlapping atoms](../basics/overlapping.md).
- Analyses given a list read it where it is, without a copy: those of the
  k nearest neighbors put the k nearest pairs of each atom first in the
  list itself, once, in parallel; those within a cutoff read the pairs
  closer than theirs and stop at the first sorted pair beyond it. One list
  within 5 Å serves every structure analysis of a metal; threads may share
  a list. A list keeps 15 bytes per pair: `i` and `vectors` are computed
  when asked for. See
  [Sharing neighbor lists](../examples/neighbor-lists.md).
- Atoms far apart, up to 1e9 box lengths: a search along a direction without
  periods puts its cells over the body of the atoms, so that an atom that flew
  off leaves the others as fast as before (0.03 s, not 12 s, for 200 000 atoms),
  and the number of cells no longer overflows, which could crash the process.
- A list of Voronoi neighbors finds its distances in parallel and, in a
  periodic box, keeps the arrays of the cells without a copy: Steinhardt
  parameters over Voronoi faces take a quarter less time and 40 % less memory.

### Calculators

- `mdapy.calculators.Calculator`, the interface of every calculator, and
  `CalculatorResult` with energies, forces, virials and stress.
  `mdapy.calculators.testing.check_calculator` tests any implementation. See
  [Calculators](../user-guide/simulation/calculators.md).
- `mdapy.calculators.EAM` for potentials in the LAMMPS eam/alloy format,
  agreeing with LAMMPS.
- `mdapy.calculators.NEP` for neuroevolution potentials (nep4 and nep4_zbl
  files of GPUMD) with every form of ZBL and per-element cutoffs, agreeing
  with NEP_CPU in double precision; `NEP.descriptors` gives the descriptors.
- `EAM.write` writes potentials in the LAMMPS eam/alloy format;
  `EAM.zhou` builds the potentials of Zhou, Johnson and Wadley for 17
  elements, as the LAMMPS database tool tabulates them; `EAM.average` gives
  the average-atom potential of a random alloy.
- `mdapy.calculators.LAMMPS` makes any interaction of LAMMPS a calculator,
  through the LAMMPS Python module.
- `mdapy.calculators.VASP` runs VASP for each calculation with a command,
  as in the job script of a cluster: it writes the inputs, keeps the wave
  functions between calls and reads the OUTCAR. `CalculationFailed` reports
  a run that stopped. See [VASP](../user-guide/simulation/vasp.md).

### Simulation

- `mdapy.simulation.minimize`: FIRE 2.0 minimization of the positions, step
  for step as LAMMPS `min_style fire`, and optionally of the box
  (`cell="isotropic"`, `"axes"` or `"full"`, at a `pressure`) to the minimum
  of LAMMPS `fix box/relax`; keeps a frame every `save_every` steps when
  asked, with the virial and the stress of each. Its steps use no BLAS, whose
  threads took the cores from the kernels under OpenBLAS (0.5 s against 3.5 s
  for 4000 atoms on 16 threads). See
  [Energy minimization](../user-guide/simulation/minimize.md).
- `mdapy.simulation.free_energy`: the free energy of solids and liquids per atom from
  molecular dynamics in LAMMPS, by switching to springs or to the Uhlenbeck-Ford model
  and following the enthalpy to other temperatures; `crossing` gives where two phases
  meet. See [Free energy](../user-guide/simulation/free-energy.md).
- `mdapy.simulation.melting_point`: the melting point from a first guess, by the
  two-phase method. `MolecularDynamics.run` takes the ensemble `"nph"`. See
  [Melting point](../user-guide/simulation/melting-point.md).
- `mdapy.simulation.binary_phase_diagram` and `plot.phase_diagram`: the phase diagram
  of two elements from the free energies of their phases. See
  [Phase diagrams](../user-guide/simulation/phase-diagram.md).
- `mdapy.simulation.pressure_temperature_diagram` and `plot.pressure_temperature`: the
  phase diagram of one element against pressure and temperature, from the free energies of
  its phases at each pressure. See
  [Pressure-temperature diagrams](../user-guide/simulation/pressure-temperature.md).
- `mdapy.simulation.stacking_fault_free_energy`: the free energy of a stacking fault at
  each temperature, from two crystals in one box. See
  [Stacking faults at finite temperature](../user-guide/simulation/stacking-fault-temperature.md).
- `slip`: an atom outside the box moves as its image in the box does; a layer of a
  relaxed crystal at the bottom of the box was split by rounding, and gave a stacking
  fault energy far too large.
- `mdapy.lattice_dynamics.elastic_constants`: the (6, 6) elastic constants
  from the stress under small strains, with relaxed positions, and Voigt,
  Reuss and Hill moduli. See [Elastic constants](../user-guide/simulation/elastic-constants.md).
- `mdapy.lattice_dynamics.phonons`: harmonic phonons from the forces on
  displaced atoms with any calculator, the force constants made symmetric
  and free of translations; frequencies and polarization vectors at any
  wave vector, band structures along standard or given paths, densities of
  states by the tetrahedron method or Gaussians, projected on the atoms,
  and the free energy, energy, entropy and heat capacity of the vibrations;
  the same dynamical matrix as LAMMPS. `primitive_cell` finds the smallest
  cell of a crystal; supercells as integer matrices, by default the one
  nearest a cube; standard paths of any crystal with seekpath, and
  branches followed through crossings; imaginary frequencies in the density
  of states; band structures written and read as phonopy's band.dat;
  `plot.band_structure` and `plot.phonon_dos`. `units.PLANCK`.
  See [Phonons](../user-guide/simulation/phonons.md).
- `mdapy.lattice_dynamics.quasi_harmonic`: the quasi-harmonic
  approximation from phonons at several volumes: the equilibrium volume,
  Gibbs energy, entropy, bulk modulus, thermal expansion and heat capacity
  at constant pressure against temperature, at any pressure, the shape of
  the box relaxed at each volume. See
  [Quasi-harmonic approximation](../user-guide/simulation/quasi-harmonic.md).
- `mdapy.lattice_dynamics.thermal_elastic_constants`: isothermal and
  adiabatic elastic constants at finite temperature, quasi-harmonic or
  quasi-static. See [Elastic constants at finite temperature](../user-guide/simulation/thermal-elastic.md).
- Sheets: a box periodic along two vectors in the x-y plane makes a system a
  two-dimensional material. `elastic_constants` gives its constants in N/m as a
  `SheetElasticResult`, with Young's, shear and area moduli and Poisson's
  ratios; `phonons` takes supercells and meshes in the plane and band paths
  through it; `primitive_cell` takes it too; `MolecularDynamics.run` controls the
  pressure in the plane with `barostat="xy"`, and runs a flat sheet, which LAMMPS
  refused when its height shrank to nothing. See [Sheets](../user-guide/simulation/sheets.md).
- `mdapy.lattice_dynamics.thermal_phonons` and `phonons_from_frames`: phonons at a
  temperature, from the forces on atoms in molecular dynamics, by the force
  constants that best give them. See
  [Phonons at finite temperature](../user-guide/simulation/thermal-phonons.md).
- `mdapy.lattice_dynamics.md_elastic_constants`: isothermal elastic
  constants at a temperature from the mean stress of molecular dynamics in
  LAMMPS under small strains, with standard errors from block averages. See
  [Elastic constants from molecular dynamics](../user-guide/simulation/md-elastic.md).
- Symmetry with spglib (optional, the extra `symmetry`): `phonons`,
  `quasi_harmonic`, `elastic_constants` and `thermal_elastic_constants`
  skip the displacements and strains that the symmetry of the crystal
  relates to others, by default whenever spglib is installed: one
  computation of forces instead of 6 for the primitive cell of fcc, 12 of
  the 73 strains of the elastic constants of a cubic crystal at finite
  temperature. The elastic constants take the rotations alone, in
  milliseconds also for cells of hundreds of atoms. See
  [Space group](../user-guide/analysis/space-group.md).
- `mdapy.simulation.MolecularDynamics`: molecular dynamics in LAMMPS with
  the interaction of a LAMMPS calculator: random velocities, runs at
  constant energy, temperature or pressure with ramps, the thermodynamic
  state as a polars table, frames and the final system back in the
  orientation of the box, any LAMMPS command in between; the same
  trajectory as the same LAMMPS input. See
  [Molecular dynamics](../user-guide/simulation/dynamics.md).

- `mdapy.simulation.vacancy_formation_energy`, `surface_energy` and
  `stacking_fault_energy` (the generalized stacking fault energy, relaxed
  along the normal), agreeing with LAMMPS; `minimize` keeps atoms still
  along the directions where a `movable` column is False. See
  [Defect energies](../user-guide/simulation/defect-energies.md).

### Analyses

- Analyses are classes whose `compute(system)` returns an immutable result;
  `System.with_columns(result)` adds its per-atom values. See
  [Analyses](../user-guide/analysis/overview.md).
- Structure identification in `mdapy.identify`:
  [`CommonNeighborAnalysis`](../user-guide/analysis/cna.md), adaptive or with a fixed
  cutoff, and [`AcklandJonesAnalysis`](../user-guide/analysis/ackland-jones.md), agreeing
  atom by atom with LAMMPS, for fcc, hcp, bcc and icosahedral atoms;
  [`DiamondStructure`](../user-guide/analysis/diamond.md) for cubic and hexagonal
  diamond and their first and second neighbors, as OVITO;
  [`ChillPlus`](../user-guide/analysis/chill-plus.md) for ice, hydrates and
  liquid water; and the [centro-symmetry](../user-guide/analysis/csp.md) and
  [common neighbor](../user-guide/analysis/cnp.md) parameters.
- [`ChemicalSpecies`](../user-guide/analysis/species.md): molecules and their Hill
  formulas by the bond rule of Jmol (covalent radii, the default) or of VMD
  and OVITO (van der Waals radii), radii given by hand or bond lengths per
  pair of elements, and the number of each along a trajectory; the same
  molecules as ASE and as OVITO. `mdapy.elements` has the covalent
  and van der Waals radii, which pictures now read from it.
- [`SpaceGroup`](../user-guide/analysis/space-group.md): the space group of a
  crystal, its symmetry operations and equivalent atoms, with spglib, for
  cells of up to 2000 atoms.
- [`PolyhedralTemplateMatching`](../user-guide/analysis/ptm.md): the structure,
  orientation, elastic deformation and chemical ordering of each atom, for
  nine structures, implemented by MDAPY itself; the same as OVITO atom by
  atom.
- [`PlanarFaults`](../user-guide/analysis/planar-faults.md): intrinsic and multilayer
  stacking faults and coherent twin boundaries of fcc crystals from the
  hcp-like atoms of polyhedral template matching, and their areas; the same
  as OVITO's Identify fcc planar faults atom by atom; optionally extrinsic
  stacking faults too.
- [`GrainSegmentation`](../user-guide/analysis/grain-segmentation.md): the grains of a
  polycrystal from the lattice orientations of polyhedral template
  matching, by node pair sampling with an automatic threshold or along a
  minimum spanning tree, with stacking faults and twin boundaries joined to
  the grains around them and orphan atoms adopted; the grain of each atom,
  and the size, structure, mean orientation, center of mass and radii of
  each grain, the same as OVITO's Grain segmentation atom by atom.
- Distributions and order parameters in `mdapy.analysis`, agreeing with
  OVITO where it has them: the
  [`RadialDistributionFunction`](../user-guide/analysis/rdf.md) and its partial
  functions; [`CoordinationNumber`](../user-guide/analysis/coordination.md) with
  cutoffs per pair of elements; the
  [`StructureFactor`](../user-guide/analysis/structure-factor.md), exact over the wave
  vectors of the box or by the Debye method, with Faber–Ziman partial
  factors; the [`SpatialCorrelation`](../user-guide/analysis/spatial-correlation.md)
  function of two properties of the atoms, by fast Fourier transforms on a
  grid and by a direct sum over neighbors, the same as OVITO's spatial
  correlation function; the [`BondAngleDistribution`](../user-guide/analysis/bond-angles.md) for
  each triplet of elements; the [`BondLengthDistribution`](../user-guide/analysis/bond-lengths.md)
  of the bonds of a topology, in total and for each pair of elements; [`RingStatistics`](../user-guide/analysis/rings.md), the
  shortest-path rings of a network of bonds up to 36 atoms, without the
  paths around the periodic box, drawn as polygons;
  [`WarrenCowley`](../user-guide/analysis/warren-cowley.md)
  short-range order, over shells of neighbors by distance or by number; [`SteinhardtParameters`](../user-guide/analysis/steinhardt.md) up
  to degree 20, optionally averaged, and with `solid=True` the solid-like
  bonds and atoms of ten Wolde, Ruiz-Montero and Frenkel, as freud's
  `SolidLiquid` (MDAPY 1.x `identify_liquid`); the
  [`EntropyFingerprint`](../user-guide/analysis/entropy.md); and the
  [`VoronoiAnalysis`](../user-guide/analysis/voronoi.md), computed by mdapy's own code,
  with the volumes and coordination of the cells and, when asked for, their
  faces and Voronoi indices (`faces=True`), which otherwise are not kept,
  the face and edge thresholds of OVITO, boxes that are not periodic, and
  the shapes of the cells as polyhedra; `NeighborList.build(system,
  voronoi=True)` for the atoms that share a face, and Steinhardt parameters
  over them, optionally weighted by the face areas.
- Dynamics in `mdapy.analysis`: the
  [`MeanSquaredDisplacement`](../user-guide/analysis/msd.md) of a trajectory, over
  every time origin or from the first frame, agreeing with LAMMPS and OVITO;
  the [`LindemannIndex`](../user-guide/analysis/lindemann.md) of each atom and of the
  frames so far; both match the atoms of the frames by their ids and
  follow them through the periodic boundaries by their image flags or by
  the nearest image, unless the positions are already unwrapped. The
  [`AtomicTemperature`](../user-guide/analysis/temperature.md) of each atom and its
  neighbors, relative to their center of mass.
  [`TimeAverage`](../user-guide/analysis/time-average.md): the mean of a column of
  the atoms, of the box or of anything computed from a frame over a
  trajectory, the atoms matched by id and their positions unwrapped, as
  OVITO's time averaging. [`trajectory_lines`](../user-guide/analysis/trajectory-lines.md):
  the paths of the atoms, unwrapped by their image flags or the nearest
  image, drawn by `render(lines=...)`.
  [`mdapy.transform.remove_drift`](../user-guide/analysis/drift.md): the frames of a
  trajectory moved to keep the center of mass of all the atoms or of some
  where it is in a reference frame, read when they are used, as OVITO's
  center of mass correction. [`TimeSeries`](../user-guide/analysis/time-series.md): a
  column reduced to one number, or anything computed, in every frame, as
  OVITO's time series; [`FrameDifference`](../user-guide/analysis/frame-difference.md),
  the change of a column of each atom from a reference frame;
  [`mdapy.transform.freeze`](../user-guide/analysis/freeze.md), the values of a column
  in a reference frame given to the atoms of another; and
  [`mdapy.transform.smooth_trajectory`](../user-guide/analysis/smooth-trajectory.md),
  the positions and other real numbers averaged over a window of frames;
  each as OVITO's modifier of the same name, the atoms matched by id.
- Statistics of a column in `mdapy.analysis`, agreeing with OVITO:
  [`Histogram`](../user-guide/analysis/histogram.md), the counts of the values of a
  column, a component or an expression in bins, of all atoms or of a mask;
  [`reduce`](../user-guide/analysis/reduce.md), one number from them: mean, sum,
  extremes, median, variance, standard deviation, the count of those not
  zero, or the atom of the lowest or highest, NaN values left out.
- Defects and deformation in `mdapy.analysis`, agreeing with OVITO:
  [`ClusterAnalysis`](../user-guide/analysis/clusters.md) for clusters of bonded atoms
  and whether they span the box, and on request their unwrapped positions,
  centers of mass and gyration tensors;
  [`AtomicDisplacement`](../user-guide/analysis/displacement.md), the displacement
  of each atom from a reference matched by id, with the minimum image and
  the affine mapping of OVITO's displacement vectors;
  [`SpatialBinning`](../user-guide/analysis/spatial-binning.md), a property in bins
  along one to three box vectors, its mean, sum, density, minimum or
  maximum, the values of OVITO's spatial binning; the
  [`isosurface`](../user-guide/analysis/isosurface.md) of a grid of such values, by
  marching tetrahedra, across the periodic boundaries, and its cells as
  colors on the faces of the box by `mdapy.visualization.voxel_faces`;
  [`WignerSeitz`](../user-guide/analysis/wigner-seitz.md) for vacancies and
  interstitials, the sites of the reference kept in a k-d tree for every
  configuration; and [`AtomicStrain`](../user-guide/analysis/atomic-strain.md), the
  von Mises and volumetric strains against a reference, its atoms matched
  by id, and on request the
  Green–Lagrange strain tensor and $D^2_\text{min}$. As in OVITO, the
  results that cost more are computed only when asked for.
- [`ElasticStrain`](../user-guide/analysis/elastic-strain.md) in `mdapy.analysis`: the
  elastic strain of each atom of an fcc, hcp, bcc or diamond crystal
  against the ideal crystal of given lattice constants, in the spatial
  frame, with the structures of the dislocation analysis; the same as
  OVITO's elastic strain calculation atom by atom.
- [`SurfaceMesh`](../user-guide/analysis/surface-mesh.md) in `mdapy.analysis`: the
  surface of the solid of atoms by the alpha-shape method of OVITO's
  Construct surface mesh, from a Delaunay tessellation of mdapy's own, with
  its area, the volume of the solid, the regions of solid and empty space
  and the voids among them, the atoms on the surface, and Taubin's
  smoothing; the same areas, volumes and atoms as OVITO. The surface is a
  `mdapy.analysis.TriangleMesh`, a mesh of triangles across periodic boundaries.
- [`DislocationAnalysis`](../user-guide/analysis/dislocations.md) in `mdapy.analysis`,
  experimental: the dislocation lines of fcc, hcp, bcc, cubic and
  hexagonal diamond crystals by OVITO's dislocation extraction algorithm,
  written anew: Burgers vectors in the lattice frame and in space, partials
  of stacking faults, junctions, loops, the defect mesh around grain
  boundaries and surfaces; the same structures atom by atom as OVITO, the
  same lines and lengths within a per cent on straight dislocations and
  loops, within several per cent in networks and at grain boundaries.

### Structure builder

- `mdapy.build.build_crystal`: perfect crystals of fifteen structures,
  cubic and hexagonal, of one element or ordered on their sublattices,
  along any crystal directions, with the shortest period along each;
  the same atoms as atomsk, ordered plane by plane without sorting them
  (2 million atoms in some 0.05 s). See [Crystals](../user-guide/build/crystals.md).
- `mdapy.build.random_alloy`: elements at random in given fractions with
  exact counts. See [Random alloys](../user-guide/build/alloys.md).
- `mdapy.build.sqs_alloy`: special quasirandom structures, the pairs of the
  nearest shells those of a random alloy. See
  [SQS](../user-guide/build/sqs.md).
- Defects in `mdapy.build`: `add_vacancies`, `add_interstitials` and
  `substitute`; `slip` for stacking faults; `add_vacuum` for surfaces;
  `insert_dislocation`, the isotropic screw and edge dislocations of
  atomsk, and `edge_dislocation` of two crystals. See
  [Point defects](../user-guide/build/defects.md) and
  [Dislocations](../user-guide/build/dislocations.md).
- Dislocations in anisotropic elasticity in `mdapy.build`:
  `insert_dislocation(..., elastic=)`, the displacements of Stroh, the same
  as atomsk with an elastic tensor; `dislocation_cylinder`, a cylinder
  around a dislocation with a fixed outer shell (`movable`), and
  `dislocation_quadrupole`, a periodic quadrupole in a box tilted by the
  plastic strain of the dipole; `DISLOCATIONS`, the common dislocations of
  bcc, fcc and diamond (glide set), or a `Dislocation` of your own; fcc and
  diamond ones split into Shockley partials around an intrinsic stacking
  fault. Burgers vectors as the dislocation analysis and OVITO find them,
  the opposite of Hirth and Lothe's (and atomsk's). See
  [Dislocations](../user-guide/build/dislocations.md).
- `mdapy.build.add_graphene`: graphene on the grain boundaries of a
  polycrystal, one or more AB-stacked layers centered on each face between
  two grains, each face once, in a random orientation, cut to the face;
  a fraction of the boundary area covered by whole faces; the atoms of the
  grains too close to the carbon removed. `build_polycrystal` keeps the
  seeds in `info["seeds"]`. See [Graphene on grain boundaries](../user-guide/build/graphene.md).
- `mdapy.build.build_polycrystal`: Voronoi grains of any periodic unit,
  columnar when the box is thin, `twin_lamellae` for twinned grains,
  `rotations_about` for the turns of columns, and `gradient_seeds` for grains
  whose size changes along one direction, from one side or both. See
  [Polycrystals](../user-guide/build/polycrystals.md), which also builds grains
  of graphene, and [Gamma surfaces](../examples/gamma-surface.md).
- `mdapy.transform.replicate` copies a system along its periodic
  directions as LAMMPS `replicate` does, with the ids, molecules and
  topology of each copy and molecules kept whole through the image flags;
  `wrap` and `unwrap` move atoms into the box or to their unwrapped
  positions. See [Periodic copies and images](../user-guide/build/transform.md).
- `mdapy.transform.slice`: the atoms on one side of any plane or in a slab
  about it, as the Slice of OVITO: a normal in Cartesian coordinates or as
  Miller indices of the box, at a distance or through a point, acting on
  all atoms or some; `slice_mask` the same as a mask. See
  [Slices](../user-guide/build/slice.md).
- `mdapy.transform.affine`: moves, rotates, shears or scales the atoms, their
  velocities, forces and displacements, and the box, all atoms or a
  selection, as OVITO's affine transformation modifier. See
  [Affine transformation](../user-guide/build/affine.md).
- `mdapy.transform.combine`: the atoms of a second system added to a
  first, in its box, as OVITO's combine datasets: the columns of either,
  ids numbered on, molecules and topology terms offset. See
  [Combining systems](../user-guide/build/combine.md).
- `mdapy.transform.create_bonds`: bonds in the topology between the atoms
  closer than a cutoff, or covalent bonds, as OVITO's create bonds; drawn by
  `render(bonds="topology")`. See [Creating bonds](../user-guide/build/bonds.md).

### MLIP

- Training sets from VASP: the ionic steps of OUTCAR files, read as
  trajectories with energies, forces, virials and convergence, written to
  extended XYZ. See [Training sets from VASP](../user-guide/mlip/vasp.md).
- `mdapy.mlip.SOAP`: the smooth overlap of atomic positions of each atom,
  the same values as DScribe's SOAP with Gaussian radial functions, in
  C++, for any structures whatever made them. See
  [SOAP descriptors](../user-guide/mlip/soap.md).
- `mdapy.mlip.structure_descriptors`, `farthest_point_sampling` and
  `principal_components`: frames chosen far from a training set and from
  each other in the space of NEP descriptors, and their projection on a
  plane. See [Choosing frames](../user-guide/mlip/selection.md).
- `mdapy.mlip.perturb`: copies of a structure with atoms displaced and the
  box strained at random. See [Perturbed structures](../user-guide/mlip/perturbation.md).
- `mdapy.mlip.predict` and `read_gpumd`: errors of any model against the
  labels of a training set, frame by frame, or read from the outputs of
  NEP training in GPUMD and of torchnep. See [Errors of a model](../user-guide/mlip/errors.md).
- `mdapy.mlip.committee` and `novelty`: the spread of the predictions of
  several models and the distance from a training set in the space of
  descriptors, to find frames worth labeling. See
  [Uncertainty and novelty](../user-guide/mlip/uncertainty.md).
- `mdapy.mlip.check_frames`: atoms too close, net forces and missing
  labels in a training set. See [Checking frames](../user-guide/mlip/checks.md).
- Training sets of many small frames: `predict`, `committee` and
  `structure_descriptors` compute NEP models and SOAP, and `check_frames`
  the closest pairs, many frames at a time in C++, a frame on a thread,
  each as it would be alone: for the 105,464 frames of UNEP16 on an Apple
  M5, the NEP descriptors in 15 s rather than 26 s, its predictions in 24 s
  rather than 35 s, and the checks in 2 s rather than 8 s. `novelty`
  screens the pairs in float32 on threads and computes the distances of
  the candidates exactly: their novelty in 5 s rather than 30 s.

### Visualization

- `mdapy.visualization.render`: pictures of systems by ray tracing, with
  ambient occlusion, shadows and outlines in four styles: fast, lit as the
  viewports of OVITO; matte, as the Tachyon renderer of OVITO (the
  default); shiny, a metallic sheen after its OSPRay renderer, with light
  bouncing between the atoms; flat, plain colors outlined as ASE draws
  atoms; colored by element,
  structure, values or by hand, with bonds, the box, Voronoi cells,
  surface meshes and dislocation lines with arrows of their Burgers
  vectors, cut at the periodic faces of the box;
  arrows of any vector of the atoms, such as displacements or forces,
  as OVITO's vector display draws them; polylines such as the paths of
  atoms; [coordination polyhedra](../user-guide/visualization/polyhedra.md), the convex
  hulls of the neighbors of atoms with their volumes, as OVITO builds them;
  molecules as balls and sticks, licorice or space filling, with covalent
  bonds found from the elements; a `Scene` of spheres, cylinders and
  polygons; the scene framed, or a given focus and span; depth of field
  and soft shadows; every setting of a style can be changed, including the
  direction of the highlights, metals and lighting in sRGB; a million atoms
  in under 5 s on the processor in the matte style, or in under a second on
  the GPU, with Metal on macOS, Vulkan on Linux or Direct3D 12 on Windows, which
  use the ray tracing units of the GPU where it has them; `show_image` shows a picture in a
  notebook; `legend=True` says what the colors mean. See
  [Rendering](../user-guide/visualization/rendering.md). A driver that runs on the processor does not count as a GPU.
- Colors of atoms in `mdapy.visualization`: `color_coding`, `color_by_type`
  and `assign_color` color all atoms or a mask and keep the colors in the
  `color` column, which `render` draws by default, as OVITO's modifiers of
  these names; `atom_colors` and `atom_radii` with colors and radii of
  elements chosen and a scale; colormaps rainbow, viridis, magma, jet, hot,
  blue_white_red and gray computed as OVITO computes them, besides plasma
  and coolwarm, and labels for discrete values. Atom types without elements, as in a LAMMPS dump of types
  only, are drawn each in a color of its own, those of OVITO (`TYPE_COLORS`),
  and all of one radius; `atom_names` gives the names these colors go by,
  and `read_trajectory(path, elements=[...])` names the types, so that files
  written afterwards hold the elements.
  See [Coloring atoms](../user-guide/visualization/colors.md).
- `mdapy.visualization.plot`: matplotlib set up for journals, in notebooks too
  (widths of columns, Nature's artwork guide, Okabe–Ito colors, panel labels, saving
  as PDF, SVG and PNG) and plots of results: parity, g(r), S(q), bond
  angles, MSD with the diffusion coefficient, stacking fault energies,
  counts of structures, and two columns of the atoms against each other
  (`scatter`). See [Plots](../user-guide/visualization/plots.md).

### Interoperability

- `mdapy.interop.to_ase` and `from_ase` convert to and from `ase.Atoms`,
  with velocities, charges, constraints, other numeric columns and the
  results of a calculation. See [ASE](../user-guide/interop/ase.md).
- `mdapy.interop.to_ase_calculator` makes any MDAPY calculator an ASE
  calculator, and `from_ase_calculator` any ASE calculator, such as that
  of torchnep, an MDAPY calculator. See
  [ASE calculators](../user-guide/interop/ase-calculators.md).
- `mdapy.interop.to_ovito` and `from_ovito` convert to and from OVITO data
  collections, with types, standard properties, the topology and the
  attributes, so that results of OVITO pipelines come back as columns.
  See [OVITO](../user-guide/interop/ovito.md).
- `mdapy.interop.to_pymatgen` and `from_pymatgen` convert to and from
  pymatgen structures and molecules. See [pymatgen](../user-guide/interop/pymatgen.md).
- `mdapy.interop.to_freud` gives the box and points of freud's analyses,
  for systems periodic in every direction, and `from_freud` takes them
  back. See [freud](../user-guide/interop/freud.md).
- Pages on exchanging files and calculators with
  [LAMMPS](../user-guide/interop/lammps.md), [VASP](../user-guide/interop/vasp.md) and
  [GPUMD](../user-guide/interop/gpumd.md).

### Performance

- `benchmarks/comparison.py` times the analyses that MDAPY shares with
  OVITO and freud, on the same systems, where each program gives the
  result of MDAPY, each program in a process of its own with limits of
  memory and time. See [MDAPY, OVITO and freud](../performance/comparison.md).
- Extended XYZ files of many small frames read faster: the comment line is
  split at once in its common form, the element column comes from the
  categories kept for the file, and the box and the columns are checked
  with less work. The 105,464 frames of UNEP16 read in 15 s rather than
  19 s on an Apple M5; small frames about 40 % faster.

### Tutorials

- Learn MDAPY: six short lessons that run on their own, from a crystal as a
  table to a nanoparticle that melts and your own LAMMPS dumps, and a cheat
  sheet. See [A crystal is a table](../getting-started/learn/table.md).
- Worked examples that combine several parts of MDAPY: a polycrystal of fcc
  and bcc grains, columnar grains on a hexagonal pattern with chosen
  misorientations, the defect energies of copper with checks of their
  convergence, when sharing neighbor lists pays, what a `System` shares
  and copies, a machine-learned potential for CrCoNi: the phonons of
  Cr, Co and Ni against DFT and the elastic constants of the alloy from 0 to
  1000 K against measurements, a high-throughput screen of alloys in many
  processes with the thread settings it needs, and a movie that flies through a crystal.
  See the Tutorials under Getting started.
- Pictures with the examples: the pages of the structure builders and of
  most analyses show what their example makes or finds, drawn by mdapy's
  renderer or plotted with `mdapy.visualization.plot`, from code on the
  page that the documentation tests run.
- [Surfaces and slip systems](../user-guide/build/slip-systems.md): the orientations of
  the common surfaces and slip systems of fcc, bcc, hcp and diamond, how to
  build their slabs and faults, their surface energies and generalized
  stacking fault energies, and the shuffle and glide planes of silicon.

### Foundations

- Released under the Mozilla Public License 2.0: MDAPY may be used in any
  software, open or closed, and changes to its own files are shared under
  the same license. MDAPY 1.x is under the GPL 3.0 or later.
- Python 3.11 or newer; NumPy 2.3.1 or newer and polars 2 are the only
  required dependencies.
- All compiled code is one extension module, built as a single abi3 wheel per
  platform for Python 3.12 and newer, optimized for speed.
- Atom indices in results are 32-bit, `mdapy.index_dtype`; a build with
  `-Ccmake.define.MDAPY_INDEX64=ON` uses 64-bit indices for more than
  2,147,483,647 atoms. See [Installation](../getting-started/installation.md#64-bit-atom-indices).
- A thread pool of mdapy's own replaces OpenMP, so MDAPY can run in the same
  process as PyTorch and LAMMPS. Results are identical for any number of
  threads. The calling thread works too, idle threads wait for the next
  step briefly before they sleep, and a step of a few pieces wakes only as
  many threads, so small systems are not slowed down by many cores. See
  [Parallelism](../basics/parallelism.md).
- `mdapy.show_versions` prints the versions of MDAPY, Python and the packages it
  depends on, and the processor, memory and GPU of the machine, to paste into an
  issue. The bug report form of the issue tracker asks for it and for the shortest
  code that shows the problem.
- `mdapy.set_num_threads`, `mdapy.get_num_threads`, `mdapy.num_threads` and
  the `MDAPY_NUM_THREADS` environment variable control the number of threads.
  The default is the logical CPUs the process may use, so that a job of a
  cluster on part of a node starts no thread for the rest of it (108 s against
  6 s for the same relaxations on 2 of 16 CPUs). A forked process gets a
  warning: polars hangs in it once its parent has used its threads, so start
  processes with `spawn`. See [Parallelism](../basics/parallelism.md#many-processes).
- `mdapy.CancelToken`, `progress` callbacks and Ctrl-C stop or follow long
  computations.
- `mdapy.visualization.write_movie` writes pictures as a movie with the FFmpeg of
  the user, taking them one by one from a generator. See
  [Movies](../user-guide/visualization/movies.md).
- `render` refuses pictures above 8192 pixels on a side or 16384 subpixels, and
  `write_movie` movies above 8192 × 4320 pixels, as larger ones can use up the memory
  or play nowhere.
- `mdapy.units` defines every physical constant and conversion factor. See
  [Units and conventions](../reference/units.md).
- `mdapy.errors` defines the exception types. `FileFormatError` reports a
  malformed file with its name and line number.
- Text files are parsed and written by compiled kernels in parallel. Numbers
  are read correctly rounded and written in the shortest form that reads back
  to the same value, whatever the locale of the process; unit conversions by
  powers of ten are exact.
- macOS wheels need macOS 14 or newer.
- `mdapy.elements` holds chemical symbols and IUPAC standard atomic weights,
  and identifies elements from masses. See [Elements](../reference/elements.md).
