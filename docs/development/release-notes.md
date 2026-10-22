# Release notes

## 2.0.0 (in development)

mdapy 2.0 is a complete rewrite and is not compatible with mdapy 1.x.

### Structures

- `mdapy.System`: one atomic configuration with a polars table of per-atom
  properties, a box, global information and an optional topology. Systems
  are immutable; `with_*` methods and `select` return new systems that share
  unchanged columns. Positions are an (N, 3) array column read without
  copying. See [Systems and topology](../getting-started/system.md).
- `mdapy.Topology`: bonds, angles, dihedrals and impropers with types and
  labels, kept consistent when atoms are selected or reordered.
- `mdapy.Box`: immutable simulation box with lattice vectors, origin and
  periodicity; fractional coordinates, wrapping with image flags, exact
  minimum images, and conversion to LAMMPS restricted triclinic form. See
  [Simulation box](../getting-started/box.md).

### Reading and writing

- `mdapy.read` and `mdapy.write` read one frame and write one or more
  frames, finding the format from the file name. Files compressed with gzip,
  bzip2, xz or zstd (Python 3.14) are read and written transparently.
  `mdapy.formats` and `mdapy.detect_format` list and identify formats. See
  [Reading and writing files](../getting-started/io.md).
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
- The mdapy file format `.mdapy`: a ZIP archive of Parquet tables that keeps
  every column with its type, the box, the information and the topology of
  each frame, a table of frames with typed columns of the information, and
  recovery of files whose writing stopped early. See
  [mdapy files](../reference/formats/mdapy.md).
- `mdapy.read_trajectory` opens the frames of any file lazily: frames are
  found and read when used, slices share the open file, and iteration starts
  at once. See [Trajectories](../getting-started/trajectory.md).

### Neighbor search

- `mdapy.NeighborList.build(system, cutoff)` and
  `NeighborList.build(system, num_neighbors=k)`: every pair within a cutoff,
  or the k nearest neighbors of each atom, in any box, with periodic images
  enumerated so that small boxes work, in CSR form with int8 image shifts
  (15 bytes per pair); `within` and `sorted` derive lists without a new
  search. Distances are computed by one formula in both, nearest neighbors are
  ordered exactly with ties by atom and image, and results are the same for
  any number of threads. A k-d tree finds nearest neighbors in clusters and
  slabs as fast as in crystals. See [Neighbor lists](../getting-started/neighbor.md).

### Calculators

- `mdapy.calculators.Calculator`, the interface of every calculator, and
  `CalculatorResult` with energies, forces, virials and stress.
  `mdapy.calculators.testing.check_calculator` tests any implementation. See
  [Calculators](../simulation/calculators.md).
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

### Simulation

- `mdapy.simulation.minimize`: FIRE 2.0 minimization of the positions, step
  for step as LAMMPS `min_style fire`, and optionally of the box
  (`cell="isotropic"`, `"axes"` or `"full"`, at a `pressure`) to the minimum
  of LAMMPS `fix box/relax`; keeps a frame every `save_every` steps when
  asked, with the virial and the stress of each. See
  [Energy minimization](../simulation/minimize.md).
- `mdapy.lattice_dynamics.elastic_constants`: the (6, 6) elastic constants
  from the stress under small strains, with relaxed positions, and Voigt,
  Reuss and Hill moduli. See [Elastic constants](../simulation/elastic-constants.md).

- `mdapy.simulation.vacancy_formation_energy`, `surface_energy` and
  `stacking_fault_energy` (the generalized stacking fault energy, relaxed
  along the normal), agreeing with LAMMPS; `minimize` keeps atoms still
  along the directions where a `movable` column is False. See
  [Defect energies](../simulation/defect-energies.md).

### Analyses

- Analyses are classes whose `compute(system)` returns an immutable result;
  `System.with_columns(result)` adds its per-atom values. See
  [Analyses](../analysis/overview.md).
- Structure identification in `mdapy.identify`:
  [`CommonNeighborAnalysis`](../analysis/cna.md), adaptive or with a fixed
  cutoff, and [`AcklandJonesAnalysis`](../analysis/ackland-jones.md), agreeing
  atom by atom with LAMMPS, for fcc, hcp, bcc and icosahedral atoms;
  [`DiamondStructure`](../analysis/diamond.md) for cubic and hexagonal
  diamond; [`ChillPlus`](../analysis/chill-plus.md) for ice, hydrates and
  liquid water; and the [centro-symmetry](../analysis/csp.md) and
  [common neighbor](../analysis/cnp.md) parameters.
- Distributions and order parameters in `mdapy.analysis`, agreeing with
  OVITO where it has them: the
  [`RadialDistributionFunction`](../analysis/rdf.md) and its partial
  functions; [`CoordinationNumber`](../analysis/coordination.md) with
  cutoffs per pair of elements; the
  [`StructureFactor`](../analysis/structure-factor.md), exact over the wave
  vectors of the box or by the Debye method, with Faber–Ziman partial
  factors; the [`BondAngleDistribution`](../analysis/bond-angles.md) for
  each triplet of elements; [`WarrenCowley`](../analysis/warren-cowley.md)
  short-range order; [`SteinhardtParameters`](../analysis/steinhardt.md) up
  to degree 20, optionally averaged; the
  [`EntropyFingerprint`](../analysis/entropy.md); and the
  [`VoronoiAnalysis`](../analysis/voronoi.md), computed by mdapy's own code,
  with the volumes, faces, coordination and Voronoi indices of the cells,
  the face and edge thresholds of OVITO, boxes that are not periodic, and
  the shapes of the cells as polyhedra; `NeighborList.build(system,
  voronoi=True)` for the atoms that share a face, and Steinhardt parameters
  over them, optionally weighted by the face areas.
- Dynamics in `mdapy.analysis`: the
  [`MeanSquaredDisplacement`](../analysis/msd.md) of a trajectory, over
  every time origin or from the first frame, agreeing with LAMMPS and OVITO;
  the [`LindemannIndex`](../analysis/lindemann.md) of each atom and of the
  frames so far; both match the atoms of the frames by their ids and
  follow them through the periodic boundaries by their image flags or by
  the nearest image, unless the positions are already unwrapped. The
  [`AtomicTemperature`](../analysis/temperature.md) of each atom and its
  neighbors, relative to their center of mass.
- Defects and deformation in `mdapy.analysis`, agreeing with OVITO:
  [`ClusterAnalysis`](../analysis/clusters.md) for clusters of bonded atoms,
  with their unwrapped positions, centers of mass, gyration tensors and
  whether they span the box;
  [`WignerSeitz`](../analysis/wigner-seitz.md) for vacancies and
  interstitials, the sites of the reference kept in a k-d tree for every
  configuration; and [`AtomicStrain`](../analysis/atomic-strain.md), the
  Green–Lagrange, von Mises and volumetric strains and $D^2_\text{min}$
  against a reference.

### Structure builder

- `mdapy.build.build_crystal`: perfect crystals of fifteen structures,
  cubic and hexagonal, of one element or ordered on their sublattices,
  along any crystal directions, with the shortest period along each;
  the same atoms as atomsk. See [Crystals](../build/crystals.md).
- `mdapy.build.random_alloy`: elements at random in given fractions with
  exact counts. See [Random alloys](../build/alloys.md).
- `mdapy.build.sqs_alloy`: special quasirandom structures, the pairs of the
  nearest shells those of a random alloy. See
  [SQS](../build/sqs.md).
- Defects in `mdapy.build`: `add_vacancies`, `add_interstitials` and
  `substitute`; `slip` for stacking faults; `add_vacuum` for surfaces;
  `insert_dislocation`, the isotropic screw and edge dislocations of
  atomsk, and `edge_dislocation` of two crystals. See
  [Point defects](../build/defects.md) and
  [Dislocations](../build/dislocations.md).
- `mdapy.build.build_polycrystal`: Voronoi grains of any periodic unit,
  columnar when the box is thin, and `twin_lamellae` for twinned grains. See
  [Polycrystals](../build/polycrystals.md).

### MLIP

- `mdapy.mlip.read_outcar` and `outcar_to_xyz`: the ionic steps of VASP
  single points, relaxations and molecular dynamics, with energies, forces
  and virials, to extended XYZ training sets, without unconverged steps or
  frames of huge forces. See [Training sets from VASP](../mlip/vasp.md).
- `mdapy.mlip.structure_descriptors`, `farthest_point_sampling` and
  `principal_components`: frames chosen far from a training set and from
  each other in the space of NEP descriptors, and their projection on a
  plane. See [Choosing frames](../mlip/selection.md).
- `mdapy.mlip.perturb`: copies of a structure with atoms displaced and the
  box strained at random. See [Perturbed structures](../mlip/perturbation.md).

### Visualization

- `mdapy.visualization.render`: pictures of systems by ray tracing, with
  ambient occlusion, shadows and outlines in six styles, colored by
  element, structure, values or by hand, with bonds, the box and Voronoi
  cells; a `Scene` of spheres, cylinders and polygons; a million atoms in
  under two seconds. See [Rendering](../visualization/rendering.md).
- `mdapy.visualization.plot`: matplotlib set up for journals (widths of
  columns, Nature's artwork guide, Okabe–Ito colors, panel labels, saving
  as PDF, SVG and PNG) and plots of results: parity, g(r), S(q), bond
  angles, MSD with the diffusion coefficient, stacking fault energies,
  counts of structures. See [Plots](../visualization/plots.md).

### Foundations

- Python 3.11 or newer; NumPy 2.3.1 or newer and polars 2 are the only
  required dependencies.
- All compiled code is one extension module, built as a single abi3 wheel per
  platform for Python 3.12 and newer, optimized for speed.
- Atom indices in results are 32-bit, `mdapy.index_dtype`; a build with
  `-Ccmake.define.MDAPY_INDEX64=ON` uses 64-bit indices for more than
  2,147,483,647 atoms. See [Installation](../getting-started/installation.md#64-bit-atom-indices).
- A thread pool of mdapy's own replaces OpenMP, so mdapy can run in the same
  process as PyTorch and LAMMPS. Results are identical for any number of
  threads. See [Parallelism](../getting-started/parallelism.md).
- `mdapy.set_num_threads`, `mdapy.get_num_threads`, `mdapy.num_threads` and
  the `MDAPY_NUM_THREADS` environment variable control the number of threads.
- `mdapy.CancelToken`, `progress` callbacks and Ctrl-C stop or follow long
  computations.
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
