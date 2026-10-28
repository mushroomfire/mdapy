# Release notes

## 2.0.0 (in development)

mdapy 2.0 is a complete rewrite and is not compatible with mdapy 1.x.

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
  information. See [Systems](../getting-started/system.md) and
  [Changing a system](../getting-started/changing.md).
- `System.filter`, `System.gather` and `System.sort` keep atoms by a
  condition, take them by index and order them, as polars does with rows.
  Conditions are Python expressions over the columns, such as
  `"positions.x**2 + positions.y**2 < 25 and element == 'Cu'"`, with the
  components of array columns by name or index, comparisons, logic,
  membership, arithmetic and common functions; keywords, masks and polars
  expressions work too. `system["positions.z"]` gives the values of a
  column, a component or such an expression as a read-only NumPy array,
  without a copy for columns and components. See
  [Selecting and ordering atoms](../getting-started/selecting.md).
- `mdapy.Topology`: bonds, angles, dihedrals and impropers with types and
  labels, kept consistent when atoms are selected or reordered. See
  [Topology](../getting-started/topology.md).
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
  at once; also the frames of several files, given as a pattern such as
  `dump.*.lammpstrj` or a list, and frames that move the atoms of a
  topology read from a data file, matched by id, with its bonds, charges
  and molecules, as OVITO's load trajectory. See
  [Trajectories](../getting-started/trajectory.md).
- Reading CIF files, crystals expanded by their symmetry operations as
  pymatgen, ASE and OVITO read them, and PDB files, with atom names,
  residues, models as frames and the bonds of `CONECT`. See
  [CIF](../reference/formats/cif.md) and [PDB](../reference/formats/pdb.md).

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
  positions in the same box (`fits`), so it outlives `with_columns`.
  See [Neighbor lists](../getting-started/neighbor.md).
- `mdapy.neighbor.expand_selection`: a selection grown by the atoms within a
  cutoff, the nearest neighbors or the bonded atoms of the selected ones,
  step by step, as OVITO's expand selection. See
  [Expanding a selection](../getting-started/expand-selection.md).
- Analyses given a list read it where it is, without a copy: those of the
  k nearest neighbors put the k nearest pairs of each atom first in the
  list itself, once, in parallel; those within a cutoff read the pairs
  closer than theirs and stop at the first sorted pair beyond it. One list
  within 5 Å serves every structure analysis of a metal; threads may share
  a list. A list keeps 15 bytes per pair: `i` and `vectors` are computed
  when asked for. See
  [Sharing neighbor lists](../getting-started/tutorials/neighbor-lists.md).

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
- `mdapy.calculators.VASP` runs VASP for each calculation with a command,
  as in the job script of a cluster: it writes the inputs, keeps the wave
  functions between calls and reads the OUTCAR. `CalculationFailed` reports
  a run that stopped. See [VASP](../simulation/vasp.md).

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
  See [Phonons](../simulation/phonons.md).
- `mdapy.lattice_dynamics.quasi_harmonic`: the quasi-harmonic
  approximation from phonons at several volumes: the equilibrium volume,
  Gibbs energy, entropy, bulk modulus, thermal expansion and heat capacity
  at constant pressure against temperature, at any pressure, the shape of
  the box relaxed at each volume. See
  [Quasi-harmonic approximation](../simulation/quasi-harmonic.md).
- `mdapy.lattice_dynamics.thermal_elastic_constants`: isothermal and
  adiabatic elastic constants at finite temperature, quasi-harmonic or
  quasi-static. See [Elastic constants at finite temperature](../simulation/thermal-elastic.md).
- `mdapy.lattice_dynamics.md_elastic_constants`: isothermal elastic
  constants at a temperature from the mean stress of molecular dynamics in
  LAMMPS under small strains, with standard errors from block averages. See
  [Elastic constants from molecular dynamics](../simulation/md-elastic.md).
- Symmetry with spglib (optional, the extra `symmetry`): `phonons`,
  `quasi_harmonic`, `elastic_constants` and `thermal_elastic_constants`
  skip the displacements and strains that the symmetry of the crystal
  relates to others, by default whenever spglib is installed: one
  computation of forces instead of 6 for the primitive cell of fcc, 12 of
  the 73 strains of the elastic constants of a cubic crystal at finite
  temperature. The elastic constants take the rotations alone, in
  milliseconds also for cells of hundreds of atoms. See
  [Space group](../analysis/space-group.md).
- `mdapy.simulation.MolecularDynamics`: molecular dynamics in LAMMPS with
  the interaction of a LAMMPS calculator: random velocities, runs at
  constant energy, temperature or pressure with ramps, the thermodynamic
  state as a polars table, frames and the final system back in the
  orientation of the box, any LAMMPS command in between; the same
  trajectory as the same LAMMPS input. See
  [Molecular dynamics](../simulation/dynamics.md).

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
  diamond and their first and second neighbors, as OVITO;
  [`ChillPlus`](../analysis/chill-plus.md) for ice, hydrates and
  liquid water; and the [centro-symmetry](../analysis/csp.md) and
  [common neighbor](../analysis/cnp.md) parameters.
- [`ChemicalSpecies`](../analysis/species.md): molecules and their Hill
  formulas by the bond rule of Jmol (covalent radii, the default) or of VMD
  and OVITO (van der Waals radii), radii given by hand or bond lengths per
  pair of elements, and the number of each along a trajectory; the same
  molecules as ASE and as OVITO. `mdapy.elements` has the covalent
  and van der Waals radii, which pictures now read from it.
- [`SpaceGroup`](../analysis/space-group.md): the space group of a
  crystal, its symmetry operations and equivalent atoms, with spglib, for
  cells of up to 2000 atoms.
- [`PolyhedralTemplateMatching`](../analysis/ptm.md): the structure,
  orientation, elastic deformation and chemical ordering of each atom, for
  nine structures, implemented by mdapy itself; the same as OVITO atom by
  atom.
- [`PlanarFaults`](../analysis/planar-faults.md): intrinsic and multilayer
  stacking faults and coherent twin boundaries of fcc crystals from the
  hcp-like atoms of polyhedral template matching, and their areas; the same
  as OVITO's Identify fcc planar faults atom by atom; optionally extrinsic
  stacking faults too.
- [`GrainSegmentation`](../analysis/grain-segmentation.md): the grains of a
  polycrystal from the lattice orientations of polyhedral template
  matching, by node pair sampling with an automatic threshold or along a
  minimum spanning tree, with stacking faults and twin boundaries joined to
  the grains around them and orphan atoms adopted; the grain of each atom,
  and the size, structure, mean orientation, center of mass and radii of
  each grain, the same as OVITO's Grain segmentation atom by atom.
- Distributions and order parameters in `mdapy.analysis`, agreeing with
  OVITO where it has them: the
  [`RadialDistributionFunction`](../analysis/rdf.md) and its partial
  functions; [`CoordinationNumber`](../analysis/coordination.md) with
  cutoffs per pair of elements; the
  [`StructureFactor`](../analysis/structure-factor.md), exact over the wave
  vectors of the box or by the Debye method, with Faber–Ziman partial
  factors; the [`BondAngleDistribution`](../analysis/bond-angles.md) for
  each triplet of elements; [`WarrenCowley`](../analysis/warren-cowley.md)
  short-range order, over shells of neighbors by distance or by number; [`SteinhardtParameters`](../analysis/steinhardt.md) up
  to degree 20, optionally averaged, and with `solid=True` the solid-like
  bonds and atoms of ten Wolde, Ruiz-Montero and Frenkel, as freud's
  `SolidLiquid` (mdapy 1.x `identify_liquid`); the
  [`EntropyFingerprint`](../analysis/entropy.md); and the
  [`VoronoiAnalysis`](../analysis/voronoi.md), computed by mdapy's own code,
  with the volumes and coordination of the cells and, when asked for, their
  faces and Voronoi indices (`faces=True`), which otherwise are not kept,
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
  [`ClusterAnalysis`](../analysis/clusters.md) for clusters of bonded atoms
  and whether they span the box, and on request their unwrapped positions,
  centers of mass and gyration tensors;
  [`AtomicDisplacement`](../analysis/displacement.md), the displacement
  of each atom from a reference matched by id, with the minimum image and
  the affine mapping of OVITO's displacement vectors;
  [`SpatialBinning`](../analysis/spatial-binning.md), a property in bins
  along one to three box vectors, its mean, sum, density, minimum or
  maximum, the values of OVITO's spatial binning;
  [`WignerSeitz`](../analysis/wigner-seitz.md) for vacancies and
  interstitials, the sites of the reference kept in a k-d tree for every
  configuration; and [`AtomicStrain`](../analysis/atomic-strain.md), the
  von Mises and volumetric strains against a reference, and on request the
  Green–Lagrange strain tensor and $D^2_\text{min}$. As in OVITO, the
  results that cost more are computed only when asked for.
- [`SurfaceMesh`](../analysis/surface-mesh.md) in `mdapy.analysis`: the
  surface of the solid of atoms by the alpha-shape method of OVITO's
  Construct surface mesh, from a Delaunay tessellation of mdapy's own, with
  its area, the volume of the solid, the regions of solid and empty space
  and the voids among them, the atoms on the surface, and Taubin's
  smoothing; the same areas, volumes and atoms as OVITO. The surface is a
  `mdapy.TriangleMesh`, a mesh of triangles across periodic boundaries.
- [`DislocationAnalysis`](../analysis/dislocations.md) in `mdapy.analysis`,
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
  (2 million atoms in some 0.05 s). See [Crystals](../build/crystals.md).
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
  [Dislocations](../build/dislocations.md).
- `mdapy.build.add_graphene`: graphene on the grain boundaries of a
  polycrystal, one or more AB-stacked layers centered on each face between
  two grains, each face once, in a random orientation, cut to the face;
  a fraction of the boundary area covered by whole faces; the atoms of the
  grains too close to the carbon removed. `build_polycrystal` keeps the
  seeds in `info["seeds"]`. See [Graphene on grain boundaries](../build/graphene.md).
- `mdapy.build.build_polycrystal`: Voronoi grains of any periodic unit,
  columnar when the box is thin, and `twin_lamellae` for twinned grains. See
  [Polycrystals](../build/polycrystals.md).
- `mdapy.transform.replicate` copies a system along its periodic
  directions as LAMMPS `replicate` does, with the ids, molecules and
  topology of each copy and molecules kept whole through the image flags;
  `wrap` and `unwrap` move atoms into the box or to their unwrapped
  positions. See [Periodic copies and images](../build/transform.md).
- `mdapy.transform.slice`: the atoms on one side of any plane or in a slab
  about it, as the Slice of OVITO: a normal in Cartesian coordinates or as
  Miller indices of the box, at a distance or through a point, acting on
  all atoms or some; `slice_mask` the same as a mask. See
  [Slices](../build/slice.md).
- `mdapy.transform.affine`: moves, rotates, shears or scales the atoms, their
  velocities, forces and displacements, and the box, all atoms or a
  selection, as OVITO's affine transformation modifier. See
  [Affine transformation](../build/affine.md).

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
  ambient occlusion, shadows and outlines in four styles: fast, lit as the
  viewports of OVITO; matte, as the Tachyon renderer of OVITO (the
  default); shiny, a metallic sheen after its OSPRay renderer, with light
  bouncing between the atoms; flat, plain colors outlined as ASE draws
  atoms; colored by element,
  structure, values or by hand, with bonds, the box, Voronoi cells,
  surface meshes and dislocation lines with arrows of their Burgers
  vectors, cut at the periodic faces of the box;
  arrows of any vector of the atoms, such as displacements or forces,
  as OVITO's vector display draws them;
  molecules as balls and sticks, licorice or space filling, with covalent
  bonds found from the elements; a `Scene` of spheres, cylinders and
  polygons; the scene framed, or a given focus and span; depth of field
  and soft shadows; every setting of a style can be changed, including the
  direction of the highlights, metals and lighting in sRGB; a million atoms
  in 4 s on the processor in the matte style, or in 1 s on the GPU with
  Metal on macOS.
  See [Rendering](../visualization/rendering.md).
- `mdapy.visualization.plot`: matplotlib set up for journals (widths of
  columns, Nature's artwork guide, Okabe–Ito colors, panel labels, saving
  as PDF, SVG and PNG) and plots of results: parity, g(r), S(q), bond
  angles, MSD with the diffusion coefficient, stacking fault energies,
  counts of structures. See [Plots](../visualization/plots.md).

### Interoperability

- `mdapy.interop.to_ase` and `from_ase` convert to and from `ase.Atoms`,
  with velocities, charges, constraints, other numeric columns and the
  results of a calculation. See [ASE](../interop/ase.md).
- `mdapy.interop.to_ase_calculator` makes any mdapy calculator an ASE
  calculator, and `from_ase_calculator` any ASE calculator, such as that
  of torchnep, an mdapy calculator. See
  [ASE calculators](../interop/ase-calculators.md).
- `mdapy.interop.to_ovito` and `from_ovito` convert to and from OVITO data
  collections, with types, standard properties, the topology and the
  attributes, so that results of OVITO pipelines come back as columns.
  See [OVITO](../interop/ovito.md).
- `mdapy.interop.to_pymatgen` and `from_pymatgen` convert to and from
  pymatgen structures and molecules. See [pymatgen](../interop/pymatgen.md).
- `mdapy.interop.to_freud` gives the box and points of freud's analyses,
  for systems periodic in every direction, and `from_freud` takes them
  back. See [freud](../interop/freud.md).
- Pages on exchanging files and calculators with
  [LAMMPS](../interop/lammps.md), [VASP](../interop/vasp.md) and
  [GPUMD](../interop/gpumd.md).

### Performance

- `benchmarks/comparison.py` times the analyses that mdapy shares with
  OVITO and freud, on the same systems, where each program gives the
  result of mdapy, each program in a process of its own with limits of
  memory and time. See [mdapy, OVITO and freud](../performance/comparison.md).

### Tutorials

- Worked examples that combine several parts of mdapy: a polycrystal of fcc
  and bcc grains, columnar grains on a hexagonal pattern with chosen
  misorientations, the defect energies of copper with checks of their
  convergence, when sharing neighbor lists pays, what a `System` shares
  and copies, and a machine-learned potential for CrCoNi: the phonons of
  Cr, Co and Ni against DFT and the elastic constants of the alloy from 0 to
  1000 K against measurements. See the Tutorials under Getting started.
- Pictures with the examples: the pages of the structure builders and of
  most analyses show what their example makes or finds, drawn by mdapy's
  renderer or plotted with `mdapy.visualization.plot`, from code on the
  page that the documentation tests run.
- [Surfaces and slip systems](../build/slip-systems.md): the orientations of
  the common surfaces and slip systems of fcc, bcc, hcp and diamond, how to
  build their slabs and faults, their surface energies and generalized
  stacking fault energies, and the shuffle and glide planes of silicon.

### Foundations

- Released under the Mozilla Public License 2.0: mdapy may be used in any
  software, open or closed, and changes to its own files are shared under
  the same license. mdapy 1.x is under the GPL 3.0 or later.
- Python 3.11 or newer; NumPy 2.3.1 or newer and polars 2 are the only
  required dependencies.
- All compiled code is one extension module, built as a single abi3 wheel per
  platform for Python 3.12 and newer, optimized for speed.
- Atom indices in results are 32-bit, `mdapy.index_dtype`; a build with
  `-Ccmake.define.MDAPY_INDEX64=ON` uses 64-bit indices for more than
  2,147,483,647 atoms. See [Installation](../getting-started/installation.md#64-bit-atom-indices).
- A thread pool of mdapy's own replaces OpenMP, so mdapy can run in the same
  process as PyTorch and LAMMPS. Results are identical for any number of
  threads. The calling thread works too, idle threads wait for the next
  step briefly before they sleep, and a step of a few pieces wakes only as
  many threads, so small systems are not slowed down by many cores. See
  [Parallelism](../getting-started/parallelism.md).
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
