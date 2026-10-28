# Glossary

These are the names mdapy uses. Every module follows them, so a concept has
one name everywhere. Some entries describe components that are still being
implemented.

## Data model

| Name | Meaning |
|---|---|
| `System` | One atomic configuration: a `Box`, a table of per-atom properties and global information. Immutable: operations return a new `System`. |
| `Box` | The simulation box: three lattice vectors, an origin and periodicity flags. |
| `box.matrix` | (3, 3) array whose rows are the lattice vectors **a**, **b**, **c**. |
| `box.pbc` | (3,) bool array, `True` where the box is periodic. |
| `Trajectory` | A sequence of frames, read lazily. |
| frame | One `System` of a `Trajectory`. |
| `table` | The per-atom table of a `System`, a polars DataFrame with one row per atom. |
| `system["name"]` | The values of a column, a component or an expression over the columns, one per atom, as a read-only NumPy array. |
| condition | A Python expression over the columns of a system that `System.filter` keeps atoms by, such as `"positions.z > 10 and element == 'Cu'"`. See [Selecting and ordering atoms](../getting-started/selecting.md). |
| component | One value of an array column, written `column.component`: `x`, `y`, `z` for 3 values, `x`, `y`, `z`, `w` for 4, `xx` … `zz` by rows for 9, and the index from 0 for any number, such as `positions.z` or `group.1`. |
| `info` | Global information of a `System`, such as `energy` or `timestep`. |
| `NeighborList` | Pairs of atoms within a cutoff, the nearest neighbors of each atom, or the atoms that share a Voronoi face, including periodic images. |
| pair | One entry of a `NeighborList`: atoms `i` and `j` and the periodic image `shifts` of `j`. |
| `shifts` | Periodic image of the second atom of a pair, (3,) int8: the pair vector is `x[j] - x[i] + shifts @ box.matrix`. |
| `index_dtype` | Integer type of atom indices in results such as `NeighborList.j`: int32, or int64 in a build with 64-bit indices. |
| `TriangleMesh` | A mesh of triangles, which may cross the periodic boundaries of a box: `vertices`, `faces` and the periodic image `shifts` of each corner. |
| `Topology` | Bonds, angles, dihedrals and impropers of a system; each a table with `atoms` (0-based row indices), `type` and optional `type_label`. |
| format | Name of a file format, passed as `format=`, such as `extxyz`. See [Reading and writing files](../getting-started/io.md). |
| `Calculator` | Interface that computes energy, forces and virial for a `System`. |
| `interop` | The module that converts to and from the objects of other packages: `to_ase`, `from_ase`, `to_ase_calculator`, `from_ase_calculator`, `to_ovito`, `from_ovito`, `to_pymatgen`, `from_pymatgen`, `to_freud`, `from_freud`. |
| average atom | The single species of an average-atom potential, standing for a random alloy; `A` in potential files. |
| `Result` | Base class of the objects returned by analyses. |
| reference | A reference configuration, for example in atomic strain analysis. |
| unwrapped positions | Positions moved by whole periodic images so that no cluster or molecule is cut by the periodic boundaries, or, along a trajectory, so that atoms move continuously through them; with an `image` column, the positions plus the image flags times the lattice vectors. |
| affine transformation | A map of positions x to M x + t, with a (3, 3) matrix M and a translation t: a rotation, shear, scaling or move, or a mix of them; vectors such as velocities turn with M alone. |
| selection | Some atoms of a system, given as a boolean mask of one value per atom or as their indices. |
| copy | One of the systems `replicate` places side by side; copy (i, j, l) of (n<sub>a</sub>, n<sub>b</sub>, n<sub>c</sub>) is number i + n<sub>a</sub>(j + n<sub>b</sub> l). |
| lag | The number of frames between two frames of a trajectory. |
| spanning cluster | A cluster that joins a periodic image of itself through its bonds, so that it has no unbroken position. |
| covalent bond | A pair of atoms closer than the sum of their covalent radii and 0.45 Å, as pictures draw them. |
| vector arrow | An arrow drawn from an atom along a vector of it, such as its displacement, velocity or force: a shaft and a head 2.5 times as wide and 4.5 times as long as the radius of the shaft. |
| global illumination | Light that reaches a surface after bouncing off others, as the shiny style of pictures traces it. |
| style of a picture | Its lights, shadows, occlusion and outlines together: fast (as the viewports of OVITO), matte (as its Tachyon renderer), shiny (after its OSPRay renderer) or flat (outlined, as ASE draws atoms). |
| highlight | The light of a lamp mirrored by a surface toward the viewer: a bright spot on a sphere. |
| metalness | From 0 to 1, how much a surface behaves as a metal: it reflects less light diffusely and its highlights take its color. |
| depth of field | In a picture through a lens, the range of depths that looks sharp around the plane in focus; nearer and farther things blur. |
| soft shadow | A shadow cast by a light of some size: dark where all of the light is hidden, fading where part of it is. |
| ionic step | One configuration of a VASP relaxation or molecular dynamics, with its energy, forces and virial; one frame of `read_outcar`. |
| training set | Frames with reference energies, forces and virials, usually from DFT, in extended XYZ, to fit a machine-learned potential. |
| structure descriptor | The mean of the descriptors of the atoms of a frame, one point per frame. |
| farthest point sampling | Choosing points one by one, each the farthest from those chosen before. |
| crystal cluster | Neighboring atoms of one structure whose neighbors map onto each other by rotations of the lattice: one region with one lattice frame, as the dislocation analysis joins them. |
| ideal vector | The vector between two atoms in the perfect, unstrained crystal, in the lattice frame of a crystal cluster. |
| good tetrahedron | A tetrahedron of the Delaunay tessellation whose six edges have ideal vectors that add up to zero around each face: crystal there maps onto the perfect lattice. |
| interface mesh | The triangles between the good tetrahedra and the rest, on which Burgers circuits run. |
| Burgers circuit | A closed path along the edges of the interface mesh; the sum of the ideal vectors of its edges is the Burgers vector of the dislocations it encloses. |
| defect mesh | The surface around the defects that are not dislocations: the interface mesh without the triangles the lines took, closed over their free ends. |
| family of Burgers vectors | The Burgers vectors of one kind, the same in any order and sign of their components, such as `1/2<110>` or `1/6<112>`. |

## Per-atom properties

Column names are lowercase snake_case. Vector quantities are stored in one
column of fixed-size arrays of length 3 and use the plural name.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `positions` | Array(f64, 3) | Å | positions |
| `velocities` | Array(f64, 3) | Å/ps | velocities |
| `forces` | Array(f64, 3) | eV/Å | forces |
| `element` | str | | chemical symbol |
| `type` | int | | LAMMPS atom type, starting at 1 |
| `id` | int | | atom ID, starting at 1 |
| `molecule` | int | | molecule ID (LAMMPS molecule-ID); from `ChemicalSpecies`, numbered from 1 by decreasing size |
| formula | str | | chemical formula of a molecule in Hill order, such as `H2O`, `CO2`, `H3N` |
| `type_label` | str | | name of the atom type, for example a force-field type |
| `mass` | f64 | amu | mass |
| `charge` | f64 | e | charge |
| `image` | Array(i32, 3) | | periodic image flags |
| `movable` | Array(bool, 3) | | whether the atom may move along x, y, z (VASP selective dynamics) |
| `structure` | int8 | | structure type from an identification such as `CommonNeighborAnalysis`; the result names the types |
| `grain` | int64 | | grain of an atom from `GrainSegmentation`, numbered from 0 by decreasing size; -1 for none |
| `planar_fault` | int8 | | planar fault type of fcc crystals from `PlanarFaults`: non-hcp, other hcp, intrinsic stacking fault, coherent twin boundary, multilayer stacking fault, extrinsic stacking fault |
| `csp` | f64 | Å² | centro-symmetry parameter |
| `cnp` | f64 | Å² | common neighbor parameter |
| `coordination` | int64 | | number of neighbors; `coordination_<element>` counts those of one element |
| `q4`, `q6`, … | f64 | | Steinhardt order parameters; `q4_average`, … when averaged over neighbors |
| `solid`, `solid_bonds` | bool, int32 | | whether an atom is solid-like, and its number of solid-like bonds (Steinhardt `solid=True`) |
| `cluster` | int64 | | cluster of bonded atoms, numbered by decreasing size |
| `surface` | bool | | whether an atom lies on the surface of the solid (`SurfaceMesh`, `surface=True`) |
| `crystal_cluster` | int | | crystal cluster of an atom in the dislocation analysis, −1 for none (`DislocationAnalysis`) |
| `voronoi_volume`, `voronoi_coordination`, `cavity_radius` | f64, int64, f64 | Å³, , Å | volume and number of faces shared with atoms of the Voronoi cell, and the distance to its farthest vertex |
| `entropy` | f64 | k<sub>B</sub> | pair entropy fingerprint; `entropy_average` when averaged over neighbors |
| `temperature` | f64 | K | temperature of an atom and its neighbors |
| `lindemann` | f64 | | Lindemann index of an atom over a trajectory |
| `displacement`, `displacement_magnitude` | Array(f64, 3), f64 | Å, Å | displacement of an atom from a reference, and its length (`AtomicDisplacement`) |
| `site`, `occupancy` | `index_dtype`, int64 | | the nearest reference site of an atom, and the number of atoms on it |
| `rmsd`, `interatomic_distance` | f64 | , Å | deviation from the best template of polyhedral template matching, and the distance between nearest neighbors of its lattice |
| `orientation` | Array(f64, 4) | | rotation of the local lattice from its standard orientation, a quaternion (x, y, z, w) |
| `deformation_gradient` | Array(f64, 9) | | elastic deformation gradient of the local lattice, by rows |
| `ordering` | int8 | | chemical ordering of the local lattice from polyhedral template matching |
| `shear_strain`, `volumetric_strain`, `d2min` | f64 | , , Å² | von Mises and volumetric atomic strain against a reference, and the non-affine squared displacement |

## Energies and virials

| Name | Meaning |
|---|---|
| `energy` | total energy (eV) |
| `atomic_energies` | per-atom energies (eV) |
| `virial` | total virial, (3, 3) (eV) |
| `atomic_virials` | per-atom virials, (N, 3, 3) (eV) |
| `stress` | stress, (3, 3) (GPa), positive under tension: −`virial` / volume |
| elastic constants | (6, 6) (GPa) in Voigt order xx, yy, zz, yz, xz, xy |
| `force_constants` | second derivatives of the energy with respect to the positions of two atoms, (n, nS, 3, 3) (eV/Å²), between the atoms of a cell and of its supercell |
| frequencies | of vibrations, in THz; imaginary ones as negative numbers |
| wave vector `q` | in units of the reciprocal lattice vectors of the cell: `(h, k, l)` is h **b**1 + k **b**2 + l **b**3, with **a**i · **b**j = 2π δij |
| `mesh` | number of wave vectors along each reciprocal lattice vector, of a mesh through Γ |
| primitive cell | the smallest periodic cell of a crystal, from `primitive_cell` |
| space group | the operations that map a crystal onto itself, from `SpaceGroup`: `number` (1 to 230), `symbol` (Hermann–Mauguin), `rotations` and `translations` (Cartesian, Å) |
| `equivalent` | for each atom, the first atom that symmetry makes it equivalent to |
| `symmetry` | whether phonons and elastic constants use the symmetry of the crystal, found by spglib: `True`, `False`, or `None`, whenever spglib is installed |
| `gibbs_energy` | Gibbs energy per atom (eV), with the static energy, the free energy of the vibrations and the pressure times the volume |
| `free_energy` | Helmholtz free energy (eV); of the vibrations in `ThermalProperties`, with the zero-point energy |
| `thermal_expansion` | volumetric thermal expansion coefficient (1/K) |
| `bulk_modulus` | isothermal bulk modulus (GPa) |
| isothermal, adiabatic | elastic constants at constant temperature, and at constant entropy (those of sound waves) |
| `errors` | standard errors of the means of a molecular dynamics result, from block averages, in its units |
| `thermo` | the thermodynamic state of a molecular dynamics run, a table of `step`, `time`, `temperature`, `potential_energy`, `kinetic_energy`, `total_energy`, `pressure`, `volume` and `stress` |
| ensemble | `"nve"`, `"nvt"` or `"npt"` |
| `heat_capacity` | heat capacity per atom (eV/K): at constant volume of the vibrations, at constant pressure in the quasi-harmonic approximation |
| descriptors | (N, dim) per-atom inputs of the network of a machine-learned potential, such as those of `NEP.descriptors` |

## Parameters

| Concept | Name | Not used |
|---|---|---|
| cutoff radius (Å) | `cutoff` | `rc`, `r_cut` |
| cutoff per pair of elements | `cutoffs: dict[tuple[str, str], float]` | `rc_dict` |
| number of nearest neighbors | `num_neighbors` | `k`, `N`, `nnn` |
| number of histogram bins | `bins` | `nbin`, `nbins` |
| upper limit of a distribution | `r_max`, `q_max` | `rc`, `k_max` |
| way of computing a quantity | `method` | `mode` |
| result for each pair or triplet of elements, ordered by atomic number | `partial[a, b]` with `pair(a, b)`; `partial[a, b, c]` (vertex `a`) with `triplet(a, b, c)` | `rdf_partial`, `adf` |
| random seed | `seed` (int or `numpy.random.Generator`) | `random_seed` |
| list of elements | `elements` | `element_list` |
| fractions of elements | `fractions` | `element_ratio` |
| crystal structure | `structure` | `lattice_type` |
| lattice constants (Å) | `a`, `c` | `lattice_constant` |
| crystal directions along the box vectors | `orientation` | `miller1`, `miller2`, `miller3` |
| number of periods along the box vectors | `repeat` | `nx`, `ny`, `nz` |
| atoms chosen by index or mask, or a number of them at random | `atoms=`, `count` | `index`, `num` |
| keep atoms by a condition, take them by index, order them | `System.filter`, `System.gather`, `System.sort` | `take`, `slice` |
| keep some columns, remove some, rename them | `System.select`, `System.drop`, `System.rename` | `keep`, `remove` |
| Burgers vector (Å), dislocation line and normal of the cut | `burgers`, `line`, `normal` | `b`, `xi`, `n` |
| elastic constants of a cubic crystal, (C11, C12, C44) or (6, 6) | `elastic` | `C11`, `C12`, `C44` |
| distance between the Shockley partials of a dislocation (Å) | `dissociation` | `partial_distance` |
| thickness of the fixed outer shell of a dislocation cylinder (Å) | `fixed` | `fix_width` |
| shells of neighbors of a lattice | `shells` | `nshell` |
| shells of neighbors, by distance or by number | `cutoff=(r1, r2, …)`, `num_neighbors=(n1, n2, …)` | `rc_list`, `shell_number` |
| seeds of grains, or their number | `grains` | `seed_number`, `seed_position` |
| prebuilt neighbor list | `neighbors=` (keyword-only) | |
| systems of a trajectory | `frames` | `pos_list`, `traj` |
| average over every time origin | `window` (bool) | `mode` |
| smallest Voronoi face (Å²), fraction of the cell's surface, shortest edge (Å) | `face_threshold`, `relative_face_threshold`, `edge_threshold` | `area_threshold` |
| neighbors that share a Voronoi face | `voronoi=True` | `use_voronoi` |
| neighbors weighted by the area of their face | `weighted=True` | `use_weight` |
| faces of the Voronoi cells | `faces=True` | |
| shapes of the Voronoi cells | `polyhedra=True` | `cell_info` |
| reduction over the atoms of a bin | `reduction` (`mean`, `sum`, `density`, `min`, `max`) | `operation` |
| keep the other atoms, the slab about a plane (Å) | `invert`, `width` | `reverse`, `slab_width` |
| a plane by Miller indices in the vectors of the box, or through a point | `miller=True`, `through` | `miller_indices`, `center` |
| nearest periodic image of a displacement | `minimum_image=True` | `minimum_image_convention` |
| radius of the probe sphere of a surface mesh (Å), smoothing steps | `radius`, `smoothing` | `probe_radius`, `smoothing_level` |
| regions of solid and empty space, voids among them | `regions=True` | `identify_regions` |
| atoms on the surface | `surface=True` | `select_surface_particles` |
| crystal of a dislocation analysis, longest Burgers circuit, edges more to follow a line, perfect dislocations only | `crystal`, `circuit_length`, `stretch`, `perfect_only` | `input_crystal_structure`, `trial_circuit_length`, `circuit_stretchability`, `only_perfect_dislocations` |
| sparseness of the points of dislocation lines, smoothing steps of the lines and of the defect mesh | `point_separation`, `smoothing`, `mesh_smoothing` | `line_point_separation`, `line_smoothing_level`, `defect_mesh_smoothing_level` |
| also the interface mesh of a dislocation analysis | `interface_mesh=True` | `output_interface_mesh` |
| arrows of vectors: the vectors, their scale, the radius of the shafts, color, point at the atom | `vectors`, `vector_scale`, `vector_radius`, `vector_color`, `vector_alignment` | `scaling`, `width`, `arrow_width`, `reverse` |
| follow atoms through the periodic boundaries | `unwrap` (bool) | `wrapped`, `unwrapped` |
| reference configuration | `reference=` | `ref` |
| progress callback, cancellation | `progress=`, `cancel=` (keyword-only) | |
| interval between kept frames | `save_every` | `dump_freq`, `interval` |
| convergence of a minimization | `max_force` (eV/Å), `max_stress` (GPa) | `fmax`, `ftol`, `etol` |
| how the box may relax | `cell` (`None`, `"isotropic"`, `"axes"`, `"full"`) | `relax_box`, `box_relax` |
| external hydrostatic pressure (GPa) | `pressure` | `press`, `p` |
| magnitude of an applied strain | `strain` | `delta`, `eps` |
| size of random displacements of atoms (Å) | `displacement` | `atom_pert_distance`, `sigma` |
| frames already chosen | `selected=` | `train_des`, `existing` |
| number of principal components | `n` | `n_components` |
| structures to look for, deviation above which an atom is other | `structures`, `rmsd_cutoff` | `check_fcc`, `rmsd_threshold` |
| also compute results that cost more: deformation gradients, chemical ordering, neighbors in template order, strain tensors, $D^2_\text{min}$, centers of mass and gyration of clusters | `deformation_gradient`, `ordering`, `template_neighbors`, `strain_tensor`, `d2min`, `center_of_mass`, `gyration` (bool), named after the result | `output_*`, `compute_*` |
| how atoms and bonds are drawn in a picture | `representation` (`"spheres"`, `"ball_and_stick"`, `"space_filling"`, `"licorice"`) | `model`, `drawing_method` |
| radii of the elements | `radius="display"`, `"covalent"`, `"vdw"` | `radii_type` |
| bounces of light in a picture | `bounces`, `samples` | `depth`, `spp` |
| direction of the highlights, metals, light everywhere, lighting the colors as given | `highlight`, `metalness`, `base_light`, `srgb` | `spec_dir`, `metallic`, `emission`, `gamma` |
| where the rays of a picture are traced | `device` (`"cpu"`, `"gpu"`, `"auto"`) | `backend`, `use_gpu` |
| whether the positions relax | `relax` (bool) | `relax_atoms`, `minimize` |
| INCAR tags, mesh of k-points, folder of POTCARs and the POTCAR of each element | `incar`, `kpoints`, `potcars`, `setups` | `params`, `kpts`, `pp_path`, `potcar_map` |
| command that runs a program, the folder it runs in | `command`, `directory` | `exe`, `workdir`, `label` |
| keep the wave functions between calls | `restart` (bool) | `istart`, `keep_files` |
| average of a polycrystal | `average` (`"voigt"`, `"reuss"`, `"hill"`) | `method` |

Boolean options are named positively, for example `include_self=True`.

## Names of functions and methods

- **Analyses** are classes named after what they compute, such as
  `CommonNeighborAnalysis`. Parameters are given to the constructor and
  validated there; `compute(system, ...)` returns a result object.
- **Changing a `System`** uses `with_*` methods and `select`, which return a
  new `System`; a `System` itself has no other methods. Geometric
  transformations such as `replicate` and `wrap` are functions in
  `mdapy.transform`.
- **Building structures** uses `build_*` functions.
- **Reading and writing** uses `read`, `write` and `read_trajectory`.
- **Conversion** to other packages lives in `mdapy.interop`, as `to_*` and
  `from_*` functions.
- `get_*` only reads a value; it never starts a computation.

Units and sign conventions are listed in
[Units and conventions](units.md).
