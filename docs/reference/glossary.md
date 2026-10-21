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
| `info` | Global information of a `System`, such as `energy` or `timestep`. |
| `NeighborList` | Pairs of atoms within a cutoff, including periodic images. |
| `Topology` | Bonds, angles, dihedrals and impropers of a system; each a table with `atoms` (0-based row indices), `type` and optional `type_label`. |
| format | Name of a file format, passed as `format=`, such as `extxyz`. See [Reading and writing files](../guide/io.md). |
| `Calculator` | Interface that computes energy, forces and virial for a `System`. |
| `Result` | Base class of the objects returned by analyses. |
| reference | A reference configuration, for example in atomic strain analysis. |

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
| `molecule` | int | | molecule ID (LAMMPS molecule-ID) |
| `type_label` | str | | name of the atom type, for example a force-field type |
| `mass` | f64 | amu | mass |
| `charge` | f64 | e | charge |
| `image` | Array(i32, 3) | | periodic image flags |
| `movable` | Array(bool, 3) | | whether the atom may move along x, y, z (VASP selective dynamics) |
| `structure` | int8 | | structure type from an identification such as `CommonNeighborAnalysis`; the result names the types |
| `csp` | f64 | Å² | centro-symmetry parameter |
| `coordination` | int64 | | number of neighbors; `coordination_<element>` counts those of one element |
| `q4`, `q6`, … | f64 | | Steinhardt order parameters; `q4_average`, … when averaged over neighbors |
| `cluster` | int64 | | cluster of bonded atoms, numbered by decreasing size |
| `voronoi_volume`, `voronoi_coordination`, `cavity_radius` | f64, int64, f64 | Å³, , Å | volume and number of shared faces of the Voronoi cell, and the distance to its farthest vertex |
| `entropy` | f64 | k<sub>B</sub> | pair entropy fingerprint; `entropy_average` when averaged over neighbors |
| `temperature` | f64 | K | temperature of an atom and its neighbors |
| `site`, `occupancy` | int64 | | the nearest reference site of an atom, and the number of atoms on it |
| `shear_strain`, `volumetric_strain`, `d2min` | f64 | , , Å² | von Mises and volumetric atomic strain against a reference, and the non-affine squared displacement |

## Energies and virials

| Name | Meaning |
|---|---|
| `energy` | total energy (eV) |
| `atomic_energies` | per-atom energies (eV) |
| `virial` | total virial, (3, 3) (eV) |
| `atomic_virials` | per-atom virials, (N, 3, 3) (eV) |

## Parameters

| Concept | Name | Not used |
|---|---|---|
| cutoff radius (Å) | `cutoff` | `rc`, `r_cut` |
| cutoff per pair of elements | `cutoffs: dict[tuple[str, str], float]` | `rc_dict` |
| number of nearest neighbors | `num_neighbors` | `k`, `N`, `nnn` |
| number of histogram bins | `bins` | `nbin`, `nbins` |
| upper limit of a distribution | `r_max`, `k_max` | |
| result for each pair or triplet of elements, ordered by atomic number | `partial[a, b]` with `pair(a, b)`; `partial[a, b, c]` (vertex `a`) with `triplet(a, b, c)` | `rdf_partial`, `adf` |
| random seed | `seed` (int or `numpy.random.Generator`) | `random_seed` |
| list of elements | `elements` | `element_list` |
| fractions of elements | `fractions` | `element_ratio` |
| prebuilt neighbor list | `neighbors=` (keyword-only) | |
| reference configuration | `reference=` | `ref` |
| progress callback, cancellation | `progress=`, `cancel=` (keyword-only) | |

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
