# Source layout

| Path | Role |
|---|---|
| `src/mdapy/__init__.py` | Version and the top-level public names. |
| `src/mdapy/errors.py` | Exception types. |
| `src/mdapy/units.py` | Physical constants and conversion factors. |
| `src/mdapy/elements.py` | Chemical symbols and standard atomic weights. |
| `src/mdapy/core/system.py` | `System`, one atomic configuration. |
| `src/mdapy/core/topology.py` | `Topology`: bonds, angles, dihedrals, impropers. |
| `src/mdapy/core/box.py` | `Box`, the simulation box. |
| `src/mdapy/core/threads.py` | Number of threads, `CancelToken`. |
| `src/mdapy/core/indices.py` | `index_dtype`, the type of atom indices in results. |
| `src/mdapy/core/result.py` | `Result`, the base class of the results of analyses. |
| `src/mdapy/io/__init__.py` | `read`, `write`, `formats`, `detect_format`; imports each format module. |
| `src/mdapy/io/_registry.py` | The format registry, format detection, `read` and `write`. |
| `src/mdapy/io/_files.py` | Opening files: compression and memory mapping. |
| `src/mdapy/io/_extxyz.py` | The extended XYZ format. |
| `src/mdapy/io/_lammps_data.py` | The LAMMPS data format. |
| `src/mdapy/io/_lammps_dump.py` | The LAMMPS dump format. |
| `src/mdapy/io/_poscar.py` | The VASP POSCAR format. |
| `src/mdapy/io/_outcar.py` | The VASP OUTCAR format, read only. |
| `src/mdapy/io/_cif.py` | The CIF format, read only: crystals expanded by their symmetry operations. |
| `src/mdapy/io/_pdb.py` | The PDB format, read only. |
| `src/mdapy/io/_native.py` | The mdapy file format. |
| `src/mdapy/io/_trajectory.py` | `Trajectory` and `read_trajectory`: frames of several files, frames on a topology. |
| `src/mdapy/interop/_ase.py` | `to_ase` and `from_ase`. |
| `src/mdapy/interop/_ase_calculator.py` | `to_ase_calculator`, `from_ase_calculator`. |
| `src/mdapy/interop/_ovito.py` | `to_ovito`, `from_ovito`. |
| `src/mdapy/interop/_pymatgen.py` | `to_pymatgen`, `from_pymatgen`. |
| `src/mdapy/interop/_freud.py` | `to_freud`, `from_freud`. |
| `src/mdapy/neighbor/_neighbor_list.py` | `NeighborList`. |
| `src/mdapy/calculators/_base.py` | `Calculator` and `CalculatorResult`. |
| `src/mdapy/calculators/testing.py` | `check_calculator`. |
| `src/mdapy/calculators/_eam.py` | `EAM`. |
| `src/mdapy/calculators/_zhou.py` | The parameters and tables of the potentials of Zhou et al. |
| `src/mdapy/calculators/_lammps.py` | `LAMMPS`, the calculator of any LAMMPS interaction. |
| `src/mdapy/calculators/_vasp.py` | `VASP`, single points of VASP run by a command. |
| `src/mdapy/calculators/_nep.py` | `NEP` and its file reader. |
| `src/mdapy/simulation/_minimize.py` | `minimize`, with FIRE 2.0, of the positions and the box. |
| `src/mdapy/lattice_dynamics/_elastic.py` | `elastic_constants`. |
| `src/mdapy/identify/_structure.py` | `CommonNeighborAnalysis`, `CentroSymmetryParameter`. |
| `src/mdapy/identify/_ptm.py` | `PolyhedralTemplateMatching`. |
| `src/mdapy/analysis/_distribution.py` | `RadialDistributionFunction`, `CoordinationNumber`, `StructureFactor`. |
| `src/mdapy/analysis/_correlation.py` | `SpatialCorrelation`, by fast Fourier transforms on a grid. |
| `src/mdapy/analysis/_steinhardt.py` | `SteinhardtParameters`. |
| `src/mdapy/analysis/_voronoi.py` | `VoronoiAnalysis`. |
| `src/mdapy/analysis/_entropy.py` | `EntropyFingerprint`. |
| `src/mdapy/analysis/_temperature.py` | `AtomicTemperature`. |
| `src/mdapy/analysis/_strain.py` | `AtomicStrain`. |
| `src/mdapy/analysis/_elastic_strain.py` | `ElasticStrain`. |
| `src/mdapy/analysis/_wigner_seitz.py` | `WignerSeitz`. |
| `src/mdapy/analysis/_dynamics.py` | `MeanSquaredDisplacement`, `LindemannIndex`, `TimeAverage`, `trajectory_lines`. |
| `src/mdapy/analysis/_polyhedra.py` | `CoordinationPolyhedra`. |
| `src/mdapy/analysis/_dislocations.py` | `DislocationAnalysis`, its results and the families of Burgers vectors. |
| `src/mdapy/build/_crystal.py` | `build_crystal` and its table of structures. |
| `src/mdapy/build/_alloy.py` | `random_alloy`. |
| `src/mdapy/mlip/_soap.py`, `src/cpp/analysis/soap.cpp` | `SOAP`, the power spectrum in Gaussian radial functions. |
| `src/mdapy/mlip/_selection.py` | descriptors of frames, farthest point sampling, principal components, `perturb`. |
| `src/mdapy/visualization/_scene.py`, `_render.py`, `_elements.py` | `Scene`, `render`, styles, colors. |
| `src/mdapy/visualization/_voxels.py` | `voxel_faces`, the cells of a grid on the faces of the box. |
| `src/mdapy/analysis/_rings.py`, `src/cpp/analysis/rings.cpp` | `RingStatistics`, shortest-path rings of bonds. |
| `src/mdapy/analysis/_isosurface.py`, `src/cpp/analysis/isosurface.cpp` | `isosurface`, by marching tetrahedra. |
| `src/cpp/visualization/render.h`, `render.cpp` | the ray tracer: bounding volume hierarchy, shading, outlines. |
| `src/cpp/visualization/render_gpu.h`, `render_metal.mm` | the rays of pictures on the GPU: Metal on macOS, its shader compiled when first used. |
| `src/mdapy/build/_sqs.py`, `src/cpp/build/sqs.cpp` | `sqs_alloy` and its annealing. |
| `src/mdapy/build/_defects.py` | point defects, `slip`, `add_vacuum`. |
| `src/mdapy/build/_dislocation.py` | `insert_dislocation`, `edge_dislocation`. |
| `src/mdapy/build/_polycrystal.py` | `build_polycrystal`, `twin_lamellae`. |
| `src/mdapy/transform/_periodic.py` | `replicate`, `wrap`, `unwrap`. |
| `src/mdapy/transform/_combine.py` | `combine`, the atoms of two systems in one. |
| `src/mdapy/transform/_drift.py` | `remove_drift`, the frames of a trajectory moved by the drift of their center of mass. |
| `src/mdapy/analysis/_bonds.py` | `ClusterAnalysis`, `BondAngleDistribution`, `WarrenCowley`, and bonds by cutoff per pair of elements. |
| `src/mdapy/io/_text.py` | Line and table reading and writing shared by the text formats. |
| `src/cpp/core/parallel.h`, `parallel.cpp` | The thread pool: `parallel_for`, `parallel_reduce`, cancellation and progress. |
| `src/cpp/core/index.h` | `index_t`, the type of atom indices, 32 or 64 bits (`MDAPY_INDEX64`). |
| `src/cpp/core/buffer.h` | Vectors left uninitialized, for large results written by threads. |
| `src/cpp/io/text.h`, `text.cpp` | Parsing and formatting of text tables, in parallel over lines; numbers are correctly rounded and independent of the locale. |
| `src/cpp/neighbor/frame.h` | The box as the neighbor searches see it, and the vector of a pair. |
| `src/cpp/neighbor/cell_list.h`, `cell_list.cpp` | The search within a cutoff, with cells. |
| `src/cpp/neighbor/kd_tree.h`, `kd_tree.cpp` | The search of nearest neighbors, with a k-d tree. |
| `src/cpp/calculators/eam.h`, `eam.cpp` | The embedded atom method. |
| `src/cpp/calculators/nep.h`, `nep.cpp` | Neuroevolution potentials. |
| `src/cpp/identify/structure.h`, `structure.cpp` | Common neighbor analysis and the centro-symmetry parameter. |
| `src/cpp/identify/ptm.h`, `ptm.cpp` | Polyhedral template matching of every atom: the neighbors in the order of their Voronoi faces, the matching, the results. |
| `src/cpp/identify/ptm_tools.h`, `ptm_tools.cpp` | Its pieces: convex hulls, canonical forms of graphs, the rotation that fits a template, the codes of correspondences. |
| `src/cpp/identify/ptm_data.h`, `ptm_data.cpp` | Its templates and what is built from them once: symmetry groups, the graphs of their hulls, the fundamental zones. |
| `src/cpp/core/voronoi_cell.h` | A convex cell cut down plane by plane, for Voronoi cells. |
| `src/cpp/analysis/steinhardt.h`, `steinhardt.cpp` | Real spherical harmonics and Steinhardt order parameters. |
| `src/cpp/analysis/distribution.h`, `distribution.cpp` | Histograms of pair distances counted during the neighbor search, and the structure factor. |
| `src/cpp/analysis/strain.h`, `strain.cpp` | Atomic strain from neighbor vectors. |
| `src/cpp/analysis/elastic_strain.h`, `elastic_strain.cpp` | Elastic strain from the neighbors of the structure identification of the dislocation analysis. |
| `src/cpp/analysis/geometry.h`, `geometry.cpp` | Clusters of bonded atoms, bond angles and the pair entropy. |
| `src/cpp/analysis/voronoi.h`, `voronoi.cpp` | Voronoi cells, cut plane by plane. |
| `src/cpp/analysis/dislocations.h`, `dislocations.cpp` | The dislocation analysis from the atoms to the lines and the defect mesh. |
| `src/cpp/analysis/dxa_structure.h`, `dxa_structure.cpp` | Its crystal atoms by common neighbor analysis, their neighbors in the order of their lattice. |
| `src/cpp/analysis/dxa_clusters.h`, `dxa_clusters.cpp` | The lattices, crystal clusters, the transitions between them and grains. |
| `src/cpp/analysis/dxa_mapping.h`, `dxa_mapping.cpp` | Ideal vectors of the edges of the tessellation and the good tetrahedra. |
| `src/cpp/analysis/dxa_mesh.h`, `dxa_mesh.cpp` | The interface mesh between the good tetrahedra and the rest. |
| `src/cpp/analysis/dxa_tracer.h`, `dxa_tracer.cpp` | Burgers circuits, the lines and junctions they find, their smoothing. |
| `src/cpp/analysis/dxa_math.h` | Small vectors and matrices of the dislocation analysis. |
| `src/cpp/identify/cna.h` | Signatures of common neighbor analysis, shared by the structure analyses. |
| `src/cpp/io/powers_of_five.h` | Generated table used by the number parser. |
| `src/cpp/bindings/module.cpp` | The extension module `mdapy._core` and its submodules. |
| `src/cpp/bindings/control.h` | `run_released`: runs a kernel without the GIL, with progress and cancellation. |
| `src/cpp/bindings/core.cpp` | Bindings for thread control. |
| `src/cpp/bindings/io.cpp` | Bindings for the text table kernels. |
| `src/cpp/bindings/neighbor.cpp` | Bindings for the neighbor search. |
| `src/cpp/bindings/calculators.cpp` | Bindings for the compiled calculators. |
| `src/cpp/bindings/identify.cpp` | Bindings for structure identification. |
| `src/cpp/bindings/analysis.cpp` | Bindings for the analyses. |
| `src/cpp/bindings/testing.cpp` | Kernels used only by the test suite. |
| `tests/` | Test suite; mirrors the package layout. |
| `tests/lattices.py` | Perfect crystals and clusters shared by the tests. |
| `tests/dislocation_models.py` | Dislocation loops from the solid angle they subtend, for the tests of the dislocation analysis. |
| `tests/cpp/thread_checks.cpp` | Checks of the thread pool and the parallel kernels, run under ThreadSanitizer. |
| `docs/` | This documentation. |
| `benchmarks/comparison.py` | Timings of mdapy against OVITO and freud; see [mdapy, OVITO and freud](../performance/comparison.md). |

All compiled code is one extension module, `mdapy._core`. Each functional
area adds a submodule, for example `mdapy._core.core`. Compiled kernels never
create threads or use OpenMP themselves; they call `mdapy::parallel_for` or
`mdapy::parallel_reduce`.
