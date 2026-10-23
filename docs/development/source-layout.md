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
| `src/mdapy/io/_native.py` | The mdapy file format. |
| `src/mdapy/io/_trajectory.py` | `Trajectory` and `read_trajectory`. |
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
| `src/mdapy/analysis/_steinhardt.py` | `SteinhardtParameters`. |
| `src/mdapy/analysis/_voronoi.py` | `VoronoiAnalysis`. |
| `src/mdapy/analysis/_entropy.py` | `EntropyFingerprint`. |
| `src/mdapy/analysis/_temperature.py` | `AtomicTemperature`. |
| `src/mdapy/analysis/_strain.py` | `AtomicStrain`. |
| `src/mdapy/analysis/_wigner_seitz.py` | `WignerSeitz`. |
| `src/mdapy/analysis/_dynamics.py` | `MeanSquaredDisplacement`, `LindemannIndex`. |
| `src/mdapy/build/_crystal.py` | `build_crystal` and its table of structures. |
| `src/mdapy/build/_alloy.py` | `random_alloy`. |
| `src/mdapy/mlip/_vasp.py` | `read_outcar`, `outcar_to_xyz`. |
| `src/mdapy/mlip/_selection.py` | descriptors of frames, farthest point sampling, principal components, `perturb`. |
| `src/mdapy/visualization/_scene.py`, `_render.py`, `_elements.py` | `Scene`, `render`, styles, colors. |
| `src/cpp/visualization/render.h`, `render.cpp` | the ray tracer: bounding volume hierarchy, shading, outlines. |
| `src/cpp/visualization/render_gpu.h`, `render_metal.mm` | the rays of pictures on the GPU: Metal on macOS, its shader compiled when first used. |
| `src/mdapy/build/_sqs.py`, `src/cpp/build/sqs.cpp` | `sqs_alloy` and its annealing. |
| `src/mdapy/build/_defects.py` | point defects, `slip`, `add_vacuum`. |
| `src/mdapy/build/_dislocation.py` | `insert_dislocation`, `edge_dislocation`. |
| `src/mdapy/build/_polycrystal.py` | `build_polycrystal`, `twin_lamellae`. |
| `src/mdapy/transform/_periodic.py` | `replicate`, `wrap`, `unwrap`. |
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
| `src/cpp/identify/ptm.h`, `ptm.cpp` | Polyhedral template matching of every atom with the PTM library. |
| `src/cpp/identify/ptm/` | The PTM library of P. M. Larsen (MIT license), as OVITO ships it; see its README for the changes. |
| `src/cpp/analysis/steinhardt.h`, `steinhardt.cpp` | Real spherical harmonics and Steinhardt order parameters. |
| `src/cpp/analysis/distribution.h`, `distribution.cpp` | Histograms of pair distances counted during the neighbor search, and the structure factor. |
| `src/cpp/analysis/strain.h`, `strain.cpp` | Atomic strain from neighbor vectors. |
| `src/cpp/analysis/geometry.h`, `geometry.cpp` | Clusters of bonded atoms, bond angles and the pair entropy. |
| `src/cpp/analysis/voronoi.h`, `voronoi.cpp` | Voronoi cells, cut plane by plane. |
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
| `tests/cpp/thread_checks.cpp` | Checks of the thread pool and the parallel kernels, run under ThreadSanitizer. |
| `docs/` | This documentation. |
| `benchmarks/comparison.py` | Timings of mdapy against OVITO and freud; see [mdapy, OVITO and freud](../performance/comparison.md). |

All compiled code is one extension module, `mdapy._core`. Each functional
area adds a submodule, for example `mdapy._core.core`. Compiled kernels never
create threads or use OpenMP themselves; they call `mdapy::parallel_for` or
`mdapy::parallel_reduce`.
