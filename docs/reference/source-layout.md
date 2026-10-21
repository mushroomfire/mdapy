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
| `src/mdapy/io/__init__.py` | `read`, `write`, `formats`, `detect_format`; imports each format module. |
| `src/mdapy/io/_registry.py` | The format registry, format detection, `read` and `write`. |
| `src/mdapy/io/_files.py` | Opening files: compression and memory mapping. |
| `src/mdapy/io/_extxyz.py` | The extended XYZ format. |
| `src/mdapy/io/_lammps_data.py` | The LAMMPS data format. |
| `src/mdapy/io/_lammps_dump.py` | The LAMMPS dump format. |
| `src/mdapy/io/_poscar.py` | The VASP POSCAR format. |
| `src/mdapy/io/_native.py` | The mdapy file format. |
| `src/mdapy/io/_text.py` | Line and table reading and writing shared by the text formats. |
| `src/cpp/core/parallel.h`, `parallel.cpp` | The thread pool: `parallel_for`, `parallel_reduce`, cancellation and progress. |
| `src/cpp/io/text.h`, `text.cpp` | Parsing and formatting of text tables, in parallel over lines; numbers are correctly rounded and independent of the locale. |
| `src/cpp/io/powers_of_five.h` | Generated table used by the number parser. |
| `src/cpp/bindings/module.cpp` | The extension module `mdapy._core` and its submodules. |
| `src/cpp/bindings/control.h` | `run_released`: runs a kernel without the GIL, with progress and cancellation. |
| `src/cpp/bindings/core.cpp` | Bindings for thread control. |
| `src/cpp/bindings/io.cpp` | Bindings for the text table kernels. |
| `src/cpp/bindings/testing.cpp` | Kernels used only by the test suite. |
| `tests/` | Test suite; mirrors the package layout. |
| `tests/cpp/thread_checks.cpp` | Checks of the thread pool and the parallel kernels, run under ThreadSanitizer. |
| `docs/` | This documentation. |

All compiled code is one extension module, `mdapy._core`. Each functional
area adds a submodule, for example `mdapy._core.core`. Compiled kernels never
create threads or use OpenMP themselves; they call `mdapy::parallel_for` or
`mdapy::parallel_reduce`.
