# Release notes

## 2.0.0 (in development)

mdapy 2.0 is a complete rewrite and is not compatible with mdapy 1.x.

### Structures

- `mdapy.System`: one atomic configuration with a polars table of per-atom
  properties, a box, global information and an optional topology. Systems
  are immutable; `with_*` methods and `select` return new systems that share
  unchanged columns. Positions are an (N, 3) array column read without
  copying. See [Systems and topology](guide/system.md).
- `mdapy.Topology`: bonds, angles, dihedrals and impropers with types and
  labels, kept consistent when atoms are selected or reordered.
- `mdapy.Box`: immutable simulation box with lattice vectors, origin and
  periodicity; fractional coordinates, wrapping with image flags, exact
  minimum images, and conversion to LAMMPS restricted triclinic form. See
  [Simulation box](guide/box.md).

### Reading and writing

- `mdapy.read` and `mdapy.write` read one frame and write one or more
  frames, finding the format from the file name. Files compressed with gzip,
  bzip2, xz or zstd (Python 3.14) are read and written transparently.
  `mdapy.formats` and `mdapy.detect_format` list and identify formats. See
  [Reading and writing files](guide/io.md).
- Extended XYZ with the conventions of ASE and GPUMD: velocities from `vel`
  (Å/fs) or ASE `momenta`, stress in GPa, any per-atom column and any global
  value, written so that they read back unchanged. See
  [Extended XYZ](formats/extxyz.md).
- LAMMPS data files of the atom styles atomic, charge, bond, angle, molecular
  and full, following LAMMPS stable_30Sep2026: type labels, masses with
  element guessing, bonds to impropers as the topology, force-field
  coefficients kept unchanged, image flags, velocities, metal and real units,
  and restricted and general triclinic boxes. See
  [LAMMPS data](formats/lammps-data.md).
- LAMMPS text dumps of the atom and custom styles, with any number of frames:
  units and time items, orthogonal, triclinic and general triclinic boxes,
  scaled and unwrapped coordinates, elements and type labels, and every other
  column, array columns such as `c_stress[1]` to `c_stress[6]` kept together.
  See [LAMMPS dump](formats/lammps-dump.md).

### Foundations

- Python 3.10 or newer; NumPy 2 and polars 2 are the only required
  dependencies.
- All compiled code is one extension module, built as a single abi3 wheel per
  platform for Python 3.12 and newer.
- A thread pool of mdapy's own replaces OpenMP, so mdapy can run in the same
  process as PyTorch, NumPy and LAMMPS. Results are identical for any number of
  threads. See [Parallelism](guide/parallelism.md).
- `mdapy.set_num_threads`, `mdapy.get_num_threads`, `mdapy.num_threads` and
  the `MDAPY_NUM_THREADS` environment variable control the number of threads.
- `mdapy.CancelToken`, `progress` callbacks and Ctrl-C stop or follow long
  computations.
- `mdapy.units` defines every physical constant and conversion factor. See
  [Units and conventions](reference/units.md).
- `mdapy.errors` defines the exception types. `FileFormatError` reports a
  malformed file with its name and line number.
- Text files are parsed and written by compiled kernels in parallel. Numbers
  are read correctly rounded and written in the shortest form that reads back
  to the same value, whatever the locale of the process; unit conversions by
  powers of ten are exact.
- macOS wheels need macOS 14 or newer.
- `mdapy.elements` holds chemical symbols and IUPAC standard atomic weights,
  and identifies elements from masses. See [Elements](reference/elements.md).
