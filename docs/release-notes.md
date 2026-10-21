# Release notes

## 2.0.0 (in development)

mdapy 2.0 is a complete rewrite and is not compatible with mdapy 1.x.

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
- `mdapy.errors` defines the exception types.
- `mdapy.elements` holds chemical symbols and IUPAC standard atomic weights,
  and identifies elements from masses. See [Elements](reference/elements.md).
