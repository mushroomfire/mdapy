# mdapy

mdapy is a Python package for analysing and modelling atomistic simulations.

> **mdapy 2.0 is under development** on the `v2` branch. It is a complete
> rewrite and is not compatible with mdapy 1.x. For the stable release, see
> [mdapy 1.0.8](https://pypi.org/project/mdapy/1.0.8/) and the `master` branch.

- Documentation: sources in [`docs/`](docs/); preview with `mkdocs serve`
- Issues: https://github.com/mushroomfire/mdapy/issues

## Install from source

```bash
pip install ".[dev]"
```

mdapy needs Python 3.10 or newer, NumPy 2 and polars 2. Its compiled kernels
use their own thread pool rather than OpenMP, so they run safely in the same
process as PyTorch, NumPy and LAMMPS.

## License

BSD 3-Clause. See [LICENSE](LICENSE).
