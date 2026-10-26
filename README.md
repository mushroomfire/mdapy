<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/mdapy-logo-dark.svg">
    <img alt="mdapy" src="docs/assets/mdapy-logo.svg" width="560">
  </picture>
</p>


mdapy, short for Molecular Dynamics Analysis in Python, analyses and builds
atomistic systems.

> **mdapy 2.0 is under development** on the `v2` branch. It is a complete
> rewrite and is not compatible with mdapy 1.x. For the stable release, see
> [mdapy 1.0.8](https://pypi.org/project/mdapy/1.0.8/) and the `master` branch.

- Documentation: sources in [`docs/`](docs/); preview with `mkdocs serve`
- Issues: https://github.com/mushroomfire/mdapy/issues

## Install from source

```bash
pip install ".[dev]"
```

mdapy needs Python 3.11 or newer, NumPy 2.3.1 or newer and polars 2. Its
compiled kernels use their own thread pool rather than OpenMP, so they run
safely in the same process as PyTorch and LAMMPS.

## License

Mozilla Public License 2.0 (MPL-2.0). See [LICENSE](LICENSE): mdapy may be
used in any software, open or closed, and changes to its own files are shared
under the same license. mdapy 1.x is under the GPL 3.0 or later.
