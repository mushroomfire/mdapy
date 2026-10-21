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

## 夜间进展汇报（临时，2026-10-21，明早删除）

### 22:35
- 已推送到 origin/v2 共 6 个提交：元素、Box、System/Topology、文本解析内核、读写框架和 extxyz、LAMMPS data。全部 645 个测试通过。
- 今晚按顺序做：LAMMPS dump（文本 atom/custom）→ POSCAR/CONTCAR → `.mdapy` 原生格式 → `read_trajectory`；时间够的话在单独分支写邻居列表草稿，明天给你看过 API 再合并。
