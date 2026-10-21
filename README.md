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

### 22:57
- **CI 已修好，最新提交 8 个任务全部通过。** 原因有两个：
  - Linux 上 GCC 的 `-Wshadow` 报了一个 clang 不报的警告（构造函数参数和成员同名），已改。本机装有 g++-15，以后提交前先用它查一遍。
  - Linux 上把 TSan 预加载进 Python 后，一导入 polars 就段错误（在 CI 里逐个导入确认过，和 mdapy 代码无关；降低 mmap 随机化、改用 macOS 都试过，不行）。现在 TSan 改为检查一个不依赖 Python 的 C++ 程序（`tests/cpp/thread_checks.cpp`），覆盖线程池和并行文本内核；ASan 照常跑完整的 Python 测试。
- **LAMMPS dump（文本 atom/custom）完成并已推送**：多帧、UNITS 只在第一帧、TIME、三斜外接盒换算、一般三斜、缩放/展开坐标、`c_x[1..k]` 合并成数组列。用 LAMMPS 2026 实测了两个方向（LAMMPS 写→mdapy 读逐位一致；mdapy 写→LAMMPS `read_dump` 一致）。
- 正在做 POSCAR/CONTCAR，测试写好了，还在修几个细节。

### 23:16
- **今晚计划内的读写部分全部完成并已推送：** LAMMPS dump、VASP POSCAR/CONTCAR、`.mdapy` 原生格式、惰性轨迹 `read_trajectory`。每项都有测试、文档页和发布说明，新文件覆盖率都是 100%。
  - POSCAR：支持三种缩放系数、POTCAR 标签、Selective dynamics（存成 `movable` 列）、CONTCAR 速度块；用 ASE 双向验证过。
  - `.mdapy`：ZIP+Parquet，所有列类型和 info 都能精确往返；`frames.parquet` 可以直接读出整条轨迹的能量等；写到一半进程被杀，也能从各帧的元数据恢复。
  - 轨迹：按需扫描，切片共享同一个打开的文件，不再引用时自动关闭文件。
- **邻居列表草稿在分支 `neighbor-draft`（未合并）：** 支持任意盒子，盒子比截断半径小也行，结果和暴力枚举逐位一致，和线程数无关。25.6 万个原子、截断 5 Å，单线程 0.39 s，10 线程 0.13 s。API 明天给你确认。
- 修了 Python 3.10 才有的问题：3.10 的 mmap 没有 `seekable()`。本地在建 3.10 环境，以后提交前先跑一遍。

### 23:29
- **计算器草稿在分支 `calculator-draft`（基于 `neighbor-draft`，未合并）：**
  - 接口：`Calculator`、`CalculatorResult`，有 energy、atomic_energies、forces、virial、atomic_virials，`stress()` 给 GPa，`attach()` 把结果写回 System。
  - 通用检查工具 `mdapy.calculators.testing.check_calculator(calc, system)`：检查求和一致、力和 virial 与有限差分一致、平移/旋转/重排不变、超胞能量加倍、传入邻居列表结果不变。torchnep 那边可以直接拿来测。每一项都用故意写错的计算器验证过能抓到错误。
  - `EAM`（C++）：读 LAMMPS eam/alloy 文件，样条插值和 LAMMPS 一样。和独立的 NumPy 实现相差 1e-11；和 LAMMPS 2026 本机实测相比，能量差 1e-12，力和 virial 差 1e-9。比较 virial 时发现 LAMMPS metal 单位用的换算常数是四舍五入过的（1.6021765e6），测试里已注明。
  - 所有 C++ 编译时关闭了 FMA 合并，同样的输入在不同编译器和 CPU 上结果逐位相同（邻居列表分支里改的）。
- 接下来做 FIRE 最小化（含晶胞弛豫），同样放在草稿分支。

