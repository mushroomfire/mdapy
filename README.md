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

### 23:43
- **`mdapy.minimize` 草稿已推送到 `calculator-draft`**：FIRE（参数同 ASE），每个原子单步位移有上限，可传回调中途停止，结果和线程数无关。测试：谐振子二聚体收敛到键长（1e-8 Å），扰动过的 fcc 弛豫回完美晶格。**目前只弛豫原子位置，不动晶胞**：晶胞弛豫的收敛判据（应力阈值、是否保持形状）要和你商量，所以先没做。
- **开始写 NEP 计算器草稿**（新分支 `nep-draft`）。为避免 GPL 代码进 BSD 的 mdapy，所有常数都由我自己从数学定义推导：角向用正交实球谐，三体项 $q_{nl}=\sum_m A_{nlm}^2$，四体项由 Wigner 3j 耦合得到，五体项正好等于 $\tfrac{7}{15}q_{n1}^2$（中间耦合 L=0,1,2 求和）。推导结果和 GPUMD 的常数表逐项一致（相对差 1e-12，来自对方表格只印了 15 位）。
- 测试计划：独立的 NumPy 参考实现（角向用 Legendre 双重求和，与 C++ 路线不同），对比 GPUMD 的描述符、能量、力、virial 参考数据（测试时从本地 torchnep 目录读取，不复制文件），再加 `check_calculator`。

### 23:56
- **NEP 计算器草稿完成并已推送到分支 `nep-draft`**（基于 `calculator-draft`，未合并）：`NEP("nep.txt")` 支持 nep4 / nep4_zbl、三种 ZBL（通用、按元素截断、flexible）、每种元素单独截断半径，能算能量、力、virial、每原子 virial，`NEP.descriptors(system)` 直接给描述符。C++ 实现，结果和线程数无关。
- **精度**：和 NEP_CPU（双精度）对比 14 个势函数（1~4 种元素、各种 ZBL），能量、力、virial、每原子 virial 都相差约 1e-13（相对最大值），最大的力 2.4e5 eV/Å 也一样。和 GPUMD（单精度、只打印 6 位）对比，能量差 1e-5 eV/原子。NumPy 参考实现和 C++ 描述符差 1e-15。
  - 中途发现：GPUMD 的 ZBL 屏蔽长度常数用的是四舍五入的 2.134563（不是 1/0.46848），不照搬的话大 ZBL 能量会差 1e-7 相对误差，已注明。
- **性能**：5.5 万原子、CrCoNi 模型（描述符 63 维、80 个神经元），单线程 35 µs/原子，10 线程 8 µs/原子。还有优化空间，先保证正确。
- 测试：CI 里用随机参数的合成势函数（48 个测试，覆盖率 100%）；对比 GPUMD/NEP_CPU 的 22 个测试只在本机有 torchnep 数据时运行。已手动触发 `nep-draft` 分支的 CI，三个草稿分支一起在 Linux/Windows/macOS 上验证。
- **需要你明天决定**：每原子 virial 的分配方式。我让 NEP 跟 GPUMD/NEP_CPU 一致（i 的能量对 r_ij 的导数记到 j 上），这样每原子应力能和 GPUMD 对上；EAM 则是每对一人一半（和 LAMMPS 一致）。两种总 virial 一样，只是每原子不同。
- 顺手修了文档 `units.md` 里应力定义的符号错误（写成了 −1/V·∂E/∂ε，和"拉伸为正"矛盾），在 v2 上。

### 00:15
- （更正：上两条的时间标错了，实际是 23:43 和 23:56，已改。）
- **新分支 `analysis-draft`**（基于 `neighbor-draft`，未合并），把第 3 阶段分析的框架先搭出来给你看：
  - `mdapy.core.Result`：所有分析结果的基类，不可变、数组只读；`system.with_columns(result)` 直接把逐原子结果加成列。分析类的用法按术语表定的：构造时给参数并检查，`compute(system, *, neighbors=None)` 返回结果。
  - `mdapy.identify.CommonNeighborAnalysis`（自适应 a-CNA，或固定截断）：fcc/hcp/bcc/ico/other，C++ 实现。和我按定义写的纯 Python 参考实现逐原子一致（有扰动、部分原子识别失败的晶体上也一致）；空位周围正好 12 个原子变成 other；单原子的 fcc 原胞也能识别。
  - `mdapy.identify.CentroSymmetryParameter`：C++ 实现，和 LAMMPS 的定义一致；完美 fcc/bcc 为 0，理想 hcp 正好是 a²，与参考实现差 1e-15。
  - `mdapy.analysis.RadialDistributionFunction`（含偏 RDF、累积配位数）和 `CoordinationNumber`（可按元素对给截断）。
  - 每项都有测试（覆盖率 100%）、文档页。
- **新分支 `cell-draft`**（基于 `nep-draft`）：`minimize` 支持晶胞弛豫，`cell=None/"isotropic"/"axes"/"full"`，可加外压 `pressure`（GPa），收敛判据 `max_stress`（GPa，默认 0.01）。非周期方向不动。测试：拉伸、四方、剪切的 fcc 盒子都弛豫回立方，且用能量有限差分验证是 E(a) 的极小点；5 GPa 下应力为 −5 GPa，且等于能量对体积的有限差分。
- **NEP 性能对比**：同一个 CrCoNi 模型，NEP_CPU 单线程 28~31 µs/原子（含邻居搜索），mdapy 35 µs/原子（不含）。差距主要是为了"结果和线程数无关"，力用 gather 方式，每对原子的导数算了两遍。如果邻居列表能提供"反向对"的索引，可以只算一遍，估计能快到 20 µs 左右——这涉及邻居列表的 API，明天和你商量。
- 三个草稿分支早先的 CI 失败只有 Python 3.10 一项，原因是它们从 v2 分出来时还没有那个 3.10 修复；已把 v2 合并进草稿分支，CI 重新在跑。NEP 和 EAM 内核已加进 TSan 检查，本机 clang TSan 和 GCC 都通过。

### 00:42
- **CI：四个草稿分支（nep / cell / analysis / interop）全部通过**（三平台、Python 3.10~3.14、ASan、TSan）。
- **NEP 提速**（`nep-draft`）：每对原子的导数只算一次，再在核内找"反向对"汇总力和 virial，不改任何接口，结果仍与线程数无关、与 NEP_CPU 一致到 1e-13。单线程 35 → 28 µs/原子（和 NEP_CPU 持平），10 线程 8.2 → 5.3 µs/原子。
- **`analysis-draft` 又加了**：
  - `SteinhardtParameters`（q_l，l ≤ 20，可做邻居平均）：和文献值、加法定理公式（不经球谐）一致；和 **LAMMPS orientorder/atom 一致到 1e-10**。
  - 和 LAMMPS 2026 对比：固定截断 CNA 与 `cna/atom` 在扰动晶体上逐原子一致，CSP 与 `centro/atom` 一致到 1e-10（本地测试，CI 跳过）。
  - `ClusterAnalysis`（按截断或按元素对成键的团簇）、`BondAngleDistribution`（键角分布）、`WarrenCowley`（化学短程序）。
  - `IdentifyDiamond`（立方/六方金刚石，Maras 2016 的二近邻 CNA 方法）。
  - Ackland-Jones 没做：LAMMPS 现行版本里有一条"全部不满足则 unknown"的判断总会被后面覆盖，还有旧版本，要按论文还是按 LAMMPS 需要你定。
- **新分支 `voronoi-draft`**（基于 analysis-draft）：**自己写的 Voronoi**（不再依赖 voro++）。每个原子从立方体开始按距离用中垂面切割，切到更远的原子不可能再切为止；邻居不够时自动加大范围。非周期方向用盒子的面封闭。fcc 正好是菱形十二面体、bcc 是截角八面体、金刚石 12 个三角形 + 4 个六边形，体积和面积到 1e-12；随机原子的胞体积之和等于盒子体积；每个面在两侧原子里都出现且面积相同；用 4 万个随机点按最近原子归属检验每个胞的体积。性能：10.8 万原子单线程 13 µs/原子、10 线程 4.9 µs/原子（含邻居搜索）。
- **新分支 `interop-draft`**（基于 v2）：`mdapy.interop.to_ase / from_ase`，速度、电荷、约束（movable）、额外数组、盒子原点、info、能量/力/应力都能往返。
- CI 提示 GitHub 的 Node 20 已弃用，已把 `actions/checkout` 升到 v5、`actions/setup-python` 升到 v6（在 v2 上）。

### 01:10
- **邻居列表提速**（`neighbor-draft`，已合并到所有下游草稿）：`sorted()` / `nearest()` 改为 C++ 按原子并行排序，10 万原子 900 万对：sorted 1.9 → 0.1 s，nearest(14) 1.86 → 0.18 s。另把格子搜索重构成可复用的"按原子访问"接口（结果逐位不变），RDF 现在**边搜索边计数、不存邻居对**：100 万原子、8 Å 的 RDF 从 6.2 s、数 GB 内存降到 0.72 s（10 线程）。
- **`analysis-draft` 新增**：
  - `EntropyFingerprint`（Piaggi–Parrinello 熵指纹）：网格和积分同 LAMMPS，但用完整高斯（LAMMPS 只展开 ±3 格，完美晶体里各原子值都不同），与 LAMMPS 差 <1%。
  - `AtomicTemperature`（原子及其近邻相对质心的温度）：自由度用 3(n−1)（1.x 用 3n），300 K 的麦克斯韦分布给出 300 K；整体漂移不影响结果。
- **100 万原子实测**（10 线程，含邻居搜索）：CNA 1.1 s、CSP 0.8 s、Steinhardt 0.9 s、熵 1.0 s、团簇 0.4 s、键角 0.8 s、Voronoi 5.2 s、RDF(8 Å) 0.7 s。
- **Voronoi 稳健性**：完美晶格加 1e-14 ~ 1e-5 Å 扰动、二维层、强烈倾斜的盒子、稀疏气体中的致密团、几乎重合的原子，体积之和都等于盒子体积（误差 ≤ 3e-11）。
- **新分支 `build-draft`**（基于 v2）：`build_crystal`（sc/bcc/fcc/diamond 任意正交 Miller 取向，hcp）、`random_alloy`（按比例精确计数的随机合金）。
- **新分支 `polycrystal-draft`**（= voronoi + build）：`build_polycrystal`，每个晶粒填满种子的周期 Voronoi 胞，取向随机或指定，晶界处距离过近的原子删除；测试里晶粒内部全部被 CNA 识别为 fcc。
- 所有草稿的公开 API 一览已整理（本地 `design/api-overview.md`），明早给你看。

### 01:35
- **和 mdapy 1.0.8 实测对比后做了一轮提速**（100 万原子、10 线程、各自从头算）。一开始 2.0 草稿在 CNA、CSP、Steinhardt、Voronoi 上反而比 1.0.8 慢 2~6 倍，原因不在计算核，而在 Python 层构建邻居列表和计算键向量。现在 k 近邻搜索和键向量都改成 C++，Voronoi 直接在格子上搜邻居、不存邻居对：

  | 分析 | 1.0.8 | 2.0 |
  |---|---|---|
  | CNA | 0.29 s | 0.39 s |
  | CSP | 0.22 | 0.24 |
  | Steinhardt | 0.28 | 0.31 |
  | RDF 8 Å | 3.69 | **0.73** |
  | 熵 | 0.77 | 0.99 |
  | 团簇 | 0.33 | 0.37 |
  | Voronoi | 0.80（voro++） | 1.43 |

  剩下的差距：Voronoi 要改成 voro++ 那样的顶点-边图结构才能再快；CNA/熵可以把邻居搜索和计算合并到一个核里。这些明天和你商量要不要做。
- **新增**：`ChillPlus`（冰/水合物/液态水，规则同 1.x，测试用加法定理写的独立实现逐分子对照）、`StructureFactor`（静态结构因子，按盒子的倒格矢直接求和，含 Ashcroft–Thornton 偏结构因子；完美 fcc 只在 111/200/220/311 处有峰且峰高精确为 N）。
- **新分支 `elastic-draft`**（基于 cell-draft）：`elastic_constants` 弹性常数（±应变、内部弛豫、应力中心差分；Voigt/Reuss/Hill 模量），C11、C44、体模量和能量曲率一致到 1e-4。
- **新分支 `preview-all`**：把所有草稿合在一起（1114 个测试通过），方便你明天一次性试用；CI 也验证组合后没有冲突。

### 02:25
- **CI**：上一轮还在跑的 preview-all、dynamics、elastic、polycrystal 全部通过。
- **Voronoi 再提速**（`voronoi-draft`，已合进 polycrystal-draft 和 preview-all，preview-all 1131 个测试通过）。先纠正一个数：1.0.8 的 0.80 s 是 voro++ 在 10 线程上的时间，单线程 voro++ 是 4.79 s。单线程原来是 8.4 s 对 4.79 s，同样差 1.75 倍，所以差距在计算核本身，不在并行。
  - 剖析发现时间主要花在处理器猜错分支上：同一个原子重复算 100 万次每个胞 2.5 µs，100 万个不同原子要 6.6 µs。排序的比较和"顶点是否保留"的判断都没法预测。
  - 改法：顶点只存一份，面存顶点编号，被切的边上的新顶点只算一次；邻居按距离分 64 个箱（计数排序）代替比较排序；保留顶点不用分支。
  - 结果：单线程 8.4 → **6.3 s**（voro++ 4.8 s），10 线程 1.43 → **1.10 s**（voro++ 0.80 s）；扰动金刚石 10.8 → 8.0 s，随机气体 18.1 → 13.7 s。
  - 拓扑完全不变：20 组体系（完美/扰动的 fcc、bcc、hcp、金刚石，气体，非周期层，面积阈值 0 和 0.01）的邻居、周期像、每个面的顶点数与改前逐一相同，体积和面积差 ≤ 6e-14。只是每个胞内面的先后顺序变了。
  - 还剩的差距主要在格子搜邻居（约 1.2 s/百万原子）：把每个格子里的原子坐标连续存放可以再快，但这会改 `CellGrid`，影响所有分析，明天再定。

### 02:50
- **邻居搜索提速**（`neighbor-draft`，已合并到所有下游草稿和 preview-all）。结果与改前逐位相同：在三斜盒子、非周期方向、截断大于盒子、原子乱序等情形下对比了 59 MB 的邻居数据，完全一致。
  - 格子内部按格子顺序另存一份原子坐标，搜索也按格子顺序进行，所以读内存是连续的。LAMMPS 多进程跑出来的 dump 原子顺序常常是乱的，这种情况提升最大。
  - 先用折回盒子内的坐标粗筛候选（同一个格子的周期偏移只算一次，留有远大于舍入误差的余量），通过的再按文档里的公式精确计算。
  - k 近邻的选取改为按距离分箱的计数排序加插入排序，代替堆排序。
  - 建格子的逐原子部分改为并行。
  - 100 万原子、10 线程：

    | | 顺序原子 | 乱序原子 |
    |---|---|---|
    | 截断 3 Å | 0.155 → 0.145 s | 0.356 → 0.207 s |
    | 截断 5 Å | 0.552 → 0.386 | 1.100 → 0.458 |
    | 12 近邻 | 0.178 → 0.146 | 0.470 → 0.167 |
- **分析核提速**（`analysis-draft`）：
  - CNA 签名计算去掉了分支，核耗时减半。
  - 熵指纹：高斯在相邻格点间的比值本身按 1/e 递减，每对原子只需两次 exp，不再每个格点算一次；核耗时降到三分之一。与 LAMMPS 和参考实现的对比仍然通过。
  - 团簇和配位数：邻居列表正好是所需截断时不再重复过滤；元素查找从排序改为线性。
  - 用户给了邻居列表时，"取最近 k 个"改为一次 C++ 调用（原来是排序后再用 NumPy 截取）。
- **更正与 1.0.8 的对比方法**：1.0.8 会把邻居列表缓存在 System 里，后面的分析直接复用，所以在同一个 System 上依次计时会低估它的邻居搜索时间。下面的对比是每项都新建 System（100 万原子、10 线程）：

  | 分析 | 1.0.8 | 2.0（01:35） | 2.0（现在） |
  |---|---|---|---|
  | CNA | 0.30 s | 0.39 | **0.27** |
  | CSP | 0.22 | 0.24 | **0.21** |
  | Steinhardt | 0.29 | 0.31 | **0.28** |
  | RDF 8 Å | 2.6–3.3 | 0.73 | **0.41** |
  | 熵 | 0.94 | 0.99 | **0.54** |
  | 团簇 | 0.30 | 0.37 | **0.20** |
  | Voronoi | 0.82（voro++） | 1.43 | 1.04 |

  除了 Voronoi，2.0 现在都比 1.0.8 快。

### 03:10
- **NEP 提速**（`nep-draft`）：
  - 每个原子的描述符和"能量对键向量的导数"合并成一遍计算，每对原子的基函数和球谐只算一次。
  - 神经网络和径向收缩改成跨神经元、跨径向函数的向量化循环；求和顺序不变，结果逐位相同。
  - 角向导数先对径向函数求和，这一步改变了舍入，力和 virial 与改前差 ≤ 2e-15。能量逐位相同，与 NEP_CPU、GPUMD 的 22 个对比测试全部通过。
  - CrCoNi，5.5 万原子：单线程 28.8 → **24.7 µs/原子**（NEP_CPU 28–31），10 线程 0.317 → 0.267 s。
  - （该提交的说明里写成"少三分之一"，不准确，实测整体快 14%；改提交说明需要 force push，所以没动。）
- **新功能：按元素三元组的键角分布**（`analysis-draft`，对应 1.x 的 ADF）。`BondAngleDistribution(cutoffs={("O", "H"): 1.2})` 可以给每对元素单独设键长，结果新增 `partial[(顶点, b, c)]` 和 `triplet("O", "H", "H")`。测试用三元无序晶体的 NumPy 暴力枚举逐项对照，并用水分子验证 104.5°。1.x 的 ADF 每条键还能设最短距离；要不要也支持，明天请你定。
- **CI**：neighbor-draft 和 calculator-draft 之前缺 v2 上 Python 3.10 的修复，CI 在 3.10 上失败。已把 v2 合并进去，重跑全绿。今晚改过代码的草稿分支（neighbor、calculator、analysis、voronoi、nep、preview-all）都手动触发了 CI，已完成的全部通过。

### 03:35
- **新分支 `cnp-draft`**：公共近邻参数 `CommonNeighborParameter(cutoff)`（Tsuzuki 2007，1.x 有、2.0 计划表没列，所以单独建了分支，请你定去留）。
  - fcc、bcc、金刚石为 0，理想 hcp 正好 2a²/3。
  - 与按原子号+周期像在两份邻居表里找公共邻居的独立 Python 实现一致到 1e-12，小盒子多像、旋转平移下也成立。
  - 已合进 preview-all。
- **把 100 万原子的各项分析都跑了一遍，找出几处慢的并修好了：**

  | | 之前 | 现在 | 说明 |
  |---|---|---|---|
  | `build_polycrystal`，66 万原子、20 个晶粒 | 18.3 s | **0.73 s** | 晶粒归属改用格子搜最近种子，结果逐位相同 |
  | `AtomicTemperature`，100 万原子 | 0.87 s | **0.18 s** | 改为 C++，结果逐位相同 |
  | `StructureFactor`，3.2 万原子、q_max = 8 Å⁻¹ | 141 s | **6.1 s** | 见下 |
  | `LindemannIndex` | 内存随原子数平方增长，100 万原子直接内存不足 | 1 万原子 × 50 帧 2.3 s，内存只与轨迹大小有关 | 改为 C++ 逐对计算 |

  - 结构因子慢的原因：`std::complex` 的每次乘法都会调用一个库函数，相位表也是跳着读。现在用实数展开计算、顺序读表，结果与原来差 1e-14。
  - **需要你定**：1.x 的 `StructureFactor` 默认用 Debye 法（由 g(r) 做正弦变换，很快），还提供 X 射线、中子、电子衍射的加权；2.0 目前只有精确的直接求和。要不要加 Debye 法和这些加权？
- **全面检查**（preview-all）：ASan 下全量测试通过，GCC 15 严格警告通过，TSan 线程检查覆盖了所有新核。CI 全绿。
- 本地 Python 3.10 环境（numpy 2.2.6 + macOS Accelerate）会报假的 matmul 除零警告，导致 16 个测试失败。这是环境问题，Linux CI 的 3.10 全过，详见明早的总结。

### 04:00
- **熵指纹不再建邻居表**（`analysis-draft`）：没有传入邻居表时，在格子上找到邻居就直接用掉，不存 4300 万对。100 万原子 5 Å：0.54 → **0.28 s**（1.0.8 0.94 s）。与给定邻居表的结果一致到 1e-14，与 LAMMPS 的对比仍通过。
- **能量最小化提速 2 倍**（`calculator-draft`）：邻居搜索到截断外 1 Å，原子移动不到 0.5 Å 时直接复用已有的对，只重算距离。给计算器的邻居表与重新搜索得到的逐位相同，所以最小化轨迹也逐位相同（有测试对照）。EAM，3.2 万原子，100 步：3.3 → **1.65 s**。弹性常数里的弛豫也一起变快。晶胞弛豫每步都改盒子，仍然每步重新搜索。
- **稳健性检查**：所有分析在 0、1、2 个原子的体系上运行，EAM、NEP、最小化、晶胞弛豫在空体系、单原子、比截断还小的盒子上运行，都不崩溃，不支持的情况给出明确的错误。顺带修了一处：MSD 在没有原子时改为报错（原来是 NaN 加 NumPy 警告）。ASan 全量测试、GCC 严格警告、TSan 均通过。
- **合并演练**：在本地把 preview-all 合进 v2，没有冲突，合并后的文件树与 preview-all 相同。如果所有草稿都接受，早上一次合并就行；只接受部分的话，要按链逐个合并，共享文件会有冲突，preview-all 里有现成的解决结果。

### 04:25
- **NEP 只算描述符快 2.5 倍**（`nep-draft`）：`NEP.descriptors()` 在得到描述符后就停，不再跑神经网络、导数和力。5.5 万原子单线程 1.23 → 0.50 s，结果逐位相同。
- **配位数和团簇也不再建邻居表**（`analysis-draft`）：在格子上找到邻居就直接计数或连键，结果与使用邻居表时完全相同（有测试对照）。100 万原子：配位数 0.20 → 0.074 s，团簇 0.19 → 0.10 s（1.0.8 0.30 s）。按截断的 Warren–Cowley 参数也用同一个计数核。现在 RDF、熵、配位数、团簇、Voronoi 都可以不建表，大体系不会因邻居表占满内存。
- **与 1.0.8 的结果对照**（同一份 100 万原子数据）：自适应 CNA 逐原子相同，CSP、Steinhardt q4/q6 差 ≤ 1.3e-15，团簇结果相同。
- **检查**：ASan 全量测试、TSan、GCC 严格警告都通过；文档在 13 个分支上 `--strict` 都能构建；CI 全绿；所有分支没有残留的冲突标记。
- 明早给你的总结里，每个待决定的问题都附上了我的建议，方便直接拍板。

### 04:38
- **又有两项分析不再建邻居表**（`analysis-draft`）：
  - 键角分布：直方图与使用邻居表时完全相同，100 万原子 0.25 → 0.18 s。
  - 原子温度：结果与使用邻居表时差 ≤ 1e-15（只是求和顺序不同），0.18 → 0.11 s。
  - 至此，RDF、熵、配位数、团簇、键角、原子温度和 Voronoi 在不传邻居表时都不存表。
- **Steinhardt** 的球谐归一化系数改为预先算好查表，结果逐位相同，多个阶数时快约 20%。CHILL+ 用同一套球谐，也一起变快。
- **全面验证**：
  - 随机一致性测试：60 个随机体系，包括各种斜盒子、混合周期、截断大于半个盒子、原子在盒外。格子算法与邻居表算法的结果全部一致，Voronoi 体积和等于盒子体积；80 组随机位移下，最小化复用的邻居表与重新搜索得到的逐位相同。
  - 每个分支单独重跑了全部测试和逐文件覆盖率门槛，全部通过。
  - 从源码打 wheel，装进全新的虚拟环境，冒烟测试通过。

### 04:52
- **新草稿分支 `ase-calculator-draft`**：对应待定项"ASE 计算器适配器要不要做"，先做出来给你看。`to_ase_calculator(EAM(...))` 让任何 mdapy 计算器都能当 ASE 计算器用：能量、力、应力（eV/Å³，ASE 的分量顺序）、逐原子能量和应力，可以直接用 ASE 的 BFGS、`FrechetCellFilter` 和 MD。测试结果：
  - 与 mdapy 直接计算的结果相同；
  - 应力与 ASE 的有限差分一致到 1e-8 eV/Å³；
  - ASE 的 BFGS 弛豫后能量与 `mdapy.minimize` 一致到 1e-8 eV；
  - ASE 晶胞弛豫后应力为零。
  - 覆盖率 100%，CI 通过。
- **`mdapy.transform` 的设计草案**（本地 design 目录，不进 git）：replicate、wrap、unwrap 的 id、molecule、image 和拓扑处理，照 LAMMPS `replicate` 的做法（按 image 摆放原子，这样拓扑只需加偏移），列了 6 个需要你定的点。明早一起看。
- `cnp-draft` 的公共近邻参数也改成不建邻居表；文档的分析总览页写明了哪些分析在格子上直接算。

### 05:14
- **又有两项不再建邻居表**，各放在单独的草稿分支，方便取舍：
  - `steinhardt-grid-draft`：Steinhardt 按截断找邻居、且不传邻居表时，直接在格子上算。100 万原子 3 Å：0.23 → 0.17 s，峰值内存 1126 → 302 MB。与使用邻居表时差 ≤ 1e-12，只是求和顺序不同。
  - `entropy-grid-draft`（基于上一个）：熵指纹的邻居平均（`average_cutoff`）也在格子上算，用的是一个通用核，可对任意逐原子量求邻居平均。100 万原子 5 Å + 4 Å 平均：0.56 → 0.30 s，峰值内存 1.6 → 0.16 GB。
  - 两个分支都有测试，包括斜盒子、非周期方向、截断大于盒子三种情况，以及 TSan 线程检查；文档已更新；已合进 `preview-all`。
- **发现一个问题，明早一起定**：进度回调 `progress(done, total)` 的文档写着 done 不会减少、结束时只报一次 done == total，但有两段并行循环的计算没做到。
  - 邻居搜索、Steinhardt、EAM 都会报两次 done == total；EAM 的 done 还会从 0 重新开始，大体系的 Steinhardt 也会。
  - 后果：GUI 的进度条会走到 100% 后又跳回去。
  - 修法建议已写进明早的总结。这个修改要动的核分布在多条草稿分支上，所以等草稿合并后再在 v2 上一次改完。
  - 今晚新写的熵平均已按这个思路处理，有测试。
- **最后一轮全面检查已做完**：17 个分支逐个重新编译，跑全部测试、逐文件覆盖率门槛、文档严格构建、ruff，并查冲突标记，全部通过。 现在有 1163 个测试。
