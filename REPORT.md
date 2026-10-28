# mdapy 2.0 夜间进展报告（2026-10-28 夜 → 29）

你下班前的安排里和 mdapy 本体有关的部分：

1. 以 OVITO 3.17.1 为准重新对比，补上 MSD、RDF（含分量）、结构因子等还没对比过的分析；速度也对比（在 Triton 上跑）。
2. 第四步还剩的功能：Load trajectory（dump 套用 data 的拓扑）、读取 CIF / PDB、时间平均、轨迹线、配位多面体、任意矢量列的箭头。
3. 你说今晚测试全过的直接合进 v2，合并后的分支删掉。

本地工作单独汇报。

## 早上先看这里

**需要你决定的几件事（10-28 21:00 更新）：**

1. **OUTCAR 未收敛的离子步**：10-23 的记录是“扫描时跳过未收敛步、`unconverged=True` 保留”，10-28 改成文件格式后我实现成“全部读出，每帧带 `info["converged"]`，训练集例子里自己过滤”（文件格式没法按选项在扫描时跳帧）。要不要改回默认跳过？
2. **第 5 步 MLIP 工具进图形界面**的具体做法还没讨论（见下面“第 5 步方案”），我没有动手。
3. **OVITO 有、我们还没有的大功能**（见下面清单），哪些要做？


- 今天已合入 v2 的：弹性常数对称性提速（15 s → 0.03 s）、`transform.affine`、`neighbor.expand_selection`、OVITO 3.17.1 作参考。
- 已合并的远程分支 dislocations、gui-superset、slip-systems、solid-liquid、surface-mesh 已删除；保留 master、night-report、mdapy_old。

## 今晚合入 v2 的提交

| 提交 | 内容 |
|---|---|
| `761ffde` | 弹性常数只取对称旋转，不再逐原子映射：4×4×4 fcc 胞 15 s → 0.03 s |
| `36bf1bc` | `transform.affine`：仿射变换，和 OVITO 的 AffineTransformationModifier 逐项一致 |
| `2c18e1e` | `neighbor.expand_selection`：按截断、最近邻、成键扩展选择，三种模式和 OVITO 一致 |
| `997d6e4` | OVITO 3.17.1 作参考：全部对比测试照样通过，CI 改装 3.17.1 |
| `7f5aa45` | MSD 和结构因子对比 OVITO 3.17.1 的测试，结构因子页写明和 OVITO 的差别 |
| `511ee82` | 性能对比加上 MSD（20 帧 dump 读入 + MSD，对比 OVITO 的 DiffusionModifier） |
| `07c337c` | `render(vectors=...)`：任意矢量列（位移、速度、力……）画箭头，和 OVITO 的 Vector display 一致 |
| `2ac7391` | `read_trajectory`：多文件（`dump.*.lammpstrj` 模式或列表）、`topology=` 让 dump 帧带上 data 文件的键、电荷、分子号（OVITO 的 Load trajectory） |
| `06f48b3` | 读取 CIF（按对称操作展开）和 PDB（原子名、残基、MODEL 多帧、CONECT 键） |
| `3ef6e94` | `analysis.TimeAverage`：任意列、盒子或函数结果的时间平均（OVITO 的 Time averaging） |
| `f9e7bde` | `analysis.trajectory_lines` + `render(lines=...)`：轨迹线 |
| `fb40c6c` | `analysis.CoordinationPolyhedra`：配位多面体（凸包）和体积 |
| `83736f7` | VASP OUTCAR 改为只读文件格式，取代 `mlip.read_outcar` / `outcar_to_xyz` |

## 进展

### 10-28 21:00
- **第 4 步全部完成**（mdapy 本体部分都已合入 v2，每项都和 OVITO 3.17.1 现场对比、有文档页和图）：任意矢量箭头、Load trajectory、CIF/PDB、时间平均、轨迹线、配位多面体；另外 OUTCAR 改成了文件格式。
- **配位多面体**：`CoordinationPolyhedra(cutoff | num_neighbors | bonds=True).compute(system, atoms)`，给出三角网格（`render(mesh=...)` 直接画）、每个三角形属于哪个多面体、各多面体体积。凸包自己写（邻居点很少，枚举三点平面，共面的点合成一个多边形再扇形三角化）。和 OVITO 的 CoordinationPolyhedraModifier 逐个三角形对比一致；立方八面体、立方体、菱形十二面体、八面体的体积和解析值一致到 1e-10。

  ![SrTiO3 的 TiO6 八面体](images/polyhedra.png)

- **OUTCAR 格式**：`mdapy.read_trajectory("OUTCAR")`，每个离子步一帧：`forces` 列、`info["energy"]`（TOTEN）、`info["virial"]`（FORCE on cell 的 Total 行）、`info["converged"]`；VASP 粘在一起的数字（`-10293.69148-10013.02274`）能分开；VASP 6 机器学习力场的 “ML FREE ENERGIE …” 行不算离子步。和 ASE 读同一文件对比一致；本地另用 4 个真实 OUTCAR（MD、弛豫、单点、BEC，来自 GPUMD 的工具目录，没有放进仓库）核对过，元素、坐标、力、晶胞、能量与 ASE 完全一致。`mlip.read_outcar`/`outcar_to_xyz` 删掉，MLIP 的 VASP 页改成用 io 读写的几行代码，VASP 计算器也改走这个格式。
- **Triton 上 OVITO 3.17.1 的速度重测已完成**（分析全组 4 000–200 万原子含 DXA 和 MSD、文件读写全组）。结论和之前基本一致：25.6 万原子以上所有分析都比 OVITO 快，邻居表 6.4–7.6 倍、Voronoi 22–25 倍、DXA 4.6–4.9 倍。
- **发现 MSD 内存问题**：200 万原子 20 帧时 mdapy 用了 9.5 GB（OVITO 1 GB）。原因是基准把 20 帧 System 都放进列表、FFT 一次处理所有原子。已改：基准直接传轨迹（逐帧读）、MSD 的 FFT 按原子分块；本地 25.6 万原子从 1172 MB 降到 645 MB（余下主要是 mmap 读文件的页）。分支 msd-memory，Triton 上只重跑 MSD 的作业在排队，跑完一起更新性能页。

### OVITO 3.17.1 有、我们还没有的（按工作量）
- **小**（可以直接做）：Freeze property（某帧的属性跟着原子走）、Center of mass correction（扣质心漂移）、Hide selected（只隐藏不删）、Find/Select overlapping particles、Color by type（按任意类型属性着色，如分子号）、Remove property、Reduce property（属性汇总成全局量）、Edit simulation cell（改盒子和周期性）、Bond length distribution。
- **中**：Smooth / Interpolate trajectory（时间窗平滑、插帧）、Combine datasets（并入另一个文件的原子）、Difference between frames、Text labels（原子上的文字）、Spatial correlation function、Find rings（环统计）、Bond order。
- **大**（需要你决定）：Elastic strain（基于 PTM/CNA 取向的晶格弹性应变）、Create isosurface 加体素网格显示（3D 分箱后的等值面）、VoroTop、Bader 电荷积分（要 CHGCAR）、蛋白质 Cartoon、Cycles/VisRTX/ANARI 渲染器（我们已有自己的光线追踪）。

### 第 5 步方案（MLIP 工具进图形界面，待讨论）
mdapy.mlip 现有：NEP 描述符、最远点采样、主成分、扰动结构。图形界面的想法：
1. “选帧”对话框：选 NEP 势文件，对当前轨迹（可再加已有训练集）算描述符，最远点采样选 N 帧，Tables 里画主成分散点（训练集/候选/选中三种颜色），导出 extxyz。
2. “扰动”建模步骤：对当前结构做 N 次随机位移加应变，导出为训练候选。
3. OUTCAR 已能直接打开，导出 extxyz 已有。
要你定：做成对话框还是修改器；描述符需要用户给 NEP 势文件。

### 10-28 20:25
- **CIF / PDB 读取**（已合入 v2，只读）：
  - CIF：自己写的解析器（引号、`;` 文本块、注释、`5.4307(2)` 这种带不确定度的数、CIF2 的点号标签都支持），晶胞参数转成 a 沿 x、b 在 xy 面的矩阵，对称操作（如 `-x+1/2,y,z+1/2`）自己解析、不用 eval，每个位点的像折回晶胞、0.01 Å 内去重。和 pymatgen、ASE、OVITO 读同一批文件对比 6 种晶体（NaCl、金红石、石英、刚玉、单斜 ZrO2、三斜）：原子和晶胞一致，位置和顺序与 ASE 一致。
  - PDB：CRYST1、ATOM/HETATM（id、元素、原子名、残基名和编号）、MODEL 多帧、CONECT 成键。和 ASE、OVITO 对比一致；唯一差别：OVITO 还会按距离补键，我们只取 CONECT（文档里写了 `covalent_bonds` 可以补）。
  - 两个格式页各一张图：读进来的金红石（3×3×4）、PDB 文本里的乙醇。

  ![CIF 读进来的金红石](images/rutile-cif.png)

- **时间平均**（已合入 v2）：`TimeAverage("positions").compute(trajectory)`，也可以平均任意列、`"box"`，或者一个函数（比如每帧的 g(r)）。按 id 对应原子、位置自动 unwrap（OVITO 要求先手动 unwrap、按下标对应）。和 OVITO 的 TimeAveragingModifier 现场对比：平均位置 1e-9、RDF 表 1e-10、时间步属性一致。文档例子：热振动的铜，单帧 CNA 只认出一半 fcc，平均 20 帧后全部 fcc：

  ![时间平均](images/time-average.png)

- **轨迹线**（分支 trajectory-lines，全量测试在跑）：`analysis.trajectory_lines(frames, atoms)` 给 (M, T, 3) 路径，`render(lines=..., wrap_lines=True)` 画出来。OVITO 的线宽 0.2 实测是直径（我们默认半径 0.1）。对比时发现 **OVITO 生成轨迹线不用 image 标志，只按最近像跟踪**，原子一帧跳超过半个盒子时它的线是错的；我们用 image 标志，和真实路径一致到 1e-15。

  ![轨迹线](images/trajectory-lines.png)

- 全量测试里遇到一次偶发失败：读 mdapy 文件时 `np.linalg.det` 对一个有限矩阵报了 “invalid value” 警告（单独跑通过），像是 macOS Accelerate 的伪浮点异常。重跑确认中，如果复现我再查。
- Triton 上 OVITO 3.17.1 的速度重测已经开始跑（作业 21268016，运行 14 分钟）。
### 10-28 20:00
- **任意矢量箭头**（第四步第一项，已合入 v2）：`render(system, vectors="displacement" 或 (N,3) 数组, vector_scale, vector_radius, vector_color（一种颜色或 "atom"）, vector_alignment（base/center/head）)`。
  - 和 OVITO 3.17.1 的 VectorVis 现场对比（同一相机、OpenGL 渲染器、Normal 着色）：三种对齐、反向、不同宽度，箭头轮廓像素重合 99%。
  - 量出来 OVITO 的 `width` 其实是箭杆**半径**（不是直径）：箭头半径 2.5w、长 4.5w，比箭头短的箭只画按比例缩小的圆锥。我们的默认值照搬（半径 0.5 Å）。OVITO 默认是 Flat 着色（总朝向观察者的扁平箭头），我们画立体箭头，对应它的 Normal 着色，文档写明。
  - 文档 Rendering 页加了 Vectors 一节，例子是刃位错周围的位移场：

  ![刃位错的位移箭头](images/vectors-edge-dislocation.png)

- **Load trajectory**（已合入 v2）：`mdapy.read_trajectory(path, topology=data)`。
  - `path` 可以是 `dump.*.lammpstrj` 这样的模式（按文件名里的数字排序，dump.10 在 dump.9 后面）或文件列表；一个文件里多帧也行；最多同时打开两个文件，几千个文件也没问题。
  - 给了 `topology`：每帧按 id 排成 data 文件的原子顺序，带上它的键、角、电荷、分子号等 dump 里没有的列；dump 缺的原子连同它们的键、角一起去掉并重编号；盒子用帧的、周期性用 data 的；帧里原子齐全时共用同一个 Topology 对象（不复制）。
  - 和 OVITO 的 LoadTrajectoryModifier 现场逐帧对比（打乱顺序、缺一个原子、盒子变化）：id 顺序、坐标、电荷、分子号、键完全一致。唯一差别：帧里有 data 没有的原子时 OVITO 追加，我们报错（写进文档）。
- 全量测试每次合并前都跑：2432 通过，覆盖率达标。
- Triton 上 OVITO 3.17.1 的速度重测作业还在排队（Priority）。

### 10-28 19:30
- **MSD**：和 OVITO 的 `DiffusionModifier` 完全一致（逐个时间窗、x/y/z 分量、有无周期翻越都比了）。本地小规模速度：4 000 原子 20 帧 mdapy 0.019 s，OVITO 0.022 s；32 000 原子 0.089 s 对 0.111 s（含读文件）。
- **结构因子**：OVITO 的两种模式和我们定义不同，测试里按它的定义比对，结果一致：
  - 直接法只对 h,k,l ≥ 0 的八分之一倒空间取平均，我们对全部 k 取平均；
  - Debye 法多了均匀密度项 1 + (4πρ/k)∫ r sin(kr) g(r) dr，我们给的是去掉这一项的结果（和定义一致），差在 1e-3 以内。
  - 这两点写进了结构因子文档的 Definition。
- 全量 2423 个测试通过，覆盖率达标，已合入 v2 并删除分支 ovito-msd、bench-msd。
- Triton 上已提交 OVITO 3.17.1 的速度重测（作业 21268016，在排队）：分析全组 4 000 到 200 万原子（含 MSD），文件读写全组。跑完后更新性能页的表和图。

### 10-28 19:20
- OVITO 升级到 3.17.1，全量 2416 个测试通过（所有 OVITO 对比都一致），CI 改装 3.17.1，已合入 v2。
- 3.17.1 里有、我们还没对比的：`DiffusionModifier`（MSD）、`RadialDistributionFunctionModifier`（RDF 及分量）、`StructureFactorModifier`、`WrapPeriodicImagesModifier`、`ReplicateModifier`。接下来补这些对比。
- 性能页里的 OVITO 计时还是 3.17.0 的，之后在 Triton 上用 3.17.1 重测。
