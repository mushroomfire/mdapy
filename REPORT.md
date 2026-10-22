# mdapy 2.0 夜间进展报告（2026-10-22 → 23）

本分支只放报告和渲染图，不含代码。今晚的代码都在各自的新分支上，明早检查后再合入 v2。

## 今晚的分支

| 分支 | 内容 | 状态 |
|---|---|---|
| `v2` | 11a 晶体建模已合入；atomsk 进 CI；删除 `overrides/` | 已推送 |
| `build-defects-draft` | 位错、点缺陷、层错、表面、SQS 合金、多晶与孪晶（11b/11c） | 已推送，待检查 |
| `defect-energies-draft` | 层错能（GSFE）、表面能、空位形成能（simulation），基于 build-defects-draft | 已推送，待检查 |
| `visualization-draft` | 渲染器与画图模板，基于 defect-energies-draft | 已推送，待检查 |
| `mlip-draft` | OUTCAR → xyz、NEP 描述符 PCA、最远点采样、扰动，基于 visualization-draft | 已推送，待检查 |
| `performance-draft` | 与 OVITO 的性能对比框架和结果；CNA（固定截断）与 Steinhardt 提速，基于 mlip-draft | 已推送，待检查 |
| `tutorials-draft` | 五篇进阶教程；System 输入数组复制的修复，基于 performance-draft | 已推送，待检查 |
| `neighbor-reuse-draft` | 读邻居表的路径改成 C++ 并行内核（问题 7 的实现），基于 tutorials-draft | 已推送，待检查 |
| `fused-nearest-draft` | 近邻类分析边找边算；原子应变和网格提速，基于 neighbor-reuse-draft | 已推送，CI 全绿 |
| `render-draft` | photo 风格、有机分子渲染、完整元素表、Metal GPU 渲染，基于 fused-nearest-draft | 已推送，CI 全绿（含 macOS 上的 Metal 测试） |
| `system-immutable-draft` | System/Topology 的表对外只给拷贝，基于 fused-nearest-draft | 已推送，CI 全绿 |
| `dxa-draft` | 位错分析（位错线、Burgers 矢量、扩展位错与不全位错）和位错线渲染，基于 render-draft | 已推送，CI 全绿 |
| `dxa-partials-draft` | 不全位错的 Burgers 矢量（Shockley、Frank 等）；面缺陷的判据更严，基于 dxa-draft | 已推送，CI 全绿 |
| `neighbor-fits-draft` | 合并 system-immutable-draft（含原子应变/网格提速）；邻居表按“坐标和盒子相同”复用（`NeighborList.fits`）；最新的 OVITO 性能对比，基于 dxa-partials-draft | 已推送 |
| `dxa-network-draft` | 位错网络（节点、线段）、扩展位错只给两条不全位错、Burgers 矢量箭头、和 OVITO 的 DXA 现场对比，基于 neighbor-fits-draft | 已推送，CI 中 |

## 进展

### 20:40
- 11a（晶体 + 随机合金）已合入 v2（cfb7a38），builder-draft 已删除。
- CI 的 coverage 任务现在会编译 atomsk Beta-0.13.1（带缓存）并检查版本，建模部分和 atomsk 的现场对比在 CI 里也会跑。
- 按你的要求删除了 `overrides/`：唯一的文件是 Simulation 卡片的图标，移到 `docs/.icons/`，主题不再需要 `custom_dir`（3eec8a6）。
- 接下来：在 `build-defects-draft` 上做位错（各向同性 Volterra，对照 atomsk `-disloc`）。

### 20:55
- `build-defects-draft`（3b18f7e，基于 v2）完成并推送，1586 个测试全过，`mdapy.build` 每个文件覆盖率 ≥ 97%：
  - **位错** `insert_dislocation(system, burgers, *, line, normal, center, poisson)`：各向同性螺型/刃型/混合，和 atomsk `-dislocation` 现场对比到 1e-9（5 例，含 b 正负、不同取向）。注意 atomsk 的刃型割面在 x1 > 0 半平面并有 ±b/2 的预/后平移，我照它做了；它的 p1 p2 按 X→Y→Z 轮换取分量（ξ=y 时 p1 是 z）。
  - **两块晶体法刃位错** `edge_dislocation(cell, repeat)`（1.x 的 fcc 不全位错对推广到任意晶胞）：用 Zhou EAM 弛豫后铜里分解成两根 Shockley 不全位错，夹着两层 hcp 的层错带，宽约 2.6 nm（测试里检查）。
  - **点缺陷** `add_vacancies` / `add_interstitials`（随机且保证最小距离，或给定位置）/ `substitute`。
  - **层错** `slip(system, displacement, *, height, axis)`：上半部滑移，周期方向盒子矢量一起倾斜，盒子里只有一个层错面；fcc 内禀层错 CNA 正好两层 hcp。
  - **表面** `add_vacuum(system, thickness, *, axis)`。
  - **SQS** `sqs_alloy(system, fractions, *, shells=3, ...)`：C++ 模拟退火，增量更新各壳层的对计数；256 原子 fcc 二元合金三个壳层的 Warren–Cowley 参数都降到 1e-12 以内（50000 步约 0.03 s）。
  - **多晶** `build_polycrystal(unit, box, grains, ...)`：unit 是任意周期体系；盒子某方向不长于 unit 时自动二维柱状（种子在一个平面上，晶粒只绕该轴转）；`twin_lamellae(cell, layers)` 生成周期孪晶片层（镜面在原子面上，CNA 每个界面一层 hcp），作为 unit 就是孪晶多晶。
  - 文档：Structure builder 下新增 4 页（SQS、点缺陷/层错/表面、位错、多晶与孪晶）。
- atomsk 已进 CI：v2 的 CI（3eec8a6）里 atomsk 从源码编译成功并缓存，建模对比测试在 CI 里实际运行（没有被跳过），9 个任务全绿。
- 发现并修复：pytest 默认的 `norecursedirs` 含 `build`，`tests/build` 之前一直没被收集（已在 v2 修好）。

### 21:20
- `defect-energies-draft`（f5b66b1）推送：`vacancy_formation_energy`、`surface_energy`、`stacking_fault_energy`（广义层错能曲线，只沿法向弛豫）。`minimize` 现在认 `movable` 列（VASP 的 selective dynamics 那一列，False 的方向不动）。
  - 与 LAMMPS 现场对比（同一张 Zhou Cu EAM 表）：空位形成能、(111) 表面能、GSFE 曲线都一致到 1e-6。注意要用 LAMMPS 的 `min_style fire`：CG 会在 slab 上走到另一个能量低 3e-4 eV 的极小点。
  - Zhou Cu 的数值：E_v = 1.275 eV，γ(100) = 1.56 J/m²，γ_isf ≈ 26 mJ/m²，γ_us ≈ 119 mJ/m²。
- `visualization-draft`（83254e8 + 7870a46）推送：**渲染器**。
  - C++ 光线追踪：球（原子）/ 圆柱（键、盒子边）/ 三角形（Voronoi 面，可半透明），SAH 分箱 BVH；环境光遮蔽、主光阴影、补光、Blinn–Phong 高光、按深度跳变的描边、超采样抗锯齿、sRGB 输出；结果与线程数无关（TSan 下检查）。
  - 6 种风格：studio（默认）、goodsell（平涂 + AO + 描边）、glossy、toon（卡通分层）、flat（每个原子描边）、preview（最快）。
  - `render(system, ...)`：按元素 / 结构类型（CNA 等结果直接传）/ 任意列或数值（viridis、plasma、coolwarm、jet、gray）/ 手动颜色着色；键（截断或邻居表，跨周期边界只画本原子那半段，像 OVITO）；盒子；Voronoi 多面体（按原子或按面的边数着色，带棱线）。`Scene` 可以自己加球、圆柱、三角形、多边形。
  - **性能**：100 万原子、1920×1080、带 AO 和阴影 1.4 s，峰值内存约 1 GB（本机 24 GB）；6 万原子 0.4 s。CPU 版已经很快，Metal/CUDA 暂时没做，先把 CPU 版做好。
  - 测试：求交与 NumPy 暴力参考逐条一致；球的投影面积与解析值一致（3%）；遮挡、半透明合成、描边位置、取景居中、透视近大远小、PNG 往返、进度与取消。
  - 文档：新的 Visualization 标签下 Rendering 一页（带 6 种风格对比图）。

#### 渲染效果图（点开看大图）

| | |
|---|---|
| 6 种风格：studio / goodsell / glossy（上）；toon / flat / preview（下） | ![](images/00_styles.png) |
| 弛豫后的铜多晶，CNA 着色（studio） | ![](images/01_polycrystal_cna_studio.png) |
| 多晶中间 16 Å 的薄片，看晶界网络（goodsell） | ![](images/02_polycrystal_boundaries_goodsell.png) |
| 按晶粒编号的 viridis 着色（glossy） | ![](images/03_polycrystal_grains_glossy.png) |
| 孪晶多晶：只显示非 fcc 原子，红色是孪晶界 | ![](images/04_twinned_polycrystal.png) |
| 二维柱状多晶，种子在六边形格点上（俯视） | ![](images/05_columnar_hexagonal.png) |
| 铜的刃位错弛豫后分解成两根 Shockley 不全位错，中间是层错带（studio） | ![](images/06_dislocation_studio.png) |
| 同上（goodsell） | ![](images/07_dislocation_goodsell.png) |
| bcc 的 Voronoi 胞：截角八面体，六边形面绿、正方形面蓝，带棱线 | ![](images/08_voronoi_bcc.png) |
| 五元 Cantor 合金 SQS（glossy） | ![](images/09_sqs_cantor_glossy.png) |
| 金纳米颗粒，透视 + 雾化（glossy） | ![](images/10_gold_nanoparticle.png) |
| h-BN 球棍模型（跨边界的键只画半段） | ![](images/11_hbn_ball_and_stick.png) |
| 97.6 万原子的多晶，1920×1080，1.4 s | ![](images/12_million_atoms.png) |

### 21:35
- `visualization-draft` 又推了 8312cb5：**画图模板** `mdapy.visualization.plot`。参考了之前的期刊模板：Nature 美术规范（5–7 pt 无衬线、细线、去上右边框、字体嵌入）、单栏 88 mm / 双栏 180 mm、Okabe–Ito 色盲友好配色、面板标签 a/b/c、一次存 PDF/SVG/PNG；结果直接画：parity（RMSE 写进图例，可选密度图）、g(r)（含分偏）、S(q)、键角分布、MSD（可拟合出扩散系数）、GSFE 曲线、结构类型计数。文档 Visualization → Plots。
- `mlip-draft`（42bd0ca）推送，并在 GitHub 上手动触发了这个分支的 CI（它包含前面三个草稿分支的全部内容，正好一起检查）：
  - **VASP → 训练集** `read_outcar(path, *, unconverged=False)`：单点、relax、MD 每个离子步一个 System（forces 列，info 里 energy = TOTEN、virial、converged），默认丢掉电子步没收敛的离子步。`outcar_to_xyz(outcars, path, *, unconverged=False, max_force=None)` 多个 OUTCAR 合成一个 extxyz，可以按最大力过滤。和 ASE 对比：本地 4 个真实 OUTCAR 和测试里生成的 OUTCAR 都逐位一致（能量、力、坐标、晶格、应力）。
  - **选帧** `structure_descriptors(frames, calculator)`（每帧原子描述符的平均）、`farthest_point_sampling(points, count, *, start=0, selected=None)`（可以给已有训练集，挑离它们最远的新帧）、`principal_components(points, n=2)`（带 `transform`，新点投影到同一组主成分上）。测试：PCA 与 SVD 一致到 1e-10，FPS 与暴力逐个挑选一致。
  - **扰动** `perturb(system, count, *, displacement=0.05, strain=0.03, style="normal"|"uniform"|"const", seed)`：对称应变 + 原子位移（参考 dpdata 的 perturb：normal / uniform 球内 / 定长三种），会去掉原来的能量、力、维里。
  - 文档：新的 **MLIP** 标签（三页），导航顺序按约定改成 Simulation → MLIP → Visualization；首页地图里 MLIP 改成实线（已完成）。
- 下图用 GPUMD 自带的 UNEP-v1 模型：四组扰动结构（fcc Cu 小/大位移、CuNi 合金、bcc Cu）的结构描述符做 PCA，黑圈是最远点采样挑出的 24 帧，能看到它优先挑边缘和稀疏区域。

| | |
|---|---|
| 画图模板：g(r) 分偏、parity、MSD 拟合、GSFE（双栏） | ![](images/13_plot_templates.png) |
| UNEP-v1 描述符的 PCA + 最远点采样 | ![](images/14_mlip_pca.png) |

- 接下来：`performance-draft`，和 OVITO 3.17 的性能对比（只比结果一致的功能；规模逐级加大，盯着内存，峰值控制在 8 GB 以内）。

### 22:05
- `mlip-draft` 的 CI 9 个任务全绿（这个分支包含前面三个草稿分支的全部内容）。
- `performance-draft`（aff447f + 4f159ec）推送，CI 已手动触发：
  - **对比框架** `benchmarks/ovito_comparison.py`：每个功能是一个 `Case`（怎么造体系、mdapy 怎么调、OVITO 怎么调、怎么比结果）。每个尺寸先在一个进程里两边都算一遍、逐原子比对（结构类型、团簇大小、占据数、直方图完全相同，浮点 1e-10），一致了才计时；计时时两边各自在**独立的新进程**里跑（同样的线程数，先用 500 原子预热，取三次最短），内存看门狗每 50 ms 查一次 RSS，超过 8 GB 直接杀掉，并按上一尺寸的峰值预测下一尺寸，超限就跳过；单进程还有超时（OVITO 的 Voronoi-Steinhardt 在 200 万原子时单次超过 5 分钟，被停掉）。结果逐行写 JSON，`table`/`plot` 子命令出表和图。单个进程的峰值内存最高 4.5 GB（OVITO 给 200 万原子建 5 Å 邻居表），低于 8 GB 的上限，本机没有卡顿。
  - **结果**（M5，10 线程，15 个功能，4 千到 200 万原子，全部逐原子一致）：大多数功能两边相差 30% 以内，mdapy 略快；Voronoi 快 5 倍，Voronoi 面积加权 Steinhardt 快 20–44 倍，团簇分析快 2.8 倍，键角分布快 1.6 倍；小体系 mdapy 快 2–6 倍（OVITO 管线每次有几毫秒固定开销）。
  - **OVITO 更快的地方**：一开始固定截断 CNA 慢 2 倍、Steinhardt 慢 1.3 倍，已优化：CNA 固定截断改成在网格上边找邻居边算（不建邻居表和向量数组），时间 −40%、内存只有原来的 1/8，现在 0.21 s 对 OVITO 0.185 s；Steinhardt 各阶球谐一次递推（运算顺序不变，结果逐位相同），不做平均时不再存所有原子的 qlm（2M×35 的大数组），给定的近邻表不再重排，现在 0.56 s 对 0.54 s。剩下**原子应变 OVITO 快 4.4 倍**：mdapy 计时里包含参考构型建邻居表（后续帧复用），单帧计算 0.14 s 对 OVITO 0.08 s。
  - **内存是 mdapy 的弱项**：mdapy 先建完整邻居表再算，基于近邻的分析（自适应 CNA、AJA、CSP、Steinhardt）在 200 万原子时多占约 1 GB，OVITO 边找边算只要 0.15 GB；固定截断 CNA 改成边找边算后两边持平。见下面的问题 4。

| | |
|---|---|
| OVITO 时间 / mdapy 时间（大于 1 表示 mdapy 快），4 个尺寸 | ![](images/15_ovito_comparison.png) |

- **发现一个不可变性的漏洞**：`System(positions)`、`with_columns(q=array)`、`with_positions(array)` 都是零拷贝把 NumPy 数组放进 polars 表，之后改原数组会悄悄改掉“不可变”的 System（也会让已经建好的邻居表失效而不报错）。`info` 本来就复制，列却没复制。我会在 `tutorials-draft` 里单独提交一个修复：NumPy 输入在进入表时复制一次（读取仍然零拷贝）。见问题 5。
- 接下来：`tutorials-draft`，五篇进阶教程（fcc+bcc 双相多晶、六边形柱状多晶、铜的缺陷能、邻居表复用、System 共享与复制），代码都会在本地跑通（docs 测试会执行页面里的每段代码）。

### 22:18
- `performance-draft` 的第一次 CI 只有 coverage 失败：Linux 上光导入 OVITO 进程就占 1 GB 多，内存预测把这部分也按原子数放大，把 2000 原子的检查跳过了。已修（604739f）：只放大随原子数增长的部分；重新跑 CI 中。
- `tutorials-draft`（aa250fa + 5653f48）推送，CI 已触发：
  - **System 复制修复**（aa250fa，单独一个提交）：`System(...)`、`with_columns`、`with_positions` 的 NumPy 输入复制一次，`info` 本来就复制；读取仍零拷贝。加了测试。
  - **五篇教程**（Getting started → Tutorials），页面里每段代码都由 docs 测试执行，全部五篇约 10 s：
    1. **fcc + bcc 双相多晶**：同样的种子和取向各建一次 Cu（fcc）和 Ta（bcc）多晶，按晶粒拼起来，删掉相界处过近的原子，CNA + polars `group_by` 统计每个晶粒的结构。原本想用 Fe 的奥氏体+铁素体，但 Zhou 的 Fe 势在 0 K 下 fcc 不稳定，一弛豫全变 bcc；Cu–Ta 用 Zhou 的通用交叉项弛豫后界面大面积无序，所以教程只建模不弛豫，并提示要用专门拟合的势。
    2. **六边形柱状多晶**：种子在三角格子上（12 个晶粒，每个 40 Å），[111] 柱轴，手选转角，EAM 弛豫，用邻居表找出相邻晶粒对并算取向差（模 60°），小角晶界显示为一串分立的位错芯。
    3. **铜的缺陷能**：先弛豫晶格常数；空位能随超胞收敛（32→500 原子：1.301→1.273 eV）；三个低指数面的表面能；GSFE 两个方向。这里发现了一个问题，见问题 6。
    4. **邻居表复用**：近邻类分析共享一张 14 近邻表快 1.5 倍；Voronoi 邻居共享快 2.7 倍；但**截断类分析给了邻居表反而慢 4–5 倍**（见问题 7），教程如实写了“不要给它们传表”。
    5. **System 共享与复制**：什么零拷贝、什么共享、什么复制，用 `np.shares_memory` 逐条演示；邻居表只认原来那个 System 对象。

| | |
|---|---|
| Cu/Ta 双相多晶（剖开一半）：按结构 | ![](images/16_cu_ta_composite.png) |
| 同上：按元素 | ![](images/17_cu_ta_elements.png) |
| 六边形柱状多晶（俯视，弛豫后）：按结构，小角晶界是分立的位错芯 | ![](images/18_columnar_cna.png) |
| 同上：按晶粒 | ![](images/19_columnar_grains.png) |
| 铜的 GSFE（自由表面 slab，18 层） | ![](images/20_gsfe_copper.png) |

### 22:35
- `performance-draft` 修好后 CI 9 个任务全绿；`tutorials-draft` 的 CI 除了 py3.11 还在跑，其余全绿。
- 问题 7 我先做了实现，放在 `neighbor-reuse-draft`（0b2960a + d0272b4），不改接口、结果逐位不变，你可以早上决定要不要：
  - `NeighborList.within` 和按元素对截断的成键选择改成 C++ 并行压缩（原来是单线程 NumPy 掩码）；RDF、配位数读表时用 C++ 并行内核，直接跳过超出截断的对。
  - 10.8 万原子：RDF 读表 41 → 1.6 ms，配位数 51 → 1.1 ms，固定截断 CNA 读表 50 → 8 ms。教程第 4 篇相应改了：截断类分析共享一张同截断的表现在也略快（42 → 34 ms）。
  - CI 第一次 TSan 任务失败：thread_checks.cpp 里新变量与外层同名，Linux 的 GCC `-Wshadow -Werror` 报错（本地只用 GCC 检查了源文件，没检查这个测试文件）。已改名重推，并把这条检查加进了本地 CLAUDE.md 的命令。
- 问题 4 也先做了实现，放在 `fused-nearest-draft`：k-d 树加了 `for_each_nearest`（`nearest_atoms` 也改用它，顺序完全一样），自适应 CNA、Ackland–Jones、CSP（k ≤ 14）、不做平均的近邻 Steinhardt 在不传邻居表时边找边算。和读表路径逐位一致（测试覆盖周期/三斜/团簇三种盒子，加 TSan 检查），和 OVITO 的逐原子对比测试照样全过。200 万原子：
  - 自适应 CNA 0.534 → 0.48 s，内存 1154 → 87 MB（OVITO 0.66 s、139 MB）；
  - Ackland–Jones 0.606 → 0.54 s，1151 → 83 MB（OVITO 0.65 s、147 MB）；
  - CSP 0.477 → 0.42 s，1001 → 78 MB（OVITO 0.52 s、111 MB）；
  - Steinhardt 0.563 → 0.52 s，1077 → 160 MB（OVITO 0.54 s、187 MB）。
  - 现在正在用这个分支的代码把 15 个功能全部重跑一遍，跑完更新 Performance 页的表和图再提交。

### 23:14（你 22:44 的决定之后）
- 按你的决定在做：邻居（给表直接并行用、没给表边找边算、AtomicStrain 提速）→ System 严格不可变 → 渲染（分子、元素表、GPU）→ DXA → 自己找改进点。都在分支里，不合并。
- **AtomicStrain**（fused-nearest-draft，7ba5a72）：
  - 构造函数不再自动建表，加了可选 `neighbors=`（参考构型的表，多帧时建一次复用）；没给表时每次 compute 在参考构型网格上边找边算；位移、应变张量、剪切、体应变、D²min 全部在一个 C++ 并行内核里（原来 NumPy 里要 85 ms）。
  - 200 万原子：0.37 s → 约 0.15 s（OVITO 0.074 s）；给表时只要 0.05 s（比 OVITO 快）。
  - 和 OVITO 的逐原子对比、与 NumPy 最小二乘的对比、给表/不给表一致的测试都过了。
- **底层网格**（所有分析都受益）：单元编号改成 x 最快（和 build_crystal、LAMMPS 的原子顺序一致，访存连续）、计数排序并行化、每个槽位的数据压进一条缓存行、去掉整数除法。建网格约快 2 倍、扫描约快 25%。
- **给表一定更快**：现在所有分析给表都不比单独算慢：RDF 11 → 1.5 ms、配位数 6 → 1.2、截断 Steinhardt 18 → 10、熵 13 → 8、原子温度 10 → 7（10.8 万原子）；近邻类分析给 14 近邻表：CNA 26 → 4.7、AJA 28 → 7.6。
- 和 OVITO 再比（200 万原子）：固定截断 CNA 现在快 1.1 倍、RDF 1.5 倍、团簇 2.9 倍；原子应变还慢 1.5–2 倍。
- **为什么原子应变还慢**：我读了 OVITO 的 CutoffNeighborFinder 源码并做了对照实验：它的查询非常简单（单元等于截断、27 个单元、只用盒内坐标比距离），在完美晶格上分支预测很规整，单线程每次查询约 55 ns；我们的网格要保证"与列表逐位相同的精确距离"，每个原子约 100 ns。基准里原子应变的参考构型恰好是完美晶格，对它最有利。在有热扰动的晶格上两边都慢一倍。再往下压需要改网格的整体设计，我先记下，不在今晚动。
- 另外：机器连续跑了几个小时，计时有明显降频漂移（同一测试从 114 ms 漂到 128 ms），报告里的数字有 ±15% 的误差。

### 23:37
- `fused-nearest-draft`（含原子应变和网格提速）CI 9 个任务全绿。
- **System 严格不可变**（`system-immutable-draft`，e827418）：实测发现 `system.table.insert_column(...)`、`replace_column`、`drop_in_place`、改 `columns` 这些 polars 原地方法会直接改掉 System 内部的表。现在 `table` 每次返回一份共享列的浅拷贝（开销只是复制列引用），构造时也存自己的一份；Topology 的四张表同样处理。测试逐个试了 polars 所有原地修改途径（插列、换列、删列、改名、extend、vstack、Series 的 scatter/append、to_numpy 写入），System 都不变。NumPy 输入复制（tutorials-draft 里那个修复）也在这条链上。
- **渲染**（`render-draft`，3088a9c + 2052d0e）：
  - **元素表**：118 个元素全了：OVITO 有的保留 OVITO 的颜色和显示半径，其余用 Jmol 颜色；新增共价半径（Cordero 2008）和范德华半径（Alvarez 2013），缺的用 1.5/2.0 Å。`element_radii(elements, kind="display"|"covalent"|"vdw")`。
  - **有机分子**：`render(..., representation=...)`：`spheres`（默认）、`ball_and_stick`（四分之一范德华半径 + 共价键）、`space_filling`（范德华半径）、`licorice`（甘草棒）；`bonds="covalent"` 按共价半径之和 + 0.45 Å 自动成键（Jmol 的判据），`covalent_bonds(system)` 单独可用。见下图（苯、乙醇、C60）。
  - **photo 风格**：路径追踪（面光源软阴影、天空光、颜色在相邻原子间反射），光强调低了一些。
  - **GPU（Metal）**：`render(..., device="gpu"|"auto")`，`gpu_available()`。着色器逐行对照 CPU 的实现、同样的单精度和同样的随机数，所以和 CPU 的图只在少数边缘像素上差一点舍入（测试里 5 种风格逐像素对比）。着色器在运行时编译，编译 mdapy 不需要 Xcode/Metal 编译器；非 macOS 或 `MDAPY_METAL=OFF` 时没有 GPU，`auto` 自动回落 CPU。按行分块提交，块之间可以取消、报告进度。
  - 同时把 BVH 改成确定性的并行构建（树与线程数无关），元素查表改用 polars 映射（原来 100 万个元素字符串排序要 0.2 s）。
  - **速度**（100 万原子）：1920×1080 studio：CPU 1.4 s → GPU 0.34 s；3840×2160 studio：4.1 s → 1.2 s；photo 1080p：7.1 s → 2.3 s。2.8 万原子 1600×1200、每像素 32×4 条路径的 photo 图 GPU 4.3 s（CPU 14 s）。峰值内存 2.4 GB。
  - 没做 CUDA：本机是 Mac，先把 Metal 做好；CUDA 版可以照着同一个着色器移植，在 Triton 上测。

| | |
|---|---|
| 有机分子：苯、乙醇、C60 × 球棍 / 甘草棒 / 空间填充 | ![](images/27_molecules.png) |
| studio vs photo（GPU）：铜多晶 | ![](images/21_photo_polycrystal.png) |
| glossy vs photo：金纳米颗粒 | ![](images/22_photo_gold.png) |
| studio vs photo：位错 | ![](images/23_photo_dislocation.png) |
| photo 大图 1600×1200（GPU 4.3 s） | ![](images/26_photo_polycrystal_large.png) |

- 接下来：DXA（位错提取与位错线可视化）。

### 00:20
- `render-draft`、`system-immutable-draft` 的 CI 全绿；GitHub 的 macOS 机器上 Metal GPU 测试也跑通了。
- **位错分析**（`dxa-draft`，dfed60d，基于 render-draft）：`DislocationAnalysis(structure, a, orientation=None, *, cutoff=None, spacing=None).compute(system, *, structures=None)`，支持 fcc 和 bcc。没有照搬 OVITO 的 DXA（它的界面网格 + 弹性映射太重），按我的理解做了一个更直接的版本：
  1. CNA 找出非完美晶体原子，去掉非周期方向自由表面附近两层；按截断聚类，每个簇是一个缺陷。
  2. **位错线**：沿一个周期方向贯穿盒子的簇按切片取核心中心，得到穿过周期边界的直线；其余的簇先轻微收缩、体素取点、最小生成树取最长路径，再判断是否成环（把路径中段切掉后两端仍连通就是环）；最后 Taubin 平滑（不会把环缩小）。沿两个以上周期方向贯穿、或原子离线太远的簇是面缺陷，不算位错线。
  3. **Burgers 矢量**：在线上 1/4、1/2、3/4 处，取核心外一圈完美晶体原子按右手方向排序，每一步换成最近的晶格矢量，求和取负（FS/RH 约定，和 `insert_dislocation` 的符号一致）；三个回路投票。回路穿过层错（hcp 原子）说明围住的是不全位错。
  4. **fcc 层错**：簇里有 hcp 原子时，先把非 hcp 原子再聚类：恰好两条 → 扩展位错，整体给出全位错 b、两条不全位错（parent 指向全位错）和扩展宽度；否则是只由不全位错包围的层错（Frank 环、Shockley 环），给出不全位错线，b 记 NaN。
  5. 结果：`lines`（点、是否闭合、周期、b、晶体坐标下的 b、族 `1/2<110>` 等、长度、与 b 的平均夹角、是否不全、parent、宽度、核心原子）、每个原子属于哪条线（列 `dislocation`）、位错密度；`table()` 一行一条线。
- **验证**（21 个测试，单文件覆盖率 97%）：fcc 螺位错 ±b（b 和方向都对）、bcc 螺位错 a/2[111]、晶体绕线旋转后结果不变、薄膜里两端止于表面的开放位错线、弛豫后的扩展刃位错（全位错 a/2[1-10]、90°、两条不全位错、宽 28.7 Å）、用立体角位移场 bΩ/4π 构造的 fcc 和 bcc 剪切位错环（闭合、半径、周长、平均夹角 45°、b 反号时线反向）、Shockley 不全位错环、弛豫后的 Frank 环、贯穿盒子的层错面（无线）、空位/间隙/完美晶体（无线）、参数错误。
- **速度**：31 万原子、5 个环：0.15 s（CNA 另算约 0.06 s）。回路只查附近的原子（简单的网格），原来每个回路扫一遍全部原子。
- **位错线渲染**：`render(..., dislocations=result, dislocation_radius=0.6)`，线画成管子，在周期边界处切断，按 Burgers 矢量族着色（1/2⟨110⟩ 蓝、1/2⟨111⟩ 绿、⟨100⟩ 品红、⟨110⟩ 紫、不全位错橙、其他红）；`line_segments(box, points, ...)` 单独可用。
- 文档：Analysis → Defects and deformation 下新增 Dislocation analysis 页（含下图），术语表、参考文献（Hirth & Lothe、Stukowski & Albe 2010）、API、发布说明都已更新。

| | |
|---|---|
| 左：铜板里的 a/2⟨110⟩ 位错环（蓝，有的穿过周期边界被切开）和一个包围层错（红）的 Shockley 环（橙）；右：弛豫后的扩展刃位错，全位错线在层错带中间，两侧是两条不全位错 | ![](images/28_dxa.png) |
| 只画位错线（photo，GPU） | ![](images/29_dxa_lines.png) |

- 接下来：自己找可以补充和完善的地方，在新分支上做（不合并）。

### 00:32
- `dxa-draft` CI 9 项全绿。
- **不全位错的 Burgers 矢量**（`dxa-partials-draft`，基于 dxa-draft）：这正是上面问题 10 的方向，我先做了，你看是否要。回路绕不全位错一圈时必然跨过层错，层错另一侧的晶体整体错开一个不全位错矢量；所以回路里跨过 hcp 原子、又对不上任何晶格矢量的那一步，改成"最近的晶格矢量 + 一个 Shockley a/6⟨112⟩ 或 Frank a/3⟨111⟩"，取误差最小的。结果：
  - 弛豫的扩展刃位错：a/2[1-10] → a/6[2-1-1] + a/6[1-21]，两条不全位错的 b 之和正好等于全位错，每条与线成 60°（教科书上的分解）。
  - Shockley 环：a/6⟨112⟩，b 的符号与构造时一致（线反向时 b 反号）。
  - 弛豫的 Frank 环：a/3⟨111⟩，垂直于环面。
  - 族名加上 `1/6<112>`（Shockley）、`1/3<111>`（Frank）、`1/6<110>`（压杆）、`1/3<100>`（Hirth）；渲染配色按 OVITO 的习惯：Shockley 绿、Frank 青、压杆紫、Hirth 黄。
- **面缺陷不再当成位错线**：之前用多个立体角环叠加做测试时发现，盒子边界上的错配面会被画出一条线。现在两个判据：核心原子到线段的距离 90% 分位不超过 2.5 个近邻距离（真实的位错核心实测都 ≤ 1.9）；线不能折回贴近自己（沿线相距三个点以上的两点不能靠得比 1.5 个点距还近）。随机圆盘测试被拒绝；所有真实位错照常识别。
- 1705 个测试全过，`_dislocations.py` 覆盖率 97%。

| | |
|---|---|
| 左：Shockley 环现在是绿色（a/6⟨112⟩）；右：扩展刃位错的两条 Shockley 不全位错 | ![](images/31_dxa_partials.png) |
| 弛豫后的 Frank 环（28 Å 空位盘塌缩）：a/3⟨111⟩，青色；0 K 弛豫后环不是平的 | ![](images/30_frank.png) |

### 02:40（你 00:40 的意见）
- **邻居表复用**（`neighbor-fits-draft`）：不再要求是同一个 System 对象。`NeighborList.fits(system)`：坐标和盒子都相同就能用（先比内存地址，不同再逐值比较），所以 `with_columns` 加列、重新读同一帧都能复用；原子动了或盒子变了才拒绝。文档（邻居表页、两篇教程、计算器页）都改了。
- **新的性能对比**（同一分支，Performance 页和下图已更新）：发现 render-draft 是从 fused-nearest-draft 较早的提交分出来的，没带上“原子应变和网格提速”；这次把 system-immutable-draft 合进来后在最终代码上重跑。200 万原子：近邻类分析（自适应 CNA、AJA、CSP、Steinhardt、金刚石）比 OVITO 快 1.2–1.6 倍，内存 80–170 MB（OVITO 130–370 MB）；RDF 1.7 倍、键角 1.9 倍、团簇 3.2 倍、Voronoi 6.5 倍、Voronoi 加权 Steinhardt 40–120 倍；原子应变 0.15 s 对 0.075 s（之前 0.38 s），是唯一还慢的。这次 OVITO 的时间比上次整体慢一点，机器发热有 ±15% 的波动。
- **DXA 重写成网络**（`dxa-network-draft`）：
  - 原来“一个缺陷簇取一条最长路径”处理不了压痕产生的位错网。现在把核心原子收缩后放进格子里建图（原子相邻的格子连边，保证连通），最小生成树 + 补回真实的环（包括穿过周期边界的），剪掉短枝，在三条及以上线相交处切成线段，每段单独做 Burgers 回路。
  - **扩展位错不再画中间的全位错线**：只给两条 Shockley 不全位错，另有 `result.dissociations` 记录成对关系、整体的 Burgers 矢量和宽度。
  - **箭头**：`render(..., dislocations=r, burgers_vectors=True, burgers_scale=3)`，在每条线中点画 Burgers 矢量箭头。
  - 表面原子改成按缺失的近邻数判断（压痕的凹坑也能识别，空洞表面也算），线的自由端会延伸到表面。
  - 回路可以穿过层错和畸变的晶体，两个回路要一致才采纳。
- **和 OVITO 的 DXA 对比**（测试里现场对比）：
  - 孤立位错（fcc/bcc 螺位错、薄膜里的开放线、fcc/bcc 剪切环、Shockley 环、弛豫的扩展刃位错）：类型和 Burgers 矢量完全一致，长度差 <10%。
  - **符号约定**：同样的线方向，OVITO 的 Burgers 矢量和我们正好相反。我核对过，我们的是 Hirth–Lothe 的 FS/RH（也和 insert_dislocation 一致），所以没改，文档里写明。
  - **压痕（LAMMPS，7.2 万原子，10 K，压深 5–17 Å）**：结构大体对得上（见下图，红是 OVITO，蓝是我们），Shockley 为主、压杆和 Hirth 都有；但定量差距明显：OVITO 线长 1113 Å 时，约 45% 在我们这边有同 Burgers 矢量的线贴着，我们的线约 55% 和 OVITO 一致。原因：相距两个近邻距离以内的几条不全位错会被我们合成一条（给出它们的净 Burgers 矢量），贴着压头高畸变区的线回路走不通被丢掉。节点处的 Frank 守恒也大多不满足。弛豫后的“Frank 环”也是这样：OVITO 看到的是 Shockley + 压杆的网络，我们给一个 a/3⟨111⟩ 的环。
  - 要和 OVITO 在密集网络上接近，需要实现 Stukowski 的完整 DXA（四面体剖分、晶格映射、在界面网格上追踪回路），大概要几天。这是下面的问题 12。
- 1724 个测试全过。

| | |
|---|---|
| 新的性能对比（最终代码） | ![](images/34_performance.png) |
| 网络版：左为位错环，右为扩展刃位错的两条 Shockley + 箭头（不再有中间那条全位错） | ![](images/35_dxa_network.png) |
| 压深 13 Å 的压痕：我们的位错线和 Burgers 矢量箭头（绿 Shockley、紫压杆、黄 Hirth、红为不在标准列表里的矢量） | ![](images/32_indent_lines.png) |
| 同一帧：红为 OVITO，蓝为我们（左俯视，右侧视） | ![](images/33_indent_vs_ovito.png) |

- 接下来：把所有草稿合成一个 preview 分支并跑通测试；WCP 多壳层；VASP 计算器。

## 早上需要你决定的事

1. 草稿分支是一条链，合并顺序：build-defects → defect-energies → visualization → mlip（后面的 performance、tutorials 也会接在 mlip 后面）。
2. 渲染的默认风格、配色（元素颜色沿用 OVITO/1.x）、风格名字，你看图后定。
3. OUTCAR 的能量取 `free  energy TOTEN`（与力一致，dpdata 也取它）；ASE 的 `get_potential_energy()` 默认取 `energy(sigma->0)`。要不要加个选项？
4. 性能：要不要把“边找邻居边算”推广到近邻类分析（自适应 CNA、AJA、CSP、Steinhardt、金刚石）？需要一个按原子找 k 近邻的 C++ 接口，内存能从约 0.5 kB/原子降到 OVITO 的水平，时间也会更快。原子应变也可以同样处理。**已在 `fused-nearest-draft` 实现（见 22:35），你决定是否合并**；原子应变和金刚石识别还没做。
5. System 输入数组复制：我的方案是 NumPy 输入复制一次（和 `info` 一致），代价是构造时多一次拷贝（200 万原子的坐标约 10 ms）。另一种是保持零拷贝、只在文档里写明“交给 System 的数组不要再改”。我倾向复制。
6. **GSFE 的周期盒子不收敛**：`stacking_fault_energy` 对沿法向周期的晶体，盒子高度固定，层错附近面间距膨胀被约束，内禀层错能随厚度慢慢下降（9/18/27/36 层：27.3/24.5/23.3/22.6 mJ/m²）；两侧自由表面的 slab 在 18 层就收敛到 22.46。未弛豫的值完全不随厚度变，确认是这个原因。另外默认 `max_force=1e-4` 会让层错能抖动约 0.3 mJ/m²，1e-6 才稳定。建议：周期情况下同时弛豫盒子沿法向的长度（像 LAMMPS 的 `fix box/relax z 0`），并把默认 `max_force` 收紧到 1e-6。这个功能在 `defect-energies-draft` 上，改它要把后面的分支都 rebase，所以今晚没动，等你定。教程里先用自由表面 slab。
7. **截断类分析给了邻居表反而慢**：RDF、配位数、固定截断 CNA 等不传表时在 C++ 里边找边算（多线程），传了表则走单线程 NumPy（掩码、`within` 子集复制、bincount），10.8 万原子上 35 ms 变 160 ms。建议给这些分析的“读表”路径也写 C++ 并行内核，并让它们直接跳过超出截断的对，而不是先 `within` 复制子集。这样复用才总是划算。**已在 `neighbor-reuse-draft` 实现（见 22:35），你决定是否合并。**
8. 网格搜索要不要再做一版"简单快速路径"（像 OVITO 那样只用盒内坐标判断，不保证与列表逐位相同的距离），用于只需要邻居集合、不需要精确距离的分析（如原子应变、计数）？大约还能快 1.5–2 倍，代价是两套代码。
9. 渲染默认 `device`：现在默认 `"cpu"`（结果与线程数无关、逐位可复现），GPU 要显式打开。要不要改成默认 `"auto"`（有 GPU 就用，结果只差舍入）？
10. **位错分析的边界**：不全位错的 Burgers 矢量已在 `dxa-partials-draft` 做了（见 00:32），你决定是否合并。还剩：只支持单晶取向（`orientation` 一个），晶界里的位错和多晶不处理；hcp、金刚石结构没做。要不要继续往这个方向做？
11. **PTM（多面体模板匹配）怎么做**：路线图第 5 阶段有"PTM 及平面缺陷识别"，它给出每个原子的局部取向（四元数）、结构（fcc/hcp/bcc/ico/sc/金刚石/六方金刚石/石墨烯）和 RMSD，是晶粒分割、多晶里的位错分析、层错/孪晶分类的基础。Larsen 的 PTM 库是 MIT 许可（LAMMPS 的 PTM 包和 OVITO 都直接带着它），约 8700 行，其中一半是数据表（模板、自同构、规范图）。两种做法：(A) 按 mdapy 的风格移植它（保留 MIT 许可声明，用我们的线程池），和 OVITO/LAMMPS 能逐原子一致，工作量大约一天；(B) 按论文自己写，凸包 + 图同构 + QCP 都要自己实现，要和 OVITO 逐原子一致很难（凸包拓扑判定、邻居排序、RMSD 归一化的细节都得一样）。我倾向 (A)。你定了我就开始。
12. **DXA 要不要做成完整版**：现在的“精简版”在孤立位错上和 OVITO 一致，在压痕这种密集网络里大约只有一半能对上（见 02:40）。完整的 DXA（Delaunay 四面体剖分 → 用晶格映射给每条边标理想矢量 → 好/坏四面体之间的界面网格 → 在网格上追踪回路并沿线推进）能做到和 OVITO 接近，但需要四面体剖分（我们有 Voronoi，可以从它导出 Delaunay）和较大的工作量（估计 2–3 天）。你觉得值得的话我就开始。
