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
| `fused-nearest-draft` | 近邻类分析边找边算（问题 4 的实现），基于 neighbor-reuse-draft | 进行中 |

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

## 早上需要你决定的事

1. 草稿分支是一条链，合并顺序：build-defects → defect-energies → visualization → mlip（后面的 performance、tutorials 也会接在 mlip 后面）。
2. 渲染的默认风格、配色（元素颜色沿用 OVITO/1.x）、风格名字，你看图后定。
3. OUTCAR 的能量取 `free  energy TOTEN`（与力一致，dpdata 也取它）；ASE 的 `get_potential_energy()` 默认取 `energy(sigma->0)`。要不要加个选项？
4. 性能：要不要把“边找邻居边算”推广到近邻类分析（自适应 CNA、AJA、CSP、Steinhardt、金刚石）？需要一个按原子找 k 近邻的 C++ 接口，内存能从约 0.5 kB/原子降到 OVITO 的水平，时间也会更快。原子应变也可以同样处理。
5. System 输入数组复制：我的方案是 NumPy 输入复制一次（和 `info` 一致），代价是构造时多一次拷贝（200 万原子的坐标约 10 ms）。另一种是保持零拷贝、只在文档里写明“交给 System 的数组不要再改”。我倾向复制。
6. **GSFE 的周期盒子不收敛**：`stacking_fault_energy` 对沿法向周期的晶体，盒子高度固定，层错附近面间距膨胀被约束，内禀层错能随厚度慢慢下降（9/18/27/36 层：27.3/24.5/23.3/22.6 mJ/m²）；两侧自由表面的 slab 在 18 层就收敛到 22.46。未弛豫的值完全不随厚度变，确认是这个原因。另外默认 `max_force=1e-4` 会让层错能抖动约 0.3 mJ/m²，1e-6 才稳定。建议：周期情况下同时弛豫盒子沿法向的长度（像 LAMMPS 的 `fix box/relax z 0`），并把默认 `max_force` 收紧到 1e-6。这个功能在 `defect-energies-draft` 上，改它要把后面的分支都 rebase，所以今晚没动，等你定。教程里先用自由表面 slab。
7. **截断类分析给了邻居表反而慢**：RDF、配位数、固定截断 CNA 等不传表时在 C++ 里边找边算（多线程），传了表则走单线程 NumPy（掩码、`within` 子集复制、bincount），10.8 万原子上 35 ms 变 160 ms。建议给这些分析的“读表”路径也写 C++ 并行内核，并让它们直接跳过超出截断的对，而不是先 `within` 复制子集。这样复用才总是划算。
