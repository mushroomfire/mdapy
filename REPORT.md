# mdapy 2.0 进展报告（2026-10-26 晚）

本分支只放报告和图，不含代码。今晚的代码在各自的新分支上，等你检查后再合入 v2。

## 今晚的分支

| 分支 | 内容 | 状态 |
|---|---|---|
| `voronoi-outputs` | Voronoi 的面按需返回（`faces=True`），默认只算体积、配位数、空腔半径 | 已推送（f91b193），Triton 计时中 |
| `faster-crystal` | `build_crystal` 不再对 200 万原子排序，元素列不经 Python 字符串 | 已推送（16bb0db），Triton 排队中 |
| `phonons-draft` | 计划 17.1 的第一步：简谐声子（力常数、色散、DOS、热力学量）、原胞 | 已推送（2209c07） |
| `qha-draft` | 17.1 第二步：准简谐近似，基于 phonons-draft | 已推送（0c8ffc8） |

## 进展

### 15:29
- **QHA**（`qha-draft`）：`quasi_harmonic(system, calculator, temperatures, *, volumes=None, pressure=0, supercell, mesh, ...)`，每个体积等比缩放并弛豫原子位置、算声子；每个温度用三阶 Birch–Murnaghan（V^(−2/3) 的三次多项式，线性最小二乘，不需要 scipy）拟合 E + F_vib + PV。结果（每原子）：V(T)、G(T,P)、S、B_T、α（由 α B_T = (∂S/∂V)_T 解析得到）、C_P = C_V + T V B α²，以及每个体积的静态能和振动自由能。
  - 测试（合成 EAM）：S = −dG/dT、α = dlnV/dT、C_P = −T d²G/dT²、dG/dP = V（差分），静态拟合的 B 与 (C11+2C12)/3 差 < 0.5%。
  - Mishin 铜：9 个体积共 0.35 s。300 K：a = 3.6320 Å（LAMMPS NPT MD 3.6319），线膨胀 1.42e-5 /K（实验 1.65e-5）。900 K、1300 K QHA 比 MD 小 0.004–0.005 Å：这就是 QHA 漏掉的非谐部分，高温和熔点要靠 MD 的自由能（计划 1e）。低温 QHA 比 MD 大，是零点振动（MD 是经典的）。
  - 注意：体积网格粗时 BM3 拟合的 B 有约 1% 误差（Mishin 势 E(V) 的高阶项），默认体积 0.97–1.09 共 9 个，文档里写明了。hcp 这类形状随体积变化的晶体，目前保持 system 的形状（c/a 不弛豫），**要你定**：要不要改成在一组压强下整体弛豫晶胞（形状自然最优）？

![QHA 与 LAMMPS NPT 分子动力学的铜晶格常数（左），QHA 的线膨胀系数和体模量（右）](images/42_qha_cu.png)

### 15:22
- **Voronoi Triton 结果**（Gold 6248，40 线程，2M 原子）：默认 0.93 s / 130 MB；`faces=True` 1.10 s / 1593 MB（内存少 12 倍）。OVITO 默认 23.0 s，带键 25.1 s；freud 37.6 s。性能页原来的表是另一型号（Gold 6230）测的，所以 analyses 整组正在排队重跑，跑完一起换表。
- **四个功能的计划**写进了 `design/outline.md` 第 17 节（子代理读了 1.x 的代码做摘要，我抽查过）。要你定的问题都标了“问”，汇总在本报告“要你定的问题”一节。
- **声子 1a**（`phonons-draft`，基于 v2）：
  - `phonons(system, calculator, *, supercell=None, displacement=0.01)` → `Phonons`：`frequencies(q)`、`modes(q)`、`band_structure("fcc" 或手给路径)`、`dos(mesh, sigma=None, projected=False)`（线性四面体法或高斯）、`thermal(T, mesh)`（每原子 F、E、S、C_V）。`primitive_cell(system)` 自己找原胞（纯平移 → HNF → LLL），不依赖 spglib/phonopy/seekpath。
  - 力常数：±h 中心差分；对称化（Φ_ij = Φ_ji^T）和平移不变（声学求和规则）；周期像按最近像等权分配。
  - 验证：简单立方弹簧解析色散（1e-8 THz）；常规胞 12 支 = 原胞在 q+G 的 3 支（折叠）；EAM 铜长波声速 = √(C11/ρ)、√(C44/ρ)（2e-3）；LAMMPS PHONON 包 `dynamical_matrix` 逐元素一致（1e-9），公度 q 点频率一致（1e-4 THz）；热力学量 S = −∂F/∂T、C = ∂E/∂T、高温极限 3k_B。38 个测试，覆盖率 100%，全套 1951 个测试通过。
  - CI 的 LAMMPS 没有 PHONON 包，那个对比测试在 CI 上自动跳过。

![fcc Cu（Mishin EAM）的声子色散和态密度，mdapy 计算](images/41_phonons_cu.png)

### 14:58
- **Voronoi 按需返回**（`voronoi-outputs`）：
  - 接口：`VoronoiAnalysis(..., faces=False, polyhedra=False)`。默认结果只有 `volume`、`coordination`、`radius`；`faces=True` 才有 `offsets/j/shifts/areas/edges` 和 `index()`（没有面时 `index()` 报错并提示 `faces=True`）；`polyhedra=True` 照旧给形状，可以和 `faces` 独立开关。
  - C++：每个原子的面数在算体积时就数出配位数，不要面时立刻丢掉，不再留在缓冲区；要面时 CSR 数组用 prefix_sum 并行拷贝。每原子的记录数组也只在需要时分配。
  - 修了一个自己引入的段错误：只要形状不要面时，形状用的块号没记录。
  - 内存：3.2 万原子 27 MB → 2 MB（本地冒烟）；200 万原子原来面数据约 1.6 GB。
  - 测试 1919 个全过，覆盖率、TSan、GCC 警告、ruff、文档 strict 都过。
  - benchmark 拆成两个用例：`voronoi`（默认，对 OVITO 默认和 freud 体积）和 `voronoi-faces`（对 OVITO `generate_bonds=True` 逐原子比键数、对 freud 邻居表）。OVITO 的 Voronoi 键和我们的面一一对应（小体系逐对检查过）。
- **build_crystal 提速**（`faster-crystal`）：
  - 瓶颈不是没并行，而是两件事：200 万个浮点三键 lexsort（约 0.13 s）和 200 万个 Python 字符串的元素列（约 0.2 s）。
  - 现在先在一个晶胞里把原子按晶面分组（第三、第二、第一个晶格矢量方向），整体顺序直接由 repeat/tile 生成，不排序；元素用 polars Enum 的 gather。
  - 本地 200 万原子 0.34 s → 0.053 s，坐标与旧版逐位相同（无取向时）。
  - 取向晶体里旧版其实没有严格按 z→y→x 排：同一晶面的 z 有 1e-15 的舍入噪声，lexsort 按噪声排了。新版严格按晶面，再按 y、x。186 个结构/取向/重复组合与旧版比较：原子集合相同（1e-9），48 个取向用例顺序不同（都是这个原因）。

## 要你定的问题

1. 声子测试要不要加 phonopy（BSD-3，pip）做现场对比？CI 的 LAMMPS 要不要加 PHONON 包？
2. 默认超胞：现在是“厚度 ≥ 2×截断”；EAM 这类多体势的力常数能到 2×截断，要不要默认更大？原胞的约化基（LLL）不唯一，超胞形状可能不对称（铜得到 7×6×6），要不要改成按常规胞取超胞？
3. 有虚频（< −0.01 THz）时 `dos`/`thermal` 直接报错，可以吗？
4. QHA 的有限温度弹性常数：准静态和完整 QHA 都做？默认哪个？
5. LAMMPS MD 接口的名字和位置（`mdapy.simulation.MolecularDynamics`？）。
6. 平面缺陷第二遍：照 OVITO 串行原地（与 OVITO 逐原子一致，但依赖原子顺序），还是与顺序无关？ESF 要不要作为选项？
7. 化学产物：默认用共价半径 ×1.2 还是 vdW ×0.6？要不要区分同分异构体？
8. 石墨烯界面：要不要弯曲片子贴合三叉晶界？

---

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
| `wcp-shells-draft` | Warren–Cowley 多壳层（按截断半径或按近邻数），基于 dxa-network-draft | 已推送 |
| `vasp-calculator-draft` | VASP 计算器（超算上用），基于 wcp-shells-draft | 已推送 |
| `ptm-draft` | PTM（多面体模板匹配，按方案 A 移植），基于 preview-2026-10-23 | 已推送，CI 全绿 |
| `dxa-full-draft` | 完整版 DXA（Delaunay 剖分、弹性映射、界面网格上的 Burgers 回路，全部 C++），基于 ptm-draft | 已推送，CI 全绿（a2b9ef1） |
| **`preview-2026-10-23`** | **上面所有草稿合在一起，已快进到 dxa-full-draft 的末端（a2b9ef1），1794 个测试全过** | 已推送，CI 全绿 |


## 早上先看这里（05:06）

- 你说的两件事都做完了：**PTM**（方案 A，`ptm-draft`）和**完整版 DXA**（`dxa-full-draft`）。`preview-2026-10-23` 已快进到最新，包含今晚全部草稿，本地 1794 个测试、覆盖率、文档、TSan、GCC 警告通过，CI 9 项全绿。
- DXA 现在是 OVITO 同样的算法：Delaunay 剖分 → 弹性映射 → 界面网格 → Burgers 回路扫掠，全部 C++。和 OVITO 3.17 对比：
  - 螺/刃（分解）/剪切环/Shockley 环/bcc 环/平板/三斜盒子/热噪声 0.15 Å：线、类别、Burgers 矢量都一样，长度差几个百分点；
  - Frank 环：OVITO 的 6 条 Shockley + 6 条压杆位错、6 个四度结点，一模一样；
  - 交叉环反应：fcc 滑移结点、Lomer 锁、bcc ⟨100⟩ 结点，都和 OVITO 一致；
  - LAMMPS 压痕（7.2 万原子）：线长的 84–99% 在 OVITO 同 Burgers 矢量的线 3 Å 内，OVITO 的 78–93% 在我们的线旁边；差别在压头下极畸变区（OVITO 的 “other” 线）和结点。
  - 速度比 OVITO 快 10–25%，内存约 1.9 倍。
- 要你定的在最后一节第 12 条（多晶、hcp/金刚石、按 PTM 模板拓扑映射）。

![交叉环：a/2[1-10](111) 与 a/2[011](11-1) 交叉处反应成 a/2[101] 的短线（mdapy 渲染，箭头是 Burgers 矢量）](images/40_crossing_loops.png)

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

### 01:36（你 00:40 的意见）
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

### 01:46
- **WCP 多壳层**（`wcp-shells-draft`）：`WarrenCowley(cutoff=(3.0, 4.0, 4.7))` 按距离分壳层（每层从上一个截断到本层截断），`WarrenCowley(num_neighbors=(12, 6, 24))` 按近邻数分壳层（热扰动下也稳）。只给一个数时结果不变；给一串时 `alpha` 是（壳层, 元素, 元素），`result["Cu", "Au"]` 给各层的数组，`result["Cu", "Au", 1]` 给某一层。测试：L1₂ 有序结构的解析值（第一层 α_AuAu = 1、第二层 −3 等），以及 NumPy 暴力逐对计数（两种分层方式）。
- **VASP 计算器**（`vasp-calculator-draft`）：`VASP(incar={...}, kpoints=(4, 4, 4), setups={"Ni": "Ni_pv"}, command="srun vasp_std", directory="relax")`。每次调用写 POSCAR/INCAR/KPOINTS/POTCAR，在目录里用 shell 执行命令（和作业脚本里一样），读 OUTCAR 的 TOTEN、力和 virial，原子顺序还原；默认保留 WAVECAR/CHGCAR，下一步直接续算（最小化很省时间）。NSW/IBRION/ISIF 固定为单点加应力，用户不能设。POTCAR 目录默认取 `VASP_PP_PATH`（或其下的 potpaw_PBE），命令默认取 `MDAPY_VASP_COMMAND`、再取 ASE 的 `ASE_VASP_COMMAND`、最后是 `vasp_std`。VASP 出错时抛新的 `mdapy.errors.CalculationFailed`，附上 VASP 最后输出的内容。没有真 VASP 也能测：测试里有一个“假 VASP”脚本，读 POSCAR、用 EAM 算、按 VASP 的格式写 OUTCAR；测试检查结果和 EAM 一致、输入文件内容、原子顺序、续算文件、出错处理，以及和 `minimize` 联用时前十步与直接用 EAM 完全一样。
- **preview-2026-10-23**：全部草稿合在一起，1733 个测试、覆盖率、文档构建都通过，CI 在跑。你审的时候可以直接看这个分支。
- 收到你的决定：接下来**先按方案 A 移植 PTM，再专心做完整版 DXA**。

### 02:50
- **PTM 完成**（`ptm-draft`，基于 preview-2026-10-23，CI 9 个任务全绿）：按方案 A 移植了 Larsen 的 PTM 库（OVITO 带的版本，保留 MIT 许可，`src/cpp/identify/ptm/`，README 记录了来源和改动），用我们的 `parallel_for`。`PolyhedralTemplateMatching(structures=("fcc", "hcp", "bcc"), rmsd_cutoff=0.1, deformation_gradient=False, ordering=False, template_neighbors=False)`：结构类型（OVITO 的编号 0–8）、RMSD、原子间距、取向四元数、弹性形变梯度、合金有序类型、模板对应关系、模板邻居。和 OVITO 3.17 现场逐原子对比（类型、RMSD、取向、形变梯度、有序类型）。200 万原子 2.83 s（OVITO 3.41 s）。
- **完整版 DXA 进行中**（还没提交）。已经能跑通的部分：
  1. 从 Voronoi 对偶得到 Delaunay 四面体（带周期像，体积和盒子体积相等）。
  2. 弹性映射：每条边取晶体原子的局部晶格（PTM 取向）下最近的晶格矢量，层错原子可加 Shockley/Frank 分矢量；**两端都不是晶体原子的边经同一四面体里的晶体原子走两步**（像 OVITO 的晶体路径），这让坏区从“所有含非晶体原子的四面体”缩成映射真正不闭合的细管——这是能把相距 3–4 Å 的 Shockley 和压杆位错分开的关键。PTM 偶尔给出转了 60° 的取向（压头下十几个原子），这些按“错配”原子处理。
  3. 表面：外接球半径大于 1.5 倍近邻距离的四面体是真空，未配对面或真空交界面上的非晶体原子才是表面（之前按配位数判断，会把空位型 Frank 环的芯部当成表面）。
  4. **在界面网格上追踪 Burgers 回路**（OVITO 的算法）：网格是好四面体的边界，对边绕棱在好区内转一圈找到；从每个顶点广度优先找最短的非零回路；所有回路一起推进，允许的最长回路一步步放宽；两个回路的边互为反向就是同一条线，三个以上回路围住一块且和为零就是结点（Frank 规则自动满足）。
- 和 OVITO 现场对比（precision = 我们的线长里附近 3 Å 内有同 b 的 OVITO 线的比例，recall 反过来）：

| 例子 | 精简版（旧） | 完整版（现在） |
|---|---|---|
| 螺/刃（分解）/剪切环/Shockley 环/bcc 环/平板螺位错 | 0.6–1.0 | 全部 ≈ 1.00 / 1.00 |
| 弛豫后的 Frank 环（OVITO：6 条 Shockley + 6 条压杆位错） | 只得到一条 Frank 矢量的环 | **1.00 / 1.00**，12 条线、四度结点 |
| 压痕 8000 步 | 0.26 / 0.04 | 0.79 / 0.83 |
| 压痕 20000 步（OVITO 1849 Å） | 0.67 / 0.39 | 0.84 / 0.76 |

![Frank 环](images/36_dxa_full_frank.png)

粗的半透明线是 OVITO，细线是我们（蓝 Shockley，紫压杆位错），红点是线端。

![压痕 20000 步](images/37_dxa_full_indent.png)

- 接下来：线端到结点的连接（现在线在结点前断开，留了小缺口）、线的平滑、继续提高压痕的召回；然后把追踪改成 C++（现在原型是 Python，7.2 万原子压痕 4 s 映射 + 4 s 追踪），接入 `DislocationAnalysis`，补测试和文档。

### 03:40
- **完整版 DXA 已提交**：新分支 `dxa-full-draft`（f9c49fd，基于 `ptm-draft`），已推送，CI 在跑。本地 1788 个测试全过，覆盖率、文档构建、TSan、GCC 警告都通过。
  - 回路追踪写成了 C++（`src/cpp/analysis/circuits.{h,cpp}`）：界面网格、并行的原始回路搜索、逐级放宽的扫掠、相遇回路的连接（边对边相反的是同一条线；三个以上围住一块且和为零的是结点；最后一轮对“互为主要邻居”的两条做配对）。7.2 万原子的压痕追踪只要 0.1 s（Python 原型 4.5 s）。
  - Python 部分：四面体剖分、弹性映射、表面、把线段串成线、按窗口平均后再平滑（和 OVITO 的粗化一样，热噪声下螺位错长度 10.3 Å，和 OVITO 相同）、芯部原子。
  - 参数改成 `spacing`（默认 2.5 Å）、`trial`（14）、`stretch`（9），对应 OVITO 的 line point separation、trial circuit length、circuit stretchability；`structures` 改为 PTM 结果（需要取向）。
  - 新测试：偶极子（两条相反的螺位错）、热噪声、平移和原子重新编号不变、盒子太小报错、Frank 环和 OVITO 对比（类别、长度 15% 内、precision/recall > 0.95）、压头下的位错网络和 OVITO 对比（准静态压痕，5184 个原子，4.5 s 生成；precision 0.98、recall 0.73）、每个结点的 Frank 规则。
  - 文档页 Definition 改写为 DXA；“Compared with OVITO” 更新了数字；术语表加了 Delaunay 剖分、理想矢量、界面网格、Burgers 回路、DXA；参考文献加了 Stukowski, Bulatov & Arsenlis (2012)。
- 和 OVITO 的对比（现在的 C++ 版本）：

| 例子 | OVITO | mdapy |
|---|---|---|
| 螺位错 / 分解的刃位错 / 剪切环 | 10.2 / 17.7 / 85.1 Å | 10.2 / 17.7 / 86.2 Å，precision 和 recall 都是 1.00 |
| Frank 环 | 6 Shockley 106 Å + 6 压杆 80 Å | 6 Shockley 117 Å + 6 压杆 83 Å，6 个四度结点，1.00 / 1.00 |
| 压痕 20000 步 | 1849 Å | 1380 Å，97 条线，13 个结点；precision 0.86，recall 0.75 |

![压痕 20000 步，mdapy 渲染](images/38_dxa_full_indent_render.png)

绿：Shockley；紫：压杆位错；黄：Hirth；红色原子：层错（hcp）。

- 还差的地方：密集网络里 OVITO 连成的结点多（我们 13 个），一些线端停在结点附近；OVITO 在压头下最畸变处还有一些 b 不是 a/6 整数倍的线（在转得很厉害的晶簇坐标系里），我们没有。
- 接下来：把四面体剖分、面的配对和弹性映射也写成 C++（现在 NumPy 版本 7.2 万原子要 4.5 s，内存约每原子几 KB，100 万原子会超过 8 GB），再做 OVITO 的速度对比。

### 03:53
- **DXA 的四面体剖分和弹性映射也写成了 C++**（`dxa-full-draft` 535ce38）：每个 Voronoi 顶点是一个四面体，只由编号最小的原子输出；跨面的邻居沿 Voronoi 胞的边找到；映射时两端都不是晶体原子的边，经过绕这条边一圈的四面体里编号最小、两步都能映射的晶体原子（这是边的局部性质，所以同一条边在任何四面体里都一样）。内层循环里原子变量的争用一开始让 10 线程比 4 线程还慢，改成每块一个局部标志后 0.87 s → 0.14 s。
- **映射容差**（1f321ab）：统计了“坏”四面体的原因，98% 是有边没映射上，只有 1% 是映射了但不闭合；其中一大类是“晶体原子 → 芯部原子”的边离最近晶格矢量超过 0.2 近邻距离。OVITO 按模板近邻拓扑映射、没有距离容差。把这类边放宽到 0.3（最近晶格矢量仍然唯一），经中转原子的两步保持 0.2：分解刃位错的界面网格面数变成和 OVITO 一样的 544，压痕的召回明显上升。
- 速度和 OVITO（同一台机器 10 线程）：

| 原子数（压痕 20000 步及其 2×2、3×3 复制） | mdapy | OVITO | mdapy 内存峰值 | OVITO |
|---|---|---|---|---|
| 72,000 | 0.39 s | 0.48 s | 534 MB | 231 MB |
| 288,000 | 1.45 s | 1.74 s | 1.6 GB | 0.55 GB |
| 648,000 | 3.56 s | 3.97 s | 2.9 GB | 1.0 GB |

  比 OVITO 快 10–25%，内存约 3 倍（Voronoi 多面体 + 剖分中间数组，约 5 KB/原子，200 万原子会超过 8 GB，待优化）。这个结果不放进性能页（那页只比两边逐原子相同的情形）。
- 与 OVITO 的最终对比（precision / recall，3 Å 内同一 Burgers 矢量）：

| 例子 | OVITO 线长 | mdapy 线长 | precision / recall |
|---|---|---|---|
| 螺位错、分解刃位错、剪切环 | 10.2 / 17.7 / 85.1 Å | 10.2 / 17.7 / 86.0 Å | 1.00 / 1.00 |
| Frank 环 | 186 Å（6 Shockley + 6 压杆） | 199 Å（同样 12 条，6 个四度结点） | 1.00 / 1.00 |
| 压痕 8000 / 12000 / 16000 / 20000 步 | 154 / 741 / 1113 / 1849 Å | 132 / 606 / 939 / 1654 Å | 0.99/0.91、0.88/0.78、0.86/0.78、0.84/0.82 |
| 测试里的准静态压痕（5184 原子） | 122 Å | 119 Å | 0.87 / 0.86 |

![压痕 20000 步：粗半透明是 OVITO，细线是 mdapy（蓝 Shockley、紫压杆、黄 Hirth、绿全位错），红点是线端，黑方块是结点](images/39_dxa_final_indent.png)

- 本地 1788 个测试、覆盖率、文档、TSan、GCC 都通过；前两次提交的 CI 除被取消的覆盖率外全绿，最新提交的 CI 在跑。
- 还差：OVITO 在压头下最畸变处的 “other” 线（约 100 Å）我们只有一半；一些线端停在结点前。

### 04:05
- **CI 发现的问题**：测试里的准静态压痕在 Linux 上和 macOS 给出不同的位错网络（precision 0.77 < 0.85 失败）。我在本地给初始坐标加 1e-8 的扰动重跑了 6 次：网络每次都不一样，precision 0.73–0.87、recall 0.34–0.88——准静态压痕是混沌的，不适合跨平台的阈值测试。其中 recall 0.34 那次是压头下大片无序（1019 个 other 原子），OVITO 给出一个有 20 多条臂的星形巨结点、许多段从结点出发又回到同一结点，这在严重无序区意义不大。
- **改成确定性的网络测试**（1d32d40）：两个 a/2⟨110⟩ 剪切环放在两个 {111} 面上互相穿过（立体角位移场叠加，不弛豫，所有平台一致）。交叉处反应成一段两端是结点的短线：
  - a/2[1-10](111) + a/2[011](11-1) = a/2[101]：滑移型结点，和 OVITO 比 precision 0.95、recall 1.00，总长 169.3 vs 170.4 Å；
  - a/2[01-1](111) + a/2[101](11-1) = a/2[110]：Lomer 锁，1.00 / 1.00。
  测试检查 Frank 规则、反应段的 b、长度 5% 内、precision/recall > 0.9。压痕测试保留，只查 Frank 规则和 Shockley。
- **内存**（9f09959）：Voronoi 内核加了“只记顶点”的模式——每个顶点只存它三个面对面的原子和周期像（网格搜索本来就知道像，不用再反射计算），留在各线程的块里不再拷贝；Delaunay 只用这些三元组。64.8 万原子峰值 2.9 → 1.9 GB（OVITO 1.0 GB），时间 3.2 s（OVITO 4.0 s）。
- `preview-2026-10-23` 等这次 CI 绿了以后快进到 `dxa-full-draft`（PTM + 完整 DXA 都在里面）。

### 04:40
- **bcc 结点**（cb45ef2）：交叉环测试加了 bcc 的 a/2[111](1-10) + a/2[-1-11](101) = a[001]：我们 ⟨100⟩ 12.0 Å、a/2⟨111⟩ 152.2 Å，OVITO 11.2 / 152.3 Å，precision/recall 0.98/1.00。
- **不给取向时自动确定**（cb45ef2）：以前不给 `orientation` 就当立方轴，任意转动的单晶所有原子都会被当成错配、什么都找不到（OVITO 不要求给取向）。现在从 PTM 求晶体的平均取向（各原子按立方对称对齐到拟合最好的那个原子后取平均、正交化），取离盒子轴最近的立方等价取向，离盒子轴 1° 以内就直接用盒子轴。结果新增 `orientation`（`lattice_burgers` 的坐标轴）。测试：任意取向的螺位错不给取向，b、芯部和给定取向时一致。
- **热运动下更稳**（211c7aa）：随机位移 0.15 Å（远高于室温）时 PTM（RMSD 截断 0.1）把四分之一的晶体判成 other，剪切环断成 5 段，而 OVITO 仍是一整条。现在 PTM 判为 other、但自适应 CNA 判为晶格（fcc 里也包括 hcp 层错）的原子也算晶体，取向用它最佳模板的；没有取向的当错配。0.15 Å 下整条环（与 OVITO 一致，1.00/1.00），0.2 Å 两边都找不到；压痕结果基本不变。加了对应测试。放宽 PTM 截断本身（0.15–0.25）试过，会损害压痕的精度，所以没用。
- cb45ef2 之前的提交 CI 全绿（9 项，含 ASan/TSan/覆盖率里的 OVITO 对比）；211c7aa 的 CI 在跑，绿了就把 `preview-2026-10-23` 快进过来。

### 05:06
- 三斜盒子的测试（剪切 4% 的偶极子，与 OVITO 一致）；网格对边找不到时当边界而不报错（9268a8e）。CI 9 项全绿，`preview-2026-10-23` 已快进到 9268a8e。
- 压痕 20000 步的最终对比图：

![压痕 20000 步：粗半透明是 OVITO，细线是 mdapy，红点线端，黑方块结点](images/39_dxa_final_indent.png)

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
10. ~~位错分析的边界~~：精简版（`dxa-draft`、`dxa-partials-draft`、`dxa-network-draft`）已被完整版取代，见第 12 条。
11. ~~PTM 怎么做~~：已按方案 A 完成（`ptm-draft`，见 02:50）。
12. ~~DXA 要不要做成完整版~~：已完成（`dxa-full-draft`，见 03:40 之后几条）。遗留几个要你定的：
    - **多晶**：现在只支持一个取向（`orientation`），PTM 取向偏离超过 20° 的原子当作错配。OVITO 按晶簇（取向一致的连通原子）各自建坐标系，所以晶界两侧的位错都能找到。要不要做？工作量约一天。
    - **hcp、金刚石结构**的 DXA（OVITO 都支持）：要不要做？
    - **映射容差**：现在“晶体原子 → 芯部原子”的边容差 0.3 近邻距离、经中转原子的两步 0.2。OVITO 不用距离容差而按 PTM 模板近邻的拓扑映射；照做需要 PTM 的模板对应关系（我们的 PTM 已经能输出），能更接近 OVITO（尤其是压头下的 “other” 线），但工作量不小。
    - 和 OVITO 的差别都写在 `docs/analysis/dislocations.md` 的 “Compared with OVITO” 一节。
