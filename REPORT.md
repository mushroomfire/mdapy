# mdapy 2.0 夜间进展报告（2026-10-22 → 23）

本分支只放报告和渲染图，不含代码。今晚的代码都在各自的新分支上，明早检查后再合入 v2。

## 今晚的分支

| 分支 | 内容 | 状态 |
|---|---|---|
| `v2` | 11a 晶体建模已合入；atomsk 进 CI；删除 `overrides/` | 已推送 |
| `build-defects-draft` | 位错、点缺陷、层错、表面、SQS 合金、多晶与孪晶（11b/11c） | 已推送，待检查 |
| `defect-energies-draft` | 层错能（GSFE）、表面能、空位形成能（simulation），基于 build-defects-draft | 已推送，待检查 |
| `visualization-draft` | 渲染器（已完成）与画图模板（进行中），基于 defect-energies-draft | 渲染器已推送 |
| `mlip-draft` | OUTCAR → xyz、NEP 描述符 PCA、最远点采样、扰动 | 待开始 |
| `performance-draft` | 与 OVITO 的性能对比 | 待开始 |
| `tutorials-draft` | 进阶教程 | 待开始 |

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

### 21:25
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

## 早上需要你决定的事

1. 可视化分支基于 defect-energies-draft（为了渲染新的建模结构），合并顺序：build-defects → defect-energies → visualization。
2. 渲染的默认风格、配色（元素颜色沿用 OVITO/1.x）、风格名字，你看图后定。
