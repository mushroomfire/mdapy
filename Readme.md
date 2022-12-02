# @Author: HerrWu 

## 关于
- 该库用于分子动力学轨迹后处理，包括邻域列表，温度，结构熵，切片等处理，目前暂无开源打算
- python版本: 兼容py3.7到py3.10
- 系统要求：win10,ubuntu,其他linux系统应该也可以, macos在M1芯片上无法编译
- 部分模块支持GPU加速
- 更新：重新下载源码安装即可。
## 依赖
- taichi
- numpy
- scipy
- pandas
- matplotlib
- SciencePlots
- pyfftw (optional) for fast FFT 
- pytorch (optional) for fast sort
## 安装
0. 下载源码
    ```bash
    git clone https://github.com/mushroomfire/mdapy.git
    cd mdapy
    ```
1. 建议使用conda先创建一个虚拟环境避免依赖库冲突
    ```bash
    conda create -n mda python==3.8.0 # 版本3.7以上都可以
    conda activate mda
    # 目前可以自主选择是否安装pytorch
    # conda install ipykernel # 如果使用jupyter的话可以把该虚拟环境激活
    # python -m ipykernel install --name mda # jupyter 激活虚拟环境
    ```
2. 安装依赖
   ```bash
   pip install -r requirements.txt
   ```
3. 安装本库
    ```python
    python install.py
    ```

## 使用
具体参考example说明
```python
import mdapy as mp
mp.init("cpu")
```

## To do...
1. 目前没有完全考虑GPU的计算,导致内存很多时候使用64位,在之后的计算中可以优化起来.
2. 64位的GPU操作需要使用N卡的,3080是可以的,其他的卡我也没有

## 更新记录 
## 2022-12-2
1. 增加了unwrap_pos，用于计算MSD和lindemann index的前置处理
2. 更新MSD的bug
3. 更新版本号0.7.1

## 2022-11-30
1. update了indemann_parameter模块
2. update了MultiSystem模块
3. 更新版本号0.6.9
4. 更新版本号0.7.0, 保留serial版本的lindemann index
   
## 2022-11-28
1. update了System模块，可以支持data文件格式的读取和导出，可以使用box和pos来创建system.
2. 增加了MeanSquaredDisplacement模块,依赖增加了pyfftw来加速快速傅里叶变换.
3. 更新版本0.6.8.

## 2022-11-26
1. 增加benchmark测试
2. 增加了不同电脑的测试

## 2022-11-20
1. 增加CreatePolycrystalline模块来创建多晶以及带石墨烯晶界的结构
2. 更新版本号0.6.7
   
## 2022-11-15
1. 暂时删除a-CNA,结果一直不对,增加了voronoi在ubuntu的编译
2. 移除cluster中的fortran代码为c++版本,移除numba版本以及去除numba的依赖
3. 更新版本号0.6.6
   
## 2022-11-13
1. 增加了voronoi来计算原子体积
2. 增加了不同元素间的g(r)计算
3. 更新版本号0.6.5
   
## 2022-11-8
1. 增加了void_distribution.py来统计空洞数目,体积,空间分布
2. 增加了WCP参数计算,用来表征高熵合金中的短程有序程度
3. 更新版本号0.6.4

## 2022-11-6
1. 增加了eam_generate.py文件,方便直接生成EAM文件
2. 增加了eam_average.py文件,可用于生成平均EAM势函数文件
3. 更新版本号0.6.3

## 2022-11-5
1. 修改了neighbor的计算,在很小体系时候也能正确计算结果,但是实际上这个时候应该去掉并行并且使用简单的暴力搜索来建立邻域
2. 现在小尺寸使用串行暴力搜索, 大尺寸使用算法并行搜索
3. 更新了一个计算eos的例子

## 2022-11-4
1. 增加了EAM class, 目前支持eam.alloy格式的势函数
2. 增加了Calculator class, 可以使用eam.alloy来计算原子的受力和能量
3. 更新版本号 0.6.2

## 2022-11-1
1. 使用scipy来构建kdtree处理近邻原子问题
2. 修改了csp和a-cna的代码
3. 更新版本号0.6.1
4. 更新example文档
5. 下一步更新EAM计算能量和受力进来

## 2022-10-30
1. 更新adaptive_common_neighbor_analysis
2. 更新lattice_maker的pos显示
3. 更新版本号0.6.0

## 2022-10-29
1. 更新common_neighbor_analysis的功能, fixed rc的模式目前,下一步继续更新adaptive rc 的模式
2. 修复lattice_maker中GRA的层间距bug,现在是正常的3.4
3. 更新版本号至0.5.3

## 2022-10-27
1. 更新cluster_analysis功能,使用fortran加速,使用预编译分发.
2. 版本号更新0.5.2
3. 内存不会泄露了,taichi版本升级到1.2.0

## 2022-10-20
1. 增加了团簇分析功能,cluster_analysis,添加了numba加速,在一些特定场合numba也还是不错的.
2. 感觉有一点看得懂ovito的代码了,这点很重要.
3. 更新版本到0.5.1

## 2022-10-16 21:53
增加了径向分布函数的计算模块.

## 2022-10-16
对于csp计算中的排序部分使用了pytorch,提高了计算效率,之后考虑把整个过程的array都转移到pytorch上面来.

## 2022-10-15
再次修改csp的计算,目前瓶颈依然在于如何排序,决定还是不引用外部库freud，发现计算有错误.

## 2022-10-14
重写了计算的内存布局,局部使用ndarray可以大量节省内存.

## 2022-10-11
1. 修复了温度计算中的int32 warnning.
   
## 2022-10-10
1. 修复了邻域列表生成中的bug.

## 2022-10-07
1. 更新了中心对称参数的计算程序

## 2022-10-04
1. 更新了System.py
2. 增加了计算模块的说明
3. 更新了example使用说明

## 2022-9-28
1. 更新了System.py的读取文件方式，减少内存占用，增加了保存为dump和data的方法
2. 修改了LatticeMaker的保存方法

## 2022-9-10
1. 更新了中心对称参数计算
2. 更新了使用说明

## 2022-9-8 16:12
1. 重写了entropy计算
2. 修改名字
   
## 2022-9-8
1. 重写了邻域列表，温度计算等文件
   
## 2022-9-7 21:01
1. 大面积删改所有内容，从头开始写taichi

## 2022-9-7 
1. 更新了lattice_maker
2. 更新了requirements.txt 

## 2022-8-31
1. 更新了requirements.txt
2. 修改了make_fortran.py
3. 修改了install.py
4. 修改了one_binning.py
5. 增加version='0.1.0'

## 2022-8-29
1. 修改了binning模块
2. 增加了requirements.txt

## 2022-3-25
1. 增加了计算wcp串行模块
2. 修正了make_fortran.py在win的适配性
3. 更新了check.py

## 2022-3-24
1. 增加了自动编译fortran文件的脚本，make_fortran.py 

## 2022-3-22
1. 增加了选择导入freud，以后可以与其他计算模块整合
2. 分离了生成领域列表的步骤，减少内存消耗
3. 修改了OneBinning模块，更名为binning
4. 增加了Twobinning模块
5. 增加install.py

## 2021-12-17
1. 增加了识别融化的计算（使用freud库）
2. 增加了模块识别和setup.py文件
3. 修改了core.py结构，增加易用性
4. 增加了timer计时器功能
5. 增加了一维划分功能和画图

## 2021-12-9
1. 修改了每个原子的最大邻域个数
2. 修改dump文件中有nan值的bug
3. 增加了颜色显示
