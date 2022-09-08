# @Author: HerrWu 

## 关于
- 该库用于分子动力学轨迹后处理，包括邻域列表，温度，结构熵，切片等处理，目前暂无开源打算。
- 系统要求：目前仅测试了win10, ubuntu。
- 支持GPU加速
- 更新：重新下载源码安装即可。

## 安装
0. 下载源码
    ```bash
    git clone https://github.com/mushroomfire/mdapy.git
    cd mdapy
    ```
1. 建议使用conda先创建一个虚拟环境避免依赖库冲突
    ```bash
    conda create -n mda
    conda activate mda
    ```
2. 首先安装依赖
    ```bash
    pip install -r requirements.txt
    ```
3. 安装本库
    ```python
    python install.py
    ```

## 使用
1. 导入库
   ```python
   import mdapy as mda
   ```

## 更新记录

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
