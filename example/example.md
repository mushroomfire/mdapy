# 本库的基本用法

## 目前主要包含以下几个功能

- 计算原子温度
- 计算原子结构熵
- 计算中心对称参数
- 建立简单晶体模型
- 建立原子近邻列表

## 导入必要的package


```python
import mdapy as mp # 无特殊说明本库简写为 mp
import numpy as np
import taichi as ti
ti.init(arch=ti.cpu) #arch=ti.cpu 使用CPU. # arch=ti.gpu, device_memory_GB=5. 使用GPU
```

    [Taichi] version 1.1.2, llvm 10.0.0, commit f25cf4a2, win, python 3.7.3
    [Taichi] Starting on arch=x64
    

## 查看版本号


```python
mp.__version__
```




    '0.4.0'



## 读取一个系统
[可在此处下载example.dump文件](http://yunpan.mushroomfire.com/s/GzTv)


```python
%%time
system = mp.System('example.dump') 
```

    Wall time: 2.96 s
    

## 查看系统数据


```python
system.data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>type</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>vx</th>
      <th>vy</th>
      <th>vz</th>
      <th>c_1</th>
      <th>c_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>192295</td>
      <td>3</td>
      <td>-27.9096</td>
      <td>42.0121</td>
      <td>-22.2491</td>
      <td>-2.35674</td>
      <td>0.481940</td>
      <td>-0.257686</td>
      <td>5.06650</td>
      <td>-3.71151</td>
    </tr>
    <tr>
      <th>1</th>
      <td>255932</td>
      <td>5</td>
      <td>-30.0338</td>
      <td>43.2841</td>
      <td>-22.1938</td>
      <td>1.55510</td>
      <td>-1.125920</td>
      <td>-1.983760</td>
      <td>5.16323</td>
      <td>-3.69935</td>
    </tr>
    <tr>
      <th>2</th>
      <td>319374</td>
      <td>2</td>
      <td>-29.4179</td>
      <td>42.0487</td>
      <td>-20.1799</td>
      <td>-1.56154</td>
      <td>-0.105773</td>
      <td>-1.052900</td>
      <td>6.02419</td>
      <td>-3.51732</td>
    </tr>
    <tr>
      <th>3</th>
      <td>383068</td>
      <td>3</td>
      <td>-27.3053</td>
      <td>43.2945</td>
      <td>-20.1681</td>
      <td>1.75865</td>
      <td>0.962938</td>
      <td>-2.273980</td>
      <td>5.15436</td>
      <td>-3.90756</td>
    </tr>
    <tr>
      <th>4</th>
      <td>700644</td>
      <td>2</td>
      <td>-27.9671</td>
      <td>44.5056</td>
      <td>-22.1799</td>
      <td>2.11749</td>
      <td>-0.684720</td>
      <td>-0.795077</td>
      <td>6.15314</td>
      <td>-3.85788</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2287143</th>
      <td>1715363</td>
      <td>1</td>
      <td>329.5120</td>
      <td>257.9240</td>
      <td>323.0480</td>
      <td>1.93039</td>
      <td>1.215740</td>
      <td>-1.869160</td>
      <td>5.89311</td>
      <td>-3.92515</td>
    </tr>
    <tr>
      <th>2287144</th>
      <td>1905765</td>
      <td>5</td>
      <td>327.2950</td>
      <td>256.6070</td>
      <td>318.9410</td>
      <td>2.16960</td>
      <td>-4.590730</td>
      <td>1.395140</td>
      <td>4.94041</td>
      <td>-4.08289</td>
    </tr>
    <tr>
      <th>2287145</th>
      <td>1966370</td>
      <td>3</td>
      <td>329.5330</td>
      <td>257.9350</td>
      <td>318.9310</td>
      <td>2.65692</td>
      <td>-3.539740</td>
      <td>-1.125020</td>
      <td>5.70667</td>
      <td>-3.83374</td>
    </tr>
    <tr>
      <th>2287146</th>
      <td>2030121</td>
      <td>1</td>
      <td>328.9320</td>
      <td>256.6390</td>
      <td>321.0620</td>
      <td>1.25297</td>
      <td>0.868006</td>
      <td>1.104770</td>
      <td>5.42057</td>
      <td>-3.39951</td>
    </tr>
    <tr>
      <th>2287147</th>
      <td>2093740</td>
      <td>5</td>
      <td>326.7530</td>
      <td>257.7980</td>
      <td>320.8960</td>
      <td>2.40437</td>
      <td>-0.944472</td>
      <td>3.358390</td>
      <td>6.04798</td>
      <td>-2.90699</td>
    </tr>
  </tbody>
</table>
<p>2287148 rows × 10 columns</p>
</div>



## 生成邻域列表


```python
%%time
system.build_neighbor(rc=5., max_neigh=60)
```

    Wall time: 3.82 s
    

## 计算中心对称参数


```python
%%time
system.cal_centro_symmetry_parameter(N=12)
```

    Wall time: 8.7 s
    

## 计算原子构型熵


```python
%%time
system.cal_atomic_entropy()
```

    Wall time: 4.31 s
    

## 计算平均原子温度


```python
%%time
system.cal_atomic_temperature(amass=np.array([58.933, 58.693, 55.847, 26.982, 63.546]))
```

    Wall time: 1.21 s
    

## 查看计算结果


```python
system.data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>type</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
      <th>vx</th>
      <th>vy</th>
      <th>vz</th>
      <th>c_1</th>
      <th>c_2</th>
      <th>csp</th>
      <th>atomic_entropy</th>
      <th>atomic_temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>192295</td>
      <td>3</td>
      <td>-27.9096</td>
      <td>42.0121</td>
      <td>-22.2491</td>
      <td>-2.35674</td>
      <td>0.481940</td>
      <td>-0.257686</td>
      <td>5.06650</td>
      <td>-3.71151</td>
      <td>5.066506</td>
      <td>-3.704394</td>
      <td>271.311096</td>
    </tr>
    <tr>
      <th>1</th>
      <td>255932</td>
      <td>5</td>
      <td>-30.0338</td>
      <td>43.2841</td>
      <td>-22.1938</td>
      <td>1.55510</td>
      <td>-1.125920</td>
      <td>-1.983760</td>
      <td>5.16323</td>
      <td>-3.69935</td>
      <td>5.163225</td>
      <td>-3.693706</td>
      <td>244.521713</td>
    </tr>
    <tr>
      <th>2</th>
      <td>319374</td>
      <td>2</td>
      <td>-29.4179</td>
      <td>42.0487</td>
      <td>-20.1799</td>
      <td>-1.56154</td>
      <td>-0.105773</td>
      <td>-1.052900</td>
      <td>6.02419</td>
      <td>-3.51732</td>
      <td>6.024189</td>
      <td>-3.511253</td>
      <td>240.937302</td>
    </tr>
    <tr>
      <th>3</th>
      <td>383068</td>
      <td>3</td>
      <td>-27.3053</td>
      <td>43.2945</td>
      <td>-20.1681</td>
      <td>1.75865</td>
      <td>0.962938</td>
      <td>-2.273980</td>
      <td>5.15436</td>
      <td>-3.90756</td>
      <td>5.154361</td>
      <td>-3.897303</td>
      <td>230.167953</td>
    </tr>
    <tr>
      <th>4</th>
      <td>700644</td>
      <td>2</td>
      <td>-27.9671</td>
      <td>44.5056</td>
      <td>-22.1799</td>
      <td>2.11749</td>
      <td>-0.684720</td>
      <td>-0.795077</td>
      <td>6.15314</td>
      <td>-3.85788</td>
      <td>6.153219</td>
      <td>-3.849362</td>
      <td>243.981369</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2287143</th>
      <td>1715363</td>
      <td>1</td>
      <td>329.5120</td>
      <td>257.9240</td>
      <td>323.0480</td>
      <td>1.93039</td>
      <td>1.215740</td>
      <td>-1.869160</td>
      <td>5.89311</td>
      <td>-3.92515</td>
      <td>5.893107</td>
      <td>-3.916707</td>
      <td>260.688812</td>
    </tr>
    <tr>
      <th>2287144</th>
      <td>1905765</td>
      <td>5</td>
      <td>327.2950</td>
      <td>256.6070</td>
      <td>318.9410</td>
      <td>2.16960</td>
      <td>-4.590730</td>
      <td>1.395140</td>
      <td>4.94041</td>
      <td>-4.08289</td>
      <td>4.940413</td>
      <td>-4.073879</td>
      <td>320.674103</td>
    </tr>
    <tr>
      <th>2287145</th>
      <td>1966370</td>
      <td>3</td>
      <td>329.5330</td>
      <td>257.9350</td>
      <td>318.9310</td>
      <td>2.65692</td>
      <td>-3.539740</td>
      <td>-1.125020</td>
      <td>5.70667</td>
      <td>-3.83374</td>
      <td>5.706692</td>
      <td>-3.825464</td>
      <td>344.000671</td>
    </tr>
    <tr>
      <th>2287146</th>
      <td>2030121</td>
      <td>1</td>
      <td>328.9320</td>
      <td>256.6390</td>
      <td>321.0620</td>
      <td>1.25297</td>
      <td>0.868006</td>
      <td>1.104770</td>
      <td>5.42057</td>
      <td>-3.39951</td>
      <td>5.420572</td>
      <td>-3.393086</td>
      <td>241.169174</td>
    </tr>
    <tr>
      <th>2287147</th>
      <td>2093740</td>
      <td>5</td>
      <td>326.7530</td>
      <td>257.7980</td>
      <td>320.8960</td>
      <td>2.40437</td>
      <td>-0.944472</td>
      <td>3.358390</td>
      <td>6.04798</td>
      <td>-2.90699</td>
      <td>6.048002</td>
      <td>-2.902135</td>
      <td>245.842743</td>
    </tr>
  </tbody>
</table>
<p>2287148 rows × 13 columns</p>
</div>



## 查看平均温度


```python
system.data['atomic_temp'].mean() # K
```




    293.87168742301066



## 与LAMMPS计算结果进行对比


```python
# csp 对比
error = abs(system.data['c_1'] - system.data['csp'])
error[error>0.1], error.max()
```




    (Series([], dtype: float64), 0.0049196543884280075)




```python
# 构型熵对比, 这里由于中间数值积分的地方我不清楚lammps的具体方式,结果数值上会有一些区别。
error = abs(system.data['c_2'] - system.data['atomic_entropy'])
error[error>0.1], error.max()
```




    (Series([], dtype: float64), 0.08606090263366717)



## 保存计算结果到dump文件中


```python
%%time
system.write_dump() # 生成一个新的dump文件
```

    Wall time: 30.1 s
    

## 建立简单晶体结构


```python
%%time
FCC = mp.LatticeMaker(4.05, 'FCC', 50, 50, 50)
FCC.compute()
```

    Wall time: 506 ms
    

## 保存为data格式


```python
%%time
FCC.write_data()
```

    Wall time: 3.23 s
    
