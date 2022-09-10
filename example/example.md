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
ti.init(ti.cpu)
```

    [Taichi] version 1.1.2, llvm 10.0.0, commit f25cf4a2, win, python 3.7.13
    [Taichi] Starting on arch=x64
    

## 读取一个系统
[可在此处下载example.dump文件](http://yunpan.mushroomfire.com/s/GzTv)


```python
system = mp.System('example.dump') 
```

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



## 初始化邻域列表


```python
neighbor = mp.Neighbor(system.pos, system.box, 5., max_neigh=60)
```

## 生成邻域列表


```python
%%time
neighbor.compute()
```

    Wall time: 3.18 s
    

## 初始化中心对称参数


```python
CSP = mp.CentroSymmetryParameter(neighbor.pos_ti, neighbor.box_ti, neighbor.boundary, neighbor.verlet_list_ti, neighbor.distance_list_ti, 12)
```

## 计算中心对称参数


```python
%%time
CSP.compute()
```

    Wall time: 2.25 s
    

## 初始化原子构型熵


```python
Entropy = mp.AtomicEntropy(neighbor.box_ti, neighbor.distance_list_ti, 5.)
```

## 计算原子构型熵


```python
%%time
Entropy.compute()
```

    Wall time: 6.73 s
    

## 初始化原子温度


```python
Temperature = mp.AtomicTemperature(amass=np.array([58.933, 58.693, 55.847, 26.982, 63.546]), vel=system.vel, verlet_list = neighbor.verlet_list_ti, atype_list=system.data['type'].values)
```

## 计算原子温度


```python
%%time
Temperature.compute()
```

    Wall time: 336 ms
    

## 查看系统平均温度


```python
Temperature.T.to_numpy().mean() # K
```




    293.7486024263631



## 添加计算结果到系统中


```python
system.data['csp'] = CSP.csp.to_numpy()
system.head[-1] = system.head[-1].strip() + ' csp'
system.data['entropy'] = Entropy.entropy.to_numpy()
system.head[-1] = system.head[-1].strip() + ' entropy'
system.data['temperature'] = Temperature.T.to_numpy()
system.head[-1] = system.head[-1].strip() + ' temperature'
```

## 保存计算结果到dump文件中


```python
%%time
mp.write_dump(system)
```

    Wall time: 30.3 s
    
