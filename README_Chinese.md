DaPy - 别让语法束缚了思想
====
![](https://img.shields.io/badge/Version-1.3.3-green.svg)  ![](https://img.shields.io/badge/Download-PyPi-green.svg)  ![](https://img.shields.io/badge/License-GNU-blue.svg)  

作为一个基于Python原生数据结构搭建的数据分析和数据挖掘库，**DaPy**致力于节约数据科学家的时间并提高他们的研究效率，同时它也在尽其所能为他们提供更舒适和流畅的操作体验。

[安装](#安装) | [特性](#特性) | [快速开始](https://github.com/JacksonWuxs/DaPy/blob/master/快速开始.md) | [远期规划](#远期规划) | [更新日志](#更新日志) | [版权归属](#版权归属) | [English](https://github.com/JacksonWuxs/DaPy/blob/master/README.md)

## 安装
最新版本1.3.3已上传至PyPi。
```
pip install DaPy
```  

用下面的代码将DaPy更新至1.3.3版本。
```
pip install -U DaPy
```

## 特性
#### Ⅰ. 舒适的体验
  从设计之初，我们就尽可能地让DaPy使用更多Python原生的数据结构，并
让他能支持更多Pythonic的写法特性。因此，你可以快速地适应何使用DaPy
中的数据结构和操作。另外，为了能让用户更流畅地实现他们的想法，我们尽可能
简化了DaPy中的公式或方法参数。 
  
  按照不同的字段及标准排序记录是了解数据集的常用方式。在这个功能中，DaPy支持
你使用多个不同的排序要求进行排序。 
```Pyton
 data.sort(('A_col', 'DESC'), ('B_col', 'ASC'), ('D_col', 'DESC'))
 ```
  
#### Ⅱ. 高效性  
我们在数据处理库中最常用的三个操作（加载数据、排序数据和遍历数据）测试
了DaPy的性能水平。相较于其他使用C语言优化的库，DaPy在测试中表现出了惊人的
效率。在所有的测试项目中，DaPy始终保持着与最快的C语言优化的库2倍内的耗时。 

我们在搭载Intel i7-6560U处理器的平台上，通过64位2.7.13版本的Python进行了测试。
测试数据集(https://pan.baidu.com/s/1kK3_V8XbbVim4urDkKyI8A)  包含多达
450万条记录，并且总的大小为240.2MB。

<table>
<tr>
	<td>测试结果</td>
	<td>DaPy</td>
	<td>Pandas</td>
	<td>Numpy</td> 
</tr>
<tr>
	<td>加载数据</td>
	<td> 29.3s (2.4x)</td>
	<td> 12.3s (1.0x)</td>
  <td>169.0s (13.7x)</td>
</tr>
<tr>
	<td>遍历数据</td>
	<td>0.34s (1.6x)</td>
<td>3.10s (14.8x)</td>
	<td>0.21s (1.0x)</td>
</tr>
<tr>
	<td>排序数据</td>
	<td>1.41s (1.65x)</td>
	<td>0.86s (1.0x)</td>
	<td>5.37s (10.1x)</td>
	</tr>
<tr>
	<td>总耗时</td>
	<td>31.1s (1.8x)</td>
	<td>17.4s (1.0x)</td>
	<td>174.6s (10.0x)</td>
	</tr>
<tr>
	<td>版本信息</td>
	<td>1.3.3</td>
	<td>0.22.0</td>
	<td>1.14.0</td>
	</tr>
</table>  


## 远期规划  
* 描述性统计
* 推断性统计
* 特征工程
	- 主成分分析
	- LDA (Linear Discriminant Analysis)
	- MIC (Maximal information coefficient)
* 模型
  	- 朴素贝叶斯
	- 支持向量机
	- K-Means
	- Lasso Regression 

## 更新日志
* V1.3.3 (2018-06-20)
	- 添加 外部数据文件读取能力拓展: Excel, SPSS, SQLite3, CSV;
	- 重构 DaPy架构, 提高了远期拓展能力;
	- 重构 DaPy.DataSet类, 一个DataSet实例可以批量管理多个数据表;
	- 重构 DaPy.Frame类, 删除了格式验证, 适配更多类型的数据集;
	- 重构 DaPy.SeriesSet类, 删除了格式验证, 适配更多类型的数据集;
	- 移除 DaPy.Table类;
	- 优化 DaPy.Matrix类, 效率提升接近2倍;
	- 优化 DaPy.Frame 及 Data.SeriesSet类的展示, 数据呈现更为清晰美观;
	- 添加 `线性回归`及`方差分析`至DaPy.mathematical_statistics;
	- 添加 DaPy.io.encode()函数, 更好地适配中文数据;
	- 替换 read_col(), read_frame(), read_matrix() 为 read()函数;

* V1.3.2 (2018-04-26)
	- 优化 数据加载的效率;
	- 添加 更多实用的功能到DaPy.DataSet中;
	- 添加 新的数据结构DaPy.Matrix,支持常规的矩阵运算;
	- 添加 常用描述数据的函数 (例如： corr, dot, exp);
	- 添加 `支持向量机`至DaPy.machine_learn;
	- 添加 一些标准数据集.
	
* V1.3.1 (2018-03-19)
	- 修复 在加载数据及中的bug;
	- 添加 支持保存数据集的功能.
	
* V1.2.5 (2018-03-15)
	- DaPy的第一个版本！

## 版权归属
Copyright (C) 2018 Xuansheng Wu
<br>
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.</br>
<br>
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.</br>
<br>
You should have received a copy of the GNU General Public License
along with this program.  If not, see https:\\www.gnu.org\licenses.# datapy
A light Python library for data processing and analysing.</br>
