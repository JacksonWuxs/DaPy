DaPy - 一个数据分析和数据挖掘的Python库
====
![](https://img.shields.io/badge/Version-1.3.2-green.svg)  ![](https://img.shields.io/badge/Download-PyPi-green.svg)  ![](https://img.shields.io/badge/License-GNU-blue.svg)  

作为一个数据分析和数据处理的库，**DaPy**致力于节约数据科学家的时间并提高他们的研究效率，同时它也在尽其所能为你提供舒适的体验。我们希望通过DaPy证明，中国人也能开发出高质量的底层库。

[安装](#installation) | [特性](#features) | [快速开始](https://github.com/JacksonWuxs/DaPy/blob/master/Quick%20Start.md ) | [远期规划](#todo) | [更新日志](#version-log) | [版权归属](#license)

## Installation
最新版本1.3.2已上传至PyPi。
```
pip install DaPy
```  

用下面的代码将DaPy更新至1.3.2版本。
```
pip install -U DaPy
```

## Features
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
	<td> 23.4s (1.9x)</td>
	<td> 12.3s (1.0x)</td>
  <td>169.0s (13.7x)</td>
</tr>
<tr>
	<td>遍历数据</td>
	<td>0.53s (2.5x)</td>
<td>4.18s (20.9x)</td>
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
	<td>25.4s (1.5x)</td>
	<td>17.4s (1.0x)</td>
	<td>174.6s (10.0x)</td>
	</tr>
<tr>
	<td>版本信息</td>
	<td>1.3.2</td>
	<td>0.22.0</td>
	<td>1.14.0</td>
	</tr>
</table>  


## TODO  
* 描述性统计
	- 汇总表（交叉表）
	- 条件查询
* 推断性统计
	- 均值估计
	- 假设检验
	- Univariate linear regression model
* 特征工程
	- 主成分分析
	- LDA (Linear Discriminant Analysis)
	- MIC (Maximal information coefficient)
	
* 算法
  - 朴素贝叶斯
	- 支持向量机
	- K-Means
	- Lasso Regression  

## Version-Log
* V1.3.2 (2018-04-26)
	- 显著提高了数据加载的效率;
	- 为DaPy.DataSet添加了更多实用的功能;
	- 添加了新的数据结构DaPy.Matrix,支持常规的矩阵运算;
	- 添加了一些数据分析的函数 (例如： corr, dot, exp);
	- 添加了第一个DaPy中的机器学习算法：DaPy.multilayer_periceptron.MLP;
	- 添加了一些标准数据集.
* V1.3.1 (2018-03-19)
	- 修复了在加载数据及中的bug;
	- 添加了支持保存数据集的功能.
* V1.2.5 (2018-03-15)
	- DaPy的第一个版本！

## License
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
