DaPy - 别让语法束缚了思想
====
![](https://img.shields.io/badge/Version-1.3.3-green.svg)  ![](https://img.shields.io/badge/Download-PyPi-green.svg)  ![](https://img.shields.io/badge/License-GNU-blue.svg)  

作为一个基于Python原生数据结构搭建的数据分析和数据挖掘库，**DaPy**致力于节约数据科学家的时间并提高他们的研究效率，同时它也在尽其所能为他们提供更舒适和流畅的操作体验。

[安装](#安装) | [特性](#特性) | [快速开始](https://github.com/JacksonWuxs/DaPy/blob/master/快速开始.md) | [远期规划](#远期规划) | [更新日志](#更新日志) | [版权归属](#版权归属) [用户手册](https://github.com/JacksonWuxs/DaPy/blob/master/Guide%20Book/Chinese/README.md)| [English](https://github.com/JacksonWuxs/DaPy/blob/master/README.md)

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
**便捷**和**高效**是DaPy的设计基石。
从一开始，我们就让
DaPy使用尽可能多的Python原生数据结构，且试图让
它支持更多的Python语法习惯。所以你可以快速地适应DaPy。
此外，我们尽力简化
其中的公式或数学模型，以便让你
流利地实现你的想法。

* 以下是DaPy做得很好的一些事情：
	- 使用`DataSet`结构高效管理各种数据文件。
	- 快速完成"增删改查"操作。
	- 使您可以轻松访问数据集的一部分，不仅可以通过索引或变量名称，还可以通过特殊条件访问。
	- 从CSV文件，Excel文件，数据库甚至SPSS文件加载数据的强大I/O工具。
	- 按多种条件对记录进行排序。
	- 使用内置分析模型快速验证您的想法（例如“方差分析”，“多层感知机”，“线性回归”）。
	- 多种方法可帮助您轻松感知数据集。
  
即使它使用Python原生数据结构，
DaPy的效率仍然与C写的一些库相当。
我们在平台上测试了DaPy
Intel i7-6560U虽然Python版本是2.7.13-64Bit。数据集
（[下载](https://pan.baidu.com/s/1kK3_V8XbbVim4urDkKyI8A)）
有超过450万条记录，总规模是
240.2 MB。

如果您想跟进最新进展，可以访问[这里](https://www.teambition.com/project/5b1b7bd40b6c410019df8c41/tasks/scrum/5b1b7bd51e4661001838eb10)。
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
	- PCA (主成分分析)
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
