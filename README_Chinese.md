<img src="https://github.com/JacksonWuxs/DaPy/blob/master/doc/material/DaPy.png">
<i>本开源项目流利地实现你在数据挖掘中的想法</i>

# DaPy - 享受你的数据挖掘之旅

![](https://img.shields.io/badge/Version-1.10.1-green.svg)  ![](https://img.shields.io/badge/Python2-pass-green.svg)![](https://img.shields.io/badge/Python3-pass-green.svg)![](https://img.shields.io/badge/Download-PyPi-green.svg)  ![](https://img.shields.io/badge/License-GNU-blue.svg)

[英文版](https://github.com/JacksonWuxs/DaPy/blob/master/README.md)

### 简介

DaPy是一个在设计时就非常关注易用性的数据分析库。通过为您提供设计合理的**数据结构**和丰富的**机器学习模型**，它能帮您快速地实现数据分析思路。简单来说，本项目能帮你完成数据挖掘任务中的每一步，导入导出数据、预处理数据、特征工程、模型训练和模型评估等。

### 示例

本示例简单展示了DaPy的功能。我们的任务是为鸢尾花分类任务训练一个分类器。更详细的信息可以参阅[这里](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Quick%20Start/English.md)。

![](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Quick%20Start/quick_start.gif)

### Why I need DaPy?

我们已经有了例如Numpy和Pandas这样优秀的数据分析库，为什么我们还需要DaPy？ 

上面那个问题的答案就是， <u>*DaPy专为数据分析师设计，而不是程序员.*</u>  DaPy的用户只需要关注于他们解决问题的思路，而不必太在意数据结构这些编程陷阱。

例如，由于Pandas最早是为了处理时间序列而设计的，它不支持对于`DataFrame.iterrows()`迭代出来的行进行操作，故在Pandas中对行操作数据是一个不太好的想法。然而，DaPy依靠“视图”这个概念解决了这个问题，能让大家轻松地按照符合人们习惯的方式按行处理数据。

```python
>>> import DaPy as dp
>>> sheet = dp.SeriesSet({'A': [1, 2, 3], 'B': [4, 5, 6]})
>>> for row in sheet:
		print(row.A, row[0])   # 按照下标或者列名访问行数据的值
		row[1] = 'b'   # 用下表为行赋值
1, 1
2, 2
3, 3
>>> sheet.show()   # 你的操作会映射到原表中
 A | B
---+---
 1 | b 
 2 | b 
 3 | b 
>>> row0 = sheet[0]   # 拿到行的索引 
>>> row0
[1, 'b']
>>> sheet.append_col(series=[7, 8, 9], variable_name='newColumn') # 为表添加新列
>>> sheet.show()
 A | B | newColumn
---+---+-----------
 1 | b |     7     
 2 | b |     8     
 3 | b |     9     
>>> row0   # 表的操作会时时刻刻反映到行上
[1, 'b', 7]
```

### 特性

我们希望DaPy是一个对用户友好的框架，为此, 我们极力优化DaPy的API接口设计，以便让你更快地适应和灵活地使用它。下面是DaPy较为友好的功能:  

- 多种在CMD中呈现数据的方式
- 符合Python语法习惯的二维数据表结构
- 与SQL语法相似的函数封装方法
- 封装了许多常用的数据预处理或者特征工程方法
- 支持多种文件格式的I/O工具 (支持格式：.html, .xls, .db, .csv, .sav)
- 内建基本机器学习模型(决策树、多层感知机、线性回归等)

另外, 为了让DaPy能应付真实世界中的任务, 我们还时刻关注DaPy的*性能表现*。虽然DaPy目前是由纯Python语言实现的，但它与现有的数据处理框架在性能上也具有可比性。下图展示了使用具有432万条记录及7个变量的数据集的性能测试结果。

![](https://github.com/JacksonWuxs/DaPy/blob/master/doc/material/Result.jpg)

### 安装

最新版的DaPy-1.10.1已经上传到了PyPi
```
pip install DaPy
```
DaPy中的部分功能依赖于下述这些第三方库：

- **xlrd**: loading data from .xls file【必要】
- **xlwt**: export data to a .xls file【必要】
- **repoze.lru**: speed up loading data from .csv file【必要】
- **savReaderWrite**: loading data from .sav file【可选】
- **bs4.BeautifulSoup**: auto downloading data from a website【可选】
- **numpy**: dramatically increase the efficiency of ML models【推荐】 

### 用法说明

- 加载数据 & 数据探索
  - 从本地csv, sav, sqlite3或xls文件中加载数据: ```sheet = DaPy.read(file_addr)```
  - 显示数据的前后5条记录: ```sheet.show(lines=5)```
  - 汇总每一个变量的统计指标（均值、方差等）: ```sheet.info```
  - 统计某分类变量的取值分布情况: ```sheet.count_values('gender')```
  - 探索分类变量不同取值之间的差异: ```sheet.groupby('city')```
  - 计算连续变量间的相关性: ```sheet.corr(['age', 'income'])```
- 预处理数据 & 数据清洗
  - 删除重复记录: `sheet.drop_duplicates(col, keep='first')`
  - 用线性插值法填充缺失值: ```sheet.fillna(method='linear')``` 
  - 去除缺失值数量超过50%的记录```sheet.dropna(axis=0, how=0.5)```
  - 移除一些无用变量（如. 客户*ID*）: ```sheet.drop('ID', axis=1)```
  - 基于某一列数据进行排序: ```sheet = sheet.sort('Age', 'DESC')```
  - 合并另一张表中新的字段: ```sheet.merge(sheet2, keep_same=False)```
  - 合并另一张表中新的记录: `sheet.join(sheet2)`
  - 逐条添加记录: `sheet.append_row(new_row)`
  - 逐个添加变量: `sheet.append_col(new_col)`
  - 按索引去除部分数据: `sheet[:10, 20: 30, 50: 100]`
  - 按列名去除部分数据: `sheet['age', 'income', 'name']`
- 特征工程

  

  - 使用日期变量构造一些分类变量（季节、周末等）: `sheet.label_date('birth')`
  - 将连续变量通过“封箱”转换为分类变量: `sheet.get_categories(cols='age', cutpoints=[18, 30, 50], group_name=['Juveniles', 'Adults', 'Wrinkly', 'Old'])`
  - 将单个分类变量转换为多个虚拟变量: `sheet.get_dummies(['city', 'education'])`
  - 为你选定的变量之间构建高阶交叉项: `sheet.get_interactions(n_power=3, col=['income', 'age', 'gender', 'education'])`
  - 为每个变量中的记录添加排名: `sheet.get_ranks(cols='income', duplicate='mean')`
  - 归一化一些连续变量: ```sheet.normalized(col='age')```
  - 对数归一化一些连续变量: ```sheet.normalized('log', col='salary')```
  - 使用符合您业务需求的函数构造新变量: ```sheet.apply(func=calculate_tax, col=['salary', 'income'])```
  - 使用差分让时间序列平稳: `DaPy.diff(sheet.income)`
- 模型训练
  - 选择并初始化一个模型: ```m = MLP()```, ```m = LinearRegression()```, ```m = DecisionTree()``` or  ```m = DiscriminantAnalysis()``` 
  - 训练模型参数: ```m.fit(X_train, Y_train)```
- 模型评估
  - 使用参数检验对模型进行评估（仅限线性回归和判别分析）: ```m.report.show()```
  - 通过可视化评估模型: ```m.plot_error()``` or ```DecisionTree.export_graphviz()```
  - 使用测试集评估模型: ```DaPy.methods.Performance(m, X_test, Y_test, mode)```.
- 保存结果
  - 保存模型: ```m.save(addr)```
  - 保存数据: ```sheet.save(addr)```


### TODO  

:heavy_check_mark: = 已完成      :running: = 正在开发       ​ :calendar:  = 规划中       :thinking: = 未排期

* 数据结构

  * DataSet (3-D data structure) :heavy_check_mark:
  * Frame (2-D general data structure)​ :heavy_check_mark:
  * SeriesSet (2-D general data structure) :heavy_check_mark:
  * Matrix (2-D mathematical data structure) :heavy_check_mark:
  * Row (1-D general data structure) :heavy_check_mark:
  * Series (1-D general data structure) :heavy_check_mark:
  * TimeSeries (1-D time sequence data structure)​ :running:

* 统计

  * 基本统计功能 (mean, std, skewness, kurtosis, frequency, fuantils)​ :heavy_check_mark:
  * 相关性分析 (spearman & pearson) :heavy_check_mark:

  * 方差分析 :heavy_check_mark:
  * 均值比较 (simple T-test, independent T-test) :thinking:

* 操作

  * 易用的API设计 (create, Retrieve, Update, Delete)  :heavy_check_mark:
  * 灵活的I/O工具 (supporting multiple source data for input and output) :heavy_check_mark:
  * 虚拟变量 :heavy_check_mark:
  * 差分序列模型:heavy_check_mark:
  * 数据标准化 (log, normal, standard, box-cox):heavy_check_mark:
  * 数据去重 :heavy_check_mark:
  * 聚合函数 :heavy_check_mark:

* 模型

  - 判别分析 :heavy_check_mark:
  - 线性回归  :heavy_check_mark:
  - 多层感知机 :heavy_check_mark:
  - 决策树 :heavy_check_mark:
  - K-Means :running:
  - PCA (Principal Component Analysis) :running:
  - ARIMA (Autoregressive Integrated Moving Average) :calendar:
  - SVM ( Support Vector Machine) :thinking:
  - Bayes Classifier :thinking:

* 其他

  * 手册 :running:
  * 示例 :running:
  * 单元测试 :running:

### 项目成员

- ###### 负责人:

  Xuansheng WU (@JacksonWoo: wuxsmail@163.com )

- ###### 开发者：

  1. Xuansheng WU
  2. Feichi YANG  (@Nick Yang: yangfeichi@163.com)  

### 版本日志

* V1.10.1 (2019-06-13)
  * 添加 ```SeriesSet.update()```, 更新某些数据中的一些记录信息;
  * 添加 ```BaseSheet.tolist()``` and ```BaseSheet.toarray()```, 将表转换为list嵌套list的结构或者numpy结构;
  * 添加 ```BaseSheet.query()```, 通过一个Python句法书写的字符串筛选符合条件的记录;
  * 添加 ```SeriesSet.dropna()```, 提出包含缺失值的记录或变量;
  * 添加 ```SeriesSet.fillna()```, 为缺失值填补固定值或者线性插值法填补;
  * 添加 ```SeriesSet.label_date()```, 为时间变量构造新的解释变量;
  * 添加 ```DaPy.Row```, 原始数据表一条记录的视图;
  * 添加 ```DaPy.methods.DecitionTree```, C4.5决策树分类器算法的实现;
  * 添加 ```DaPy.methods.SignTest```, 符号检验;
  * 重构 ```DaPy.core.base```;
  * 优化 ```BaseSheet.groupby()```, 以前性能的18倍;
  * 优化 ```BaseSheet.select()```, 以前性能的14倍;
  * 优化 ```BaseSheet.sort()```, 以前性能的2倍;
  * 优化 ```dp.save()```, 保存.csv的性能是以前1.6倍;
  * 优化 ```dp.read()```, 加载数据的性能是以前1.1倍;
* V1.9.2 (2019-04-23)
  * 添加 `BaseSheet.groupby()`, 基于特定列为记录进行分类分析;
  * 添加 `DataSet.apply()`, 对数据集映射一个函数;
  * 添加 `DataSet.drop_duplicates()`, 自动去除数据集中的重复值;
  * 添加 `DaPy.Series`, 用于保存序列数据的新数据结构;
  * 添加 `DaPy.methods.Performance()`, 自动评价一个机器学习模型的性能;
  * 添加 `DaPy.methods.Kappa()`, 计算给定混淆矩阵的Kappa系数;
  * 添加 `DaPy.methods.ConfuMat()`, 基于给定真实值和预测值生成混淆矩阵;
  * 更新 `BaseSheet.select()`, 支持新的字段 limit 和 columns;
* V1.7.2 Beta (2019-01-01)
  * 添加 `get_dummies()` , 引入虚拟变量方式处理名义变量;
  * 添加 `DaPy.show_time`, DaPy开始具备日志功能;
  * 添加 `boxcox()` , Box-Cox转换;
  * 添加 `diff()`, 对时间序列进行差分;
  * 添加 `DaPy.methods.LDA`, 判别分析模型（支持线性判别法和Fisher判别法）;
  * 添加 `row_stack()`, 纵向合并多个数据表;
  * 添加 `Row`，新数据结构更好地以*视图*方式访问一行数据;
  * 添加 `LinearRegression.report`,  访问该模型训练集上的参数检验统计报告;
  * 更新 `read()`, 支持自动从网页中爬取数据;
  * 更新 `SeriesSet.merge()`, 更多可用的参数;
  * 重命名 `DataSet.pop_miss_value()`  为 `DataSet.dropna()`;
  * 重构 `methods`, more stable and more scalable in the future;
  * 重构 `methods.LinearRegression`, it can prepare a statistic report for you after training;
  * 重构 `BaseSheet.select()`, 5 times faster and more pythonic API design;
  * 重构 `BaseSheet.replace()`, 20 times faster and more pythonic API design;
  * 开始支持Python 3！
  * 修复了一些小Bug;
* V1.5.1 (2018-11-17)
  * 添加 `select()`, 快速基于某些条件筛选数据;
  * 添加 `delete()`, 按照某个坐标轴删除一个非DaPy数据结构的数据;
  * 添加 `column_stack()`, 横向合并多个数据表;
  * 添加 DaPy.P() 和 DaPy.C()函数，用于计算排列数和组合数；
  * 添加 语法特性，使得用户可以通过data.title来访问表结构中的列;
  * 重构 DaPy.BaseSheet类，精简代码体积并提高了拓展性;
  * 重构 DaPy.DataSet.save()函数，提高了代码稳定性及拓展能力；
  * 重写 部分基本数学函数的算法；
  * 修复 一些细小的bug;
* V1.3.3 (2018-06-20)
  - 添加 外部数据文件读取能力: Excel, SPSS, SQLite3, CSV;
  - 重构 DaPy架构, 提高了远期拓展能力;
  - 重构 DaPy.DataSet类, 一个DataSet实例可以批量管理多个数据表;
  - 重构 DaPy.Frame类, 删除了格式验证, 适配更多类型的数据集;
  - 重构 DaPy.SeriesSet类, 删除了格式验证, 适配更多类型的数据集;
  - 移除 DaPy.Table类;
  - 优化 DaPy.Matrix类, 效率提升接近2倍;
  - 优化 DaPy.Frame 及 Data.SeriesSet类的展示, 数据呈现更为清晰美观;
  - 添加 `线性回归`及`方差分析`至DaPy.stats;
  - 添加 DaPy.io.encode()函数, 更好地适配中文数据;
  - 替换 read_col(), read_frame(), read_matrix() 为 read()函数;
* V1.3.2 (2018-04-26)
  - 优化 数据加载的效率;
  - 添加 更多实用的功能到DaPy.DataSet中;
  - 添加 新的数据结构DaPy.Matrix,支持常规的矩阵运算;
  - 添加 常用描述数据的函数 (例如： corr, dot, exp);
  - 添加 `多层感知机`至DaPy.machine_learn;
  - 添加 一些标准数据集.
* V1.3.1 (2018-03-19)
  - 修复 在加载数据时的bug;
  - 添加 支持保存数据集的功能.
* V1.2.5 (2018-03-15)
  - DaPy的第一个版本！

### 开源协议

Copyright (C) 2018 - 2019 Xuansheng Wu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https:\\www.gnu.org\licenses.