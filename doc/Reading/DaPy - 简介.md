# 比Pandas好用的数据分析框架：DaPy

![](https://img.shields.io/badge/Version-1.10.1-green.svg)  ![](https://img.shields.io/badge/Python2-pass-green.svg)![](https://img.shields.io/badge/Python3-pass-green.svg)![](https://img.shields.io/badge/Download-PyPi-green.svg)  ![](https://img.shields.io/badge/License-GNU-blue.svg)

# 一、项目介绍

### 1. DaPy 是什么？

​	DaPy是一个在设计时就非常关注易用性的数据分析库。通过提供设计合理的**数据结构**和丰富的**机器学习模型**，它能帮您快速地实现数据分析思路。简单来说，DaPy能帮助你完成数据挖掘任务中的每一步，导入导出数据、预处理数据、特征工程、模型训练和模型评估等。DaPy让数据分析师不用成为一个程序员。
	项目地址: [DaPy](https://github.com/JacksonWuxs/DaPy)

### 2. 为什么用DaPy？

​	总的来说，DaPy通过一系列精心设计的APIs接口和对于这些接口的优化，显著降低了数据分析过程中编程人员对于数据结构等编程技巧的要求。

* 符合人们习惯的数据结构

​	按行处理数据是符合我们每一个人想法的，因此几乎所有的数据库设计都是按照按行存储的。Pandas是Python语言中最常用的数据分析/数据处理框架。由于Pandas最早是为了处理时间序列数据而开发的，所以他的数据是按列存储的。虽然按列存储在全局处理上有不错的性能，但在进行“行操作”时却是一场灾难。由于缺乏替代品，人们不得不去适应Pandas的编程思维。比如，Pandas禁止对`DataFrame.iterrows()`迭代出来的每一行数据进行赋值操作。

​	DaPy看到了列存储在全局处理时的高效性，但同时也关注到了这类反直觉操作所耗费的大量时间与精力。针对这个问题，DaPy通过引入“视图”的概念使得人们不但可以高效地按照符合人们习惯的方式进行数据操作，同时也具备全局处理时的高性能。

* 多种在CMD中展示数据的方案

  下面我们将导入DaPy自带的经典鸢尾花数据集作为数据展示的样例。

```
>>> from DaPy.datasets import iris
>>> sheet, info = iris()
 - read() in 0.001s.
>>> sheet
sheet:data
==========
sepal length: <5.1, 4.9, 4.7, 4.6, 5.0, ... ,6.7, 6.3, 6.5, 6.2, 5.9>
 sepal width: <3.5, 3.0, 3.2, 3.1, 3.6, ... ,3.0, 2.5, 3.0, 3.4, 3.0>
petal length: <1.4, 1.4, 1.3, 1.5, 1.4, ... ,5.2, 5.0, 5.2, 5.4, 5.1>
 petal width: <0.2, 0.2, 0.2, 0.2, 0.2, ... ,2.3, 1.9, 2.0, 2.3, 1.8>
       class: <setos, setos, setos, setos, setos, ... ,virginic, virginic, virginic, virginic, virginic>
>>> sheet.info
sheet:data
==========
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=150 | Variables=5
3. Miss Value: 0 elements
                               Descriptive Statistics                                   
=======================================================================================
    Title     | Miss |    Min    |    Mean   |     Max     |     Std      |    Mode    
--------------+------+-----------+-----------+-------------+--------------+------------
 sepal length |  0   |  4.300001 | 5.8433333 | 7.900000095 | 0.8253012767 |          5
 sepal width  |  0   |         2 | 3.0540000 | 4.400000095 | 0.4321465798 |          3
 petal length |  0   |         1 | 3.7586666 | 6.900000095 |  1.758529178 |        1.5
 petal width  |  0   | 0.1000000 | 1.1986666 |         2.5 | 0.7606126088 |        0.2
    class     |  0   |         - |         - |           - |            - |      setos
=======================================================================================
>>> sheet.show(3)
sheet:data
==========
 sepal length | sepal width | petal length | petal width |  class  
--------------+-------------+--------------+-------------+----------
     5.1      |     3.5     |     1.4      |     0.2     |  setos   
     4.9      |     3.0     |     1.4      |     0.2     |  setos   
     4.7      |     3.2     |     1.3      |     0.2     |  setos     
                          .. Omit 144 Ln ..                         
     6.5      |     3.0     |     5.2      |     2.0     | virginic 
     6.2      |     3.4     |     5.4      |     2.3     | virginic 
     5.9      |     3.0     |     5.1      |     1.8     | virginic 
```

* 支持优雅的链式表达

  让我们来做一个稍微有趣点的链式表达! 我希望对于前面的鸢尾花数据集在一行代码中完成下面的6个操作。

（1）对于每一列数据分别进行标准化操作；

（2）然后找到在标准化以后满足sepal length小于petal length的记录；

（3）对于筛选出来的数据集按照鸢尾花的类别class进行分组；

（4）对于每个分组都按照petal width进行升序排序；

（5）对于排好序后的分组选取前10行记录；

（6）对于每个由前十行记录构成的子数据集进行描述性统计；

```
>>> sheet.normalized().query('sepal length < petal length').groupby('class').sort(' petal width')[:10].info   # 这就是链式表达的代码
 - normalized() in 0.005s.
 - query() in 0.000s.
 - groupby() in 0.000s.
 - sort() in 0.000s.
sheet:('virginic',)
===================
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=10 | Variables=5
3. Miss Value: 0 elements
                                Descriptive Statistics                                 
=======================================================================================
    Title     | Miss |    Min    |   Mean   |    Max     |     Std      |     Mode     
--------------+------+-----------+----------+------------+--------------+--------------
 sepal length |  0   |   0.16666 | 0.572218 | 0.83333331 |  0.177081977 | 0.5555555173
 sepal width  |  0   | 0.0833333 | 0.295832 | 0.41666665 |  0.102824685 | 0.3749999851
 petal length |  0   |  0.593220 | 0.747457 | 0.89830505 | 0.0852358577 | 0.8135593089
 petal width  |  0   |  0.541666 | 0.654166 | 0.70833331 | 0.0619419457 | 0.7083333332
    class     |  0   |         - |        - |          - |            - |     virginic
=======================================================================================
sheet:('setos',)
================
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=6 | Variables=5
3. Miss Value: 0 elements
                                Descriptive Statistics                                 
=======================================================================================
    Title     | Miss |    Min    |   Mean   |    Max    |     Std      |      Mode     
--------------+------+-----------+----------+-----------+--------------+---------------
 sepal length |  0   | -5.29e-08 | 0.050925 |  0.138888 | 0.0465272020 | 0.02777772553
 sepal width  |  0   |     0.375 |  0.45833 |  0.583333 | 0.0680413746 |  0.4166666501
 petal length |  0   |    0.0169 | 0.070621 |  0.152542 | 0.0419945431 | 0.05084745681
 petal width  |  0   | -6.20e-10 | 0.034722 | 0.0416666 | 0.0155282505 | 0.04166666607
    class     |  0   |         - |        - |         - |            - |         setos
=======================================================================================
sheet:('versicolo',)
====================
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=10 | Variables=5
3. Miss Value: 0 elements
                                Descriptive Statistics                                 
=======================================================================================
    Title     | Miss |   Min    |   Mean   |    Max     |     Std      |     Mode     
--------------+------+----------+----------+------------+--------------+--------------
 sepal length |  0   |  0.16666 | 0.308333 | 0.47222217 |   0.10126514 | 0.1944443966
 sepal width  |  0   |        0 |  0.16666 | 0.29166665 | 0.0790569387 |   0.16666666
 petal length |  0   | 0.338983 | 0.442372 | 0.52542370 | 0.0564434752 | 0.3898305022
 petal width  |  0   |    0.375 |  0.38749 | 0.41666665 | 0.0190940608 | 0.3749999996
    class     |  0   |        - |        - |          - |            - |    versicolo
=======================================================================================
```

* 强大的I/O功能

  我们都会遇到过一个问题，怎么把csv转换成Excel；或者反过来，Excel转回csv?

```
>>> sheet.groupby('class').save('iris.xls') # 对！直接链式表达转成了xls! 别忘了Excel是支持多子表的，所以刚刚groupby之后DaPy给你存了三个子表！
 - groupby() in 0.000s.
 - save() in 0.241s.
>>> import DaPy as dp
>>> dp.read('iris.xls').shape # DaPy竟然又一次性读完了三个表！！！
 - read() in 0.004s.
sheet:('virginic',)
===================
sheet(Ln=50, Col=5)
sheet:('setos',)
================
sheet(Ln=50, Col=5)
sheet:('versicolo',)
====================
sheet(Ln=50, Col=5)
```

​	除此以外，DaPy的I/O工具还支持更为灵活的数据源。

```
>>> import DaPy as dp
>>> dp.read('iris.xls').save('iris.db') # Excel 转 Sqlite3
>>> dp.read('iris.sav').save('iris.html') # SPSS 转 HTML
>>> # 爬取FIFA球员数据并存入MySQL数据库
>>> dp.read('http://sofifa.com/players').save('mysql://root:123@localhost:3306/db') 
>>> dp.read('mysql://root:123@localhost:3306/db').save('fifa.csv') # MySQL 转 CSV
```

* 全面的数据预处理和特征工程

  部分数据预处理的函数

```
>>> sheet.drop_duplicates(keep='first') #删除重复记录
>>> sheet.fillna(method='linear') #线性插值法填充缺失值
>>> sheet.drop('ID', axis=1) # 删除无用变量
>>> sheet.count_values('gender') # 对于某个变量进行计数统计
```

​	部分特征工程的函数

```
>>> sheet.get_date_label('birth') # 对日期变量做变化，会自动生成一大堆周期性变量
>>> sheet.get_categories(cols='age', cutpoints=[18, 30, 50], group_name=['青年', '壮年', '中年', '老年']) # 对于连续型变量进行封箱操作
>>> sheet.get_dummies(['city', 'education']) # 对于分类变量进行虚拟变量的引入
>>> sheet.get_interactions(n_power=3, col=['income', 'age', 'gender', 'education']) # 为你选定的变量之间构成高阶交叉项，阶数n_power可以随便填！！！
```

### 3. 什么时候应该使用DaPy

* 针对这个数据我知道一个baseline级别的思路，我想**快速实现思路**的方向；
* 我希望解决项目论文中的数据分析工作，并**不用于工程项目**；
* 不要让数据格式错误或者找不到接口这些编码级别的问题打断我的思路；
* Pandas的操作太不人性了，但是Excel又解决不了着几百万级别的数据，我有其他选择吗？
* 我希望进行更多统计学的相关假设检验的实验；

---

# 二、快速上手

- Python 版本: 3.5
- 运行方式：命令行

## 2.1 安装

1. 安装 DaPy

```shell
pip install DaPy
```


安装完成之后, 就可以直接在命令行中使用了! 

## 2.2 快速体验一个机器学习

​	（1）导入经典的鸢尾花数据集，并且查看该数据集的基本描述性统计信息。

```
>>> from DaPy.datasets import iris
>>> data, info = iris()
 - read() in 0.017s.
>>> data.info
sheet:data
==========
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=150 | Variables=5
3. Miss Value: 0 elements
                                Descriptive Statistics                                 
=======================================================================================
    Title     | Miss |    Min     |    Mean   |    Max     |     Std      |    Mode    
--------------+------+------------+-----------+------------+--------------+------------
 sepal length |  0   |  4.3000191 | 5.8433333 | 7.90000095 | 0.8253012767 |          5
 sepal width  |  0   |          2 | 3.0540003 | 4.40000095 | 0.4321465798 |          3
 petal length |  0   |          1 | 3.7586666 | 6.90000095 |  1.758529178 |        1.5
 petal width  |  0   | 0.10000015 | 1.1986658 |        2.5 | 0.7606126088 |        0.2
    class     |  0   |          - |         - |          - |            - |      setos
=======================================================================================
```

​	基于上述结果我们可以看到，我们总共有4个自变量，总计150条数据，不存在缺失值。

​	（2）接下来我们比较一下各个目标类别的差别

```
>>> data.groupby('class').info
 - groupby() in 0.001s.
sheet:('virginic',)
===================
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=50 | Variables=5
3. Miss Value: 0 elements
                                 Descriptive Statistics                                
=======================================================================================
    Title     | Miss |    Min    |    Mean   |     Max     |     Std      |    Mode    
--------------+------+-----------+-----------+-------------+--------------+------------
 sepal length |  0   | 4.9000095 | 6.5880002 | 7.900000095 | 0.6294886326 |        6.3
 sepal width  |  0   | 2.2000048 | 2.9739999 | 3.799999952 | 0.3192553797 |          3
 petal length |  0   |       4.5 | 5.5519999 | 6.900000095 | 0.5463478596 |        5.1
 petal width  |  0   | 1.3999999 | 2.0259999 |         2.5 | 0.2718896914 |        1.8
    class     |  0   |         - |         - |           - |            - |   virginic
=======================================================================================
sheet:('setos',)
================
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=50 | Variables=5
3. Miss Value: 0 elements
                                Descriptive Statistics                                 
=======================================================================================
    Title     | Miss |    Min     |    Mean    |    Max     |    Std      |    Mode    
--------------+------+------------+------------+------------+-------------+------------
 sepal length |  0   |  4.3000001 |  5.0060004 |  5.8000091 | 0.348946968 |          5
 sepal width  |  0   |  2.2999999 |  3.4180007 |  4.4000095 |   0.3771949 |        3.4
 petal length |  0   |          1 |  1.4639996 |  1.8999999 | 0.171767294 |        1.5
 petal width  |  0   | 0.10000015 | 0.24400048 | 0.60000008 | 0.106131996 |        0.2
    class     |  0   |          - |          - |          - |           - |      setos
=======================================================================================
sheet:('versicolo',)
====================
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=50 | Variables=5
3. Miss Value: 0 elements
                                 Descriptive Statistics                                
======================================================================================
    Title     | Miss |    Min    |    Mean   |     Max     |     Std      |    Mode    
--------------+------+-------------+-------------+-------------+--------------+------------
 sepal length |  0   | 4.9000000 | 5.9359999 |           7 | 0.5109833734 |        5.5
 sepal width  |  0   |         2 |  2.770000 | 3.400000095 | 0.3106444926 |          3
 petal length |  0   |         3 | 4.2599999 | 5.099999905 | 0.4651881313 |        4.5
 petal width  |  0   |         1 | 1.3259999 | 1.799999952 | 0.1957651626 |        1.3
    class     |  0   |         - |         - |           - |            - |  versicolo
=======================================================================================
```

​	上面的信息告诉我们，三个类别各自有50条记录，不同的类别下的自变量差异显著。

​	（3）下面我们要进行标准化处理，并且把自变量和因变量独立开来

```
>>> sheet = sheet.shuffle().normalized()
 - shuffle() in 0.001s.
 - normalized() in 0.005s.
>>> X, Y = sheet[:'petal width'], sheet['class']
```

​	（4）导入经典的多层感知机分类器并训练模型，最后保存训练好的模型

```
>>> from DaPy.methods.classifiers import MLPClassifier
>>> mlp = MLPClassifier().fit(X[:120], Y[:120]).save('mymodel.pkl')
 - Structure | Input:4 - Dense:4 - Output:3
 - Finished: 0.2%	Epoch: 1	Rest Time: 0.24s	Accuracy: 0.33%
 - Finished: 99.8%	Epoch: 500	Rest Time: 0.00s	Accuracy: 0.88%
 - Finish Train | Time:1.9s	Epoch:500	Accuracy:88.33%
```

​	（5）性能测试

```
>>> from DaPy.methods.evaluator import Performance
>>> Performance(mlp, X[120:], Y[120:], mode='clf') # 性能测试包括了正确率、kappa系数和混淆矩阵，二分类任务会包含AUC
 - Classification Accuracy: 86.6667%
 - Classification Kappa: 0.8667
┏                   ┓
┃ 11   0    0    11 ┃
┃ 0    8    1    9  ┃
┃ 0    3    7    10 ┃
┃11.0 11.0 8.0  30.0┃
┗                   ┛
```

