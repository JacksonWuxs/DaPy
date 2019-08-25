### DaPy - 带你领略从未有过的丝滑般体验
总因为Pandas严格的数据结构要求让你感受到很苦恼？为了实现一个简单的操作也要查阅很多的文档而头疼？

DaPy来解放你啦！你可以用[DaPy](https://github.com/JacksonWuxs/DaPy/blob/master/README_Chinese.md)流利地实现脑子早已思索好的想法，不再因为**找不到API**或者**数据格式报错**而打断你的思路！DaPy是一个从设计开始就非常关注易用性的数据分析框架，它专为数据分析师而设计，而不是程序员。对于数据分析师而言，你的价值是解决问题思路！而不是害得你996的几百行代码！

### DaPy有多友好？

##### 1. 多种在CMD中呈现数据的方式
不要小看浏览数据的方式！对于数据分析师而言，感知数据是非常重要的！
```python
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
>>> sheet.show(5)
sheet:data
==========
 sepal length | sepal width | petal length | petal width |  class  
--------------+-------------+--------------+-------------+----------
     5.1      |     3.5     |     1.4      |     0.2     |  setos   
     4.9      |     3.0     |     1.4      |     0.2     |  setos   
     4.7      |     3.2     |     1.3      |     0.2     |  setos   
     4.6      |     3.1     |     1.5      |     0.2     |  setos   
     5.0      |     3.6     |     1.4      |     0.2     |  setos   
                          .. Omit 140 Ln ..                          
     6.7      |     3.0     |     5.2      |     2.3     | virginic 
     6.3      |     2.5     |     5.0      |     1.9     | virginic 
     6.5      |     3.0     |     5.2      |     2.0     | virginic 
     6.2      |     3.4     |     5.4      |     2.3     | virginic 
     5.9      |     3.0     |     5.1      |     1.8     | virginic 
```
##### 2. 符合人们习惯的二维数据表结构
按行处理数据是符合我们每一个人想法的，因此几乎所有的数据库设计都是按照按行存储的。由于Pandas最早是为了处理时间序列数据而开发的，所以他的数据是以列进行的存储。虽然这种存储方式在全局处理上表现出了不错的性能，但没优化情况下行操作却让人较为难以忍受的。由于没有什么更好的替代品，人们不得不花很多时间去适应Pandas的编程思维。比如，Pandas不支持对于`DataFrame.iterrows()`迭代出来的行进行赋值操作。这个功能即使如此常用，在NumPy中也是原生支持的功能在Pandas里却是被禁止的。

针对这类由行操作引发的问题，DaPy通过引入“视图”的概念重新优化了按行操作这个符合人们习惯的操作方式。

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

##### 3. 对了，听说有人喜欢链式表达？
让我们来做一个稍微有趣点的链式表达! 我希望对于经典的鸢尾花数据集在一行代码中完成下面的6个操作。

（1）对于每一列数据分别进行标准化操作；

（2）然后找到在标准化以后满足sepal length小于petal length的记录；

（3）对于筛选出来的数据集按照鸢尾花的类别class进行分组；

（4）对于每个分组都按照petal width进行升序排序；

（5）对于排好序后的分组选取前10行记录；

（6）对于每个由前十行记录构成的子数据集进行描述性统计；
```python
>>> from DaPy.datasets import iris
>>> sheet, info = iris()
 - read() in 0.001s.
>>> sheet.normalized().query('sepal length < petal length').groupby('class').sort('petal width')[:10].info
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
##### 4. 一些numpy和pandas优良的特性他也保留了
```python
>>> sheet.A + sheet.B # 下标访问列并且做四则运算
>>> sheet[sheet.A > sheet.B] # 这个非常Pythonic的切片写法！
```
### 除了语法特性上的优化，还有没有其他的硬家伙？

##### 1. 超级NB的、鲁棒性极强的I/O工具！！！
我们都会遇到过一个问题，怎么把csv转换成Excel；或者反过来，Excel转回csv?
```python
>>> from DaPy.datasets import iris
>>> sheet, info = iris()
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
你以为read函数就这点水平吗？让我们来看看更骚的！！！
```python
>>> import DaPy as dp
>>> dp.read('iris.xls').save('iris.db') # Excel 转 Sqlite3
>>> dp.read('iris.sav').save('iris.html') # SPSS 转 HTML
>>> # 爬取FIFA球员数据并存入MySQL数据库
>>> dp.read('https://sofifa.com/players').save('mysql://root:123@localhost:3306/fifa_db') 
>>> dp.read('mysql://root:123123@localhost:3306/fifa_db').save('fifa.csv') # MySQL 转 CSV
```
##### 2. 支持超级多的数据预处理或者特征工程的操作
先来一些数据预处理的
```python
>>> sheet.drop_duplicates(keep='first') #删除重复记录
>>> sheet.fillna(method='linear') #线性插值法填充缺失值
>>> sheet.drop('ID', axis=1) # 删除无用变量
>>> sheet.count_values('gender') # 对于某个变量进行计数统计
```
再来一些特征工程的
```python
>>> sheet.get_date_label('birth') # 对日期变量做变化，会自动生成一大堆周期性变量
>>> sheet.get_categories(cols='age', cutpoints=[18, 30, 50], group_name=['青年', '壮年', '中年', '老年']) # 对于连续型变量进行封箱操作
>>> sheet.get_dummies(['city', 'education']) # 对于分类变量进行虚拟变量的引入
>>> sheet.get_interactions(n_power=3, col=['income', 'age', 'gender', 'education']) # 为你选定的变量之间构成高阶交叉项，阶数n_power可以随便填！！！
```
##### 3. 最最后，重中之重，机器学习模块！
在DaPy里面，已经内置了四个模型，分别是线性回归、逻辑回归、多层感知机和C4.5决策树。在模型这一块的话，DaPy的开发团队认为sklearn和tensorflow已经做得很好了。出于开发团队主要成员是统计系学生的关系，他们的思路是增加更多的统计学检验报告~
我们先看看一个demo级别的样例好了

```python
>>> from DaPy.datasets import iris
>>> sheet, info = iris()
 - read() in 0.001s.
>>> sheet = sheet.shuffle().normalized()
 - shuffle() in 0.001s.
 - normalized() in 0.005s.
>>> X, Y = sheet[:'petal width'], sheet['class']
>>> 
>>> from DaPy.methods.classifiers import MLPClassifier
>>> mlp = MLPClassifier().fit(X[:120], Y[:120])
 - Structure | Input:4 - Dense:4 - Output:3
 - Finished: 0.2%	Epoch: 1	Rest Time: 0.24s	Accuracy: 0.33%
 - Finished: 99.8%	Epoch: 500	Rest Time: 0.00s	Accuracy: 0.88%
 - Finish Train | Time:1.9s	Epoch:500	Accuracy:88.33%
>>> 
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

### 最后，如果你觉得DaPy不错的话，[去Github点个Star吧](https://github.com/JacksonWuxs/DaPy)！也很欢迎来提Issue哟！！！