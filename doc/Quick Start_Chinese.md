## 快速开始
#### Ⅰ. 加载数据集
DaPy自带了少量著名的数据集，比如用于分类问题的**红酒分类**和**鸢尾花**数据集。
接下来，我们首先启动一个Python Shell并加载作为例子的红酒数据集：
```Python
>>> from DaPy import datasets
>>> from DaPy import machine_learn
>>> wine, info = datasets.wine()
```
这个函数会返回一个内部由*DaPy.SeriesSet*结构包装的数据集，同时还会返回一个
数据集的官方简介。
  
一般来说，如果要加载一个外部的数据集，你可以通过如下的语法：
```Python
>>> data = dp.read(file_name)
```
本例中，作为一个监督学习问题，所有的自变量和因变量都被包含在了一个*SeriesSet*结构中。
为此，我们可以通过如下的方式观察*红酒*数据集的信息。
```Python
>>> wine
             Alcohol: <14.23, 13.2, 13.16, 14.37, 13.24, ... ,13.71, 13.4, 13.27, 13.17, 14.13>
          Malic acid: <1.71, 1.78, 2.36, 1.95, 2.59, ... ,5.65, 3.91, 4.28, 2.59, 4.1>
                 Ash: <2.43, 2.14, 2.67, 2.5, 2.87, ... ,2.45, 2.48, 2.26, 2.37, 2.74>
   Alcalinity of ash: <15.6, 11.2, 18.6, 16.8, 21.0, ... ,20.5, 23.0, 20.0, 20.0, 24.5>
           Magnesium: <127, 100, 101, 113, 118, ... ,95, 102, 120, 120, 96>
       Total phenols: <2.8, 2.65, 2.8, 3.85, 2.8, ... ,1.68, 1.8, 1.59, 1.65, 2.05>
          Flavanoids: <3.06, 2.76, 3.24, 3.49, 2.69, ... ,0.61, 0.75, 0.69, 0.68, 0.76>
Nonflavanoid phenols: <0.28, 0.26, 0.3, 0.24, 0.39, ... ,0.52, 0.43, 0.43, 0.53, 0.56>
     Proanthocyanins: <2.29, 1.28, 2.81, 2.18, 1.82, ... ,1.06, 1.41, 1.35, 1.46, 1.35>
     Color intensity: <5.64, 4.38, 5.68, 7.8, 4.32, ... ,7.7, 7.3, 10.2, 9.3, 9.2>
                 Hue: <1.04, 1.05, 1.03, 0.86, 1.04, ... ,0.64, 0.7, 0.59, 0.6, 0.61>
               OD280: <3.92, 3.4, 3.17, 3.45, 2.93, ... ,1.74, 1.56, 1.56, 1.62, 1.6>
             Proline: <1065, 1050, 1185, 1480, 735, ... ,740, 750, 835, 840, 560>
             class_1: <1, 1, 1, 1, 1, ... ,0, 0, 0, 0, 0>
             class_2: <0, 0, 0, 0, 0, ... ,0, 0, 0, 0, 0>
             class_3: <0, 0, 0, 0, 0, ... ,1, 1, 1, 1, 1>
```
每一个*SeriesSet*对象都会自动地统计一些基本的数据集信息（缺失值、均值等）。例如，你可以通过如下的方式浏览数据集：
```Python
>>> wine.info
sheet:data
==========
1.  Structure: DaPy.SeriesSet
2. Dimensions: Ln=178 | Col=16
3. Miss Value: 0 elements
4.   Describe: 
        Title         | Miss |  Min  |  Max  |  Mean  |  Std   |Dtype
----------------------+------+-------+-------+--------+--------+-----
       Alcohol        |  0   | 11.03 | 14.83 | 13.00  |  0.81  | list
      Malic acid      |  0   |  0.74 |  5.8  |  2.34  |  1.12  | list
         Ash          |  0   |  1.36 |  3.23 |  2.37  |  0.27  | list
  Alcalinity of ash   |  0   |  10.6 |  30.0 | 19.49  |  3.34  | list
      Magnesium       |  0   |   70  |  162  | 99.74  | 14.28  | list
    Total phenols     |  0   |  0.98 |  3.88 |  2.30  |  0.63  | list
      Flavanoids      |  0   |  0.34 |  5.08 |  2.03  |  1.00  | list
 Nonflavanoid phenols |  0   |  0.13 |  0.66 |  0.36  |  0.12  | list
   Proanthocyanins    |  0   |  0.41 |  3.58 |  1.59  |  0.57  | list
   Color intensity    |  0   |  1.28 |  13.0 |  5.06  |  2.32  | list
         Hue          |  0   |  0.48 |  1.71 |  0.96  |  0.23  | list
        OD280         |  0   |  1.27 |  4.0  |  2.61  |  0.71  | list
       Proline        |  0   |  278  |  1680 | 746.89 | 314.91 | list
       class_1        |  0   |   0   |   1   |  0.33  |  0.47  | list
       class_2        |  0   |   0   |   1   |  0.40  |  0.49  | list
       class_3        |  0   |   0   |   1   |  0.27  |  0.45  | list
=====================================================================
```
#### Ⅱ. 预处理数据
在我们开始一个机器学习对象之前，为了能让数据符合模型的要求，我们需要进行预处理操作。  
  
在刚刚观察数据集时我们发现原始的数据集按照“类”变量被排序好了。为了我们能提供一个
平衡样本的训练集，我们可以通过*shuffles()*函数打乱我们的数据集。另外，在我们浏览
数据集时还发现，不同变量之间的量纲差异显著，因此我们认为进行标准化处理会更好：
```Python
>>> wine.shuffles()
>>> wine.normalized()
```
在打乱数据集后，我们要将目标变量和特征变量分离：
```Python
>>> target = wine.pop_col('class_1', 'class_2', 'class_3')
```
#### Ⅲ. 学习和预测
在*红酒分类*数据集中，我们的任务是给定一个新的纪录，预测它属于哪一个类。我们为每一个可能的类
都提供了相应的已有记录来训练分类器，以此分类器便能分辨出那些它未曾见过的样本了。 
  
在DaPy中，一个常用的分类器是来自于DaPy.multilayer_perseptron类中实现的多层感知机模型：
```Python
>>> mlp = machine_learn.MLP()
>>> mlp.create(input_cell=13, output_cell=3)
 - Create structure: 13 - 7 - 3
```
因为模型叫做multilayer perceptron，因此我们将该分类器称为mlp。现在我们需要训练模型，
也就是我们必须让他从数据集中学习。我们使用142条记录（总数的80%）来作为训练集。我们通过
[:142]这种非常Pythonic的语法来提取我们的数据：
```Python
>>> mlp.train(feature[:142], target[:142])
 - Start Training...
 - Initial Error: 152.02 %
    Completed: 10.00 	Remain Time: 3.38 s	Error: 12.22%
    Completed: 20.00 	Remain Time: 2.34 s	Error: 8.61%
    Completed: 29.99 	Remain Time: 1.77 s	Error: 6.88%
    Completed: 39.99 	Remain Time: 1.27 s	Error: 5.82%
    Completed: 49.99 	Remain Time: 1.05 s	Error: 5.10%
    Completed: 59.99 	Remain Time: 0.84 s	Error: 4.56%
    Completed: 69.99 	Remain Time: 0.62 s	Error: 4.15%
    Completed: 79.98 	Remain Time: 0.41 s	Error: 3.82%
    Completed: 89.98 	Remain Time: 0.19 s	Error: 3.55%
    Completed: 99.98 	Remain Time: 0.00 s	Error: 3.33%
 - Total Spent: 2.0 s	Error: 3.3268 %
```
   ![Page Not Found](https://github.com/JacksonWuxs/DaPy/blob/master/doc/material/QuickStartResult.png 'Result of Training')  
  
现在，*mlp*已经训练好了。值得注意的是，最后一行中的*Errors*并不意味着分类的正确率，
而是它与目标向量的绝对误差。  

让我们用我们的模型去分类那些红酒数据集中剩余的它不曾接触过的数据：
```Python
>>> mlp.test(feature[142:], target[142:])
'Classification Correct: 94.4444%'
```
正如你们所见到的，我们的模型具备了一定的分类能力。
#### Ⅳ. 后记
为了能在下一次任务中快速地调用训练好的模型，DaPy中支持了模型的保存方法：
```Python
>>> mlp.topkl('First_mlp.pkl')
```
在一次正式的工作中，你可以通过如下方式快速地使用训练好的模型预测新的案例：
```Python
>>> mlp = machine_learn.MLP()
>>> mlp.readpkl('First_mlp.pkl')
>>> mlp.predict(My_new_data)
```
