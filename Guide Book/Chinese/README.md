# DaPy - 享受数据挖掘之旅
[简介](https://github.com/JacksonWuxs/DaPy/blob/master/Guide%20Book/README.md#introduction)   
   - [什么是DaPy？](https://github.com/JacksonWuxs/DaPy/tree/master/Guide%20Book#what-is-dapy)  
   - [为什么要使用DaPy？](https://github.com/JacksonWuxs/DaPy/tree/master/Guide%20Book#why-use-dapy)   
   - [如何使用DaPy？](https://github.com/JacksonWuxs/DaPy/tree/master/Guide%20Book#how-to-use-dapy)  

[特征](https://github.com/JacksonWuxs/DaPy/blob/master/Guide%20Book/Features%20Introduction.md)
   - [可视化管理各种数据](https://github.com/JacksonWuxs/DaPy/blob/master/Guide%20Book/Features%20Introduction.md#visibly-manage-diverse-data)
   - 快速完成“增删改查"操作
   - 轻松访问部分数据
   - 灵活的I/O工具
   - 易于使用内建模型算法
  
[快速入门](https://github.com/JacksonWuxs/DaPy/blob/master/doc/GuideBook.md#quick-start)
   - 加载数据集
   - 预处理数据
   - 模型建立与训练
   - 显示结果
  
[数据结构]
  - 介绍
   - DataSet 结构
   - sheet 结构
   - matrix 结构
  
[集成算法]
   - 机器学习模型
   - 数学统计模型
  
# 介绍
#### 什么是DaPy？
DaPy是一个由Python原生数据结构设计的高效且易于使用的数据挖掘库。
它旨在成为一个基础和友好的数据处理工具。此外，我们还在 DaPy 中建立了统计和机器学习中的一些常用数据分析算法，以帮助您尽快验证您的想法。

#### 为什么要使用DaPy？
现在有不少可以有效地支持科学计算和数据分析。但是，这些类型的库对Python中的新手来说并不友好，因为他们不得不花费大量时间熟悉这些数据结构。

例如，遍历数据集，大多数人会使用``for``语法作为他们的第一个想法。但是Pandas只会迭代列名，而不是记录。此外，当用户尝试选择一些有条件的记录时，Pandas首先需要设置`Bool`序列，这导致用户无法知道数据是如何操作的。

毫无疑问，这些可敬的库在数据科学领域发挥着重要作用。但是，他们仍然需要使用一些互补品。与这些数据处理或计算库相比，DaPy专注于特定方面，快速开发。DaPy 希望通过流畅的用户操作提升开发效率。简而言之，DaPy适合刚开始新研究的用户。借助DaPy的帮助，科学家们可以流畅地实现他们的想法，而不受复杂语法的限制。

使用DaPy的一种推荐方法是在新研究中对数据集进行预处理时使用它。在您的demo证明之后，您可以使用numpy或tensorflow数据结构来重写您的想法。但这并不意味着DaPy无法处理大数据。相反，他在某些方面也具有很强的运作效率。

#### 如何使用DaPy？
当您在程序中使用DaPy作为数据处理工具时，您只需要将其想象为Excel文件。在大多数情况下，从您的脑海中跳出的第一个想法是在处理数据时正确的语法。这是一个简单的例子来证明语法是否符合您的想法。

首先，我们制作一个`Frame`结构如下。 Frame是DaPy中的一种`sheet`，另一种`sheet`结构是`SeriesSet`。
```
>>>将DaPy导入为dp
>>> data = dp.Frame（[
[1,2,3,4,5,6]，
[1,3,5,7,9,11]，
[2,4,6,8,10,12]]，
   ['A_col'，'B_col'，'C_col'，'D_col'，'E_col'，'F_col']）
>>>数据
 A_col | B_col | C_col | D_col | E_col | F_col
------- ------- + ------- + ------- + ------- + ------- +
   1 | 2 | 3 | 4 | 5 | 6
   1 | 3 | 5 | 7 | 9 | 11
   2 | 4 | 6 | 8 | 10 | 12
```
现在，我们的任务是挑选以下列：'B_col'，'C_col'，'D_col'和'F_col'。我们发现'B_col'，'C_col'和'D_col'连在一起。想想使用原生Python结构（例如list）中的`slice`，我们的操作如下：
```
>>>数据['B_col'：'D_col'，'F_col']
 B_col | C_col | D_col | F_col
------- ------- + ------- + ------- +
   2 | 3 | 4 | 6
   3 | 5 | 7 | 11
   4 | 6 | 8 | 12
 ```
