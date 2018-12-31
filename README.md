<img src="https://github.com/JacksonWuxs/DaPy/blob/master/doc/material/DaPy.png">

DaPy - Enjoy the Tour in Data Mining
====
![](https://img.shields.io/badge/Version-1.7.2-Beta-green.svg)  ![](https://img.shields.io/badge/Python2.x-pass-green.svg)![](https://img.shields.io/badge/Python3.x-testing-red.svg)![](https://img.shields.io/badge/Download-PyPi-green.svg)  ![](https://img.shields.io/badge/License-GNU-blue.svg)

**DaPy**'s easy-to-use API designs and professinal statistical reports of models can help you enjoy the journey of data mining. In order to approach this goal, DaPy provides you some flexible and powerful high-level __data structures__, some __statistical functions__ and __machine learning models__.

[Installation](#installation) | [Features](#features) | [Quick Start](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Quick%20Start/English.md ) | [To Do List](#todo) | [Version Log](#version-log) | [License](#license) | [Guide Book](https://github.com/JacksonWuxs/DaPy/tree/master/doc/Guide%20Book/README.md) | [中文版](https://github.com/JacksonWuxs/DaPy/blob/master/README_Chinese.md)

## Installation
The latest version 1.7.2 had been upload to PyPi.
```
pip install DaPy
```
Updating your last version to 1.7.2 with PyPi as follow.
```
pip install -U DaPy
```

It should be noticed that DaPy still works on Python 2.x only. We are trying to make it avaliable on Python 3 as soon as possible. If you are interesting in using DaPy on Python 3, some of function may cause unexpect issues.

## Features

**Convinience** and **efficiency** are the cornerstone of DaPy data structures. 
Since the very beginning, we try to make it supports more Python syntax habits. Therefore you can adapt to DaPy quickly. In addition, we do our best to simplify the formulas or mathematical models in it, so that it can be easy to use and helps you to implement your idea fluentely.

* Here are just a few of the things that DaPy does well:  
	- [Efficiently manage diverse data with clearly perceiving approach](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Guide%20Book/English/Features.md#visually-manage-diverse-data)
	- [Quick insert and delete a mount of new records or new variables](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Guide%20Book/English/Features.md#easily-insert-and-delete-a-large-number-of-data)
	- [Make it easy to access a part of your dataset, not only by index or variable names, but also by specefic conditions]
	- [Functional IO tools for loading data from CSV files, Excel files, database and even SPSS files]
	- [Sort your records by multiple conditions]
	- [Fast verify your ideas with the built-in analysis models (e.g. `ANOVA`, `MLP`, `Linear Regression`)]
	- A variety of ways to help you easily perceive your dataset.
  
Even if it uses Python original data structures, DaPy still has efficiency comparable to some libraries which are written by C. We have tested DaPy on the platform with Intel i7-8550U while the Python version is 2.7.15-64Bit. The [dataset](http://www.wuxsweb.cn/Library/DaPy$Test_data/read_csv.csv) has more than 2 million records and total size is 240 MB.  

<table style='text-align:center'>
<thead>
    <th>Programs</th>
    <th>DaPy(V1.5.3)</th>
    <th>Pandas(V0.23.4)</th>
    <th>Numpy(V1.15.1)</th>
</thead>
<tbody>
<tr>
	<td>Loading</td>
	<td>12.85s (3.2x)</td>
	<td> 4.02s (1.0x)</td>
  	<td>22.63s (5.6x)</td>
</tr>
<tr>
	<td>Traversing</td>
	<td>0.09s (1.9x)</td>
	<td>1.15s (14.4x)</td>
	<td>0.08s (1.0x)</td>
</tr>
<tr>
	<td>Sorting</td>
	<td>0.97s (1.7x)</td>
	<td>0.25s (1.0x)</td>
	<td>3.03s (9.2x)</td>
</tr>
<tr>
	<td>Saving</td>
	<td>7.20s (3.5x)</td>
	<td>7.60s (3.7x)</td>
	<td>2.05s (1.0x)</td>
</tr>
<tr>
	<td>Total</td>
	<td>21.11s (1.6x)</td>
	<td>13.02s (1.0x)</td>
	<td>27.79s (2.1x)</td>
</tr>
</tbody>
</table>   


## TODO  
:heavy_check_mark: = Done      :running: = Coding & Testing       ​ :calendar:  = Put on the agenda       :thinking: = Not sure

* Data Structures

  * DataSet (3-D data structure) :heavy_check_mark:
  * Frame (2-D general data structure)​ :heavy_check_mark:
  * SeriesSet (2-D general data structure) :heavy_check_mark:
  * Matrix (2-D mathematical data structure) :heavy_check_mark:
  * Row (1-D general data structure) :heavy_check_mark:
  * Series (1-D general data structure) :running:
  * TimeSeries (1-D time sequence data structure)​ :running:

* Statistics

  * Basic Statistics (mean, std, skewness, kurtosis, frequency, fuantils)​ :heavy_check_mark:
  * Correlation (spearman & pearson) :heavy_check_mark:

  * Analysis of variance :heavy_check_mark:
  * Compare Means (simple T-test, independent T-test) :thinking:

* Operations

  * Beautiful CRUD APIs (create, Retrieve, Update, Delete)  :heavy_check_mark:
  * Flexible I/O Tool(supporting multiple source data for input and output) :heavy_check_mark:
  * Dummy Variables (auto parse norminal variable into dummy variable) :heavy_check_mark:
  * Difference Sequence Data :heavy_check_mark:
  * Normalize Data (log, normal, standard, box-cox):heavy_check_mark:
  * Drop Dupilicate Records :running:

* Methods

  - LDA (Linear Discriminant Analysis) :heavy_check_mark:
  - LR (Linear Regression)  :heavy_check_mark:
  - ANOVA (Analysis of Variance)  :heavy_check_mark:
  - MLP (Multi-Layers Perceptron)  :heavy_check_mark:
  - K-Means :running:
  - PCA (Principal Component Analysis) :running:
  - ARIMA (Autoregressive Integrated Moving Average) :calendar:
  - SVM ( Support Vector Machine) :thinking:
  - Bayes Classifier :thinking:
  - MIC (Maximal information coefficient) :thinking:   

* Others

  * Manual :running:
  * Example Notebook :running:
  * Unit Test :calendar:

## Version-Log
* V1.7.2 Beta (2019-01-01)

  * Added `get_dummies()` , supports to auto process norminal variables;
  * Added `show_time` attribute, auto timer for DataSet object;
  * Added `boxcox()` , supports Box-Cox transformation to a sequence data;
  * Added `diff()`, supports calculate the differences to a sequence data;
  * Added `DataSet.map()`, maps a function to a specific variable;
  * Added `methods.LDA`, supports DIscriminant Analysis on two methods (Fisher & Linear);
  * Added `row_stack()`, supports to combine multiple data structures with out DataSet;
  * Added `Row` structure for handling a record in sheet;
  * Added `report` attribute to all classes in `methods`,  you can read a statistical report after training a model;
  * More on `read()`, supports to auto parse data from a web address;
  * More on `DataSet.merge()`, supports for specifying how to save match keywords and the duplicate keywords.
  * Rename `DataSet.pop_miss_value()` into `DataSet.dropna()`;
  * Refactored `methods`, more stable and more scalable in the future;
  * Refactored `methods.LinearRegression`, it can prepare a statistic report for you after training;
  * Refactored `BaseSheet.select()`, 5 times faster and more pythonic API design;
  * Refactored `BaseSheet.replace()`, 20 times faster and more pythonic API design;
  * Supported Python 3.x platform;
  * Fixed a lot of bugs;

* V1.5.3 (2018-11-17)

  * Added `select()` function for quickly access partial data with some conditions;
  * Added more supported external data types: html and SQLite3 for saving data;
  * Added `delete()` and `column_stack()` for deleting and merging a un-DaPy object;
  * Added `P()` and `C()` for calculating permutation numbers and combination numbers;
  * Added new syntax, therefore users can view values in a column with statement as `data.title`.
  * Refactored `BaseSheet`, less codes and more flexsible in the future;
  * Refactored `DataSet.save()`, more stable and more flexsible in the future;
  * Rewrite a part of basic mathematical functions;
  * Fixed some bugs;

* V1.4.1 (2018-08-19)
  - Added `replace()` for high-speed transering your data;
  - Optimized the speed in reading .csv file;
  - Refactored the` methods.MLP`, customized with any layers, any active functions and any cells now;
  - Refactored the `Frame` and `SeriesSet` to improve the efficiency;
  - Supported to initialize Pandas and Numpy data structures;
  - Fixed some bugs;

* V1.3.3 (2018-06-20)
  - Added `methods.LinearRegression` and `methods.ANOVA` ;
  - Added `io.encode()` for better adepting to Chinese;
  - Optimized `SeriesSet.__repr__()` and `Frame.__reprt__()` to show data in beautiful way;
  - Optimized the `Matrix`, so that the speed in calculating is two times faster;
  - More on `read()` , supports external file as: Excel, SPSS, SQLite3, CSV;
  - Renamed `DataSet.read_col()`, `DataSet.read_frame()`, `DataSet.read_matrix()` by `DataSet.read()`;
  - Refactored the `DataSet`, which can manage multiple sheets at the same time;
  - Refactored the `Frame` and `SeriesSet`, delete the attributes' limitations;
  - Removed `DaPy.Table`;

* V1.3.2 (2018-04-26)
  - Added more useful functions for DaPy.DataSet;
  - Added a new data structure called DaPy.Matrix;
  - Added some mathematic formulas (e.g. corr, dot, exp);
  - Added `Multi-Layers Perceptrons` to DaPy.machine_learn;
  - Added some standard dataset;
  - Optimized the loading function significantly;

* V1.3.1 (2018-03-19)
  - Added the function which supports to save data as a csv file;
  - Fixed some bugs in the loading data function;

* V1.2.5 (2018-03-15)

  - First public beta version of DaPy!   

    # Contributors

    ###### Originator: Jackson Woo

    ###### Maintainor: Jackson Woo & Nick Yang

## License

Copyright (C) 2018 - 2019 Xuansheng Wu
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
