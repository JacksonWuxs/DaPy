<img src="https://github.com/JacksonWuxs/DaPy/blob/master/doc/material/DaPy.png">

DaPy - Enjoy the Tour in Data Mining
====
![](https://img.shields.io/badge/Version-1.5.3-green.svg)  ![](https://img.shields.io/badge/Download-PyPi-green.svg)  ![](https://img.shields.io/badge/License-GNU-blue.svg)

As a data analysis and processing library based on the original data structures in Python, **DaPy** is not only committed to save the time of data scientists and improve the efficiency of research, but also try it best to offer you a new experience in data science.

[Installation](#installation) | [Features](#features) | [Quick Start](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Quick%20Start/English.md ) | [To Do List](#todo) | [Version Log](#version-log) | [License](#license) | [Guide Book](https://github.com/JacksonWuxs/DaPy/tree/master/doc/Guide%20Book/README.md) | [中文版](https://github.com/JacksonWuxs/DaPy/blob/master/README_Chinese.md)

## Installation
The latest version 1.5.3 had been upload to PyPi.
```
pip install DaPy
```
Updating your last version to 1.5.3 with PyPi as follow.
```
pip install -U DaPy
```

## Features
**Convinience** and **efficiency** are the cornerstone of DaPy. 
Since the very beginning, we have designed DaPy to Python's 
native data structures as much as possible and we try to make 
it supports more Python syntax habits. Therefore you can 
adapt to DaPy quickly, if you imagine you are opearting an Excel table.
In addition, we do our best to simplify
the formulas or mathematical models in it, in order to let you 
implement your ideas fluently.  

* Here are just a few of the things that DaPy does well:  
	- [Efficiently manage diverse data with clearly perceiving approach](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Guide%20Book/English/Features.md#visually-manage-diverse-data)
	- [Quick insert and delete a mount of new records or new variables](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Guide%20Book/English/Features.md#easily-insert-and-delete-a-large-number-of-data)
	- [Make it easy to access a part of your dataset, not only by index or variable names, but also by specefic conditions]
	- [Functional IO tools for loading data from CSV files, Excel files, database and even SPSS files]
	- [Sort your records by multiple conditions]
	- [Fast verify your ideas with the built-in analysis models (e.g. `ANOVA`, `MLP`, `Linear Regression`)]
	- A variety of ways to help you easily perceive your dataset.
  
Even if it uses Python original data structures, 
DaPy still has efficiency comparable to some libraries which are written by C.
We have tested DaPy on the platform with
Intel i7-8550U while the Python version is 2.7.15-64Bit. The [dataset](http://www.wuxsweb.cn/Library/DaPy&Test_data/read_csv.csv)
has more than 2 million records and total size is 
119 MB.  

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
* Descriptive Statistics
* Inferential statistics
  - Time Sequence;
  - T-test
* Feature Engineering
  - PCA (Principal Component Analysis)
  - LDA (Linear Discriminant Analysis)
  - MIC (Maximal information coefficient)
* Algorithm
  - SVM ( Support Vector Machine)
  - K-Means
  - Lasso Regression  
  - Bayes Classification

If you want to follow up the latest developments, you can visit [here](https://www.teambition.com/project/5b1b7bd40b6c410019df8c41/tasks/scrum/5b1b7bd51e4661001838eb10).

## Version-Log
* V1.5.3 (2018-11-17)
  * Added `select()` function for quickly access partial data with some conditions;
  * Added more supported external data types: html and SQLite3 for saving data;
  * Added `DaPy.delete()` and `DaPy.column_stack()` for deleting and merging a un-DaPy object;
  * Added `DaPy.P()` and `DaPy.C()` for calculating permutation numbers and combination numbers;
  * Added new syntax, therefore users can view values in a column with statement as `data.title`.
  * Refactored `DaPy.BaseSheet`, less codes and more flexsible in the future;
  * Refactored `DaPy.DataSet.save()`, more stable and more flexsible in the future;
  * Rewrite a part of basic mathematical functions;
  * Fixed some bugs;
* V1.4.1 (2018-08-19)
  - Added `replace()` function for high-speed transering your data;
  - Optimized the speed in reading .csv file;
  - Refactored the `DaPy.machine_learn.MLP`, which can be formed with any layers, any active functions and any cells now;
  - Refactored the DaPy.Frame and DaPy.SeriesSet in order to improve the efficiency;
  - Supported to initialize Pandas and Numpy data structures;
  - Fixed some bugs;
* V1.3.3 (2018-06-20)
  - Added more supported external data types: Excel, SPSS, SQLite3, CSV;
  - Added `Linear Regression` and `ANOVA` to DaPy.Mathematical_statistics;
  - Added `DaPy.io.encode()` for better adepting to Chinese;
  - Optimized the presentations of SeriesSet and Frame in a more beautiful way;
  - Optimized the DaPy.Matrix so that the speed in calculating is two times faster;
  - Replaced read_col(), read_frame(), read_matrix() by read();
  - Refactored the DaPy.DataSet, which can manage multiple sheets at the same time;
  - Refactored the DaPy.Frame and DaPy.SeriesSet, delete the attribute limitation of types;
  - Removed DaPy.Table;
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
  - First public version of DaPy!

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
