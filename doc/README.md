DaPy - don't limit your mind by syntax
====
![](https://img.shields.io/badge/Version-1.3.3-green.svg)  ![](https://img.shields.io/badge/Download-PyPi-green.svg)  ![](https://img.shields.io/badge/License-GNU-blue.svg)  

As a data analysis and processing library based on the origion data structures in Python, **DaPy** is not only committed to saving the time of data scientists and improving the efficiency of research, but also try it best to offer you a new experience in data science.

[Installation](#installation) | [Features](#features) | [Quick Start](https://github.com/JacksonWuxs/DaPy/blob/master/Quick%20Start.md ) | [To Do List](#todo) | [Version Log](#version-log) | [License](#license) | [中文版](https://github.com/JacksonWuxs/DaPy/blob/master/README_Chinese.md)

## Installation
The Latest version 1.3.3 had been upload to PyPi.
```
pip install DaPy
```
Updating your last version to 1.3.3 with PyPi as follow.
```
pip install -U DaPy
```

## Features
#### Ⅰ. Comfortable Experience
Since the very beginning, we have designed DaPy to Python's 
native data structures as much as possible and we try to make 
it support more Python syntax habits. Therefore you can 
adapt to DaPy quickly. In addition, we do our best to simplify
the formulas or functions in it in order to let users 
implement their ideas fluently.  
  
  
Sorting records obeyed different arranging orders is a 
common way to help you recognize your dataset. In this case,
DaPy supports you set up more than one conditions to arrangement 
your dataset. 
```Pyton
 data.sort(('A_col', 'DESC'), ('B_col', 'ASC'), ('D_col', 'DESC'))
 ```
  
#### Ⅱ. Efficiency  
We have testified the performance of DaPy in three fields 
(load data, sort data & traverse data), 
those were most useful functions to a data processing library.
In contrast with those packages written by C programe languages,
DaPy showed an amazing efficiency in testing. In all subjects of
test, DaPy just spends less than twice time as long as the 
fastest C language library.   
  
  
We tested DaPy on the platform with
Intel i7-6560U while the Python version is 2.7.13-64Bit. The 
dataset (https://pan.baidu.com/s/1kK3_V8XbbVim4urDkKyI8A)
has more than 4.5 million records and total size is 
240.2 MB. 

<table>
<tr>
	<td>Result of Testing</td>
	<td>DaPy</td>
	<td>Pandas</td>
	<td>Numpy</td> 
</tr>
<tr>
	<td>Loading Time</td>
	<td>29.3s (2.4x)</td>
	<td>12.3s (1.0x)</td>
	<td>169.0s (13.7x)</td>
</tr>
<tr>
	<td>Traverse Time</td>
	<td>0.34s (1.6x)</td>
	<td>3.10s (14.8x)</td>
	<td>0.21s (1.0x)</td>
</tr>
<tr>
	<td>Sort Time</td>
	<td>1.41s (1.65x)</td>
	<td>0.86s (1.0x)</td>
	<td>5.37s (10.1x)</td>
	</tr>
<tr>
	<td>Total Spent</td>
	<td>25.4s (1.5x)</td>
	<td>17.4s (1.0x)</td>
	<td>174.6s (10.0x)</td>
	</tr>
<tr>
	<td>Version</td>
	<td>1.3.3</td>
	<td>0.22.0</td>
	<td>1.14.0</td>
	</tr>
</table>  


## TODO  
* Descriptive Statistics
* Inferential statistics
* Feature Engineering
	- PCA (Principal Component Analysis)
	- LDA (Linear Discriminant Analysis)
	- MIC (Maximal information coefficient)
* Algorithm
	- SVM ( Support Vector Machine)
	- K-Means
	- Lasso Regression  

## Version-Log
* V1.3.3 (2018-06-20)
	- Added more external data file: Excel, SPSS, SQLite3, CSV;
	- Added `Linear Regression` and `ANOVA` to DaPy.Mathematical_statistics;
	- Added `DaPy.io.encode()` for better adepted to Chinese;
	- Replaced read_col(), read_frame(), read_matrix() by read();
	- Optimized the DaPy.Matrix so that the speed in calculating is two times faster;
	- Expreesed SeriesSet and Frame in more beautiful way;
	- Refactored the DaPy.DataSet, which can manage multiple sheets at the same time;
	- Refactored the DaPy.Frame and DaPy.SeriesSet, delete the attribute limitation of types.
	- Removed DaPy.Table;
* V1.3.2 (2018-04-26)
	- Increased the efficiency of loading data significantly;
	- Added more useful functions for DaPy.DataSet;
	- Added a new data structure called DaPy.Matrix;
	- Added some mathematic formulas (e.g. corr, dot, exp);
	- Added `Multi-Layers Perceptrons` to DaPy.machine_learn;
	- Added some standard dataset.
* V1.3.1 (2018-03-19)
	- Fixed some bugs in the loading data function;
	- Added the function which supports to save data as a file.
* V1.2.5 (2018-03-15)
	- First version of DaPy!

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
