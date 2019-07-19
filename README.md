<img src="https://github.com/JacksonWuxs/DaPy/blob/master/doc/material/DaPy.png">
<i>This open source framework fluently implements your ideas for data mining.</i>

# DaPy - Enjoy the Tour in Data Mining

![](https://img.shields.io/badge/Version-1.10.1-green.svg)  ![](https://img.shields.io/badge/Python2-pass-green.svg)![](https://img.shields.io/badge/Python3-pass-green.svg)![](https://img.shields.io/badge/Download-PyPi-green.svg)  ![](https://img.shields.io/badge/License-GNU-blue.svg)

[中文版](https://github.com/JacksonWuxs/DaPy/blob/master/README_Chinese.md)

### Overview

DaPy is a data analysis library designed with ease of use in mind, which lets you smoothly implement your thoughts by providing well-designed **data structures** and abundant  **professional ML models**. In short, this project can help you complete data mining tasks in each step, such as loading data, preprocessing data, feature engineering, developing models, model evaluation and result export.

### Example

This example simply shows you what DaPy can do for you. Our goal in this example is to train a classifier for Iris classification task. Detail information can be read from [here](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Quick%20Start/English.md).

![](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Quick%20Start/quick_start.gif)

### Comparison



### Features

We hope DaPy is an user-friendly tool. Therefore, we pay attention to the design of API in DaPy in order to let you quickly adept it and use it flexibly. Here are just a few of things that make DaPy simple:  

- Variety of ways to visualize data in CMD

- 2D data sheet structures following Python syntax habbits
- SQL-like APIs to process data
- Flexible IO tools for loading and saving data (e.g. Website, Excel, Sqlite3, SPSS, Text)
- Built-in basic models (e.g. Decision Tree, Multilayer Perceptron, Linear Regression, ...)

Also, we hope it can be used in some real-world tasks, thereby we are keeping an eye on its *efficiency*. Although DaPy is implemented by pure Python, it has comparable efficiency to some exists libraries. Following dialog shows a testing result and the data had 432 thousands rows and 7 columns.

![](https://github.com/JacksonWuxs/DaPy/blob/master/doc/material/Result.jpg)

### Install

The latest version 1.10.1 had been updated to PyPi.

```
pip install DaPy
```

Some of functions in DaPy depend on requirements.

- **xlrd**: loading data from .xls file【Necessary】
- **xlwt**: export data to a .xls file【Necessary】
- **repoze.lru**: speed up loading data from .csv file【Necessary】
- **savReaderWrite**: loading data from .sav file【Optional】
- **bs4.BeautifulSoup**: auto downloading data from a website【Optional】
- **numpy**: dramatically increase the efficiency of ML models【Recommand】 


### Example Useages

- Loading Source Data
  - Load data from a local csv, sav, sqlite3 or xls file: ```sheet = DaPy.read(file_addr)```
  - Display the first five and the last five records: ```sheet.show(lines=5)```
- Preprocessing Data
  - Remove duplicate records: ```sheet.drop_duplicates(col, keep='first')```
  - Merge external features from another table: ```sheet.merge(sheet2, left_key='ID', other_key='ID', keep_key='self', keep_same=False)```
  - Use linear interpolation to fill in NaN or remove them: ```sheet.fillna(method='linear')``` , ```sheet.dropna(axis=0)```
  - Remove some meaningless columns (e.g. *ID*): ```sheet.drop('ID', axis=1)```
  - Sort records by some columns: ```sheet = sheet.sort('Age', 'DESC')```
- Explore Data
  - Summary the statistic information of each column: ```sheet.info```
  - Count distribution of categorical variable: ```sheet.count_values('gender')```
  - Find differences of the labels in categorical variables: ```sheet.groupby('city')```
  - Calculate the correlation between the continuous variables: ```sheet.corr(['age', 'income'])```
- Feature Engineering
  - Constructing some meaningful variables to use date time: ```sheet.label_date('birth')```
  - Transfer categorical variables into dummy variables: ```sheet.get_dummies(['city', 'education'])```
  - Standardize some normal continuous variables: ```sheet.normalized(col='age')```
  - Special processing for some special variables: ```sheet.normalized('log', col='salary')```
  - Create new variables by some business logical formulas: ```sheet.apply(func=tax_rate, col=['salary', 'income'])```
  - Difference process to make time-series stable: ```DaPy.diff(sheet.income)```
  - Constructing cross terms: ```sheet.get_cross_terms(col=['income', 'age', 'gender', 'education'])```
- Developing Models
  - Choose a model and initialize it: ```m = MLP()```, ```m = LinearRegression()```, ```m = DecisionTree()``` or  ```m = DiscriminantAnalysis()``` 
  - Train the model parameters: ```m.fit(X_train, Y_train)```
- Model Evaluation
  - Evaluate model with  parameter tests: ```m.report.show()```
  - Evaluate model with  visualization: ```m.plot_error()``` or ```DecisionTree.export_graphviz()```
  - Evaluate model with test set: ```DaPy.methods.Performance(m, X_test, Y_test, mode)```.
- Saving Result
  - Save the model: ```m.save(addr)```
  - Save the final dataset: ```sheet.save(addr)```
  
  
### TODO  

:heavy_check_mark: = Done      :running: = In Development       ​ :calendar:  = Put On the Agenda       :thinking: = Not Sure

* Data Structures

  * DataSet (3-D data structure) :heavy_check_mark:
  * Frame (2-D general data structure)​ :heavy_check_mark:
  * SeriesSet (2-D general data structure) :heavy_check_mark:
  * Matrix (2-D mathematical data structure) :heavy_check_mark:
  * Row (1-D general data structure) :heavy_check_mark:
  * Series (1-D general data structure) :heavy_check_mark:
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
  * Drop Duplicate Records :heavy_check_mark:
  * Group By (analysis the dataset under controlling a group variable):heavy_check_mark:

* Methods

  - LDA (Linear Discriminant Analysis) :heavy_check_mark:
  - LR (Linear Regression)  :heavy_check_mark:
  - ANOVA (Analysis of Variance)  :heavy_check_mark:
  - MLP (Multi-Layers Perceptron)  :heavy_check_mark:
  - DT (Decision Tree):heavy_check_mark:
  - K-Means :running:
  - PCA (Principal Component Analysis) :running:
  - ARIMA (Autoregressive Integrated Moving Average) :calendar:
  - SVM ( Support Vector Machine) :thinking:
  - Bayes Classifier :thinking:

* Others

  * Manual :running:
  * Example Notebook :running:
  * Unit Test :running:

### Contributors

- ###### Directors:

  Xuansheng WU (@JacksonWoo: wuxsmail@163.com )

- ###### Developers

  1. Xuansheng WU
  2. Feichi YANG  (@Nick Yang: yangfeichi@163.com)  

### Version-Log

* V1.10.1 (2019-06-13)
  * Added ```SeriesSet.update()```, update some values of specific records;
  * Added ```BaseSheet.tolist()``` and ```BaseSheet.toarray()```, transfer your data to list or numpy.array;
  * Added ```BaseSheet.query()```, select records with a python statement in string;
  * Added ```SeriesSet.dropna()```, drop rows or variables which contain NaN;
  * Added ```SeriesSet.fillna()```, fill missing values in the dataset with constant value or linear model;
  * Added ```SeriesSet.label_date()```, transfer a datetime object to several columns;
  * Added ```DaPy.Row```, a view of a row record of the original data;
  * Added ```DaPy.methods.DecitionTree```, classifier implemented with C4.5 algorithm;
  * Added ```DaPy.methods.SignTest```, supported some of sign test algorithms;
  * Refactored the structure of ```DaPy.core.base``` package;
  * Optimized ```BaseSheet.groupby()```, 18 times faster than ever before;
  * Optimized ```BaseSheet.select()```, 14 times faster than ever before;
  * Optimized ```BaseSheet.sort()```, 2 times faster than ever before;
  * Optimized ```dp.save()```, 1.6 times faster than ever before to saving data to a .csv;
  * Optimized ```dp.read()```, 10% faster than ever before to loading data from .csv;
* V1.9.2 (2019-04-23)
  * Added `BaseSheet.groupby()`, regroup your observations with specific columns;
  * Added `DataSet.apply()`, mapping a function to the dataset by axis;
  * Added `DataSet.drop_duplicates()`, automatically dropout the duplicate records in the dataset;
  * Added `DaPy.Series`, a new data structure to obtain a sequence of data;
  * Added `DaPy.methods.Performance()`, automatically testify the performance of ML models;
  * Added `DaPy.methods.Kappa()`, calculate the Kappa index with a confusing matrix;
  * Added `DaPy.methods.ConfuMat()`, calculate the Confusing matrix with your result;
  * Added `DaPy.methods.DecitionTree()`, implement the C4.5 decision tree algorithm;
  * Refactored the structure of `DaPy.core.base` package;
  * More on `BaseSheet.select()`, supports new keywords "limit" and "columns";
* V1.7.2 Beta (2019-01-01)
  * Added `get_dummies()` , supports to auto process norminal variables;
  * Added `show_time` attribute, auto timer for DataSet object;
  * Added `boxcox()` , supports Box-Cox transformation to a sequence data;
  * Added `diff()`, supports calculate the differences to a sequence data;
  * Added `DaPy.methods.LDA`, supports DIscriminant Analysis on two methods (Fisher & Linear);
  * Added `row_stack()`, supports to combine multiple data structures with out DataSet;
  * Added `Row` structure for handling a record in sheet;
  * Added `report` attribute to all classes in `methods`,  you can read a statistical report after training a model;
  * More on `read()`, supports to auto parse data from a web address;
  * More on `SeriesSet.merge()`, more options when we merge to SeriesSets;
  * Rename `DataSet.pop_miss_value()` into `DataSet.dropna()`;
  * Refactored `methods`, more stable and more scalable in the future;
  * Refactored `methods.LinearRegression`, it can prepare a statistic report for you after training;
  * Refactored `BaseSheet.select()`, 5 times faster and more pythonic API design;
  * Refactored `BaseSheet.replace()`, 20 times faster and more pythonic API design;
  * Supported Python 3.x platform;
  * Fixed a lot of bugs;
* V1.5.3 (2018-11-17)

  * Added `select()`, quickly access partial data with some conditions;
  * Added `delete()`, delete data along the axis from a un-DaPy object;
  * Added `column_stack()`, merging several un-DaPy objects together;
  * Added `P()` & `C()` , calculating permutation numbers and combination numbers;
  * Added new syntax, therefore users can view values in a column with statement as `data.title`.
  * Optimized ```DaPy.save()```, supported external saving data types: html and SQLite3;
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
  - Added more useful functions for `DaPy.DataSet`;
  - Added a new data structure called `DaPy.Matrix`;
  - Added some mathematic formulas (e.g. corr, dot, exp);
  - Added `Multi-Layers Perceptrons` to DaPy.machine_learn;
  - Added some standard dataset;
  - Optimized the loading function significantly;
* V1.3.1 (2018-03-19)
  - Added the function which supports to save data as a csv file;
  - Fixed some bugs in the loading data function;
* V1.2.5 (2018-03-15)
  - First public beta version of DaPy!  

### License

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
along with this program.  If not, see https:\\www.gnu.org\licenses.# datapy
A light Python library for data processing and analysing.

