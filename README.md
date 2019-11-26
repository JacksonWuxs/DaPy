<img src="https://github.com/JacksonWuxs/DaPy/blob/master/doc/material/DaPy.png">
<i>This open source framework fluently implements your ideas for data mining.</i>

# DaPy - Enjoy the Tour in Data Mining

![](https://img.shields.io/badge/Version-1.11.1-green.svg)  ![](https://img.shields.io/badge/Python2-pass-green.svg)![](https://img.shields.io/badge/Python3-pass-green.svg)![](https://img.shields.io/badge/Download-PyPi-green.svg)  ![](https://img.shields.io/badge/License-GNU-blue.svg)

[中文版](https://github.com/JacksonWuxs/DaPy/blob/master/README_Chinese.md)

### Overview

DaPy is a data analysis library designed with ease of use in mind and it lets you smoothly implement your thoughts by providing well-designed **data structures** and abundant  **professional ML models**. There has been a lot of famous data operation modules already like Pandas, but there is no module, which

* supports writing codes in Chain Programming;
* multi-threading safety data containers;
* operates feature engineering methods with simple APIs;
* handles data as easily as using Excel (do not pay attention to data structures);
* shows the log of each steps on console like MySQL.

Thus, DaPy is more suitable for data analysts, statistic professors and who works with big data with limited  computer knowledge than the engineers. In DaPy, our data structure offers 70 APIs for data mining, including 40+ data operation functions, 10+ feature engineering functions and 15+ data exploring functions.

### Example

This example simply shows the characters of DaPy of **chain programming**, **working log** and **simple feature engineering methods**. Our goal in this example is to train a classifier for Iris classification task. Detail information can be read from [here](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Quick%20Start/English.md).

![](https://github.com/JacksonWuxs/DaPy/blob/master/doc/Quick%20Start/quick_start.gif)

### Features of DaPy

We already have abundant of great libraries for data science, why we need DaPy? 

The answer is <u>*DaPy is designed for data analysts, not for coders.*</u>  In DaPy, users only need to focus on their thought of handling data, and pay less attention to coding tricks. For example, in contrast with Pandas, DaPy supports you manipulating data by rows as same as using SQL. Here are just a few of things that make DaPy simple:  

- Variety of ways to visualize data in CMD
- 2D data sheet structures following Python syntax habits
- SQL-like APIs to process data
- Thread-safety data container
- Variety functions for preprocessing and feature engineering
- Flexible IO tools for loading and saving data (e.g. Website, Excel, Sqlite3, SPSS, Text)
- Built-in basic models (e.g. Decision Tree, Multilayer Perceptron, Linear Regression, ...)

Also, DaPy has high efficiency to support you solving real-world situations. Following dialog shows a testing result which provides that DaPy has comparable efficiency than some exists C written libraries. The detail of test can be found from here.

![Performance Test](https://github.com/JacksonWuxs/DaPy/blob/master/doc/material/Result.jpg)

### Install

The latest version 1.11.1 had been updated to PyPi.

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


### Usages

- Load & Explore Data
  - Load data from a local csv, sav, sqlite3, mysql server, mysql dump file or xls file: ```sheet = DaPy.read(file_addr)```
  - Display the first five and the last five records: `sheet.show(lines=5)`
  - Summary the statistical information of each columns: ```sheet.info```
  - Count distribution of categorical variable: ```sheet.count_values('gender')```
  - Find differences of the labels in categorical variables: ```sheet.groupby('city')```
  - Calculate the correlation between the continuous variables: ```sheet.corr(['age', 'income'])```
- Preprocessing & Clean Up Data
  - Remove duplicate records: `sheet.drop_duplicates(col, keep='first')`
  - Use linear interpolation to fill in NaN : ```sheet.fillna(method='linear')``` 
  - Remove the records which contains more than 50% variables are NaN: `sheet.dropna(axis=0, how=0.5)`
  - Remove some meaningless columns (e.g. *ID*): ```sheet.drop('ID', axis=1)```
  - Sort records by some columns: `sheet = sheet.sort('Age', 'DESC')`
  - Merge external features from another table: `sheet.merge(sheet2, left_key='ID', other_key='ID', keep_key='self', keep_same=False)`
  - Merge external records from another table: `sheet.join(sheet2)`
  - Append records one by one: `sheet.append_row(new_row)`
  - Append new variables one by one: `sheet.append_col(new_col)`
  - Get parts of records by index: `sheet[:10, 20: 30, 50: 100]`
  - Get parts of columns by column name: `sheet['age', 'income', 'name']`
- Feature Engineering
  - Transfer a date time into  categorical variables: `sheet.get_date_label('birth')`
  - Transfer numerical variables into categorical variables: `sheet.get_categories(cols='age', cutpoints=[18, 30, 50], group_name=['Juveniles', 'Adults', 'Wrinkly', 'Old'])`
  - Transfer categorical variables into dummy variables: `sheet.get_dummies(['city', 'education'])`
  - Create higher-order crossover terms between your selected variables: `sheet.get_interactions(n_power=3, col=['income', 'age', 'gender', 'education'])`
  - Introduce the ranks of each records: `sheet.get_ranks(cols='income', duplicate='mean')`
  - Standardize some normal continuous variables: ```sheet.normalized(col='age')```
  - Special processing for some special variables: ```sheet.normalized('log', col='salary')```
  - Create new variables by some business logical formulas: ```sheet.apply(func=tax_rate, col=['salary', 'income'])```
  - Difference process to make time-series stable: `DaPy.diff(sheet.income)`
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

### Contributors

- ###### Contributors:

    1. Xuansheng WU (@JacksonWoo: wuxsmail@163.com)   
       2. Feichi YANG  (@Nick Yang: yangfeichi@163.com)  

- ###### Members:

    1. Rong XING (@TerenceXing: terencexing@126.com)

### Related

Following programs are also great data analyzing/ manipulating frameworks in Python:

* [Agate](https://github.com/wireservice/agate): Data analysis library optimized for humans
* [Numpy](https://github.com/numpy/numpy): fundamental package for scientific computing with Python
* [Pandas](https://github.com/pandas-dev/pandas): Python Analysis Data 
* [Scikit-Learn](https://github.com/scikit-learn/scikit-learn): Machine Learn in Python  

### Further-Info

-  Version change log 
- Todo List
- License



