Description
========================================================
DataPy is a light data processing library which could
be used in loading data from data profiles easily and 
contain or reduce part of the data easily. Additionly,
it contains some basic formulas that help the developers
to understand the basic variables about the data set.
The first Stable version (V1.2.3) will be uploaded to 
pypi by March 15.

Advantage and Disadvantage
========================================================
We have tested the performance of Datapy in loading file.
It was an amazing result that Datapy has the fastes speed and using less memory.
Despite this, because Datapy is not yet able to support matrix operations, it has a long way to go to achieve Numpy's achievements in scientific computing. The testing code has been uploaded already.

       		Datapy	Pandas    Numpy   File Info
	 Loading Time  17.77s | 4.55s  | 62.23s |    N/A
	Traverse Time   0.5s  | 316.5s |  0.1s  |    N/A
 	  Total Spent  18.2s  | 321.1s |  62.3  |    N/A
 	  Memory Size   14MB  | 174MB  |  69MB  |    96MB

Installation
========================================================
Download using pip via pypi.
> pip install datapy

Instructions for use
========================================================
```
DataPy.DataSet(addr='data.csv', title=True, split='AUTO',
		db=None, name='Data', firstline=1, 
		miss_value=None)
```
- This class is the core function for processing data, which supports user to pretreatment the database.
distribution of the data.
- [addr] Your database path and name;
- [title] "True" means your database includes title in some line. To the contrary, "False" means the there is no title on it.
- [split] This variable means the separator of the database in every line. 'AUTO' will ask system find the best match separator to split the file. The system will set separator as ',' if the file type is '.cvs' and set separator as ' ' if the file type is '.txt', while will set separator as '\t' if the file type is '.xls'.
- [db] You could set a canned database.
- [name] Your database name.
- [firstline] Your data starts from "firstline".
- [miss_value] The missing value in your data will be replace into the simble of this variable.

```
dataset.readframe(col=all)
```
- This function will help you load data from a file as a DataFrame, which implements with some named-tuple in a list. You could pickout the  data with line number and columen name.
- [col] Giving a iterable variable which contains the column number you would like to choose.

```
dataset.readtable(col=all)
```
- This function will help you load data from a file as a DataFrame, which implements with lists only. So it will not be allowed in pick out data with the column name.
- [col] Giving a iterable variable which contains the column number you would like to choose.

```
dataset.readcol(col=all)
```
- This function supports user to load data by column which has the data structure as a diction and the keyword is column and the series is value. Additonly, you could pick series of  data in one column.
- [col] Giving a iterable variable which contains the column number you would like to choose.

```
dataset.data
```
- Return the data you just loaded before.

```
dataset.titles
```
- Return the title of this data set.

```
str(dataset.data)
```
- Return the name of this dataset.

```
len(dataset.data)
```
- Return the number of records.

```
DataPy.CountFrequancy(data, cut=0.5)
```
- This function is used in counting the distribution of a data series. The variables "cut" means the cut point you want.If you would like to calculate the proportion of series, you can get help from this function.
- Variable "data" expects a iterble item, such as tuple() or list(). Unnecessary variable "cut" expects a number. It will return a float means the proportion of data which is larger than "cut".
- Your data is distribute in "1" or "0", you would like to calculate the proportion of "1".You can use it as follow.

```
Datapy.CountDistribution(data, shapes=[0.05,0.1,0.25,0.5,0.75,0.9,0.95])
```
- This function could help you find the distribution of the data.
- Return the value of each quantile.

```
Datapy.Statistic(data)
```
- Return the basic statistics of the data set as NamedTuple.
- The tuple includes 'Mean','Std','CV','Min','Max' and 'Range'.

Examples
========================================================

```
<Example 1>
>>> import datapy as dp
>>> data = dp.dataset('testdb.csv')
>>> data.readframe()
>>> data.data
[Data(A_col=3, B_col=2, C_col=1, D_col=4),
 Data(A_col=4, B_col=3, C_col=2, D_col=2), 
 Data(A_col=1, B_col=3, C_col=4, D_col=2), 
 Data(A_col=3, B_col=3, C_col=1, D_col=2), 
 Data(A_col=4, B_col=5, C_col=4, D_col=3),
 Data(A_col=2, B_col=1, C_col=1, D_col=5)]
```
From Example 1, you have loaded a data set as a frame 
from your data profile. Easily, right? Anyway, there is 
another way to load data by column as series. Here is 
the example.

```
<Example 2>
>>> data.readcol()
>>> data.titles
['A_col', 'B_col', 'C_col', 'D_col']
>>> for title in data.titles:
	print title,data[title]
	
A_col [3, 4, 1, 3, 4, 2]
B_col [2, 3, 3, 3, 5, 1]
C_col [1, 2, 4, 1, 4, 1]
D_col [4, 2, 2, 2, 3, 5]	
```
Example 1 and Example 2  have showed two ways to load data.
But a simpler data structure is always neccessary. We offer
the third way to load data in bellowed.

```
<Example 3>
>>> data.readtable()
>>> data.data
[[3, 2, 1, 4], 
 [4, 3, 2, 2], 
 [1, 3, 4, 2], 
 [3, 3, 1, 2], 
 [4, 5, 4, 3], 
 [2, 1, 1, 5]]
```
Afther we load the data in menmery, we would like to 
characterize the data set and try to destribe it. See the 
example as follow.


```
<Example 4>
>>> len(data)
6
>>> str(data)
'Data'
>>> dp.CountDistribution(data['A_col'],[0.25,0.5,0.75])
[2,3,4]
>>> dp.CountFrequancy(data['A_col'],2)
0.16666666666666666
>>> STAT(Mean=2.8333333333333335, Std=1.1690451, 
CV=0.4126041862764749, Min=1, Max=4, Range=3)
```
Finally, we also support the client opearts the data with
'-' or '+'. It will help you change the data size.


```
<Example 5>
>>> data.readtable()
>>> data - 2
>>> len(data)
4
>>> data.data
[[3, 2, 1, 4], 
 [4, 3, 2, 2], 
 [1, 3, 4, 2], 
 [3, 3, 1, 2]]	
```

License
========================================================
Copyright (C) 2018 Xuansheng Wu

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
A light Python library for data processing.
