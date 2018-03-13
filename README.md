Description
========================================================
DaPy is a light data processing and analyzing library which could
be used in `loading dataset` from data profiles amenity and 
contain or reduce part of the dataset conveniency. We hope that
Datapy can help data scientists process their datasets
more quickly and easily. Additionly,it contains some 
basic formulas that help the developers to understand 
the basic attributes about the data set. In the future, 
we will implement functions such as `matrix operations` 
for DaPy, and we will include more `statistic formulas`.
<br>The first public stable version (V1.2.4) will be uploaded
to `pypi` by `March 15`.</br>

Characteristic
========================================================
We have tested the performance of DaPy in loading file.
It was an amazing result that DaPy has the `fastes speed` with `less memory`.
Despite this, because DaPy is not yet able to support matrix operations, it has a long way to go to achieve Numpy's and Pandas' achievements in scientific computing. The testing code has been uploaded already.
```
            Environment:
    File Size  96MB
     Platform  Win10
     Language  Python2.7-64Bit

            Result of Testing
       	        DaPy	Pandas    Numpy   
 Loading Time  17.77s | 4.55s  | 62.23s 
Traverse Time   0.5s  | 316.5s |  0.1s  
  Total Spent  18.2s  | 321.1s |  62.3   
  Memory Size   14MB  | 174MB  |  69MB    
```
Examples
========================================================

```Python
<Example 1>
>>> import DaPy as dp
>>> data = dp.DataSet('testdb.csv')
>>> data.readframe()
>>> data.data
[Data(A_col=3, B_col=2, C_col=1, D_col=4),
 Data(A_col=4, B_col=3, C_col=2, D_col=2), 
 Data(A_col=1, B_col=3, C_col=4, D_col=2), 
 Data(A_col=3, B_col=3, C_col=1, D_col=2), 
 Data(A_col=4, B_col=5, C_col=4, D_col=3),
 Data(A_col=2, B_col=1, C_col=1, D_col=5)]
>>> data[0].B_col # Data in the second column of the first record
2
>>> data[-1].C_col # Data in the third column of the last record
1
```

From Example 1, you have loaded a data set as a frame 
from your data profile. Easily, right? Anyway, there is 
another way to load data by column as series. Here is 
the example.

```Python
<Example 2>
>>> data.readcol()
>>> data.titles
['A_col', 'B_col', 'C_col', 'D_col']
>>> for title in data.titles:
	print title, data[title]
	
A_col [3, 4, 1, 3, 4, 2]
B_col [2, 3, 3, 3, 5, 1]
C_col [1, 2, 4, 1, 4, 1]
D_col [4, 2, 2, 2, 3, 5]
```

Example 1 and Example 2  have showed two ways to load data.
But a simpler data structure is always neccessary. We offer
the third way to load data in bellowed.

```Python
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


```Python
<Example 4>
>>> len(data) # How much records does dataset include
6
>>> str(data) # The name of dataset
'Data'
>>> dp.CountDistribution(data['A_col'],[0.25,0.5,0.75])
[2,3,4]
>>> dp.CountFrequancy(data['A_col'],2)
0.16666666666666666
>>> dp.Statistic(data['A_col'])
STAT(Mean=2.8333333333333335, Std=1.1690451, CV=0.4126041862764749, Min=1, Max=4, Range=3)
```

Finally, we also support the client opearts the data with
'-' or '+'. It will help you change the data size.


```Python
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
Installation
========================================================
Download using pip via pypi.
> pip install datapy


Data Structure
========================================================
Since the very beginning, we have brought about DaPy to Python's native data structure as much as possible, so that it is easier for users to adapt to our library. At the same time, we believe that the original data structure of Python will definitely perform better than our own. The results of the experiment also prove this idea.
- Type of Value
	- [float] 
		If the symbol of '.' is inside of the value, Datapy will try to transfrom the value into 'float' at first, while you load a new dataset from file. The type of float are descided by Python.
	- [integer]
		If the value contains digit number only, Datapy will transfrom it in to 'int'.
	- [Bool]
		If the value is similar as blanket, 'None', 'NA', 'N/A' and '?', Datapy will transfrom the value into None or anything else you set as missing value.
	- [string] 
		If the value doesn't meet any condition, it will be transfrom into 'string'. 

- Type of DataSet
	- [frame] It is a list with amount of named-tuples inside. You can easily get the records you want with this data structure. And it is write-protected by "tuples", and you can be more at ease with your data. Therefore, we recommend that you use this data structure when you only need to traverse the data without changing it.
	- [table] It is a list with amount of lists inside. This is a simple data structure for you. While you use this structure, you always expect to process the source data set.
	- [series] It is a diction while the column name are 'Key' and datas are included in a list as 'Value'. When you need to analyze a variable in the dataset and not all the variables in each record, you can easily extract one column in this structure.


Instructions for use
========================================================
```Python
DaPy.DataSet(addr='data.csv', title=True, split='AUTO',db=None, name='Data', firstline=1, miss_value=None)
```
This class is the core function for processing data, which supports user to pretreatment the database.
distribution of the data.
* [addr] Your database path and name;
* [title] "True" means your database includes title in some line. To the contrary, "False" means the there is no title on it.
* [split] This variable means the separator of the database in every line. 'AUTO' will ask system find the best match separator to split the file. The system will set separator as ',' if the file type is '.cvs' and set separator as ' ' if the file type is '.txt', while will set separator as '\t' if the file type is '.xls'.
* [db] You could set a canned database.
* [name] Your database name.
* [firstline] Your data starts from "firstline".
* [miss_value] The missing value in your data will be replace into the simble of this variable.
```Python
DataSet.readframe(col=all)
```
This function will help you load data from a file as a DataFrame, which implements with some named-tuple in a list. You could pickout the  data with line number and columen name.
- [col] Giving a iterable variable which contains the column number you would like to choose.

```Python
DataSet.readtable(col=all)
```
This function will help you load data from a file as a DataFrame, which implements with lists only. So it will not be allowed in pick out data with the column name.
- [col] Giving a iterable variable which contains the column number you would like to choose.

```Python
DataSet.readcol(col=all)
```
This function supports user to load data by column which has the data structure as a diction and the keyword is column and the series is value. Additonly, you could pick series of  data in one column.
- [col] Giving a iterable variable which contains the column number you would like to choose.

```Python
DataSet.data
```
- Return the data you just loaded before.

```Python
DataSet.titles
```
- Return the title of this data set.

```Python
str(DataSet.data)
```
- Return the name of this dataset.

```Python
len(DataSet.data)
```
- Return the number of records.

```Python
DaPy.CountFrequancy(data, cut=0.5)
```
This function is used in counting the distribution of a data series. 
- [cut] means the cut point you want.If you would like to calculate the proportion of series, you can get help from this function.
- [data] expects a iterble item, such as tuple() or list(). Unnecessary variable "cut" expects a number. It will return a float means the proportion of data which is larger than "cut".

```Python
Dapy.CountDistribution(data, shapes=[0.05,0.1,0.25,0.5,0.75,0.9,0.95])
```
This function could help you find the distribution of the data.
- Return the value of each quantile.

```Python
Dapy.Statistic(data)
```
- Return the basic statistics of the data set as NamedTuple.
- The tuple includes 'Mean','Std','CV','Min','Max' and 'Range'.

License
========================================================
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
A light Python library for data processing.</br>
