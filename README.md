
Description
========================================================
As a light data **processing** and **analysis** library，**DaPy** is
committed to saving analyzing time of data scientists 
and improving the efficiency of research.
We hope that DaPy can help data scientists process or 
analysis their datasets more *quickly* and *easily*.  

In terms of **data loading**, DaPy's data structure 
is so clear and concise that data scientists could "feel" data;
functions are feature-rich and efficient, saving data
scientists the processing time for complex data. In terms
of **descriptive statistics**, DaPy has provided comprehensive
calculation formulas that can help data scientists quickly
understand data characteristics.  

In the future, DaPy will add more data cleansing and
***inferential statistics functions***; implement more formulas
used in ***mathematical modeling***; and even
includes more ***machine learning models*** (multilayer
perceptrons, support vector machines, etc.). DaPy is
continuously improving according to the data analysis process.  

If you think DaPy is interesting or helpful,
don't forget to share DaPy with your frineds! If you have 
any suggestions, please tell us with "Issues". Besides, 
*giving us a 'Star'* will be the best way to encourage us!  

Installation
========================================================
The Latest version 1.3.2 had been upload to PyPi.
```
pip install DaPy
```
Updating your last version to 1.3.2 with PyPi as follow.
```
pip install -U DaPy
```

Characteristic
========================================================
We have testified the performance of DaPy in three fields 
(load data, sort data & traverse data), 
which were most useful function to a data processing library.
In contrast with those packages written by C programe languages,
DaPy showed amazing efficiency in testing. In all subjects of
test, DaPy spent almost only twice as long as the fastest
C program language library.


<table>
<tr>
	<td>Date: 2018-4-21</td>
	<td>Testing Information</td>
	</tr>
<tr>
	<td>CPU</td>
	<td>Intel Core i7-6560U</td>
	</tr>
<tr>
	<td>Hard Disk</td>
	<td>NVMe THNSN5256GPU7 NV</td>
	</tr>
<tr>
	<td>File</td>
	<td>240MB / 4.32 million records</td>
	</tr>
<tr>
	<td>Platform</td>
	<td>Win10 / Python2.7-64Bit</td>
	</tr>
</table>
<table>
<tr>
	<td>Result of Testing</td>
	<td>DaPy</td>
	<td>Pandas</td>
	<td>Numpy</td> 
</tr>
<tr>
	<td>Loading Time</td>
	<td>23.4s (1.9x)</td>
	<td>12.3s (1.0x)</td>
	<td>169.0s (13.7x)</td>
</tr>
<tr>
	<td>Traverse Time</td>
	<td>0.53s (2.5x)</td>
	<td>4.18s (20.9x)</td>
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
	<td>1.3.2</td>
	<td>0.22.0</td>
	<td>1.14.0</td>
	</tr>
</table>

Data Structure
========================================================
Since the very beginning, we have designed DaPy to Python's native data structure as much as possible, so that it is easier for users to adapt to our library. At the same time, we believe that the original data structure of Python will definitely perform better than our own. The results of the experiment also prove this idea.
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
DataSet.tocsv(addr)
```
This function will help you to save the dataset as a 'csv' file.
- [addr] Expects a string type value of file address.
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
DaPy.cor(data_1, data_2)
```
This function will help you calculate the correlation of two series data.
```Python
DaPy.cov(data_1, data_2)
```
This function will help you calculate the covariance of two series data.
```Python
DaPy.CountFrequency(data, cut=0.5)
```
This function is used in counting the distribution of a data series. 
- [cut] means the cut point you want.If you would like to calculate the proportion of series, you can get help from this function.
- [data] expects a iterble item, such as tuple() or list(). Unnecessary variable "cut" expects a number. It will return a float means the proportion of data which is larger than "cut".

```Python
DaPy.CountQuantiles(data, shapes=[0.05,0.1,0.25,0.5,0.75,0.9,0.95])
```
This function could help you find the Quantiles of the data.
- [shapes] Expects a iterble item which includes some decimals.
- Return the value of each quantile.
```Python
DaPy.CountDistribution(data, breaks=10)
```
This function could help you statistic the distribution of the data.
- [breaks] Expects a integer which means how many groups you would like to partition.
- Return a list that includes the frequency of each partition.
```Python
DaPy.Statistic(data)
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
A light Python library for data processing and analysing.</br>
