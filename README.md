
Description
========================================================
As a light data **processing** and **analysis** library，**DaPy** is
committed to saving analyzing time of data scientists 
and improving the efficiency of research.
  
In terms of **data loading**, DaPy's data structure 
is so clear and concise that data scientists could "feel" data;
functions are feature-rich and efficient, saving data
scientists the processing time for complicated data. In terms
of **descriptive statistics**, it has provided comprehensive
calculation formulas that can help data scientists quickly
understand data characteristics. Finally, in terms of 
**algorithms**, it has provided a simple multilayer 
perceptrons for predicting.

In the future, DaPy will add more data cleansing and
**inferential statistics** functions; implement more formulas
used in **mathematical modeling**; and even includes more 
**machine learning models** (support vector machines, liner regresion, etc.). DaPy is
continuously improving according to the data analysis process.  

If you think DaPy is interesting or helpful,
don't forget to share DaPy with your frineds! If you have 
any suggestions, please tell us with "Issues". Besides, 
**giving us a 'Star'** will be the best way to encourage us!  

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

Features
========================================================
### Ⅰ. Comfortable Experience
Since the very beginning, we have designed DaPy to Python's 
native data structures as much as possible and we try to make 
it support more Python operaing habits. Therefore you can 
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
  
### Ⅱ. Efficiency  
We have testified the performance of DaPy in three fields 
(load data, sort data & traverse data), 
those were most useful functions to a data processing library.
In contrast with those packages written by C programe languages,
DaPy showed an amazing efficiency in testing. In all subjects of
test, DaPy just spends less than twice time as long as the 
fastest C language library.   
  
  
We tested DaPy on the platform with
Intel i7-6560U while the Python version is 2.7.13-64Bit. The 
dataset (Download Addr: https://pan.baidu.com/s/1kK3_V8XbbVim4urDkKyI8A)
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

Quick Start
========================================================
## Loading a dataset
DaPy comes with a few famous datasets, for examples the **iris** 
and **wine** datasets for classification.   
  
In the following, we will launch a Python shell and then 
load the wine datasets. 
```Python
>>> from DaPy import datasets
>>> wine, info = datasets.wine()
```
This function will return a *DaPy.SeriesSet* structure that holds 
all the data while a description of data will be returned at the 
same time. 
  
In general, to load from an external dataset, you can use these 
statements, please refer to GuideBook for more details.
```Python
>>> import DaPy as dp
>>> data = dp.DataSet(file_name)
>>> data.readcol()
```
In this case, as a supervised problem, all of the 
independent variables and dependent variables are stored in the 
*SeriesSet* menber. For instance, the data of the *wine* could be accessed using:
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
-------------------------------------
SeriesSet{178 Records & 16 Variables}
```
Every object of *SeriesSet* will auto concluses some basic information of the 
dataset (number of miss value, number of records & variable names). For exaples, 
you can browse the dataset of *wine* as:
```Python
>>> wine.info
1.  Structure: DaPy.SeriesSet
2.   Set Name: MySeries
3. Dimensions: Ln=178 | Col=16
4. Miss Value: 0 elements
5.    Columns:        Title        |  Miss Value  |  Column Type  |  Value Type 
               --------------------+--------------+---------------+-------------
                     Alcohol       |       0      |     <list>    |   <float>   
                    Malic acid     |       0      |     <list>    |   <float>   
                       Ash         |       0      |     <list>    |   <float>   
                Alcalinity of ash  |       0      |     <list>    |   <float>   
                    Magnesium      |       0      |     <list>    |    <int>    
                  Total phenols    |       0      |     <list>    |   <float>   
                    Flavanoids     |       0      |     <list>    |   <float>   
               Nonflavanoid phenols|       0      |     <list>    |   <float>   
                 Proanthocyanins   |       0      |     <list>    |   <float>   
                 Color intensity   |       0      |     <list>    |   <float>   
                       Hue         |       0      |     <list>    |   <float>   
                      OD280        |       0      |     <list>    |   <float>   
                     Proline       |       0      |     <list>    |    <int>    
                     class_1       |       0      |     <list>    |    <int>    
                     class_2       |       0      |     <list>    |    <int>    
                     class_3       |       0      |     <list>    |    <int>    
               --------------------+--------------+---------------+-------------
```
## Preparing data
Before we start a machine learning subject, we should process our 
data so that the data can meet the requirements of the models.   
  
By just accessed our data we found that our dataset is arrangement 
by class. For supporting a proportionate training data, we can 
mass our data with *shuffles()*.
```Python
>>> wine.shuffles()
```
After disrupting the data, we should separte our data according to the 
target variables and feature variables. In addition, we will convert 
the data into a *DaPy.Matrix* structure Simultaneously.
```Python
>>> import DaPy as dp
>>> target = dp.Matrix(wine.pop_col('class_1', 'class_2', 'class_3')) # contains the target
>>> wine.tomatrix()  # contains the feature variables
```
## Learning and predicting
In the case of the wine dataset, the task is to predict, given a new record, 
which class it represents. We are given samples of each of the 3 possible classes on 
which we fit an estimator to be able to predict the classes to which unseen samples belong.  
  
In DaPy, and example of an estimator is the class DaPy.MLP that implements *mutilayers perceptrons*. 
```Python
>>>

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
