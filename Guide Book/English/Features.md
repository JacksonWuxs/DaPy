## Features
**Convinience** and **efficiency** are the cornerstone of DaPy. 
Since the very beginning, we have designed DaPy to Python's 
native data structures as much as possible and we struggle to make 
it supports more Python syntax habits. Therefore you can 
adapt to DaPy quickly, if you imagine you are opearting an 2-dimentions table.
In addition, we do our best to simplify
the formulas or mathematical models in it, in order to let you 
implement your ideas fluently.   

#### Visually manage diverse data
Every data scientist should have at least one experience in handling the needed datas 
with multiple sources. It is inconvenient to manage or access datas with amount of 
variables names. In this section, we will simply introduce a data container, which 
represented the ideology of designing DaPy, called *DataSet*.

Both data scientist and a young kid in primary school are skillful in 
MS Office Excel software. In this software, every data should be contained in a 
*sheet* structures. We draw on ideas from Excel and proposed a data managing structure  
that is *DataSet*. 

Here is a example how does DaPy work basically to manage diverse dataset. We have prepared a [students.xlsx](http://www.wuxsweb.cn/Library/DaPy&Examples_data/students.xlsx) file as a example, which has 3 sheets insides, named "Info", "Course", and "Scholarship". Now, we will use DaPy to read this file into a DataSet object and access the data.
```Python2
>>> import DaPy as dp
>>> data = dp.read('students.xlsx', 'frame')
>>> data
sheet:Info
==========
   ID   |   Name  | Gender | Age 
--------+---------+--------+------
 1801.0 |  Olivia |   F    | 14.0 
 1802.0 |  James  |   M    | 14.0 
 1803.0 | Charles |   M    | 15.0 
 1804.0 |   Emma  |   F    | 16.0 
 1805.0 |   Mary  |   F    | 13.0 
 1806.0 |  Kevin  |   M    | 14.0 
 1807.0 |  Jeanne |   F    | 15.0 

sheet:Course
============
   ID   |   Course   | Score
--------+------------+-------
 1801.0 | Chemistry  |  90.0 
 1802.0 |  Biology   |  87.0 
 1803.0 |  Biology   |  88.0 
 1804.0 |  Geology   |  85.0 
 1805.0 | psychology |  92.0 
 1806.0 | Chemistry  |  93.0 
 1807.0 |  Geology   |  87.0 

sheet:Scholarship
=================
   ID   | Scholarship
--------+-------------
 1801.0 |    Third    
 1805.0 |    Second   
 1806.0 |    First    
 ```
And now, we have a new sheet named "tuition" that needs to be added into data and save it as "new_students.xlsx". One of the sheet structure in DaPy is *Frame*. You can initialize a new Frame object with records and column names. 
 ```Python3
>>> tuition = dp.Frame(
	[[1801, 3000],
	 [1802, 3500],
	 [1803, 3000],
	 [1804, 2500],
	 [1805, 2500],
	 [1806, 2500],
	 [1807, 3000]],
	['ID', 'Tuition'])
>>> data.add(tuition, 'Tuition')
>>> data.save("new_students.xlsx")
``` 
#### Easily insert and delete a large number of data  
As far as we are concerned, DaPy is a kind of data manage system, therefore, we learned from the thinking as 'CRUE'(Create, Retrieve, Update and Delete). We followed some of the 'list' data structure supported functions and extended them appropriately to fit the two-dimensional data structure. In this section, we will begin with briefly review all these functions.

```DaPy.DataSet.add()``` is the hightest level data function, which is used to add a new 2-dimentional data structure into DataSet structure. With this function, DataSet can support multiple sheets inside. Following example shows how to add a new sheet.
```Python2
>>> data = dp.DataSet([[1, 1, 1], [1, 1, 1]], 'sheet-1')
>>> data.add([[2, 2, 2], [2, 2, 2]], 'sheet-2')
>>> data.toframe()
>>> data
sheet:sheet-1
=============
 C_0 | C_1 | C_2
-----+-----+-----
  1  |  1  |  1  
  1  |  1  |  1  
  
sheet:sheet-2
=============
 C_0 | C_1 | C_2
-----+-----+-----
  2  |  2  |  2  
  2  |  2  |  2  
```

Now, we are going to introduce two pairs of functions to you. One of a pair of functions are ```append()``` and ```append_col()```, and which is obviously to see the meanings. ```append()``` can help you append a new record at the tail of each sheet and ```append_col()``` can support you to append a new variable at the tail of each sheet in DataSet. On the other hand, ```extend()``` and ```extend_col()``` were designed to add amount of records or amount of variables at the tail of each sheets in dataset.
```Python2
>>> from DaPy import datasets
>>> example = datasets.example()
>>> example.append([None, None, None, None])
>>> example.append_col(range(example.shape[0].Ln), 'New_col')
>>> example.show()
sheet:sample
============
 A_col | B_col | C_col | D_col | New_col
-------+-------+-------+-------+---------
   3   |   2   |   1   |   4   |    0    
   4   |   3   |   2   |   2   |    1    
   1   |   3   |   4   |   2   |    2    
   3   |   3   |   1   |   2   |    3    
   4   |   5   |   4   |   3   |    4    
   2   |   1   |   1   |   5   |    5    
   6   |   4   |   3   |   2   |    6    
   4   |   7   |   8   |   3   |    7    
   1   |   9   |   8   |   3   |    8    
   3   |   2   |   6   |   5   |    9    
   2   |   9   |   1   |   5   |    10   
   3   |   4   |   1   |   6   |    11   
  None |  None |  None |  None |    12  
>>>
>>> # we will add 3 new records at the same time, and especially the third record has miss values.
>>> example.extend([ 
	['A', 'A', 'A', 'A', 'A'],
	['B', 'B', 'B', 'B', 'B'],
	['C', 'C', 'C']])
>>> example.show(5)
sheet:sample
============
 A_col | B_col | C_col | D_col | New_col
-------+-------+-------+-------+---------
   3   |   2   |   1   |   4   |    0    
   4   |   3   |   2   |   2   |    1    
   1   |   3   |   4   |   2   |    2    
   3   |   3   |   1   |   2   |    3    
   4   |   5   |   4   |   3   |    4    
             .. Omit 6 Ln ..              
   3   |   4   |   1   |   6   |    11   
  None |  None |  None |  None |    12   
   A   |   A   |   A   |   A   |    A    
   B   |   B   |   B   |   B   |    B    
   C   |   C   |   C   |  None |   None  
 ```
>>>
