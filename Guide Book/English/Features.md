## Features
**Convinience** and **efficiency** are the cornerstone of DaPy. 
Since the very beginning, we have designed DaPy to Python's 
native data structures as much as possible and we try to make 
it supports more Python syntax habits. Therefore you can 
adapt to DaPy quickly, if you imagine you are opearting an Excel table.
In addition, we do our best to simplify
the formulas or mathematical models in it, in order to let you 
implement your ideas fluently.   

#### Visibly manage diverse data
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
#### 
