### DaPy - Provides You With Smooth Data Processing Experience You've  Never Had
Are you upset by the strict data structure requirements of Pandas from time to time? Are you having a headache of consulting various documents for a simple operation？

DaPy is here to emancipate you!!! Using DaPy, you can realise ideas with ease and you don't have to worry about being puzzled by an unfamiliar API or being interrupted by a popping up data format error. 

DaPy is a data analysis framework that pays close attention to usability from the beginning of its design. It is designed for data analysts, not programmers. What makes data analysts different is their problem-solving ideas, not hundreds of lines of code that make them work overtime!

### How Friendly DaPy Is ?

##### 1. Various Ways of Presenting Data in The Command Console
Do not underestimate the way you browse data!!! As for data analysts, data perception plays a prominent part in their everyday work.
```python
>>> from DaPy.datasets import iris
>>> sheet, info = iris()
 - read() in 0.001s.
>>> sheet
sheet:data
==========
sepal length: <5.1, 4.9, 4.7, 4.6, 5.0, ... ,6.7, 6.3, 6.5, 6.2, 5.9>
 sepal width: <3.5, 3.0, 3.2, 3.1, 3.6, ... ,3.0, 2.5, 3.0, 3.4, 3.0>
petal length: <1.4, 1.4, 1.3, 1.5, 1.4, ... ,5.2, 5.0, 5.2, 5.4, 5.1>
 petal width: <0.2, 0.2, 0.2, 0.2, 0.2, ... ,2.3, 1.9, 2.0, 2.3, 1.8>
       class: <setos, setos, setos, setos, setos, ... ,virginic, virginic, virginic, virginic, virginic>
>>> sheet.info
sheet:data
==========
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=150 | Variables=5
3. Miss Value: 0 elements
                                   Descriptive Statistics                                   
============================================================================================
    Title     | Miss |     Min      |     Mean    |     Max     |     Std      |    Mode    
--------------+------+--------------+-------------+-------------+--------------+------------
 sepal length |  0   |  4.300000191 | 5.843333327 | 7.900000095 | 0.8253012767 |          5
 sepal width  |  0   |            2 | 3.054000003 | 4.400000095 | 0.4321465798 |          3
 petal length |  0   |            1 | 3.758666655 | 6.900000095 |  1.758529178 |        1.5
 petal width  |  0   | 0.1000000015 | 1.198666658 |         2.5 | 0.7606126088 |        0.2
    class     |  0   |            - |           - |           - |            - |      setos
============================================================================================
>>> sheet.show(5)
sheet:data
==========
 sepal length | sepal width | petal length | petal width |  class  
--------------+-------------+--------------+-------------+----------
     5.1      |     3.5     |     1.4      |     0.2     |  setos   
     4.9      |     3.0     |     1.4      |     0.2     |  setos   
     4.7      |     3.2     |     1.3      |     0.2     |  setos   
     4.6      |     3.1     |     1.5      |     0.2     |  setos   
     5.0      |     3.6     |     1.4      |     0.2     |  setos   
                          .. Omit 140 Ln ..                          
     6.7      |     3.0     |     5.2      |     2.3     | virginic 
     6.3      |     2.5     |     5.0      |     1.9     | virginic 
     6.5      |     3.0     |     5.2      |     2.0     | virginic 
     6.2      |     3.4     |     5.4      |     2.3     | virginic 
     5.9      |     3.0     |     5.1      |     1.8     | virginic 
```
##### 2. Well-Designed Table Structure for Human
Many database systems, such as MySQL, Excel and SAS, are designed to store data line by line, since processing data that way tallies with the thinking method of human. However,  Pandas is first designed to handle time series data, so data are stored column by column, a method which does not tie in with human brain and which makes Pandas difficult to master. For example, Pandas does not support very useful operations such us assigning values to lines that are iterated by`DataFrame.iterrows()`, an operation that is, in turn, available in Numpy. Many data analysts have to live with this steep learning curve because there is almost no substitute for Pandas. 

To make the steep curve flat, DaPy bring this common practice back by introducing the concept of "View".

```python
>>> import DaPy as dp
>>> sheet = dp.SeriesSet({'A': [1, 2, 3], 'B': [4, 5, 6]})
>>> for row in sheet:
	print(row.A, row[0])   # access values by index or column
	row[1] = 'b'   # assign values by index
1, 1
2, 2
3, 3
>>> sheet.show()   # your operation to the row is actually working on the sheet
 A | B
---+---
 1 | b 
 2 | b 
 3 | b 
>>> row0 = sheet[0]   # get the view of that row 
>>> row0
[1, 'b']
>>> sheet.append_col(series=[7, 8, 9], variable_name='newColumn')
>>> sheet.show()
 A | B | newColumn
---+---+-----------
 1 | b |     7     
 2 | b |     8     
 3 | b |     9     
>>> row0   # your operation to the sheet will react to the row
[1, 'b', 7]
```

##### 3. By The Way, Did Anyone Say He Likes Chain Programming? 
Here is an interesting Chain Programming example. 

Following are 6 operations that we want to apply to the Anderson's Iris data set.

（1）Normalise each column;

（2）Screen out records that meet the criteria that ‘sepal length’ is shorter than ‘petal length’;

（3）Group the filtered data set by the class of Iris;

（4）Sort the 'petal width' of each subgroup in ascending order;

（5）Select the first 10 rows of each sorted subgroup;

（6）Show descriptive statistical information of each subsets;

```python
>>> from DaPy.datasets import iris
>>> sheet, info = iris()
 - read() in 0.001s.
>>> sheet.normalized().query('sepal length < petal length').groupby('class').sort('petal width')[:10].info
 - normalized() in 0.005s.
 - query() in 0.000s.
 - groupby() in 0.000s.
 - sort() in 0.000s.
sheet:('virginic',)
===================
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=10 | Variables=5
3. Miss Value: 0 elements
                                      Descriptive Statistics                                      
==================================================================================================
    Title     | Miss |      Min      |     Mean     |     Max      |      Std      |     Mode     
--------------+------+---------------+--------------+--------------+---------------+--------------
 sepal length |  0   |   0.166666612 | 0.5722221836 | 0.8333333135 |  0.1770819779 | 0.5555555173
 sepal width  |  0   | 0.08333332837 | 0.2958333254 | 0.4166666567 |  0.1028246885 | 0.3749999851
 petal length |  0   |  0.5932203531 | 0.7474576116 | 0.8983050585 | 0.08523585797 | 0.8135593089
 petal width  |  0   |  0.5416666865 | 0.6541666567 | 0.7083333135 | 0.06194194576 | 0.7083333332
    class     |  0   |             - |            - |            - |             - |     virginic
==================================================================================================
sheet:('setos',)
================
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=6 | Variables=5
3. Miss Value: 0 elements
                                         Descriptive Statistics                                         
========================================================================================================
    Title     | Miss |       Min        |      Mean     |      Max      |      Std      |      Mode     
--------------+------+------------------+---------------+---------------+---------------+---------------
 sepal length |  0   | -5.298190686e-08 | 0.05092587401 |  0.1388888359 | 0.04652720208 | 0.02777772553
 sepal width  |  0   |            0.375 |  0.4583333184 |  0.5833333135 | 0.06804137465 |  0.4166666501
 petal length |  0   |    0.01694915257 | 0.07062146782 |  0.1525423676 | 0.04199454314 | 0.05084745681
 petal width  |  0   | -6.208817349e-10 | 0.03472222315 | 0.04166666791 | 0.01552825054 | 0.04166666607
    class     |  0   |                - |             - |             - |             - |         setos
========================================================================================================
sheet:('versicolo',)
====================
1.  Structure: DaPy.SeriesSet
2. Dimensions: Lines=10 | Variables=5
3. Miss Value: 0 elements
                                      Descriptive Statistics                                     
=================================================================================================
    Title     | Miss |     Min      |     Mean     |     Max      |      Std      |     Mode     
--------------+------+--------------+--------------+--------------+---------------+--------------
 sepal length |  0   |  0.166666612 | 0.3083332881 | 0.4722221792 |   0.101265146 | 0.1944443966
 sepal width  |  0   |            0 |  0.166666659 | 0.2916666567 | 0.07905693876 |   0.16666666
 petal length |  0   | 0.3389830589 | 0.4423728734 | 0.5254237056 | 0.05644347527 | 0.3898305022
 petal width  |  0   |        0.375 |  0.387499997 | 0.4166666567 | 0.01909406084 | 0.3749999996
    class     |  0   |            - |            - |            - |             - |    versicolo
=================================================================================================
```
##### 4. Some Good Features of Numpy & Pandas Are Inherited
```python
>>> sheet.A + sheet.B
>>> sheet[sheet.A > sheet.B]
```
### What's More？

##### 1. Super Powerful and Robust I/O Tools
We are sometimes faced with a same problem----how to convert .csv files to Excel files and the reverse?
```python
>>> from DaPy.datasets import iris
>>> sheet, info = iris()
>>> sheet.groupby('class').save('iris.xls') # this step actually save three sub-sheets in Excel
 - groupby() in 0.000s.
 - save() in 0.241s.
>>> import DaPy as dp
>>> dp.read('iris.xls').shape # DaPy read 3 table in once
 - read() in 0.004s.
sheet:('virginic',)
===================
sheet(Ln=50, Col=5)
sheet:('setos',)
================
sheet(Ln=50, Col=5)
sheet:('versicolo',)
====================
sheet(Ln=50, Col=5)
```
Do you think this is all what `read()` can do？Let's see something more impressive！！！
```python
>>> import DaPy as dp
>>> dp.read('iris.xls').save('iris.db') # Excel to Sqlite3
>>> dp.read('iris.sav').save('iris.html') # SPSS to HTML
>>> dp.read('https://sofifa.com/players').save('mysql://root:123123@localhost:3306/fifa_db') # Scrpe the FIFA players information and save htem into MySQL
>>> dp.read('mysql://root:123123@localhost:3306/fifa_db').save('fifa.csv') # MySQL to CSV
```
##### 2. DaPy Supports Super-many Data Preprocessing and Feature Engineering Operations
Data Preprocessing
```python
>>> sheet.drop_duplicates(keep='first') #drop out duplicating records
>>> sheet.fillna(method='linear') # fillng NaN with linear interception
>>> sheet.drop('ID', axis=1) # Remove some useless variables
>>> sheet.count_values('gender') # Counting the frequency
```
Feature Engineering
```python
>>> sheet.get_date_label('birth') # Process the date variable to automatically construct more periodic variables
>>> sheet.get_categories(cols='age', cutpoints=[18, 30, 50], group_name=['Teenager', 'Mature', 'Middle Age', 'Senior']) # Seperate continuous variables into categorical variables
>>> sheet.get_dummies(['city', 'education']) # categorical variables to dummy variables
>>> sheet.get_interactions(n_power=3, col=['income', 'age', 'gender', 'education']) # form high order interaction variables 
```
##### 3. Last But Not Least, The Machine Learning Module!!!
DaPy has four built-in models: linear regression, logical regression, multi-layer perceptron and C4.5 decision tree. When it comes to statistical models, the developers of DaPy's think Sklearn and Tensorflow have already done an unsurpassable job. Our prime focus is on presenting more reports of hypothesis-testings since the main devoloper's major is  Statistics.
Let's first take a look at a demo.

```python
>>> from DaPy.datasets import iris
>>> sheet, info = iris()
 - read() in 0.001s.
>>> sheet = sheet.shuffle().normalized()
 - shuffle() in 0.001s.
 - normalized() in 0.005s.
>>> X, Y = sheet[:'petal width'], sheet['class']
>>> 
>>> from DaPy.methods.classifiers import MLPClassifier
>>> mlp = MLPClassifier().fit(X[:120], Y[:120])
 - Structure | Input:4 - Dense:4 - Output:3
 - Finished: 0.2%	Epoch: 1	Rest Time: 0.24s	Accuracy: 0.33%
                   ### Logs are omitted here. ###
 - Finished: 99.8%	Epoch: 500	Rest Time: 0.00s	Accuracy: 0.88%
 - Finish Train | Time:1.9s	Epoch:500	Accuracy:88.33%
>>> 
>>> from DaPy.methods.evaluator import Performance
>>> Performance(mlp, X[120:], Y[120:], mode='clf')
 - Classification Accuracy: 86.6667%
 - Classification Kappa: 0.8667
┏                   ┓
┃ 11   0    0    11 ┃
┃ 0    8    1    9  ┃
┃ 0    3    7    10 ┃
┃11.0 11.0 8.0  30.0┃
┗                   ┛
```



### Finally, if you find DaPy useful，[give us a star on Github](https://github.com/JacksonWuxs/DaPy)! Issues are also very welcomed!!!

