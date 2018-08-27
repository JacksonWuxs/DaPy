## Introduction
#### What is DaPy?
DaPy is a Python package providing efficiency and readily usable data structures designed by Python origin data strucutres. 
It aims to be a fundamental and friendly tools for real-world data processing in Python. Additionally, we also built some common data analysis algorithms, both in statistics and machine learning, in the library, in order to help you testify your idea as soon as posible.  

#### Why use DaPy?  
A large number of eminent libraries are now available to efficiently support scientific computing and data analysis. However, these kind of libraries are not friendly to a freshman in Python, because they have to spend a lot of time getting familiar with these data structures. 

Traversing the data set for example, most people would use ``for`` syntax as their first idea. But Pandas would iterate the column names only, not the records. Moreover, when users try to select some of the records with conditions, Pandas needs a `Bool` set at first. It is not visible for user to know how the data opearted. 

There is no doubt that these deferential libraries play major roles in the data science filed. However, they would need some complementary products at the same time. In contrast with these data processing or computing libraries, DaPy focus on a specific aspect, which is defined as rapid development. In a simple word, DaPy is suitable for the users who are in the begining of a new research. With the DaPy help, scientists can fluently implements their ideas without limitation of complex syntax.  

A recommend way to use DaPy is that using it while you do some pre-process of your data set in a new research. After you testify your demo, you can use numpy or tensorflow data structures to rewrite your idea. But it doesn't means that DaPy is not up to the processing of big data. On the contrary, he also has strong operational efficiency in some aspects.


#### Who use DaPy?
DaPy is a powerful and flexible data processing tools, therefore it suitable most of users who have the demand in data mining. DaPy can help a expert develop and testify their own method quickly, can help reserachers handle their data conviniencetly, and help a novice complete a heavy data task in a short time. By the way, in Shanghai University of International Business and Economics, DaPy has been widely used in some programs.


#### How to use DaPy?
While you using DaPy as a data processing tool in your programe, you just need to imagine it as a Excel file. And in the most of time, the first idea which jumped out from your mind is the correct syntax while you processing the data. Here is a simple example to testify if the syntax matches your idea. 

First of all, we make a `Frame` structure as follow. Frame is a kind of `sheet` in DaPy, the another `sheet` structure is `SeriesSet`.
```
>>> import DaPy as dp
>>> data = dp.Frame([
	[1, 2, 3, 4, 5, 6],
	[1, 3, 5, 7, 9, 11],
	[2, 4, 6, 8, 10, 12]], 
   	['A_col', 'B_col', 'C_col', 'D_col', 'E_col', 'F_col'])
>>> data
 A_col | B_col | C_col | D_col | E_col | F_col
-------+-------+-------+-------+-------+-------
   1   |   2   |   3   |   4   |   5   |   6   
   1   |   3   |   5   |   7   |   9   |   11  
   2   |   4   |   6   |   8   |   10  |   12  
```
Now, our task is picking out following columns: 'B_col', 'C_col', 'D_col', and 'F_col'. We find that 'B_col', 'C_col' and 'D_col' are connected together. Think about the `slice` using in native Python structures such as list. Here is what we do.
```
>>> data['B_col': 'D_col', 'F_col']
 B_col | C_col | D_col | F_col
-------+-------+-------+-------
   2   |   3   |   4   |   6   
   3   |   5   |   7   |   11  
   4   |   6   |   8   |   12  
 ```




