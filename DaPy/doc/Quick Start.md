## Quick Start
#### Ⅰ. Loading a dataset
DaPy comes with a few famous datasets, for examples the **iris** 
and **wine** datasets for classification.   
  
In the following, we will start a Python shell and then 
load the wine datasets as an example: 
```Python
>>> from DaPy import machine_learn
>>> from DaPy import datasets
>>> wine, info = datasets.wine()
```
This function will return a *DaPy.SeriesSet* structure that holds 
all the data while a description of data will be returned at the 
same time. 
  
In general, to load from an external dataset, you can use these 
statements, please refer to GuideBook for more details:
```Python
>>> data = dp.read(file_name)
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
```
Every object of *SeriesSet* will auto concluses some basic information of the 
dataset (number of miss value, number of records & variable names). For exaples, 
you can browse the dataset of *wine* as:
```Python
>>> wine.info
sheet:data
==========
1.  Structure: DaPy.SeriesSet
2. Dimensions: Ln=178 | Col=16
3. Miss Value: 0 elements
4.   Describe: 
        Title         | Miss | Min | Max | Mean | Std  |Dtype
----------------------+------+-----+-----+------+------+-----
       Alcohol        |  0   | 0.0 | 1.0 | 0.52 | 0.21 | list
      Malic acid      |  0   | 0.0 | 1.0 | 0.32 | 0.22 | list
         Ash          |  0   | 0.0 | 1.0 | 0.54 | 0.15 | list
  Alcalinity of ash   |  0   | 0.0 | 1.0 | 0.46 | 0.17 | list
      Magnesium       |  0   |  0  |  1  | 0.01 | 0.07 | list
    Total phenols     |  0   | 0.0 | 1.0 | 0.45 | 0.22 | list
      Flavanoids      |  0   | 0.0 | 1.0 | 0.36 | 0.21 | list
 Nonflavanoid phenols |  0   | 0.0 | 1.0 | 0.44 | 0.23 | list
   Proanthocyanins    |  0   | 0.0 | 1.0 | 0.37 | 0.18 | list
   Color intensity    |  0   | 0.0 | 1.0 | 0.32 | 0.20 | list
         Hue          |  0   | 0.0 | 1.0 | 0.39 | 0.19 | list
        OD280         |  0   | 0.0 | 1.0 | 0.49 | 0.26 | list
       Proline        |  0   |  0  |  1  | 0.01 | 0.07 | list
       class_1        |  0   |  0  |  1  | 0.33 | 0.47 | list
       class_2        |  0   |  0  |  1  | 0.40 | 0.49 | list
       class_3        |  0   |  0  |  1  | 0.27 | 0.45 | list
=============================================================
```
#### Ⅱ. Preparing data
Before we start a machine learning subject, we should process our 
data so that the data can meet the requirements of the models.   
  
By just accessed our data we found that our dataset is arrangement 
by class. For supporting a balance proportion of the training data, we can 
mass our data with *shuffles()*. In addition, for the reason that 
the dimensional difference between variables is significant, which 
we found in scanning data, we suppose to normalize the data:
```Python
>>> wine.shuffles()
>>> wine.normalized()
```
After disrupting the data, we should separte our data according to the 
target variables and feature variables: 
```Python
>>> feature, target = wine[:'Proline'], wine['class_1':] # contains the target
```
#### Ⅲ. Learning and predicting
In the case of the wine dataset, the task is to predict, given a new record, 
which class it represents. We are given samples of each of the 3 possible classes on 
which we fit an estimator to be able to predict the classes to which unseen samples belong.  
  
In DaPy, an example of an estimator is the class DaPy.MLP that 
implements *mutilayer perceptrons*: 
```Python
>>> mlp = dp.MLP()
>>> mlp.create(input_cell=13, output_cell=3)
 - Create structure: 13 - 12 - 3
```
We call our estimator instance mlp, as it is a multilayer perceptrons. 
It now must be trained to the model, that is, it must learn from the 
known dataset. As a training set, let us use 142 records from our 
dataset apart in 80% of total. We select this training set with the
[:142] Python syntax, which produces a new SeriesSet that contains 
80% records of total:  
```Python
>>> mlp.train(feature[:142], target[:142])
 - Start Training...
 - Initial Error: 150.55 %
    Completed: 10.00 	Remain Time: 1.32 s	Error: 11.82%
    Completed: 20.00 	Remain Time: 1.37 s	Error: 8.37%
    Completed: 29.99 	Remain Time: 1.26 s	Error: 6.64%
    Completed: 39.99 	Remain Time: 1.11 s	Error: 5.59%
    Completed: 49.99 	Remain Time: 0.94 s	Error: 4.88%
    Completed: 59.99 	Remain Time: 0.72 s	Error: 4.36%
    Completed: 69.99 	Remain Time: 0.54 s	Error: 3.96%
    Completed: 79.98 	Remain Time: 0.36 s	Error: 3.65%
    Completed: 89.98 	Remain Time: 0.18 s	Error: 3.39%
    Completed: 99.98 	Remain Time: 0.00 s	Error: 3.18%
 - Total Spent: 2.0 s	Error: 3.1763 %
```
   ![Page Not Found](https://github.com/JacksonWuxs/DaPy/blob/master/doc/material/QuickStartResult.png 'Result of Training')  
  
Now, *mlp* has been trained. It should be attention that the *Error* 
in last line does not means the correct proportion of classfication, 
instead that it means the absolutely error of the target vector.  
  
Let us use our model to classifier the left records in wine dataset, 
which we have not used to train the estimator:
```Python
>>> mlp.test(feature[142:], target[142:])
'Classification Correct: 97.2222%'
```
As you can see, our model has a satisfactory ability in classification. 
#### Ⅳ. Postscript
In order to save time in the next task by using a ready-made model, 
it is possible to save our model in a file:
```Python
>>> mlp.topkl('First_mlp.pkl')
```
In a real working environment, you can quickly use your trained 
model to predict a new record as:
```Python
>>> import DaPy as dp
>>> mlp = machine_learn.MLP()
>>> mlp.readpkl('First_mlp.pkl')
>>> mlp.predict(My_new_data)
```
