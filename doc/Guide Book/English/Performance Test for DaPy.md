### Performance Test for DaPy

#### Date: 2019-10-19

#### Version: 1.10.1

#### Data

* Information

  We use the data which collected the price of gold from China Future Market.  It had 4.32 million rows and 7 columns.

* Download

  You can download the data from here.

#### Standards

- 

- Task 1: load

  Libraries have to load the original data from a CSV format file. In this CSV file, it has different columns with different data types. The libraries must have the ability to automatically predict the best matched data type then transfer the values. We recorded the time consumption of each library spent on the task. The commands we used are listed as bellow.

  ```python
  >>> pandas.readcsv(addr) 
  >>> numpy.genfromtxt(addr, dtype=None, delimiter=',', encoding=None, names=True)
  >>> DaPy.read(addr)
  ```

- Task 2: Traverse

  Libraries have to traverse each row of the data loaded in Task1. We recorded the time consumption of each library spent on the task. The commands we used are listed as bellow.

  ```python
  >>> for row in pd_DataFrame.itertuples():
  		pass
  >>> for row in np_Ndarray:
  		pass
  >>> for row in dp_SeriesSet.iter_rows():
  		pass
  ```

- Task 3: Sort

  Libraries have to sort the records from the data loaded in Task 1 by one column named "Price". We recorded the time consumption of each library spend on the task. The commands we used in this task are listed as bellow.

  ```Python
  >>> pd_DataFrame.sort_values(by='Price')
  >>> np_Ndarray.sort(axis=0, order='Price')
  >>> dp_SeriesSet.sort('Price')
  ```

- Task 4: Query

  Libraries have to select the records that the keyword "Price" is greater than 99999. We recorded the time consumption of each library spent on the task. The commands we used are listed as bellow.

  ```python
  >>> pd_DataFrame.query('Price >= 99999')
  >>> numpy.extract(tuple(_['Price'] > 99999 for _ in np_Ndarray), np_Ndarray)
  >>> dp_SeriesSet.query('Price >= 99999', limit=None)
  ```

- Task 5: Groupby

  Libraries have to separate the records into groups according to the keyword of "Date", than calculate the mean of each column for each subset. Because `numpy.ndarray`  doesn't support the `groupby` operation, Numpy skips this task. We recorded the time consumption of each library spent on the task. The commands we used are listed as bellow. 

  ```python
  >>> pd_DataFrame.groupby('Date')[['Price', 'Volume', 'Token', 'LastToken', 'LastMaxVolume']].mean()
  >>> dp_SeriesSet.groupby('Date', np.mean, apply_col=['Price', 'Volume', 'Token', 'LastToken', 'LastMaxVolume'])
  ```

- Task 6: Save

  Libraries have to save their data into a CSV format file. We recorded the time consumption of each library spent on the task. The commands we used are listed as bellow. 

  ```python
  >>> pd_DataFrame.to_csv('test_Pandas.csv', index=0)
  >>> np.savetxt('test_numpy.csv', np_Ndarray, delimiter=',', fmt='%s%s%s%s%s%s%s')
  >>> dp_SeriesSet.save('test_Numpy.csv')
  ```

  