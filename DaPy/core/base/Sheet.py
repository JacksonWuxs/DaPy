'''
This file is a part of DaPy project

We define three base data structures for operating 
data like an excel. In contrsat with Pandas, it is more 
convinience and more simply to use. 

BaseSheet is a rudimentary structure and it provides some 
functions which have no different between SeriesSet and 
Frame structures.
'''

from collections import Counter
from copy import copy
from datetime import datetime
from itertools import chain, repeat
from operator import eq, ge, gt, le, lt
from re import compile as re_compile

from .BaseSheet import BaseSheet
from .constant import (DUPLICATE_KEEP, PYTHON2, PYTHON3, SHEET_DIM, STR_TYPE,
                       VALUE_TYPE)
from .constant import nan as NaN
from .DapyObject import check_thread_locked
from .IndexArray import SortedIndex
from .Row import Row
from .Series import Series
from .utils import (argsort, auto_plus_one, auto_str2value, count_nan,
                    fast_str2value, hash_sort, is_dict, is_empty, is_iter,
                    is_math, is_seq, is_str, is_value, isnan, range, split,
                    str2date, strip, xrange, zip_longest)
from .utils.utils_join_table import inner_join, left_join, outer_join
from .utils.utils_regression import simple_linear_reg

__all__ = ['SeriesSet']

def reader(file_, queue):
    for row in file_:
        queue.put(strip(row))
    queue.put(False)
            
def analyzer(queue, sep, miss, data, dtypes, miss_symbol, nan):
    while True:
        row = queue.get()
        if row is False:
            break
class SeriesSet(BaseSheet):

    '''Variable stores in sequenes
    '''

    __all__ = [
        'info', 'missing', 'T', 'append_col', 'append_row', 'apply',
        'create_index', 'corr', 'copy', 'count', 'count_nan', 'count_values',
        'describe', 'query', 'drop', 'drop_col', 'drop_row', 'dropna', 
        'drop_duplicates', 'extend', 'fillna', 'from_file', 'get', 
        'get_best_features', 'get_categories', 'get_date_label', 'get_dummies',
        'get_interactions', 'get_ranks', 'get_nan_instrument', 'get_numeric_label',
        'groupby', 'sort', 'show', 'iter_groupby', 'items', 'iter_times', 
        'iter_rows', 'iter_values', 'iloc', 'insert_row', 'insert_col', 'join',
        'keys', 'merge', 'normalized', 'pop', 'pop_row', 'pop_col', 'reverse', 
        'reshape', 'replace', 'update', 'shuffle', 'select', 'sum', 'values'
    ]

    def __init__(self, series=None, columns=None, nan=float('nan')):
        self._data = dict()
        BaseSheet.__init__(self, series, columns, nan)

    @property
    def info(self):
        '''summary the information of sheet'''
        self.describe(level=0)

    @property
    def missing(self):
        '''self.missing -> number of missing values in each column'''
        return SeriesSet([self._missing], self.columns)[0]

    @property
    def T(self):
        '''transpose the current data set -> SeriesSet'''
        return SeriesSet(self.iter_values(), None, self.nan)

    def __eq__(self, other):
        '''Sheet1 == 3 -> Bool in sheet'''
        if is_value(other):
            return self.__compare_value__(other, SeriesSet(nan=self.nan), eq)
        
        if other.shape.Ln != self.shape.Ln:
            return False
        if other.shape.Col != self.shape.Col:
            return False
        if other.columns != self.columns:
            return False
        for lval, rval in zip(other.iter_values(), self.iter_values()):
            if (lval == rval).all() is False:
                return False
        return True

    def __gt__(self, other):
        return self.__compare_value__(other, SeriesSet(nan=self.nan), gt)

    def __ge__(self, other):
        return self.__compare_value__(other, SeriesSet(nan=self.nan), ge)

    def __lt__(self, other):
        return self.__compare_value__(other, SeriesSet(nan=self.nan), lt)

    def __le__(self, other):
        return self.__compare_value__(other, SeriesSet(nan=self.nan), le)
    
    @check_thread_locked
    def append_col(self, series, variable_name=None):
        '''append_col([1, 2, 3], 'New_column') -> None
        Append a new variable named `variable_name` with a list of data
        `series` at the tail of sheet.

        Setting a series of data as a new variable at the sheet. 

        Parameters
        ----------
        series : list 

        variable_name : str or None (default=None)
            the new variable name, if it is None, it will 
            automatically add one.
        
        Returns
        -------
        None

        Examples
        --------
        >>> import DaPy as dp
        >>> sheet = dp.SeriesSet()
        >>> sheet.append_col([1, 2, 3], variable_name='A')
        >>> sheet
        A: <1, 2, 3>
        >>> sheet.append_col([1, 2, 3, 4, 5], variable_name='A')
        >>> sheet
        A: <1, 2, 3, nan, nan>
        A_1: <1, 2, 3, 4, 5>
        >>> sheet.append_col([0, 0])
        >>> sheet
          A: <1, 2, 3, nan, nan>
        A_1: <1, 2, 3, 4, 5>
        C_2: <0, 0, nan, nan, nan>

        Notes
        -----
        1. This function has been added into unit test.
        2. Function won't be locked when sheet.locked is False
        '''
        self._append_col(series, variable_name)
        return self
    
    @check_thread_locked
    def append_row(self, row):
        '''append_row(row=[1, 2, 3]) -> None
            Append a new record `row` at the tail of sheet.

        Using this to append a set of data as a new row in to 
        the tail of current sheet. It will automatically add 
        new columns if the new row is longer than the exist 
        columns. When the length of the new row is smaller 
        than the number of columns, the new row will be automatically 
        addded NaN.

        Parameters
        ----------
        row : dict, namedtuple, list, tuple 
            a new row
        
        Returns
        -------
        None

        Examples
        --------
        >>> import DaPy as dp
        >>> sheet = dp.SeriesSet(columns=['A', 'B', 'C'])
        >>> sheet.append_row([3, 4, 5]) # a list of new row
        >>> sheet.show()
         A | B | C
        ---+---+---
         3 | 4 | 5 
        >>> # a dict of new row which has much more values
        >>> sheet.append_row(dict(A=1, B=2, C=3, D=4)) 
        >>> sheet.show()
         A | B | C |  D 
        ---+---+---+-----
         3 | 4 | 5 | nan 
         1 | 2 | 3 |  4  
        >>> # length is less than the number of columns
        >>> sheet.append_row([9]) 
         A |  B  |  C  |  D 
        ---+-----+-----+-----
         3 |  4  |  5  | nan 
         1 |  2  |  3  |  4  
         9 | nan | nan | nan 

        Notes
        -----
        1. This function has been added into unit test.
        2. Function won't be locked when sheet.locked is False
        '''
        self._append_row(row)
        return self

    def apply(self, func, col=None, axis=0, inplace=False):
        '''apply(func, col=None, inplace=False, axis=0)
           apply a process to column(s) or row(s)

        Parameters
        ----------
        func : callable
            the function that you need to process the data

        col : str, str in list (default='all')
            the columns that you expect to process
        
        inplace : True or False (default=False)
            operate the values in the current sheet or on the copy
        
        Returns
        -------
        applied_sheet : SeriesSet
            if you `inplace` is True, it will return the subset 
            which drop out the values.

        Example
        -------
        >>> sheet = example()
        - read() in 0.000s.
        >>> sheet['B_col']
        Sereis([2,nan,3,3,5, ..., nan,nan,2,9,4])
        >>> power = lambda row: row.B_col ** 2
        >>> sheet.apply(power, axis=0)
        - apply() in 0.000s.
        sheet:sample
        ============
        A_col: <4, nan, 9, 9, 25, ... ,nan, nan, 4, 81, 16>

        Notes
        -----
        1. Function may be locked when `inplace` is True and 
           sheet.locked is False. When you operate the 
           column which is an Index, it will be locked.
        '''
        assert inplace in (True, False), '`inplace` must be True or False'
        assert axis in (0, 1), '`axis` must be 0 or 1'
        assert callable(func), '`func` parameter should be a callable object'
        cols = self._check_columns_index(col)

        if inplace is False:
            if axis == 0:
                return SeriesSet(map(func, self[cols]), nan=self.nan)
            ret = SeriesSet(columns=self.columns, nan=self.nan)
            row = [func(val.tolist()) if key in cols else self.nan for key, val in self.items()]
            ret.append_row(row)
            return ret[cols]
        return self._apply_inplace(func, cols, axis)

    def create_index(self, columns):
        '''create_index(column) -> None
        set a column as an Index for quickly searching

        Create an Index for quickly searching the records. 
        When you have settled down some columns as Indexes, 
        sheet.query() or sheet.update() will automatically 
        check your Indexes. If you have the Index, it will 
        use the Index to quickly select the records. When 
        you select records with Index, the time cumsulting 
        is just O(logN). 

        You must be attention that some of operation  
        functions will be unavaliable after you have 
        Indexes. And the others will be a little slower. 
        Use `sheet.locked` to make sure whether the 
        sheet is locked or not.

        Parameters
        ----------
        columns : str, str in list
            the column(s) you want to make index
        
        Returns
        -------
        None

        Examples
        --------


        See Also
        --------
        DaPy.SeriesSet.query
        DaPy.SeriesSet.update
        '''
        for col in self._check_columns_index(columns):
            error = '%s has been an index already' % col
            assert col not in self._sorted_index, error
            self._sorted_index[col] = SortedIndex(self._data[col])

    def corr(self, method='pearson', col=None):
        '''corr(method='pearson', col=None) -> SeriesSet
        calculate the correlation among the variables

        Calculate the correlation between the variables and 
        return the result as a n*n matrix.

        Parameters
        ----------
        method : str (default='pearson')
            the algorithm to calculate the correlation
            ("pearson", "spearman" and 'kendall' are supported)

        col : str or None (default=None)
            the columns to calculate
        
        Returns
        -------
        correlations : SeriesSet

        Examples
        --------
        >>> import DaPy as dp
        >>> sheet = dp.SeriesSet({
            'A_col': [1, 3, 4, 5, 6, 4, 5, 6, 8],
            'B_col': [2, 4, 6, 8, 10, 12, 13, 15, 16],
            'C_col': [-2, -3, -4, -5, -4, -7, -8, -10, -11]})
        >>> sheet.corr(col=['A', 'B', 'C']).show()
           |        A        |       B        |        C       
        ---+-----------------+----------------+-----------------
         A |       1.0       | 0.862775883665 | -0.790569415042 
         B |  0.862775883665 |      1.0       |  -0.95155789511 
         C | -0.790569415042 | -0.95155789511 |       1.0       
        '''
        from DaPy import corr as corr_
        col = self._check_columns_index(col)
        frame = [[1.0] * len(col) for i in xrange(len(col))]
        for i, current in enumerate(col):
            for j, next_ in enumerate(col):
                if next_ != current:
                    coef = corr_(self._data[current],
                                 self._data[next_],
                                 method)
                    frame[i][j] = coef
                    frame[j][i] = coef
        new_ = SeriesSet(frame, col, nan='')
        new_._insert_col(0, col, '')
        return new_

    def copy(self):
        '''copy the current sheet'''
        return SeriesSet(self, nan=self._nan)

    def count(self, value, col=None, row=None):
        '''count(X, col=None, row=None) -> Counter or int
        count the frequency of value(s) in a specific area

        Count the frequency of values appearence in a specific 
        area. You should identify an area by columns and rows 
        indexes. Otherwise, it will check the whole sheet.

        Parameters
        ----------
        val : value
            anything you want to count in the area

        col : None, str, str in list (default=None)
            columns you want to count with
            None -> all columns will be checked
        
        row : None, int, int in list (default=None)
            rows you want to count with
            None -> all rows will be checked
        
        Returns
        -------
        numbers : int / Counter object
            If you just have one value, it will return int.
            If you have more than one values, it will return a dict.
        
        Examples
        --------
        >>> sheet = dp.SeriesSet([[1, 2,    3,    4],
                                  [2, None, 3,    4],
                                  [3, 3,    None, 5],
                                  [7, 8,    9,    10]])
        >>> sheet.show()
         C_0 | C_1  | C_2  | C_3
        -----+------+------+-----
          1  |  2   |  3   |  4  
          2  | None |  3   |  4  
          3  |  3   | None |  5  
          7  |  8   |  9   |  10 
        >>> sheet.count(3) 
        4   # 3 totally appears four times in the sheet

        >>> sheet.count([3, None]) # None totally appears two times
        Counter({3: 4, None: 2})

        >>> sheet.count(3, col=0) 
        1   # 3 totally appears 1 time in the first column

        >>> sheet.count(3, col=0, row=[0, 1])
        0   # 3 never appears in the first column and the first two rows

        Notes
        -----
        1. Function won't be locked when sheet.locked is False.
        '''
        if is_value(value):
            value = (value,)

        assert is_seq(value), 'value must be stored in an iterable'
        value = set(value)
        col = self._check_columns_index(col)
        row = self._check_rows_index(row)

        counter = Counter()
        for title in col:
            sequence = self._data[title]
            for val in sequence[row]:
                if val in value:
                    counter[val] += 1

        if len(value) == 1:
            return counter[tuple(value)[0]]
        return counter
    
    def count_nan(self, axis=0):
        '''return the frequency of NaN according to `axis`'''
        assert axis in (0, 1, None)
        if axis == 1:
            return self.missing
        if axis == 0:
            return Series(sum(map(self._isnan, _)) for _ in self.iter_rows())     
        return sum(self._missing)

    def count_values(self, col=None):
        '''count_values(col=None) -> Counter

        Count the frequency of values for each variable(s).
        You could count only a part of your data set with 
        setting `col` as an iterble inluding the number 
        of column or variable names.

        Parameters
        ----------
        col : None, str or str in list (default=None)
            the column you expected to analysis
            None -> all columns
        
        Returns
        -------
        dict : the frequency of elements in each column.

        Examples
        --------
        >>> import DaPy as dp
        >>> sheet = dp.SeriesSet([[1, 2, 3, 4],
                                  [2, None, 3, 4],
                                  [3, 3, None, 5],
                                  [7, 8, 9, 10]])
        >>> sheet.count_values(col=1)
        Counter({8: 1, 
                 2: 1, 
                 3: 1, 
                 None: 1})
        >>> sheet.count_values(col=[1, 2])
        Counter({3: 3, 
                 None: 2, 
                 8: 1, 
                 9: 1, 
                 2: 1})

        Notes
        -----
        1. Function won't be locked when sheet.locked is False.
        '''
        col = self._check_columns_index(col)
        counter = Counter()
        for title in col:
            counter.update(self._data[title])
        return counter

    def describe(self, level=0):
        '''describe(lvel=0, show=True) -> None
        summary the information of current sheet
        '''
        assert level in (0, 1, 2)
        from DaPy import describe
        info = dict(mins=[], maxs=[], avgs=[], stds=[], skes=[], mode=[],
                    kurs=[], miss=list(map(str, self._missing)))
        for sequence in self.iter_values():
            des = describe(Series(filter(lambda x: not self._isnan(x), sequence)))
            for arg, val in zip(['mins', 'maxs', 'avgs', 'stds', 'mode', 'skes', 'kurs', ],
                                [des.Min, des.Max, des.Mean, des.S, des.Mode, des.Skew, des.Kurt]):
                if val is None and arg != 'mode':
                    info[arg].append('-')
                elif isinstance(val, float):
                    if val > 9999999999:
                        float_template = '%g'
                    else:
                        float_template = '%10.' + str(11 - len(str(int(val)))) + 'g'
                    info[arg].append(float_template % val)
                else:
                    info[arg].append(str(val))

        blank_size = [max(len(max(self.columns, key=len)), 5) + 2,
                      max(len(max(info['miss'], key=len)), 4) + 2,
                      max(len(max(info['mins'], key=len)), 4) + 2,
                      max(len(max(info['avgs'], key=len)), 4) + 2,
                      max(len(max(info['maxs'], key=len)), 4) + 2,
                      max(len(max(info['stds'], key=len)), 4) + 2,
                      max(len(max(info['mode'], key=len)), 4) + 2]
        
        # Draw the title line of description
        message = '|'.join(['Title'.center(blank_size[0]),
                            'Miss'.center(blank_size[1]),
                            'Min'.center(blank_size[2]),
                            'Mean'.center(blank_size[4]),
                            'Max'.center(blank_size[3]),
                            'Std'.center(blank_size[5]),
                            'Mode'.center(blank_size[6])])

        if level >= 1:
            blank_size.extend([
                max(len(max(info['skes'], key=len)), 4) + 2,
                max(len(max(info['kurs'], key=len)), 4) + 1])

            message += '|' + '|'.join([
                'Skew'.center(blank_size[6]),
                'Kurt'.center(blank_size[7])])

        message += '\n%s\n' % '+'.join(map(lambda x: '-' * x, blank_size))

        # Draw the main table of description
        for i, title in enumerate(self._columns):
            message += title.center(blank_size[0]) + '|'
            message += info['miss'][i].center(blank_size[1]) + '|'
            message += info['mins'][i].rjust(blank_size[2] - 1) + ' |'
            message += info['avgs'][i].rjust(blank_size[4] - 1) + ' |'
            message += info['maxs'][i].rjust(blank_size[3] - 1) + ' |'
            message += info['stds'][i].rjust(blank_size[5] - 1) + ' |'
            message += info['mode'][i].rjust(blank_size[6] - 1)
            if level >= 1:
                message += '|' + info['skes'][i].center(blank_size[7]) + '|'
                message += info['kurs'][i].center(blank_size[8])
            message += '\n'

        lenth = 6 + 2 * level + sum(blank_size)
        print('1.  Structure: DaPy.SeriesSet\n' +\
              '2. Dimensions: Lines=%d | Variables=%d\n' % self.shape +\
              '3. Miss Value: %d elements\n' % sum(self.missing) +\
              'Descriptive Statistics'.center(lenth) + '\n' +\
               '=' * lenth + '\n' + message + '=' * lenth)

    def query(self, expression, col=None, limit=1000):
        '''sheet.query('A_col != 1') -> SeriesSet

        Parse a string of python syntax statement and select rows which
        match the query. Two algorithms are used in this function. The first
        solution, binary select, needed sorted indexes before calling this
        function, has a high efficiency with O(logN) time comsumption. On the 
        other hand, normal linear comparing select, implemented like `where`
        function, has a linear efficiency of O(N) speed.

        Parameters
        ----------
        expression : str
            the statement you want to use to select data,
            you can write it like python condition syntax.

        col : None, str or list (default=None)
            which columns you want to select

        limit : int, None (default=1000)
            the maximum number of rows you want to select,
            this is a good way to speed up selection from
            million of rows if you need only 1 of them
            in each time.

        Return
        ------
        subset : SeriesSet
            the selection result according to your statement.

        Example
        -------
        >>> from DaPy.datasets import iris
        >>> sheet, info = iris()
        >>> sheet.query('5.5 >= sepal length > 5 and sepal width > 4').show()
         sepal length | sepal width | petal length | petal width | class
        --------------+-------------+--------------+-------------+--------
             5.2      |     4.1     |     1.5      |     0.1     | setosa
             5.5      |     4.2     |     1.4      |     0.2     | setosa
        >>> data.query('sepal length / 2.0 == sepal width').show()
         sepal length | sepal width | petal length | petal width |   class
        --------------+-------------+--------------+-------------+------------
             6.4      |     3.2     |     4.5      |     1.5     | versicolor
             7.2      |     3.6     |     6.1      |     2.5     | virginica
             6.4      |     3.2     |     5.3      |     2.3     | virginica
             5.6      |     2.8     |     4.9      |     2.0     | virginica
             6.0      |     3.0     |     4.8      |     1.8     | virginica

        See Also
        --------
        DaPy.core.base.Sheet.SeriesSet.create_index
        DaPy.core.base.IndexArray.SortedIndex
        '''
        sub_index, select_col = self._query(expression, col, limit)
        if len(sub_index) == 0:
            return SeriesSet(None, select_col, nan=self.nan)
        return self.iloc(sub_index)[select_col]

    def drop(self, index=-1, axis=0, inplace=False):
        '''remove a column or a row in the sheet

        Parameters
        ----------
        index : single value or list-like (default=-1)
            Index of column name or index of column or index of rows

        axis : 0 or 1
            drop index along the row (axis=0) or column (axis=1)
        
        inplace : True or False (default=False)
            operate the values in the current sheet or on the copy

        Return
        ------
        subset : SeriesSet
            if you `inplace` is True, it will return the subset 
            which drop out the values.

        Example
        -------
        >>> sheet = dp.SeriesSet(range(5))
        >>> sheet.drop(0, axis=0, True)
        C_0 : <1, 2, 3, 4>
        >>> sheet.drop('C_0', axis=1, True)
        empty SeriesSet instant

        Notes
        -----
        1. This function has been added into unit test.
        '''
        assert axis in (0, 1)
        if axis == 0:
            return self.drop_row(index, inplace)
        return self.drop_col(index, inplace)

    @check_thread_locked
    def drop_col(self, index=-1, inplace=False):
        '''drop_col(index=-1, inplace=False) -> SeriesSet
        remove the columns according to the index

        Parameters
        ----------
        index : int, str or item in list (default=-1)
            the columns will be removed
        
        inplace : True / False (default=False)
            remove the columns from the current sheet or 
            remove the columns from a copy of the sheet and 
            return the copy
        
        Returns
        -------
        dropped_sheet : SeriesSet
            if `inplace` is True, it will return the current sheet, 
            otherwise, it will return the copy of the sheet.
        
        Examples
        --------
        '''
        if inplace is False:
            return SeriesSet(self, nan=self.nan)._drop_col(index)
        return self._drop_col(index)

    @check_thread_locked
    def drop_row(self, index=-1, inplace=True):
        '''drop_row(index=-1, inplace=True) -> SeriesSet
        drop out rows according to the index'''
        if inplace is False:
            return SeriesSet(self, nan=self.nan)._drop_row(index)
        return self._drop_row(index)

    def dropna(self, axis=0, how='any', inplace=False):
        '''dropna(axis=0, how='any', inplace=False)
        
        Drop out all records, which contain miss value, if `axis` is
        `0`. Drop out all the variables, which contain miss value,
        if `axis` is `1`.

        Parameters
        ----------
        axis : 0 or 1 (default=0)
            drop out process along axis
            0 -> rows: any rows which contain NaN will be droped out.
            1 -> cols: any columns which contain NaN will be droped out.
        
        how : 'any', 'all' or float (default='any')
            how to delete the records or columns when they appear NaN.
            'all' -> each elements of the row or column are NaN, than 
                     we will remove the row or column.
            'any' -> If there is at least one NaN in that row or column, 
                     we will remove the row or column.
            float -> if the percentage of NaN in that row or column 
                     is greater than the float, we will remove the row 
                     or column.
        
        inplace : True or False (default=False)
            opearte on the current sheet or the copy of the sheet
        
        Returns
        -------
        dropped_sheet : SeriesSet   
            if `inplace` is True, it will return the current sheet, 
            otherwise, it will return the copy of the sheet.

        Examples
        --------
        >>> import DaPy as dp
        >>> sheet = dp.SeriesSet([[1, 2, 3, 4],
                                  [2, None, None, 4],
                                  [3, 3, None, 5],
                                  [7, 8, 9, 10]], 
                                  nan=None)
        >>> sheet.dropna(axis=0, how='any').show()
         C_0 | C_1 | C_2 | C_3
        -----+-----+-----+-----
          1  |  2  |  3  |  4  
          7  |  8  |  9  |  10 
        >>> sheet.dropna(axis=0, how=0.4).show()
         C_0 | C_1 | C_2  | C_3
        -----+-----+------+-----
          1  |  2  |  3   |  4  
          3  |  3  | None |  5  
          7  |  8  |  9   |  10 
        >>> sheet.dropna(axis=1, how='any').show()
         C_0 | C_3
        -----+-----
          1  |  4  
          2  |  4  
          3  |  5  
          7  |  10 
        >>> sheet.dropna(axis=1, how=0.4).show()
         C_0 | C_1  | C_3
        -----+------+-----
          1  |  2   |  4  
          2  | None |  4  
          3  |  3   |  5  
          7  |  8   |  10 
        '''
        assert axis in (0, 1), 'axis must be 1 or 0.'
        err = '`how` must be "any", "all" or float between 0 and 1'
        assert how in ('any', 'all') or 1 > how > 0, err

        if axis == 1:
            pop_ind, lenth = [], float(self.shape.Ln)
            for i, value in enumerate(self._missing):
                if how == 'any' and value > 0:
                    pop_ind.append(self._columns[i])
                elif value / lenth > how:
                    pop_ind.append(self._columns[i])
                elif how == 'all' and value == lenth:
                    pop_ind.append(self._columns[i])
            return self.drop_col(pop_ind, inplace)

        # if axis == 0:
        pop_ind, lenth = [], float(self.shape.Col)
        for i, row in enumerate(self.iter_rows()):
            num = count_nan(self._isnan, row)
            if how == 'any':
                if num > 0:
                    pop_ind.append(i)
            elif num / lenth > how:
                pop_ind.append(i)
            elif how == 'all':
                if num == lenth:
                    pop_ind.append(i)
        return self.drop_row(pop_ind, inplace)

    def drop_duplicates(self, col=None, keep='first', inplace=False):
        '''drop_duplicates(col=None, keep='first', inplace=False) -> SeriesSet
        '''
        assert self.locked, LOCK_ERROR
        assert keep in ('first', 'last', False)
        pop_col = self._check_columns_index(col)
        drop_ind, drop_symbol = [], DUPLICATE_KEEP[keep]

        droped_table = self._group_index_by_column_value(pop_col)  # O(n)
        for values in droped_table.values():  # O(n)
            if len(values) != 1:
                drop_ind.extend(values[drop_symbol])
        return self.drop_row(drop_ind, inplace)  # O(k*lnk + n)

    @check_thread_locked
    def extend(self, item, inplace=False):
        '''extend the current SeriesSet with records in set.


        Examples
        --------
        >>> import DaPy as dp
        >>> data1 = dp.SeriesSet(
                        [[11, 11],
                        [21, 21],
                        [31, 31],
                        [41, 41]],
                        ['C1', 'C2']), 'Table1')
        >>> data2 = dp.SeriesSet(
                        [[21, 21],
                        [22, 22],
                        [23, 23],
                        [24, 24]],
                        ['C2', 'C3']), 'Table2')
        >>> data1.extend(data2)
         C1  | C2 |  C3 
        -----+----+------
         11  | 11 | nan 
         21  | 21 | nan 
         31  | 31 | nan 
         41  | 41 | nan 
         nan | 21 |  21  
         nan | 22 |  22  
         nan | 23 |  23  
         nan | 24 |  24  

        Notes
        -----
        1. This function has been added into unit test.
        '''
        if isinstance(item, SeriesSet) is False:
            try:
                columns = item.columns if hasattr(item, 'columns') else self.columns
                item = self._extend(SeriesSet(item, columns))
            except:
                raise TypeError('could not extend a single value only.')

        if inplace is False:
            return SeriesSet(self)._extend(item)
        return self._extend(item)

    @check_thread_locked
    def fillna(self, fill_with=None, col=None, method=None, limit=None):
        '''fill nan in the dataset

        Parameters
        ----------
        fill_with : value, dict in valu (default=None)
            the value used to fill with

        cols : str or str in list (default=None)
            the columns would be operated, None means the whole dataset

        method  : str (default=None)
            which method you expect to use, if this keyword is not None,
            `fill_with` keyword will be failed to use. The data which use to
            establish the model are located around the missing value and the
            number of those data are auto-adapt.

            `linear` : establish a linear model

        limit : int (default=None)
            the maximum number of missing values to fill with, operate all
            missing value use None.

        Return
        ------
        self

        Example
        -------
        >>> data = dp.SeriesSet({'A': [dp.nan, 1, 2, 3, dp.nan, dp.nan,  6]},
                                nan=dp.nan)
        >>> data.fillna(method='linear')
        >>> data
        A: <0.0, 1, 2, 3, 4.0, 5.0, 6>
        '''
        return self._fillna(fill_with, col, method, limit)
        
    @classmethod
    def from_file(cls, addr, **kwrd):
        '''read dataset from .txt or .csv file.

        Parameters
        ----------
        addr : str
            address of source file.

        first_line : int (default=1)
            the first line with data.

        miss_symbol : str or str in list (default=['?', '??', '', ' ', 'NA', 'None'])
            the miss value symbol in csv file.

        nan : str (default=float('nan'))
            the symbol of missing value in current sheet

        title_line : int (default=0)
            the line with title, rules design as follow:
            -1 -> there is no title inside;
            >=0 -> the titleline.

        sep : str (default=",")
            the delimiter symbol inside.

        dtypes : Type name or dict of columns (default=None):
            use specific data types for parsing each column
            int -> transfer any possible values into int
            float -> transfer any possible values into float
            str -> keep all values in str type
            datetime -> transfer any possible values into datetime-object
            bool -> transfer any possible values into bool
        '''
        nan = kwrd.get('nan', NaN)
        sheet = cls(nan=nan)
        first_line = kwrd.get('first_line', 1)
        title_line = kwrd.get('title_line', first_line - 1)
        columns = list(kwrd.get('columns', []))
        sep = kwrd.get('sep', ',')

        dtypes = kwrd.get('dtypes', [])
        if is_str(dtypes):
            dtypes = [dtypes]

        miss_symbol = kwrd.get('miss_symbol', set(['nan', '?', '??', '', ' ', 'NA', 'None']))
        if is_value(miss_symbol):
            miss_symbol = [miss_symbol]
        if isinstance(miss_symbol, set) is False:
            miss_symbol = set(miss_symbol)

        miss_symbol.add(None)
        split, strip = str.split, str.strip
        if kwrd.get('careful_cut', None):
            pattern = re_compile(sep + '(?=(?:[^"]*"[^"]*")*[^"]*$)')
            split = pattern.split

        param, data, miss = {'mode': 'rU'}, (), ()
        if PYTHON3:
            param['encoding'] = kwrd.get('encoding', None)
            param['file'] = addr
            param['newline'] = kwrd.get('newline', None)
        if PYTHON2:
            param['name'] = addr

        assert first_line > title_line, '`first_line` must be larger than `title_line`'
        assert all(map(is_str, columns)), 'column names must be `'
        
        with open(**param) as file_:
            # skip the rows which are unexpected to read
            for i in xrange(first_line):
                line = file_.readline()
                if i == title_line:
                    # setup the title line
                    columns = tuple(map(strip, split(line, sep)))

            # begin to load data
            for row in file_:
                for mis, seq, transfer, val in zip_longest(miss, data, dtypes, split(strip(row), sep)):  
                    # iter value
                    try:
                        if val in miss_symbol:
                            seq.append(nan)
                            mis.append(1)
                        else:
                            seq.append(transfer(val))
                            
                    except ValueError:# different types of data in the same variable
                        seq.append(auto_str2value(val))
                        
                    except Exception: # we found a new variable
                        mis = []
                        miss += (mis,)
                        if val in miss_symbol:
                            val = nan
                            mis.append(1)
                        else:
                            val = auto_str2value(val, transfer)
                            type_name = str(val.__class__).split()[1][1:-2].split('.')[0]
                            dtypes.append(fast_str2value[type_name])
                            
                        if not data:
                            data += (Series([val]),)
                        else:
                            missed = len(data[0]) - 1
                            mis.append(missed)
                            data += (Series(chain(repeat(nan, missed), [val])),)

        sheet._dim = SHEET_DIM(len(data[0]), len(data))
        sheet._init_col_name(columns)
        for i, (missing, seq, col) in enumerate(zip(miss, data, sheet.columns)):
            add_space = sheet.shape.Ln - len(seq)
            seq.extend(repeat(sheet.nan, add_space))
            sheet._missing.append(add_space + sum(missing))
            sheet._data[col] = seq
        return sheet

    def get(self, key, default=None):
        '''get(key, default=None) -> row or Series
           select column or row from the sheet, 
           return `default` if `key` is not a column name
        '''
        return self._get(key, default)

    def get_best_features(self, method='variance', X=None, Y=None, top_k=1):
        '''get_best_features(method='variance', X=None, Y=None, top_k=1) -> SeriesSet

        Select K features which are the most important to the variable `Y`
        '''
        return self._get_best_features(method, X, Y, top_k)

    def get_categories(self, cols, cut_points, group_name,
                       boundary=(False, True), inplace=False):
        '''transfer numerical variables into categorical variable'''
        if inplace is False:
            return SeriesSet(self, nan=self.nan)._get_categories(
                cols, cut_points, group_name, boundary)
        return self._get_categories(cols, cut_points, group_name, boundary)

    def get_date_label(self, cols, daytime=True,
                       weekend=True, season=True, inplace=False):
        '''transfer a datetime object into categorical variables'''
        cols = self._check_columns_index(cols)
        if inplace is False:
            self = SeriesSet(nan=self.nan)

        def dummy_date(col_name):
            sequence = copy(self.data[col_name])
            for i, value in enumerate(sequence):
                if isinstance(value, datetime) is False:
                    sequence[i] = str2date(str(value))
            date_sheet = SeriesSet(None, ['month', 'hour', 'week'])
            for row in sequence:
                date_sheet.append_row([row.month, row.hour, row.weekday()])
            return date_sheet

        for col in cols:
            date = dummy_date(col)
            self._get_date_label(date, col, daytime, weekend, season)
        return self

    def get_dummies(self, cols=None, value=1, inplace=False):
        '''Convert categorical variable into multiple binary variables

        Parameters
        ----------
        cols : str or str in list (default=None)
            the columns would be operated, None means the whole dataset

        value : value-type (default=1)
            the value which will be used as a mark in the return object

        inplace : True or False (default=False)
            operate the values in the current sheet or on the copy

        Examples
        --------
        >>> import DaPy as dp
        >>> sheet = dp.SeriesSet([
                    ['A', 2],
                    ['B', 3],
                    ['A', 3],
                    ['C', 1],
                    ['D', 4],
                    ['C', 1]],
		            ['alpha', 'num'])
        >>> sheet.get_dummies(cols='alpha').show()
         alpha | num | alpha_A | alpha_C | alpha_B | alpha_D
        -------+-----+---------+---------+---------+---------
           A   |  2  |    1    |    0    |    0    |    0    
           B   |  3  |    0    |    0    |    1    |    0    
           A   |  3  |    1    |    0    |    0    |    0    
           C   |  1  |    0    |    1    |    0    |    0    
           D   |  4  |    0    |    0    |    0    |    1    
           C   |  1  |    0    |    1    |    0    |    0    
         '''
        
        if inplace is False:
            return SeriesSet(self, nan=self.nan)._get_dummies(cols, value)
        return self._get_dummies(cols, value)

    def get_interactions(self, n_power=2, cols=None, inplace=False):
        '''get_interactions(n_var=3, cols=None, inplace=False) -> SeriesSet
            create new variables by multipling each other

        Getting interactions of variables is a common operation
        in Feature Engineering. This function will help you
        achieve it easily.

        Parameters
        ----------
        n_power : int (default=2)
            the number of features to create interactions.
            For example, you have variables A and B. You want
            to create new variable with A * A * B, you should
            set n_var=3.

        cols : str, str in list (default=None)
            the features to create interactions.

        inplace : True / False (default=False)
            save the new_features in current sheet or not

        Returns
        -------
        new_features : SeriesSet

        Examples
        --------
        >>> sheet = SeriesSet({'A': [1, 1, 1, float('nan')],
                               'B': [2, 2, 2, 2],
                               'C': [3, 3, 3, 3]})
        >>> sheet.get_interactions(2).show()
         B*C | A^2 | B^2 | C^2 | A*B | A*C
        -----+-----+-----+-----+-----+-----
          6  |  1  |  4  |  9  |  2  |  3  
          6  |  1  |  4  |  9  |  2  |  3  
          6  |  1  |  4  |  9  |  2  |  3  
          6  | nan |  4  |  9  | nan | nan
        >>> sheet = SeriesSet({'A': [1, 1, 1, 1],
                               'B': [2, 2, 2, 2],})
        >>> sheet.get_interactions(3).show()
         B^3 | A*B^2 | A^3 | A^2*B
        -----+-------+-----+-------
          8  |   4   |  1  |   2   
          8  |   4   |  1  |   2   
          8  |   4   |  1  |   2   
          8  |   4   |  1  |   2   
        '''
        new_features = self._get_interactions(SeriesSet(nan=self.nan))
        if inplace is False:
            return new_features
        return self.join(new_features, inplace=True)

    def get_ranks(self, cols=None, duplicate='mean', inplace=False):
        '''get_ranks(cols=None, duplicate='mean', inplace=False) -> SeriesSet
        get the ranks of each row in each column

        Parameters
        ----------
        cols : str, str in list (default=None)

        duplicate : 'mean', 'first' (default='mean')
            how to rank the records which obtain the same value

        inplace : True / False (default=False)
            restore the ranks in a new SeriesSet or not

        Returns
        -------
        ranks : SeriesSet

        Examples
        --------
        >>> sheet = SeriesSet({'A': [2, 2, 1, 1],
                               'B': [2, 7, 5, 2],})
        >>> sheet.get_ranks(inplace=True).show()
         A | B | A_rank | B_rank
        ---+---+--------+--------
         2 | 2 |  3.5   |  1.5   
         2 | 7 |  3.5   |   4    
         1 | 5 |  1.5   |   3    
         1 | 2 |  1.5   |  1.5   
        '''
        ranks = self._get_ranks(SeriesSet(nan=self.nan), cols, duplicate)
        if inplace is False:
            return ranks
        return self.join(ranks, inplace=True)

    def get_nan_instrument(self, cols=None, inplace=False):
        '''create instrument variable for determining whether a variable is miss or not'''
        instruments = self._get_nan_instrument(SeriesSet(nan=self.nan), cols)
        if inplace is False:
            return instruments
        return self.join(instruments, inplace=True)

    def get_numeric_label(self, cols=None, inplace=False):
        '''encode string values into numerical values'''
        to_return = self if inplace is True else SeriesSet(nan=self.nan)
        return self._get_numeric_label(to_return, cols)           

    def groupby(self, keys, func=None, apply_col=None):
        '''groupby(keys, func=None, apply_col=None)

        It will return the result of function of each groupby object when 
        you pass a callable object in `func`. Otherwise, it will return 
        each groupby subsheet in a dict.

        Parameters
        ----------
        keys : str, str in list
            columns that will be seem as category variable

        func : None or function (default=None)
            map this function to each group by subsheet

        apply_col : str, str in list (default=None)
            The columns will be used by the function, 
            default means all columns will be used.
        
        Returns
        -------
        groupby : SeriesSet or dict
            if `func` is not None, it will be SeriesSet.
        
        Examples
        --------
        >>> from DaPy.datasets import iris
        >>> from DaPy import sum
        >>> sheet = iris()[0]
         - read() in 0.001s.
        >>> sheet.groupby('class').show(3)
        - groupby() in 0.000s.
        sheet:('virginica',)
        ====================
         sepal length | sepal width | petal length | petal width |   class  
        --------------+-------------+--------------+-------------+-----------
             6.3      |     3.3     |     6.0      |     2.5     | virginica 
             5.8      |     2.7     |     5.1      |     1.9     | virginica 
             7.1      |     3.0     |     5.9      |     2.1     | virginica 
                                 .. Omit 44 Ln ..                           
             6.5      |     3.0     |     5.2      |     2.0     | virginica 
             6.2      |     3.4     |     5.4      |     2.3     | virginica 
             5.9      |     3.0     |     5.1      |     1.8     | virginica 

        sheet:('setosa',)
        =================
         sepal length | sepal width | petal length | petal width | class 
        --------------+-------------+--------------+-------------+--------
             5.1      |     3.5     |     1.4      |     0.2     | setosa 
             4.9      |     3.0     |     1.4      |     0.2     | setosa 
             4.7      |     3.2     |     1.3      |     0.2     | setosa 
                                 .. Omit 44 Ln ..                         
             4.6      |     3.2     |     1.4      |     0.2     | setosa 
             5.3      |     3.7     |     1.5      |     0.2     | setosa 
             5.0      |     3.3     |     1.4      |     0.2     | setosa 

        sheet:('versicolor',)
        =====================
         sepal length | sepal width | petal length | petal width |   class   
        --------------+-------------+--------------+-------------+------------
             7.0      |     3.2     |     4.7      |     1.4     | versicolor 
             6.4      |     3.2     |     4.5      |     1.5     | versicolor 
             6.9      |     3.1     |     4.9      |     1.5     | versicolor 
                                     .. Omit 44 Ln ..                           
             6.2      |     2.9     |     4.3      |     1.3     | versicolor 
             5.1      |     2.5     |     3.0      |     1.1     | versicolor 
             5.7      |     2.8     |     4.1      |     1.3     | versicolor 
        >>> sheet.groupby('class', dp.mean, apply_col='petal length').show()
         - groupby() in 0.001s.
        sheet:data
        ==========
         petal length |   class   
        --------------+------------
            5.552     | virginica  
            1.464     |   setosa   
             4.26     | versicolor 
        '''
        if func is not None:
            result = tuple(self._iter_groupby(keys, func, apply_col))
            return SeriesSet(result, result[0].columns)

        result = {}
        for key, subset in self._iter_groupby(keys, func, None):
            result[key] = subset
        return result

    def sort(self, *orderby):
        '''sort(('A_col', 'DESC'), ('B_col', 'ASC')) --> Return sorted sheet

        You will be asked to offer at least one ordering conditions.
        The parameter should be like a tuple or a list with two elements,
        on behalf of the key value and arrangement condition (key, arrangement).
        e.g. ('D_col', 'ASC') means that ascending ordered the records
        with D_col.

        Parameters
        ----------
        orderby : tuple or str
            A pair of string to represent the orderby keywords,
            the tuple is kin to tuple(column_name, order).
            `order` must be 'DESC' or 'ASC'.
        
        Returns
        -------
        ordered_sheet : SeriesSet

        Examples
        --------
        >>> from DaPy import datasets
        >>> sheet = datasets.example()
        >>> sheet.sort(('B_col', 'DESC'), ('C_col', 'ASC')).show()
         - sort() in 0.000s.
        sheet:sample
        ============
                A_col        | B_col | C_col | D_col
        ---------------------+-------+-------+-------
         2017-02-11 00:00:00 |   9   |  2.2  | False 
         2017-02-05 00:00:00 |   5   |  1.6  | False 
         2017-02-07 00:00:00 |   4   |  1.8  | False 
         2017-02-12 00:00:00 |   4   |  2.3  | False 
         2017-02-03 00:00:00 |   3   |  1.4  |  True 
         2017-02-04 00:00:00 |   3   |  1.5  |  True 
         2017-02-10 00:00:00 |   2   |  nan  | False 
         2017-02-01 00:00:00 |   2   |  1.2  |  True 
         2017-02-06 00:00:00 |   1   |  1.7  | False 
         2017-02-09 00:00:00 |  nan  |  nan  | False 
         2017-02-02 00:00:00 |  nan  |  1.3  |  True 
         2017-02-08 00:00:00 |  nan  |  1.9  | False 
        '''
        return self._sort(SeriesSet(nan=self.nan), *orderby)

    def show(self, lines=None):
        '''show(lines=None) -> None
        
        Parameters
        ----------
        lines : None, int (default=None)
            number of rows you want to show
        
        Returns
        -------
        str_frame : a string shows the table data

        Examples
        --------
        >>> from DaPy.datasets import example
        >>> sheet = example()
        sheet:sample
        ============
                A_col        | B_col | C_col | D_col
        ---------------------+-------+-------+-------
         2017-02-01 00:00:00 |   2   |  1.2  |  True 
         2017-02-02 00:00:00 |  nan  |  1.3  |  True 
         2017-02-03 00:00:00 |   3   |  1.4  |  True 
         2017-02-04 00:00:00 |   3   |  1.5  |  True 
         2017-02-05 00:00:00 |   5   |  1.6  | False 
         2017-02-06 00:00:00 |   1   |  1.7  | False 
         2017-02-07 00:00:00 |   4   |  1.8  | False 
         2017-02-08 00:00:00 |  nan  |  1.9  | False 
         2017-02-09 00:00:00 |  nan  |  nan  | False 
         2017-02-10 00:00:00 |   2   |  nan  | False 
         2017-02-11 00:00:00 |   9   |  2.2  | False 
         2017-02-12 00:00:00 |   4   |  2.3  | False 
        >>> sheet.show(3)
        sheet:sample
        ============
                A_col        | B_col | C_col | D_col
        ---------------------+-------+-------+-------
         2017-02-01 00:00:00 |   2   |  1.2  |  True 
         2017-02-02 00:00:00 |  nan  |  1.3  |  True 
         2017-02-03 00:00:00 |   3   |  1.4  |  True 
                       .. Omit 6 Ln ..                
         2017-02-10 00:00:00 |   2   |  nan  | False 
         2017-02-11 00:00:00 |   9   |  2.2  | False 
         2017-02-12 00:00:00 |   4   |  2.3  | False 

        See Also
        --------
        DaPy.SeriesSet.describe()
        '''
        return self._show(lines)

    def iter_groupby(self, keys, func=None, apply_col=None):
        '''iter_groupby(keys, func=None, apply_col=None)

        It will return the result of function of each groupby object when 
        you pass a callable object in `func`. Otherwise, it will return 
        each groupby subsheet in a dict.

        Parameters
        ----------
        keys : str, str in list
            columns that will be seem as category variable

        func : None or function (default=None)
            map this function to each group by subsheet

        apply_col : str, str in list (default=None)
            The columns will be used by the function, 
            default means all columns will be used.
        
        Returns
        -------
        groupby : Iterator of SeriesSet or Iterator of tuple
            if `func` is not None, it will be SeriesSet.
        
        Notes
        -----
        1. Highly recommand to use .groupby instead of iter_groupby.
        '''
        for group in self._iter_groupby(keys, func, apply_col):
            yield group
    
    def items(self):
        '''items() -> list of tuple(column, Series)'''
        return [(_, self._data[_]) for _ in self.columns]

    def iter_items(self):
        '''iter_items() -> yield column, Series'''
        for column in self.columns:
            yield column, self._data[column]

    def iter_rows(self):
        '''iter_rows() -> yield tuple'''
        for row in zip(*(self._data[col] for col in self.columns)):
            yield row

    def iter_values(self):
        '''iter_values() -> yield Series'''
        for col in self.columns:
            yield self._data[col]

    def iloc(self, indexs):
        '''iloc(indexs) -> SeriesSet'''
        indexs = self._check_rows_index(indexs)
        return self._iloc(SeriesSet(nan=self.nan), indexs)

    @check_thread_locked
    def insert_row(self, index, new_row):
        '''insert_row(index, new_row) -> None
        Insert a new record ``item`` in position ``index``.

        Parameter
        ---------
        index : int
            the position of new record.

        item : value or iter
            an iterable object containing new record.

        Examples
        --------
        >>> sheet = dp.SeriesSet(range(12)).reshape((6, 2))
        >>> sheet.show()
         C_0 | C_1
        -----+-----
          0  |  1  
          2  |  3  
          4  |  5  
          6  |  7  
          8  |  9  
          10 |  11 
        >>> sheet.insert_row(3, ['Inserted'])
        >>> sheet.show()
           C_0    | C_1
        ----------+-----
            0     |  1  
            2     |  3  
            4     |  5  
         Inserted | nan # index 3 and automatically add NaN
            6     |  7  
            8     |  9  
            10    |  11 

        Notes
        -----
        1. This function has been added into unit test.
        '''
        self._insert_row(index, new_row)

    @check_thread_locked
    def insert_col(self, index, new_series, new_name=None):
        '''insert_col(index, new_series, new_name=None) -> None
        Insert a new variable named `variable_name` with a sequencial data
        `series` in position `index`.

        Parameter
        ---------
        variable_name : str (default=None)
            the name of new column.

        new_series : sequence-like
            a sequence containing new variable values.

        new_name : int
            the position of new variable at.

        Examples
        --------
        >>> import DaPy as dp
        >>> sheet = dp.SeriesSet(range(10)).reshape((5, 2))
        >>> sheet.show()
         C_0 | C_1
        -----+-----
          0  |  1  
          2  |  3  
          4  |  5  
          6  |  7  
          8  |  9  
        >>> sheet.insert_col(1, ['A', 'B', 'C'], 'InsertColumn')
        >>> sheet.show()
         C_0 | InsertColumn | C_1
        -----+--------------+-----
          0  |      A       |  1  
          2  |      B       |  3  
          4  |      C       |  5  
          6  |     nan      |  7  
          8  |     nan      |  9  
        
        Notes
        -----
        1. This function has been added into unit test.
        '''
        self._insert_col(index, new_series, new_name)

    @check_thread_locked
    def join(self, other, inplace=False):
        '''join(other: SeriesSet, inplace=False) -> SeriesSet
        right join another sheet to the current sheet

        This function can help you combine another sheet while it considers 
        the matching column is the index of rows. To be simple, it just 
        like you map append_col() to each variables in other sheet, 
        but it is faster and easier to use.

        Examples
        --------
        >>> import DaPy as dp
        >>> sheet1 = dp.SeriesSet(
                        [[11, 11],
                        [21, 21],
                        [31, 31],
                        [41, 41]],
                        ['C1', 'C2'])
        >>> sheet2 = dp.SeriesSet(
                        [[21, 21],
                        [22, 22],
                        [23, 23],
                        [24, 24]],
                        ['C2', 'C3'])
        >>> sheet1.join(sheet2).show()
         C1 | C2 | C2_1 | C3
        ----+----+------+----
         11 | 11 |  21  | 21 
         21 | 21 |  22  | 22 
         31 | 31 |  23  | 23 
         41 | 41 |  24  | 24 

        Notes
        -----
        1. This function has been added into unit test.
        '''
        if not isinstance(other, SeriesSet):
            if isinstance(other, Series) or all(map(is_value, other)):
                other = SeriesSet({None: other})
            if not isinstance(other, SeriesSet) and all(map(is_iter, other)):
                new_col = [title + '_1' for title in self._columns]
                other = SeriesSet(other, new_col)
        if not isinstance(other, SeriesSet):
            raise TypeError('could not extend a single value only.')
        if inplace is False:
            return SeriesSet(self, nan=self.nan)._join(other)
        return self._join(other)
        
    def keys(self):
        '''keys() - > list of column names'''
        return self.columns

    def merge(self, other, how='inner', self_on=0, right_on=0):
        '''right join another sheet and automatically arranged by key columns

        Combind two sheet together according to two keywords. 
        It exactely matches the records in both sheet. 

        Rules of Combination
        --------------------
        <1> It will compare the keywords and find the records which have
            the same value in the keywords.
        <2> It will add the new data as new variables behind the exist records.
        <3> If there is more than one record that matches the keywords of the
            two data sets, it will correspond to the sequence of the records.

        Parameter
        ---------
        other : array-likes
            the other sheet which is used to extend

        how : 'inner', 'outer', 'left', 'right' (default='inner')
            how to handle rows which not match the columns
            `left` -> Keep only all rows in the current sheet;
            `right` -> Keep only all rows in the other sheet;
            `inner` -> Keep only rows from the common parts of two tables;
            `outer` -> Keep all rows from both sheets;

        self_on : int, str (default=0)
            choose a column as the keyword in this sheet

        right_on : int, str (default=0)
            choose a column as the keyword in the other sheet

        Return
        ------
        None

        Example
        -------
        >>> left.show()
           Name  | Age  | gender
        ---------+------+--------
           Alan  |  35  |   M    
           Bob   |  27  |   M    
         Charlie |  30  |   F    
          Daniel |  29  |  None  
           None  | None |   F   
        >>> right.show()
           Name  | gender | Age 
        ---------+--------+------
           Alan  |   M    |  35  
           Bob   |   M    |  27  
         Charlie |   F    |  30  
          Janny  |   F    |  26  
           None  |  None  | None

        Notes
        -----
        1. This function has been added into unit test.
        '''
        assert how in ('inner', 'outer', 'left', 'right')
        if isinstance(other, SeriesSet) is False:
            other = SeriesSet(other)
        self_on = self._check_columns_index(self_on)
        right_on = other._check_columns_index(right_on)
        assert len(self_on) == len(right_on) == 1, 'only support single index'
        self_on, right_on = self_on[0], right_on[0]

        # match the records according to the index
        joined = SeriesSet(nan=self.nan)
        if how == 'left':
            return left_join(self, other, self_on, right_on, joined)
        if how == 'outer':
            return outer_join(self, other, self_on, right_on, joined)
        if how == 'inner':
            return inner_join(self, other, self_on, right_on, joined)
        return left_join(other, self, right_on, self_on, SeriesSet(nan=other.nan))

    @check_thread_locked
    def normalized(self, process='NORMAL', cols=None, inplace=False):
        '''normalized(process='NORMAL', cols=None, inplace=False, **kwrds):

        Parameters
        ----------
        process : str (default='NORMAL')
            which process you wish to apply
            `NORMAL` -> operate the data so that its arrange between 0 to 1.
            `STANDAR` -> operate the data so that its mean is 0 and variance is 1.
            `LOG` -> find the logarithm of X.
            `BOX-COX` -> Box-Cox operation

        col : str, str in list (default='all')
            which column you wish to operate

        min : float (default=Min(X))
            Available when process is NORMAL

        range : float, int (default=Range(X))
            Available when process is NORMAL

        mean : float (default=mean(X))
            Available when process is STANDAR

        std : float (default=std(X))
            Available when process is STANDAR

        a : float (default=0)
            Available when process is BOX-COX

        k : float (default=1)
            Available when process is BOX-COX
            
        lamda : float (default=1)
            Available when process is BOX-COX
            
        base : float (default=e)
            Available when process is LOG
            
        Examples
        --------
        >>> from DaPy import datasets
        >>> data = datasets.example()
        >>> data.info
        sheet:sample
        ============
        1.  Structure: DaPy.SeriesSet
        2. Dimensions: Lines=12 | Variables=4
        3. Miss Value: 0 elements
                        Descriptive Statistics                
        ======================================================
         Title | Miss | Min | Max | Mean | Std  | Skew | Kurt 
        -------+------+-----+-----+------+------+------+------
         A_col |  0   |  1  |  6  | 3.00 | 1.35 | 0.63 |57.05 
         B_col |  0   |  1  |  9  | 4.33 | 2.56 | 3.11 |29.84 
         C_col |  0   |  1  |  8  | 3.33 | 2.59 | 3.13 |15.63 
         D_col |  0   |  2  |  6  | 3.50 | 1.38 | 0.61 |81.70 
        ======================================================
        >>> data.normalized()
        >>> data.info
        sheet:sample
        ============
        1.  Structure: DaPy.SeriesSet
        2. Dimensions: Lines=12 | Variables=4
        3. Miss Value: 0 elements
                        Descriptive Statistics                
        ======================================================
         Title | Miss | Min | Max | Mean | Std  | Skew | Kurt 
        -------+------+-----+-----+------+------+------+------
         A_col |  0   |  0  |  1  | 0.08 | 0.28 | 0.44 |11.28 
         B_col |  0   |  0  |  1  | 0.17 | 0.37 | 0.41 | 5.64 
         C_col |  0   |  0  |  1  | 0.17 | 0.37 | 0.41 | 5.64 
         D_col |  0   |  0  |  1  | 0.08 | 0.28 | 0.44 |11.28 
        ======================================================
        '''
        if inplace is False:
            return SeriesSet(self, nan=self._nan)._normalized(process, cols)
        return self._normalized(process, cols)

    def pop(self, index=-1, axis=0):
        '''pop(index=-1, axis=0) -> SeriesSet

        Remove and return the data of `index` along the `axis`

        Parameters
        ----------
        index : int, str, str in list or int in list (default=-1)

        axis : 0 or 1 (default=-1)
            0 -> `index` is the index of rows
            1 -> `index` is the index of columns
        
        Returns
        -------
        pop_sheet : SeriesSet

        See Also
        --------
        DaPy.SeriesSet.pop_row()
        DaPy.SeriesSet.pop_col()

        Notes
        -----
        1. Function won't be locked when sheet.locked is False and 
           axis is 1.
        '''
        assert axis in (0, 1)
        if axis == 0:
            return self.pop_row(index)
        return self.pop_col(index)

    @check_thread_locked
    def pop_row(self, index=-1):
        '''pop_row(index=-1) -> SeriesSet
        pop(remove & return) record(s) from the sheet

        Delete and return the record in position ``index``.

        Parameters
        ----------
        index : int, int in list (default=-1)
            the row indexes you expected to pop with
        
        Returns
        -------
        Poped_sheet : SeriesSet

        Examples
        --------
        >>> import DaPy as dp
        >>> sheet = dp.SeriesSet(
                [[1, 1, 1, 1],
                 [2, 2, 2, 2], 
                 [3, 3, 3, 3],
                 [4, 4, 4, 4]]
            )
        >>> sheet.pop_row([1, 2]).show()
         C_0 | C_1 | C_2 | C_3
        -----+-----+-----+-----
          2  |  2  |  2  |  2  
          3  |  3  |  3  |  3  
        >>> sheet.show()
         C_0 | C_1 | C_2 | C_3
        -----+-----+-----+-----
          1  |  1  |  1  |  1  
          4  |  4  |  4  |  4  

        Notes
        -----
        1. Function is locked when sheet.locked is False!
        '''
        return self._pop_row(SeriesSet(nan=self.nan), index)

    @check_thread_locked
    def pop_col(self, col=-1):
        '''pop_col(col=-1) -> SeriesSet
        pop(remove & return) column(s) from the sheet

        Delete and return all the variables in `col`.
        `col` could assignment as a number or some variable name.

        Parameters
        ----------
        col : int, str, str in list (default=-1)
            The columns you expect to drop out, 
            int n means the nth variable. 
            And default value -1 means the last column.
        
        Returns
        -------
        Poped_sheet : SeriesSet

        Examples
        --------
        >>> import DaPy as dp
        >>> sheet = dp.SeriesSet([[1,2,3,4],
                                  [2,3,4,5],
                                  [3,4,5,6],
                                  [4,5,6,7],
                                  [5,6,7,8]])
        >>> sheet.pop_col([1, 'C_2'])
        C_1: <2, 3, 4, 5, 6>
        C_2: <3, 4, 5, 6, 7>
        >>> sheet
        C_0: <1, 2, 3, 4, 5>
        C_3: <4, 5, 6, 7, 8>

        Notes
        -----
        1. Function won't be locked when sheet.locked is False!
        '''
        return self._pop_col(SeriesSet(nan=self._nan), col)

    @check_thread_locked
    def reverse(self, axis=0, inplace=True):
        '''reverse(axis=0, inplace=True) -> SeriesSet'''
        if inplace is True:
            return self._reverse(axis)
        return SeriesSet(self)._reverse(axis)

    def reshape(self, nshape, axis=0):
        '''Gives a new shape without changing its data.
    
        Parameters
        ----------
        nshape : int or tuple of ints
            The new shape should be compatible with the original shape. If
            an integer, then the result will be a 1-D array of that length.
            One shape dimension can be -1. In this case, the value is
            inferred from the length of the array and remaining dimensions.

        axis : int, 0 or 1 (default=0)
            Appending the values from the exist sheet along which axis. 
            0 -> iter rows at first
            1 -> iter columns at first

        Returns
        -------
        reshaped_sheet : SeriesSet
    
        Examples
        --------
        >>> sheet = dp.SeriesSet(range(12))
        >>> sheet
        C_0: <0, 1, 2, 3, 4, ... ,7, 8, 9, 10, 11>
        >>> sheet.reshape((3, 4)).show()
         C_0 | C_1 | C_2 | C_3
        -----+-----+-----+-----
          0  |  1  |  2  |  3  
          4  |  5  |  6  |  7  
          8  |  9  |  10 |  11 
        >>> sheet.reshape((6, -1)).show()
         C_0 | C_1
        -----+-----
          0  |  1  
          2  |  3  
          4  |  5  
          6  |  7  
          8  |  9  
          10 |  11 
        '''
        type_error = '`new_shape` must be contained by a tuple'
        assert is_seq(nshape) or isinstance(nshape, int), type_error
        assert axis in (0, 1), 'axis must be 1 or 0'
        if axis == 0:
            iter_chain = chain(*self.iter_rows())
        else:
            iter_chain = chain(*self.iter_values())
        
        if is_seq(nshape):
            total_values = self.shape.Ln * self.shape.Col
            nshape = list(nshape)
            if len(nshape) == 1:
                nshape.append(-1)
            assert len(nshape) == 2 and nshape.count(-1) <= 1
            if -1 == nshape[0]:
                nshape[0] = total_values / float(nshape[1])
            if -1 == nshape[1]:
                nshape[1] = total_values / float(nshape[0])
            assert isinstance(nshape[0], int), isinstance(nshape[1], int)
            assert nshape[0] > 0 and nshape[1] > 0
            err = "can't reshape size %s into shape %s" % (self.shape, nshape)
            assert nshape[0] * nshape[1] == total_values, err
            
            shape_ln = nshape[1]
            sheet, row = [], []
            for i, value in enumerate(iter_chain, 1):
                row.append(value)
                if i % shape_ln == 0:
                    sheet.append(row)
                    row = []
        else:
            sheet = Series(iter_chain)
        return SeriesSet(sheet, nan=self.nan)

    @check_thread_locked
    def replace(self, old, new, col=None, regex=False):
        '''replace(old, new, col=None, regex=False)
           transform the old value(s) to new value(s) inplace
           sheet.replace('A', 'Good', col=None)
           sheet.replace(['A', 'B'], ['Good', 'Bad'], col=None)
           sheet.replace(r'\d{4}[-]\d{1,2}[-]\d{1,2}', 'Date', regex=True)
        
        In contrast with update, this function can only be used to 
        directely transform value(s) to value(s), thus it is little
        faster. It also automatically uses Index to speed up the 
        process if you have set one.

        Parameters
        ----------
        old : value or values in list 
            the value(s) to be converted
        
        new : value or values in list 
            the value(s) that is changed to
        
        col : str, int or None (default=None)
            affected columns 
        
        regex : True / False (default=False)
            the pattern is a regex or not
        
        Returns
        -------
        None

        Examples
        --------
        >>> sheet = dp.SeriesSet(
            [['Jackson', 20, 'M', 'Died by accident'],
             ['Bob', 21, 'M', 'Died by natural causes'],
             ['Alice', 19, 'F', 'Died by accident'],
             ['Brown', 19, 'M', 'Died by natural causes']],
            ['Name', 'Age', 'Gender', 'DESC'])
        >>> sheet.replace(20, 1).show() 
           Name  | Age | Gender |          DESC         
        ---------+-----+--------+------------------------
         Jackson |  1  |   M    |    Died by accident    
           Bob   |  21 |   M    | Died by natural causes 
          Alice  |  19 |   F    |    Died by accident    
          Brown  |  19 |   M    | Died by natural causes 
        >>> sheet.replace('(Died by)', '->', col='DESC', regex=True).show() 
           Name  | Age | Gender |        DESC      
        ---------+-----+--------+-------------------
         Jackson |  1  |   M    |    -> accident    
           Bob   |  21 |   M    | -> natural causes 
          Alice  |  19 |   F    |    -> accident    
          Brown  |  19 |   M    | -> natural causes 
        '''
        return self._replace(old, new, col, regex)

    @check_thread_locked
    def update(self, where, **set_values):
        '''update(where, **set_values) -> SeriesSet'''
        return self._update(where, **set_values)
    
    @check_thread_locked
    def shuffle(self, inplace=False):
        '''shuffle(inplace=True) -> SeriesSet'''
        if inplace is True:
            return self._shuffle()
        return SeriesSet(self, nan=self._nan)._shuffle()

    def select(self, where, col=None, limit=1000):
        '''Return records from the sheet depending on `where`
        sheet.select(lambda row: row.age != 30, col='Name', limit=10)
        Equal SQL: SELECT Name FROM sheet WHERE age != 30 LIMIT 10;

        You can select the records from the sheet depending on your `where`
        condition. This condition must be a callable object and return value.
        Using the parameter `col` is faster than select the whole sheet at first 
        and get columns later. Also, `limit` is useful too when you just 
        need some records. 

        Parameters
        ----------
        where : callable object
            a function to process each row

        col : None, str or list (default=None)
            which columns you want to select

        limit : int, 'all' (default=1000)
            the maximum number of rows you want to select,
            this is a good way to speed up selection from
            million of rows if you need only 1 of them
            in each time.

        Returns
        -------
        subset : SeriesSet
            the selection result according to your statement.

        Examples
        --------
        >>> from DaPy import datasets
        >>> sheet = datasets.example()
        >>> sheet.show()
                A_col        | B_col | C_col | D_col
        ---------------------+-------+-------+-------
         2017-02-01 00:00:00 |   2   |  1.2  |  True 
         2017-02-02 00:00:00 |  nan  |  1.3  |  True 
         2017-02-03 00:00:00 |   3   |  1.4  |  True 
         2017-02-04 00:00:00 |   3   |  1.5  |  True 
         2017-02-05 00:00:00 |   5   |  1.6  | False 
         2017-02-06 00:00:00 |   1   |  1.7  | False 
         2017-02-07 00:00:00 |   4   |  1.8  | False 
         2017-02-08 00:00:00 |  nan  |  1.9  | False 
         2017-02-09 00:00:00 |  nan  |  nan  | False 
         2017-02-10 00:00:00 |   2   |  nan  | False 
         2017-02-11 00:00:00 |   9   |  2.2  | False 
         2017-02-12 00:00:00 |   4   |  2.3  | False 

        >>> sheet.select(lambda row: row.A_col == 1).show()
                A_col        | B_col | C_col | D_col
        ---------------------+-------+-------+-------
         2017-02-06 00:00:00 |   1   |  1.7  | False 
        
        >>> sheet.select(lambda row: row.A_col > 2 and row.B_col > 3).show()
                A_col        | B_col | C_col | D_col
        ---------------------+-------+-------+-------
         2017-02-03 00:00:00 |   3   |  1.4  |  True 
         2017-02-04 00:00:00 |   3   |  1.5  |  True 
         2017-02-05 00:00:00 |   5   |  1.6  | False 
         2017-02-07 00:00:00 |   4   |  1.8  | False 

        See Also
        --------
        DaPy.SeriesSet.query
        '''
        if limit is None:
            limit = self.shape.Ln
        col = self._check_columns_index(col)
        sub_index = self._where_by_rows(where, limit)
        return self._iloc(SeriesSet(nan=self.nan), sub_index)[col]
    
    def sum(self, axis=0, col=None):
        assert axis in (0, 1, None)
        col = self._check_columns_index(col)

        if axis == 1:
            values = tuple(sum(_) for _ in self[col].values())
            return SeriesSet((values,), col, nan=self.nan)[0]
        
        if axis == 0:
            return Series(sum(_) for _ in self[col].iter_rows())
        
        return sum(chain(*self[col].values()))

    def values(self):
        for col in self.columns:
            yield self._data[col]

class Frame(BaseSheet):

    '''Variable stores in sequenes
    '''

    def __init__(self, series=None, columns=None, nan=float('nan')):
        raise NotImplementedError('this class has been abandened, please use SeriesSet')
