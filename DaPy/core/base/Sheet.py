'''
This file is a part of DaPy project

We define three base data structures for operating 
data like an excel. In contrsat with Pandas, it is more 
convinience and more simply to use. 

BaseSheet is a rudimentary structure and it provides some 
functions which have no different between SeriesSet and 
Frame structures.
'''

from collections import Counter, Iterator, OrderedDict
from copy import copy, deepcopy
from datetime import datetime
from itertools import chain
from itertools import combinations
from itertools import repeat
from operator import itemgetter
from random import shuffle as shuffles
from re import compile as re_compile
from re import findall, sub

from .constant import (DUPLICATE_KEEP, PYTHON2, PYTHON3, 
                       SHEET_DIM, STR_TYPE, VALUE_TYPE, nan as NaN)
from .IndexArray import SortedIndex
from .Row import Row
from .Series import Series
from .utils import (
    argsort, auto_plus_one, auto_str2value, count_nan, fast_str2value,
    is_empty, is_iter, is_math, is_seq, is_str, is_value, isnan, split,
    xrange, zip_longest, hash_sort, str2date, strip)
from .utils.utils_join_table import inner_join, left_join, outer_join
from .utils.utils_regression import simple_linear_reg

__all__ = ['SeriesSet', 'Frame']

PATTERN_AND_OR = re_compile(r'\sand\s|\sor\s')
PATTERN_COMBINE = re_compile(r'[(](.*?)(\sand\s|\sor\s)(.*?)[)]')
PATTERN_RECOMBINE = re_compile(r'[)](\sand\s|\sor\s)[(]')
PATTERN_COLUMN = r'[(|\s]{0,1}%s[\s|)]{0,1}'

PATTERN_EQUAL = re_compile(r'(.*?)(!=|==)(.*?)')
PATTERN_LESS = re_compile(r'(.*?)(<=|<)(.*?)')
PATTERN_GREAT = re_compile(r'(.*?)(>=|>)(.*?)')
PATTERN_BETWEEN1 = re_compile(r'(.+?)(>=|>)(.+?)(>=|>)(.+?)')
PATTERN_BETWEEN2 = re_compile(r'(.+?)(>=|>)(.+?)(>=|>)(.+?)')
PATTERN_EQUALS = (PATTERN_EQUAL, PATTERN_LESS, PATTERN_GREAT)
SIMPLE_EQUAL_PATTERN = ('!=', '<=', '>=')

LOCK_ERROR = 'sheet is locked by indexes, See drop_index()'

def subset_quickly_append_col(subset, col, seq, miss):
    '''append a new column to the sheet without checking'''
    col = subset._check_col_new_name(col)
    subset.data[col] = seq
    subset._columns.append(col)
    subset._missing.append(miss)
    subset._dim = SHEET_DIM(len(seq), subset._dim.Col + 1)
    return subset

def where_by_index_combine(rows, symbols):
    '''put all rows together'''
    final_rows = set(rows[0])
    for row, comb in zip(rows[1:], symbols):
        if comb.strip() == 'and':
            final_rows = final_rows & row
        else:
            final_rows = final_rows | row
    return final_rows

class BaseSheet(object):
    '''The base sheet structure for user to handle 2D data structure

    Attributes
    ----------
    shape : namedtuple(Ln, Col)
        a two dimensional span of this sheet.

    nan : value (default=Nan)
        a symbol represented miss value in current seriesset.

    columns : str in list
        names for each feature

    data : dict / list in list
        an object contains all the data by columns or row.

    missing : int in list
        number of missing value in each column.

    indexes : SortedIndex in dict
        a dict stored all indexes
    '''
    def __init__(self, obj=None, columns=None, nan=NaN):
        self._missing = []
        self._nan = nan
        self._init_nan_func()
        self._sorted_index = {}

        if isinstance(obj, BaseSheet): 
            if isinstance(self.data, dict):
                # initialize from a SeriesSet
                self._init_col(obj, columns)
            else:
                # initialize from a DataFrame
                self._init_frame(obj, columns)

        elif obj is None or is_empty(obj): 
            # initialize an empty sheet
            self._dim, self._columns = SHEET_DIM(0, 0), []
            if columns is not None:
                if is_str(columns):
                    columns = [columns, ]
                for name in columns:
                    self.append_col([], name)

        elif hasattr(obj, 'items'): 
            # initialize from a dict
            self._init_dict(obj, columns)

        elif isinstance(obj, Iterator): 
            # initialie from an iterator object
            self._init_like_iter(obj, columns)

        elif isinstance(obj, Series) or \
            (is_seq(obj) and all(map(is_value, obj))):
             # initialize from a single series
            self._init_like_seq(obj, columns)

        elif is_seq(obj) and all(map(is_iter, obj)):
            # initialize from array-like object
            self._init_like_table(obj, columns)

        else:
            raise TypeError("sheet don't support %s" % type(obj))

    @property
    def data(self):
        '''self.data -> source container of the data'''
        return self._data

    @property
    def shape(self):
        '''self.shape -> numbers of rows and columns in tuple'''
        return self._dim

    @property
    def columns(self):
        '''self.columns -> copy of the columns in the sheet'''
        return copy(self._columns)

    @columns.setter
    def columns(self, item):
        '''self.columns = ['A', 'B'] -> setting columns of sheet'''
        if self.shape.Col == 0 and item != []:
            self._dim = SHEET_DIM(0, len(item))
            self._missing = [0] * len(item)
            old_col = item

        old_col = self.columns
        self._init_col_name(item)
        if isinstance(self._data, dict):
            for old, new in zip(old_col, self.columns):
                seq = self.data.get(old, Series())
                del self.data[old]
                self.data[new] = seq

    @property
    def nan(self):
        '''self.nan -> return the symbol of missing value'''
        return self._nan

    @nan.setter
    def nan(self, item):
        '''self.nan = None -> change another missing value symbol'''
        assert is_value(item), 'sheet.nan must be a value'
        self._nan = item
        self._init_nan_func()
        for miss, seq in zip(self._missing, self.iter_values()):
            if miss != 0:
                for i, value in enumerate(seq):
                    if self._isnan(value):
                        seq[i] = item

    @property
    def missing(self):
        '''self.missing -> number of missing values in each column'''
        return SeriesSet([self._missing], self.columns)[0]

    @property
    def locked(self):
        '''self.is_mutable -> bool
        check whether sheet is clock by sorted indexes or not'''
        if self._sorted_index:
            return False
        return True
        

    def __getattr__(self, name):
        '''self.A -> return column A'''
        if name in self._columns:
            return self.__getitem__(name)
        raise AttributeError("Sheet object has no attribute '%s'" % name)

    def __len__(self):
        '''len(sheet) -> number of rows'''
        return self._dim.Ln

    def __contains__(self, cmp_):
        '''3 in sheet -> True / False'''
        if is_str(cmp_):
            return cmp_ in self._columns

        if is_seq(cmp_):
            if len(cmp_) == self._dim.Col:
                for record in self:
                    if record == cmp_:
                        return True

            elif len(cmp_) == self._dim.Ln:
                for variable in self.iter_values():
                    if variable == cmp_:
                        return True

        if is_value(cmp_):
            for record in self:
                for value in record:
                    if value == cmp_:
                        return True
        return False

    def __eq__(self, other):
        '''Sheet1 == 3 -> Bool in sheet'''
        if is_value(other):
            temp = SeriesSet()
            for title, seq in self.iter_items():
                temp[title] = [_ == other for _ in seq]
            return temp
        
        if other.shape.Ln != self.shape.Ln:
            return False
        if other.shape.Col != self.shape.Col:
            return False
        if other.columns != self.columns:
            return False
        for lval, rval in zip(other.iter_values(),
                              self.iter_values()):
            if (lval == rval).all() is False:
                return False
        return True

    def __setitem__(self, key, value):
        '''sheet['Col'] = [1, 2, 3, 4] -> None'''
        error = 'only support setting record(s) or column(s)'
        if isinstance(key, int):
            if key <= self._dim.Ln:
                self.__delitem__(key)
                self.insert_row(key, value)
            else:
                self.append_row(value)

        elif is_str(key):
            if key in self._columns:
                pos = self._columns.index(key)
                self.__delitem__(key)
                self.insert_col(pos, value, key)
            else:
                self.append_col(value, key)
        
        elif isinstance(key, slice):
            start, stop, axis = self._check_slice(key)
            if axis == 1:
                self._setitem_slice_col(start, stop, value)

            if axis == 0:
                self._setitem_slice_row(start, stop, value)
        else:
            raise TypeError(error)
        
    def _check_mixture_tuple(self, key):
        '''don't let user use sheet[:, 0] syntax'''
        left, right = key
        if left.start is None and left.stop is None:
            if isinstance(right, slice):
                rstart, rstop = right.start, right.stop
                maybe = (self.columns[rstart], self.columns[rstop])
                maybe += (rstart, rstop)
                raise SyntaxError('do you mean sheet' + 
                                  '["%s":"%s"] or sheet[%s:%s]?' % maybe)
            if isinstance(right, int):
                raise SyntaxError(
                    'do you mean sheet ' +
                    '["%s"] or sheet[%s]' % (self.columns[right], right))
    
    def _getitem_by_tuple_subcol(self, key, subset):
        '''given columns, get subset'''
        for arg in key:
            if is_str(arg):
                seq = self.data[arg]
                miss = self._missing[self._columns.index(arg)]
                subset_quickly_append_col(subset, arg, seq, miss)

            elif isinstance(arg, slice):
                start, stop = self._check_slice_col(arg.start, arg.stop)
                for col in self._columns[start:stop + 1]:
                    miss = self._missing[self._columns.index(col)]
                    seq = self.data[col]
                    subset_quickly_append_col(subset, col, seq, miss)
            else:
                raise TypeError('bad statement as sheet[%s]' % col)
        return subset
    
    def _getitem_by_tuple_subrow(self, int_args, slc_args, subset):
        '''given rows, get subset'''
        subset = self._iloc(subset, int_args)
        for row in slc_args:
            start, stop = self._check_slice_row(row.start, row.stop)
            subset.extend(self._getslice_ln(start, stop, 1))
        return subset

    def _getitem_by_tuple(self, key, subset):
        '''given columns or rows, get subset'''
        if is_seq(key) and len(key) == 2 and isinstance(key[0], slice):
            self._check_mixture_tuple(key)

        args, int_args, slc_args = [], [], []
        for arg in key:
            if isinstance(arg, slice):
                slc_args.append(arg)
                for value in (arg.start, arg.stop):
                    if value is not None:
                        args.append(value)
            else:
                args.append(arg)
                if isinstance(arg, int):
                    int_args.append(arg)

        subcol = all(map(is_str, args))
        subrow = all(map(is_math, args))
        err = "don't get subset with columns and rows at the " +\
         "same time. Use: sheet['A':'B'][3:10] or sheet[3:10]['A':'B']"
        assert subcol or subrow, err

        if subcol is True:
            return self._getitem_by_tuple_subcol(key, subset)
        return self._getitem_by_tuple_subrow(int_args, slc_args, subset)

    def __delitem__(self, key):
        ok_types = tuple([STR_TYPE] + [int, list, tuple, slice])
        assert isinstance(key, ok_types), 'not allowed type "%s"' % type(key)
        if isinstance(key, int):
            self.drop_row(key, True)

        if is_str(key):
            self.drop_col(key, True)

        if isinstance(key, (list, tuple)):
            int_keys = list(filter(is_math, key))
            str_keys = list(filter(is_str, key))
            if str_keys != []:
                self.drop_col(str_keys, True)
            if int_keys != []:
                self.drop_row(int_keys, True)
        
        if isinstance(key, slice):
            self.__delslice__(key)
    
    def __delslice__(self, key):
        start, stop, axis = self._check_slice(key)
        if axis == 1:
            self.drop_col(self.columns[start, stop + 1], True)
        elif axis == 0:
            self.drop_col(range(start, stop), True)

    def __getslice__(self, start, stop, step=1):
        start, stop, axis = self._check_slice(slice(start, stop))
        if axis == 1:
            return self._getslice_col(start, stop)
        if axis == 0:
            return self._getslice_ln(start, stop, step)
        raise TypeError('bad expression as [%s:%s]' % (start, stop))

    def __getstate__(self):
        instance = self.__dict__.copy()
        instance['_dim'] = tuple(self._dim)
        instance['_data'] = self._data
        instance['_sorted_index'] = self._sorted_index
        return instance

    def __setstate__(self, read_dict):
        '''load this object from a stream file'''
        self._dim = SHEET_DIM(*read_dict['_dim'])
        self._columns = read_dict['_columns']
        self._missing = read_dict['_missing']
        self._nan = read_dict['_nan']
        self._data = read_dict['_data']
        self._sorted_index = read_dict['_sorted_index']

    def _init_col_name(self, columns):
        if is_str(columns) and self._dim.Col == 1:
            self._columns = [columns, ]
            
        elif is_str(columns) and self._dim.Col != 1:
            self._columns = [columns + '_%d' % i for i in range(self._dim.Col)]

        elif columns is None or str(columns).strip() == '':
            self._columns = ['C_%d' % i for i in xrange(self._dim.Col)]
            
        elif is_iter(columns) is True:
            self._columns, columns = [], list(columns)
            bias = self._dim.Col - len(columns)
            columns.extend(['C_%d' % i for i in range(bias)])
            for col in columns[:self._dim.Col]:
                self._columns.append(self._check_col_new_name(col))
            for _ in xrange(self._dim.Col - len(self._columns)):
                self._columns.append(self._check_col_new_name(None))
        else:
            raise TypeError('column names should be stored in a iterable')

    def _trans_where(self, where, axis=0):
        assert axis in (1, 0), 'axis 1 for value, 0 for sequence'
        if axis == 0:
            if where is None:
                return lambda x: True
            where = ' ' + where + ' '
            for i in argsort(self._columns, key=len, reverse=True):
                where = sub(self._columns[i], '___x___[%d]' % i, where)
            where = 'lambda ___x___: ' + where

        if axis == 1:
            opeartes = {' and': 4, ' or': 3}
            for opearte, bias in opeartes.items():
                counts = where.count(opearte)
                index = 0
                for i in xrange(counts):
                    index = where.index(opearte, index) + bias
                    where = where[: index] + ' ___x___' + where[index:]
                    index += bias
            where = 'lambda ___x___: ___x___ ' + where
        return eval(where)

    def _add_row(self, row):
        if is_value(row):
            row = [row] * self._dim.Col

        if hasattr(row, '_asdict'):
            row = row._asdict()

        if isinstance(row, (dict, OrderedDict)):
            row, dic_row = [row.get(col, self.nan) for col in self.columns], row
            for key, value in dic_row.items():
                if key not in self._data:
                    row.append(value)
                    self._data[key] = Series(repeat(self.nan, self.shape.Ln))
                    self._missing.append(self.shape.Ln)
                    self._columns.append(key)
                    self._dim = SHEET_DIM(self._dim.Ln, self.shape.Col + 1)

        lenth_bias = len(row) - self._dim.Col
        if lenth_bias > 0 and self.shape.Ln == 0:
            for i in xrange(lenth_bias):
                self.append_col(Series())
        elif lenth_bias > 0:
            self.join([[self.nan] * lenth_bias for i in xrange(self.shape.Ln)],
                      inplace=True)

        miss, row = self._check_sequence(row, self.shape.Col)
        self._dim = SHEET_DIM(self._dim.Ln + 1, max(self._dim.Col, len(row)))
        if miss != 0:
            for i, value in enumerate(row):
                if self._isnan(value):
                    self._missing[i] += 1
        return row

    def _check_sequence(self, series, size):
        '''check the shape of the sequence and fill nan if not long enough

        if sequence is record, size = self.shape.Ln;
        else size = self.shape.Col
        '''
        if is_value(series):
            return 0, [series] * size

        assert is_iter(series), "append item should be an iterable object"
        series = Series(series)
        if len(series) < size:
            series = chain(series, [self._nan] * (size - len(series)))
            series = Series(series)
        return count_nan(self._isnan, series), series

    def _check_col_new_name(self, new_name):
        if new_name is None:
            return self._check_col_new_name('C_%d' % len(self._columns))

        new_name = str(new_name)
        if is_str(new_name) and new_name not in self._columns:
            return new_name
        return auto_plus_one(self._columns, new_name)

    def _check_slice(self, slc):
        start, stop = slc.start, slc.stop
        types_1 = is_str(start) == is_str(stop)
        types_2 = is_math(start) == is_math(stop)
        types_3 = None in (start, stop)
        error = 'only support delete row or column at each time'
        assert types_1 or types_2 or types_3, error
        if isinstance(start, int) or isinstance(stop, int):
            start, stop = self._check_slice_row(start, stop)
            return start, stop, 0

        # is_str(start) or is_str(stop):
        start, stop = self._check_slice_col(start, stop)
        return start, stop, 1

    def _check_slice_row(self, start, stop):
        lenth = self.shape.Ln
        if start is None:
            start = 0
        elif start < 0:
            start += lenth
        elif start > lenth:
            start = lenth

        if stop is None:
            stop = lenth - 1
        elif stop < 0:
            stop += lenth
        elif stop > lenth:
            stop = lenth
        error = 'Index out of range'
        assert 0 <= start <= lenth and 0 <= stop <= lenth, error
        return start, stop

    def _check_slice_col(self, start, stop):
        if start in self._columns:
            start = self._columns.index(start)
        elif start is None:
            start = 0
        else:
            raise ValueError('%s is not a title in this sheet' % start)

        if stop in self._columns:
            stop = self._columns.index(stop)
        elif stop is None:
            stop = self._dim.Col - 1
        else:
            raise ValueError('%s is not a title in this sheet' % stop)
        return start, stop

    def _check_operation_key(self, keys):
        '''transfer the string key name into itemgetter object'''
        return itemgetter(*tuple(map(self.columns.index, keys)))

    def _init_nan_func(self):
        if isnan(self._nan) is True:
            self._isnan = isnan
        else:
            self._isnan = lambda val: val == self._nan

    def _check_columns_index(self, col):
        if col is None:
            return tuple(self._columns)

        if is_str(col):
            error = '%s is not a title in current dataset' % col
            assert col in self._columns, error
            return (col,)

        if isinstance(col, int):
            assert abs(col) < self.shape.Col, 'title index is out of range'
            return (self._columns[col],)

        if is_seq(col):
            return tuple(self._check_columns_index(_)[0] for _ in col)

        # isinstance(col, slice):
        start, stop, axis = self._check_slice(col)
        assert axis == 1, "don't put a row index in here"
        return self.columns[start, stop]
    
    def _check_rows_index(self, row):
        assert is_str(row) is False, 'row index must not be a str'
        if row is None:
            return range(self.shape.Ln)
        
        if isinstance(row, int):
            assert abs(row) < self.shape.Ln, 'row index is out of range'
            return (row,)

        if is_seq(row):
            return tuple(self._check_rows_index(_)[0] for _ in row)
        
        # isinstance(row, slice):
        start, stop, axis = self._check_slice(row)
        assert axis == 0, "don't put a column index in here"
        return range(start, stop)

    def _match_column_from_str(self, statement):
        sorted_column = sorted(self.columns, key=len, reverse=True)
        pattern = '|'.join([PATTERN_COLUMN % x for x in sorted_column])
        useful_col = [col.strip() for col in findall(pattern, statement)]
        return [col.replace('(', '').replace(')', '') for col in useful_col]

    def _group_index_by_column_value(self, columns, engine=list):
        subset = {}
        for i, row in enumerate(zip(*(self._data[col] for col in columns))):
            subset.setdefault(row, engine()).append(i)
        return subset

    def copy(self):
        '''copy the current sheet'''
        if isinstance(self.data, dict):
            return SeriesSet(self)
        return Frame(self)

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
        col = self._check_columns_index(col)
        assert self._isnan(new) is False, 'transfer value cannot be NaN'

        if regex is True:
            assert is_str(old), '`where` must be str when regex=True'
            condition = re_compile(old)
            where = lambda val: condition.sub(new, val)

        if is_value(old) and regex is False:
            list_old = [old]
            where = lambda val: new if old == val else val

        if is_seq(old) and is_seq(new):
            assert len(old) == len(new), 'length of keys != length of value'
            list_old = old
            condition = dict(zip(old, new))
            where = lambda val: condition.get(val, val)

        for column in col:
            sequence = self.data[column]
            if column in self._sorted_index and regex is False:
                # speed up process with Index
                for old_ in list_old:
                    to_replace = where(old_)
                    for i in self._sorted_index.equal(old_):
                        sequence[i] = to_replace

            else:
                # iter each row and compare
                for i, val in enumerate(sequence):
                    sequence[i] = where(val)
        return self

    def tolist(self):
        '''return the data as lists in list'''
        if isinstance(self.data, list):
            return deepcopy(self.data)
        return list(map(list, self.iter_rows()))

    def toarray(self):
        '''return the data as a numpy.array object'''
        try:
            from numpy import array
        except ImportError:
            raise ImportError("can't find numpy library")

        if isinstance(self.data, list):
            return array(self.data)
        return array(tuple(self.iter_rows()))

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
        DaPy.core.base.Sheet.SeriesSet._where_by_index
        DaPy.core.base.Sheet.SeriesSet._where_by_rows
        DaPy.core.base.Sheet.SeriesSet.create_index
        DaPy.core.base.IndexArray.SortedIndex
        '''
        assert is_str(expression), '`expression` should be a python statement'
        select_col = self._check_columns_index(col)
        useful_col = self._match_column_from_str(expression)
        assert useful_col, "can't match any column from `expression`"
        if limit is None:
            limit = self.shape.Ln

        if all([col in self._sorted_index for col in useful_col]) is False:
            subset = self[useful_col]
            where = subset._trans_where(expression, axis=0)
            sub_index = subset._where_by_rows(where, limit)
        else:
            sub_index = sorted(self._where_by_index(expression))
            sub_index = sub_index[:limit]
        try:
            return self._iloc(SeriesSet(nan=self.nan), sub_index)[select_col]
        except TypeError:  # there is no row match the query
            return SeriesSet(None, select_col, nan=self.nan)
    
    def _where_by_index_bigcombine(self, combines, string):
        '''process statement like: (A > 1 and B < 2) and (C < 2)'''
        string = PATTERN_RECOMBINE.findall(string)
        symbols = [_[1:-1] for _ in string]
        rows = [self._where_by_index(''.join(_)) for _ in combines]
        return where_by_index_combine(rows, symbols)
    
    def _where_by_index_subcombine(self, subcombine, string):
        '''process statement like: A > 1 and B < 2'''
        symbols = PATTERN_AND_OR.findall(string)
        rows = [self._where_by_index(_) for _ in subcombine]
        return where_by_index_combine(rows, symbols)
    
    def _where_by_index_simple(self, column, string):
        '''process statement like: A > 1'''
        index = self._sorted_index[column]
        operater = (index.equal, index.lower, index.upper)

        def clear_pattern(_):
            return _.strip().replace('"', '').replace("'", '')

        for i, (pattern, symbol, func) in enumerate(zip(PATTERN_EQUALS,
                                                        SIMPLE_EQUAL_PATTERN,
                                                        operater)):
            pattern = [_ for _ in pattern.split(string) if _]
            if len(pattern) == 3:
                val = clear_pattern(pattern[2])
                val = auto_str2value(val)
                if pattern[1] == symbol:
                    if i == 0:
                        return set(index.unequal(val))
                    return set(func(val))
                try:
                    return set(func(val, False))
                except TypeError:
                    return set(func(val))

        pattern = [_ for _ in PATTERN_BETWEEN1.split(string) if _.strip()]
        lvalue = auto_str2value(clear_pattern(pattern[4]))
        hvalue = auto_str2value(clear_pattern(pattern[0]))
        boundary = True, True
        for i, pattern in enumerate([pattern[1], pattern[3]]):
            if pattern == '>':
                boundary[i] = False
        return set(index.between(lvalue, hvalue, boundary))

    def _where_by_index(self, string):
        '''select records according to the sorted index

        Analysis the purposes of statements, then return
        the indexes of the records which match statements.

        Parameters
        ----------
        substring : pythonic string statement
            you can write very complex statement like:
            eg.1 : A_col >= 3
            eg.2 : (B_col <= 2 and 3 >= D_col >= 1) or (A_col == 2 and B_col == 3)

        column_pattern : a compiled regex object
            used to match the column name in the value

        Returns
        -------
        final_rows : indexes in the list
        '''
        combines = PATTERN_COMBINE.findall(string)
        if combines:
            return self._where_by_index_bigcombine(combines, string)

        subcombine = PATTERN_AND_OR.split(string)
        if len(subcombine) > 1:
            return self._where_by_index_subcombine(subcombine, string)

        column = self._match_column_from_str(string)
        assert len(column) <= 1, 'indexes are only used in processing single column'
        assert len(column) == 1, "can't match any column from `%s`" % string
        column = column[0]
        assert column in self._sorted_index, "`%s` isn't in statement `%s`" % (column, string)
        return self._where_by_index_simple(column, string)

    def _where_by_rows(self, where, limit):
        assert isinstance(limit, int)
        assert callable(where), '`where` is not callable, try: Sheet.query(where)'
        rows = self.iter_rows()

        try:
            where(tuple())
        except AttributeError:
            rows = self.__iter__()

        selected = 0
        for i, row in enumerate(rows):
            if where(row):
                selected += 1
                yield i
                if selected == limit:
                    break

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
        type_error = '`new_shape` must contain in a tuple'
        assert is_seq(nshape) or isinstance(nshape, int), type_error
        assert axis in (0, 1), 'axis must be 1 or 0'
        if axis == 0:
            iter_chain = chain(*self.iter_rows())
        else:
            iter_chain = chain(*self.iter_values())
        
        if is_seq(nshape):
            total_values = self.shape.Ln * self.shape.Col
            nshape = list(nshape)
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
            sheet = SeriesSet(Series(iter_chain), nan=self.nan)

        if isinstance(self.data, dict):
            return SeriesSet(sheet, nan=self.nan)
        return Frame(sheet, nan=self.nan)
        
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

    def get(self, key, default=None):
        '''get(key, default=None)
           select column from the sheet, 
           return default if key is not a column name
        '''
        if key in self.columns:
            return self[key]
        return default

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
            result = tuple(self.iter_groupby(keys, func, apply_col))
            return SeriesSet(result, result[0].columns)

        result = {}
        for key, subset in self.iter_groupby(keys, func, None):
            result[key] = subset
        return result

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
        def operate_subset(subset, key):
            ret = subset.apply(func, col=apply_col, axis=1)
            for key_value, key_name in zip(key, keys):
                if key_name not in ret.columns:
                    pos = self.columns.index(key_name)
                    ret.insert_col(pos, key_value, key_name)
            return ret[0]

        keys = self._check_columns_index(keys)
        assert keys, 'must give at least 1 key column to group by'
        if len(keys) == 1 and keys[0] in self._sorted_index:
            subsets = {}
            index = self._sorted_index[keys[0]]
            for group_value in set(self._data[keys[0]]):
                subsets[(group_value,)] = index.equal(group_value)
        else:
            subsets = self._group_index_by_column_value(keys)

        if func is not None:
            apply_col = self._check_columns_index(apply_col)
            for keyword, rows in subsets.items():
                subset = self._iloc(SeriesSet(nan=self._nan), rows)
                yield operate_subset(subset, keyword)
        else:
            for keyword, rows in subsets.items():
                yield keyword, self._iloc(SeriesSet(nan=self._nan), rows)

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
        if not self._columns:
            frame = 'empty sheet instant'
        else:
            error = '`lines` must be an int or None.'
            assert (is_math(lines) and lines > 0) or lines is None, error
            if lines is None or 2 * lines >= self._dim.Ln:
                lines, omit = -1, 0
                temp = self
            else:
                omit = self._dim.Ln - 2 * lines
                temp = self[:lines]
                temp.extend(self[-lines:], inplace=True)
            temporary_series = [[title, ] for title in self._columns]
            for i, col in enumerate(temp.values()):
                temporary_series[i].extend(map(str, col))

            col_size = [len(max(col, key=len)) for col in temporary_series]
            frame = ' ' + ' | '.join([title.center(col_size[i]) for i, title in
                                      enumerate(self._columns)]) + '\n'
            frame += '+'.join(['-' * (_ + 2) for _ in col_size]) + '\n'

            for i, item in enumerate(temp.iter_rows()):
                if i == lines:
                    frame += ('.. Omit %d Ln ..' % omit).center(len(line)) + '\n'
                line = ''
                for j, value in enumerate(item):
                    line += ' ' + str(value).center(col_size[j]) + ' |'
                frame += line[:-1] + '\n'
        print(frame)

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
        err = "orderby must be a sequence of conditions like ('A_col', 'DESC')"
        assert all(map(lambda x: (is_seq(x) and len(x) == 2)
                       or is_str(x), orderby)), err
        symbols = ['ASC' if is_str(_) else str(_[1]) for _ in orderby]
        if is_str(orderby[0]):
            orderby = [(_,) for _ in orderby]
        keys = self._check_columns_index([_[0] for _ in orderby])

        if not (len(symbols) == 1 and isinstance(self._data, dict)):
            columns = tuple(map(self.columns.index, keys))
            temp = hash_sort(self.iter_rows(), *zip(columns, symbols))
            if isinstance(self._data, dict):
                return SeriesSet(temp, self.columns)
            return Frame(temp, self.columns)

        reverse = False
        if symbols[0] == 'DESC':
            reverse = True

        new_index = argsort(list(self.data[keys[0]]), reverse=reverse)
        sort_subset = SeriesSet(nan=self.nan)
        
        for i, (key, seq) in enumerate(self.iter_items()):
            subset_quickly_append_col(sort_subset, 
                                      key,
                                      seq[new_index],
                                      self._missing[i])
        return sort_subset


class SeriesSet(BaseSheet):

    '''Variable stores in sequenes
    '''

    def __init__(self, series=None, columns=None, nan=float('nan')):
        self._data = dict()
        BaseSheet.__init__(self, series, columns, nan)

    @property
    def info(self):
        '''summary the information of sheet'''
        self.describe(level=0)

    @property
    def T(self):
        '''transpose the current data set -> SeriesSet'''
        return SeriesSet(self.iter_values(), None, self.nan)

    def _init_col(self, series, columns):
        '''initialzie from a SeriesSet

        Notes
        -----
        1. This function has been added into unit test.
        '''
        if columns is None:
            columns = series.columns
        self._dim = copy(series.shape)
        self._init_col_name(columns)
        for col, seq in zip(self.columns, series.values()):
            self._data[col] = copy(seq)
        self._missing = list(series.missing)
        self.nan = series.nan

    def _init_dict(self, series, columns):
        '''initialize from a dict object

        Notes
        -----
        1. This function has been added into unit test.
        '''
        if columns is None:
            columns = list(series.keys())
        for key, value in series.items():
            if is_value(value) is True:
                series[key] = [value]
        max_ln = max(map(len, series.values()))
        self._dim = SHEET_DIM(max_ln, len(series))
        self._init_col_name(columns)
        for col in self.columns:
            miss, sequence = self._check_sequence(series[col], self._dim.Ln)
            self._missing.append(miss)
            self._data[col] = sequence

    def _init_frame(self, series, columns):
        '''initialize from a Frame

        Notes
        -----
        1. This function has been added into unit test.
        '''
        if columns is None:
            columns = series.columns
        self._dim = copy(series.shape)
        self._missing = list(series.missing)
        self._init_col_name(columns)
        for sequence, title in zip(zip(*series), self._columns):
            self._data[title] = Series(sequence)
        self.nan = series.nan

    def _init_like_seq(self, series, columns):
        '''initialize from a single sequence

        Notes
        -----
        1. This function has been added into unit test.
        '''
        self._dim = SHEET_DIM(len(series), 1)
        self._init_col_name(columns)
        miss, series = self._check_sequence(series, self._dim.Ln)
        self._missing = [miss, ]
        self._data[self._columns[0]] = series

    def _init_like_table(self, series, columns):
        '''initialize from an array-like object

        Notes
        -----
        1. This function has been added into unit test.
        '''
        lenth_col = len(max(series, key=len))
        self._dim = SHEET_DIM(len(series), lenth_col)
        self._init_col_name(columns)
        self._missing = [0] * self._dim.Col
        for j, sequence in enumerate(zip_longest(fillvalue=self.nan, *series)):
            miss, series = self._check_sequence(sequence, self._dim.Ln)
            self._missing[j] += miss
            self._data[self._columns[j]] = series

    def _init_like_iter(self, series, columns):
        '''initialize from an iterator'''
        data_columns = []
        self._missing = []
        for i, row in enumerate(series):
            bias = len(row) - len(data_columns)
            self._missing.extend(repeat(0, bias))
            more_col = [Series(repeat(self.nan, i)) for _ in xrange(bias)]
            data_columns.extend(more_col)
            for j, (ser, val) in enumerate(zip(data_columns, row)):
                ser.append(val)
                if self._isnan(val):
                    self._missing[j] += 1
        self._dim = SHEET_DIM(len(data_columns[0]), len(data_columns))
        self._init_col_name(columns)
        self._data = dict(zip(self.columns, data_columns))

    def __repr__(self):
        if self._dim.Ln > 10:
            def write_line(title, blank):
                item = self._data[title]
                msg = ' ' * blank + title + ': <'
                msg += ', '.join([str(value) for value in item[:5]])
                msg += ', ... ,'
                msg += ', '.join([str(value) for value in item[-5:]])
                msg += '>\n'
                return msg
            
        elif self._dim.Ln != 0:
            def write_line(title, blank):
                item = self._data[title]
                msg = ' ' * blank + title + ': <'
                msg += ', '.join([str(value) for value in item])
                msg += '>\n'
                return msg
        else:
            return 'empty SeriesSet instant'

        msg = ''
        size = len(max(self._columns, key=len))
        for title in self._columns:
            msg += write_line(title, size - len(title))
        return msg[:-1]

    def _getslice_col(self, i, j):
        subset = SeriesSet(nan=self.nan)
        for ind, column in enumerate(self._columns[i: j + 1], i):
            subset_quickly_append_col(
                subset, 
                column, 
                self.data[column], 
                self._missing[ind]
                )
        return subset

    def _getslice_ln(self, i, j, k):
        subset = SeriesSet(None, None, self.nan)
        for miss, col in zip(self._missing, self._columns):
            seq = self._data[col][i:j:k]
            if miss != 0:
                miss = count_nan(self._isnan, seq)
            subset_quickly_append_col(subset, col, seq, miss)
        return subset

    def _setitem_slice_col(self, start, stop, value):
        columns = self.columns[start:stop]
        err = "number of columns don't match number of given data"
        assert len(columns) == len(value), err
        for i, column, seq in zip(range(start, stop), columns, value):
            miss, seq = self._check_sequence(seq, self.shape.Ln)
            bias = len(seq) - self.shape.Ln
            if bias > 0:
                self._dim = SHEET_DIM(self.shape.Ln + bias, 
                                      self.shape.Col)
                for j, (col, src_seq) in enumerate(self.iter_items()):
                    if col not in columns:
                        src_seq.extend(repeat(self.nan, bias))
                        self._missing[j] += bias
            self._data[column] = seq
            self._missing[i] = miss

    def __getitem__(self, interval):
        if isinstance(interval, int):
            return Row(self, interval)

        if isinstance(interval, Series):
            assert len(interval) == self.shape.Ln
            return self.iloc([i for i, val in enumerate(interval) if val])

        if isinstance(interval, (tuple, list)):
            return self._getitem_by_tuple(interval,
                                          SeriesSet(nan=self._nan))

        if isinstance(interval, slice):
            start, stop = interval.start, interval.stop
            return self.__getslice__(start, stop)

        if is_str(interval):
            return self._data[interval]

        raise TypeError('SeriesSet index must be int, str and slice, ' +
                        'not %s' % str(type(interval)).split("'")[1])

    def __iter__(self):
        for i in xrange(self._dim.Ln):
            yield Row(self, i)

    def __reversed__(self):
        for i in xrange(self._dim.Ln - 1, -1, -1):
            yield Row(self, i)

    def _arrange_by_index(self, self_new_index=None, other_new_index=None):
        assert (self_new_index, other_new_index).count(None) == 1
        if self_new_index:
            getter = self_new_index
        else:
            getter = argsort(other_new_index)

        for title, sequence in self._data.items():
            self._data[title] = sequence[getter]

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
        miss, series = self._check_sequence(series, self._dim.Ln)
        size = len(series)
        if size > self._dim.Ln:
            bias = size - self._dim.Ln
            for i, title in enumerate(self._columns):
                self._missing[i] += bias
                self._data[title].extend([self._nan] * bias)
        subset_quickly_append_col(self, variable_name, series, miss)

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
        row = self._add_row(row)
        for val, col, seq in zip(row, self.columns, self.iter_values()):
            seq.append(val)
            if col in self._sorted_index:
                self._sorted_index[col].append(val)

    def apply(self, func, col=None, inplace=False, axis=0):
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
                return SeriesSet(map(func, self[cols]), cols, self.nan)

            ret = SeriesSet(columns=self.columns, nan=self.nan)
            row = [func(self[_]) if _ in cols else self.nan for _ in self.columns]
            ret.append_row(row)
            return ret[cols]

        if axis == 1:
            err = 'Your are operating columns which are Index. '
            err += 'Please delete that column at first!'
            assert all(lambda x: x not in self._sorted_index, cols), err
            for name in cols:
                seq = Series(map(func, self.data[name]))
                ind = self.columns.index(seq)
                self._missing[ind] = count_nan(self._isnan, seq)
                self.data[name] = seq

        if axis == 0:
            new_col = self._check_col_new_name(None)
            new_seq = (func(row) for row in self[cols])
            self.append_col(new_val, new_col)
        return self

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
        new_.insert_col(0, col, '')
        return new_

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
            rows you want to count with,
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
            counter.update(Counter(self._data[title]))
        return counter

    def describe(self, level=0):
        '''describe(lvel=0, show=True) -> None
        summary the information of current sheet
        '''
        assert level in (0, 1, 2)
        from DaPy import describe
        info = dict(mins=[], maxs=[], avgs=[], stds=[], skes=[],
                    kurs=[], miss=list(map(str, self._missing)))
        for sequence in self.iter_values():
            des = describe(sequence)
            for arg, val in zip(['mins', 'maxs', 'avgs', 'stds', 'skes', 'kurs'],
                                [des.Min, des.Max, des.Mean, des.S, des.Skew, des.Kurt]):
                if val is None:
                    info[arg].append('-')
                elif isinstance(val, float):
                    info[arg].append('%.2f' % val)
                else:
                    info[arg].append(str(val))

        blank_size = [max(len(max(self.columns, key=len)), 5) + 2,
                      max(len(max(info['miss'], key=len)), 4) + 2,
                      max(len(max(info['mins'], key=len)), 3) + 2,
                      max(len(max(info['maxs'], key=len)), 3) + 2,
                      max(len(max(info['avgs'], key=len)), 4) + 2,
                      max(len(max(info['stds'], key=len)), 3) + 2, ]
        
        # Draw the title line of description
        message = '|'.join(['Title'.center(blank_size[0]),
                            'Miss'.center(blank_size[1]),
                            'Min'.center(blank_size[2]),
                            'Max'.center(blank_size[3]),
                            'Mean'.center(blank_size[4]),
                            'Std'.center(blank_size[5])])

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
            message += info['mins'][i].center(blank_size[2]) + '|'
            message += info['maxs'][i].center(blank_size[3]) + '|'
            message += info['avgs'][i].center(blank_size[4]) + '|'
            message += info['stds'][i].center(blank_size[5])
            if level >= 1:
                message += '|' + info['skes'][i].center(blank_size[6]) + '|'
                message += info['kurs'][i].center(blank_size[7])
            message += '\n'

        lenth = 5 + 2 * level + sum(blank_size)
        line = '=' * lenth
        print('1.  Structure: DaPy.SeriesSet\n' +\
              '2. Dimensions: Lines=%d | Variables=%d\n' % self.shape +\
              '3. Miss Value: %d elements\n' % sum(self.missing) +\
              'Descriptive Statistics'.center(lenth) + '\n' +\
               line + '\n' + message + line)

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
        assert self.locked, LOCK_ERROR
        pop_index = self._check_rows_index(index)
        pop_item = SeriesSet(nan=self.nan)
        for i, (title, seq) in enumerate(self.iter_items()):
            pop_item[title] = seq.pop(pop_index)
            if self._missing[i] != 0:
                self._missing[i] -= pop_item[title].count(self._nan)
        self._dim = SHEET_DIM(self._dim.Ln - len(index), self._dim.Col)
        return pop_item

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
        pop_name = self._check_columns_index(col)
        pop_data = SeriesSet(nan=self._nan)
        for title in pop_name:
            pos = self._columns.index(title)
            subset_quickly_append_col(
                subset=pop_data, 
                col=self._columns.pop(pos), 
                seq=self._data.pop(title), 
                miss=self._missing.pop(pos)
            )
            if title in self._sorted_index:
                pop_ind = self._sorted_index.pop(title)
                pop_data._sorted_index[title] = pop_ind
        self._dim = SHEET_DIM(self._dim.Ln, self._dim.Col - len(pop_name))
        return pop_data

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
            self = SeriesSet(self, nan=self.nan)

        pop_name = list(set(self._check_columns_index(index)))
        line, col = self.shape
        for title in pop_name:
            pos = self._columns.index(title)
            del self._data[title], self._missing[pos], self._columns[pos]
            if title in self._sorted_index:
                del self._sorted_index[title]

        col -= len(pop_name)
        if col == 0:
            line = 0
        self._dim = SHEET_DIM(line, col)
        return self

    def drop_row(self, index=-1, inplace=True):
        '''drop_row(index=-1, inplace=True) -> SeriesSet
        drop out rows according to the index'''
        assert self.locked, LOCK_ERROR
        if inplace is False:
            self = SeriesSet(self, nan=self.nan)

        index = self._check_rows_index(index)
        for i, seq in enumerate(self.iter_values()):
            del seq[index]
            self._missing[i] = count_nan(self._isnan, seq)
        self._dim = SHEET_DIM(self._dim.Ln - len(index), self._dim.Col)
        return self

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
            if how == 'any' and num > 0:
                pop_ind.append(i)
            elif num / lenth > how:
                pop_ind.append(i)
            elif how == 'all' and num == lenth:
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

    def extend(self, item, inplace=False):
        '''extend the current SeriesSet with records in set.

        Notes
        -----
        1. This function has been added into unit test.
        '''
        if isinstance(item, SeriesSet):
            if inplace is False:
                self = SeriesSet(self)

            other_miss = item.missing
            for title, sequence in item.iter_items():
                miss = other_miss[title]
                if title not in self.columns:
                    self._columns.append(self._check_col_new_name(title))
                    self._missing.append(self._dim.Ln + miss)
                    seq = Series(repeat(self._nan, self._dim.Ln))
                else:
                    self._missing[self.columns.index(title)] += miss
                    seq = self.data[title]
                seq.extend(sequence)
                self._data[title] = seq
            self._dim = SHEET_DIM(self.shape.Ln + item.shape.Ln, 
                                  len(self._columns))

            for i, sequence in enumerate(self.values()):
                if len(sequence) != self._dim.Ln:
                    add_miss_size = self._dim.Ln - len(sequence)
                    sequence.extend(repeat(self.nan, add_miss_size))
                    self._missing[i] += add_miss_size
            return self

        if isinstance(item, Frame):
            return self.extend(SeriesSet(item), inplace)

        if all(filter(is_iter, item)):
            return self.extend(SeriesSet(item, self._columns), inplace)

        raise TypeError('could not extend a single value only.')

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
        None

        Example
        -------
        >>> data = dp.SeriesSet({'A': [dp.nan, 1, 2, 3, dp.nan, dp.nan,  6]},
                                nan=dp.nan)
        >>> data.fillna(method='linear')
        >>> data
        A: <0.0, 1, 2, 3, 4.0, 5.0, 6>
        '''
        col = self._check_columns_index(col)
        isnan_fun = self._isnan
        if limit is None:
            limit = self.shape.Ln
        assert limit >= 1, 'fill with at least 1 missing value, not limit=%s' % limit
        assert method in ('linear', 'polynomial', 'quadratic', None)
        if method is None:
            self._fillna_value(fill_with, col, isnan_fun, limit)
        else:
            func = simple_linear_reg
            self._fillna_simple_function(col, isnan_fun, limit, func)

    def _fillna_value(self, fill_with, col, _isnan, all_limit):
        err = '`fill_with` must be a value'
        assert isinstance(fill_with, (dict,) + VALUE_TYPE), err
        if isinstance(fill_with, dict) is False:
            fill_with = dict(zip(col, repeat(fill_with)))

        for key, fill_val in fill_with.items():
            limit = all_limit
            key_index = self.columns.index(key)
            if key in col and self._missing[key_index] != 0:
                sequence = self._data[key]
                for i, val in enumerate(sequence):
                    if _isnan(val) is True:
                        sequence[i] = fill_val
                        self._missing[key_index] -= 1
                        limit -= 1
                        if limit == 0:
                            break

    def _fillna_simple_function(self, col, _isnan, all_limit, func):
        '''establish a linear model to predict the missing value

        This function will predict the missing value with a linear model,
        which is established by the arounding records.
        '''
        for key in col:
            limit = all_limit
            key_index = self.columns.index(key)
            if self._missing[key_index] == 0:
                continue

            seq = self._data[key]
            for i, value in enumerate(seq):
                if _isnan(value) is False and _isnan(seq[i + 1]) is False:
                    break

            if i != 0:
                xlist = []
                for value in seq[i:2 * i + 1]:
                    if _isnan(value) is True:
                        break
                    xlist.append(value)
                slope, constant = func(xlist, range(i, 2 * i + 1))
                for ind in xrange(0, i):
                    seq[ind] = slope * ind + constant
                    self._missing[key_index] -= 1
                    limit -= 1
                    if limit == 0:
                        break

            start = None
            for stop, value in enumerate(seq):
                if limit == 0:
                    break
                    
                if _isnan(value) is True:
                    if start is None:
                        start = stop

                elif start is not None:
                    empty_length = stop - start
                    back = max(start - empty_length, 0)
                    fore = min(stop + empty_length, len(seq))
                    left_length = start - back + 2
                    ylist = seq[back:start] + seq[stop:fore]
                    xlist = range(left_length) +\
                            range(left_length + empty_length - 1,
                                  len(ylist) + empty_length + 1)

                    slope, constant = func(*zip(*[
                        (_x, _y) for _x, _y in zip(xlist, ylist) \
                                if _isnan(_y) is False]))
                    left_length += 2
                    for ind, _ in enumerate(xrange(left_length, left_length + empty_length), start):
                        seq[ind] = slope * _ + constant
                        self._missing[key_index] -= 1
                        if limit == 0:
                            break
                    start = None

    @classmethod
    def from_file(cls, addr, **kwrd):
        '''read dataset from .txt or .csv file.

        Parameters
        ----------
        addr : str
            address of source file.

        first_line : int (default=1)
            the first line with data.

        nan : str (default="NA")
            the symbol of missing value in csv file.

        title_line : int (default=0)
            the line with title, rules design as follow:
            -1 -> there is no title inside;
            >=0 -> the titleline.

        sep : str (default=",")
            the delimiter symbol inside.

        prefer_type : type-object (default=float):
            int -> transfer any possible values into int
            float -> transfer any possible values into float
            str -> keep all values in str type
            datetime -> transfer any possible values into datetime-object
            bool -> transfer any possible values into bool

        dtypes : Type name or dict of columns (default=None):
            use one type to
        '''
        sheet = cls()
        first_line = kwrd.get('first_line', 1)
        title_line = kwrd.get('title_line', 0)
        columns = list(kwrd.get('columns', []))
        nan = kwrd.get('nan', set(['nan', '?', '??', '', ' ', 'NA', 'None']))
        if is_value(nan):
            nan = set([nan])
        sep = kwrd.get('sep', ',')
        prefer_type = kwrd.get('prefer_type', None)
        col_types = kwrd.get('dtypes', [])
        param, temp_data, miss = {'mode': 'rU'}, (), ()
        if PYTHON3:
            param['encoding'] = kwrd.get('encoding', None)
            param['file'] = addr
            param['newline'] = kwrd.get('newline', None)
        if PYTHON2:
            param['name'] = addr

        assert first_line > title_line, '`first_line` must be larger than `title_line`'
        assert all(map(is_str, columns)), 'column names must be `str`'

        with open(**param) as file_:
            # skip the rows which are unexpected to read
            for i in xrange(first_line):
                line = file_.readline()
                if i == title_line:
                    # setup the title line
                    columns = tuple(map(str.strip, split(line, sep)))

            # begin to read the data
            for line in file_:  # iter row
                for missing, seq, transfer, value in zip_longest(miss, 
                                                           temp_data, 
                                                           col_types, 
                                                           split(line.strip(), sep)):  # iter value


                    if value in nan:
                        seq.append(sheet.nan)
                        missing.append(1)
                        continue
                    try:
                        seq.append(transfer(value))
                    except ValueError:
                        # different types of data in the same column
                        seq.append(auto_str2value(value))
                    except:
                        transed = auto_str2value(value, prefer_type)
                        missing = []
                        miss += (missing,)
                        if transed in nan:
                            transed = sheet.nan
                            missing.append([1])
                        else:
                            tranfer_n = str(transed.__class__).split()[1][1:-2].split('.')[0]
                            col_types.append(fast_str2value[tranfer_n])

                        if not temp_data:
                            temp_data += (Series([transed]),)
                        else:
                            missed = len(temp_data[0]) - 1
                            missing.append(missed)
                            temp_data += (Series(chain(repeat(sheet.nan, missed), 
                                [transed])),)

        sheet._dim = SHEET_DIM(len(temp_data[0]), len(temp_data))
        sheet._init_col_name(columns)
        for i, (missing, seq, col) in enumerate(zip(miss, temp_data, sheet.columns)):
            add_space = sheet.shape.Ln - len(seq)
            seq.extend(repeat(sheet.nan, add_space))
            sheet._missing.append(add_space + sum(missing))
            sheet.data[col] = seq
        return sheet

    def get_categories(self, cols, cut_points, group_name,
                       boundary=(False, True), inplace=False):
        '''transfer numerical variables into categorical variable'''
        from DaPy.operation import get_categories
        cols = self._check_columns_index(cols)
        if inplace is False:
            self = SeriesSet(nan=self.nan)

        for col in cols:
            categories = get_categories(
                self[col], cut_points, group_name, boundary)
            self.append_col(categories, '%s_category' % col)

    def get_date_label(self, cols, daytime=True,
                       weekend=True, season=True, inplace=False):
        '''transfer a datetime object into categorical variable'''
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
            if daytime is True:
                new = self._check_col_new_name('%s_daytime' % col)
                self[new] = date['hour']
                self.update('%s in set([23] + range(7))' % new, {new: 'latenight'})
                self.update('%s in set(range(7, 11))' % new, {new: 'morning'})
                self.update('%s in set(range(11, 14))' % new, {new: 'noon'})
                self.update('%s in set(range(14, 18))' % new, {new: 'afternoon'})
                self.update('%s in set(range(18, 23))' % new, {new: 'night'})

            if weekend is True:
                new = self._check_col_new_name('%s_weekend' % col)
                self[new] = date['week']
                self.replace([0, 1, 2, 3, 4, 5, 6], 
                             [1, 0, 0, 0, 0, 0, 1], 
                             col=new)

            if season is True:
                new = self._check_col_new_name('%s_season' % col)
                self[new] = date['month']
                self.update('%s in set([3, 4, 5])' % new, {new: 'spring'})
                self.update('%s in set([6, 7, 8])' % new, {new: 'summer'})
                self.update('%s in set([9, 10, 11])' % new, {new: 'autumn'})
                self.update('%s in set([12, 1, 2])' % new, {new: 'winter'})
        return self

    def get_dummies(self, cols=None, value=1, inplace=False):
        '''Convert categorical variable into dummy variables

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
        from DaPy import get_dummies
        if inplace is False:
            self = SeriesSet(self, nan=self.nan)

        cols = self._check_columns_index(cols)
        for title in cols:
            dummies = get_dummies(self._data[title], value, 'set')
            dummies.columns = [title + '=' + _ for _ in dummies.columns]
            self.join(dummies, inplace=True)
        return self

    def get_interactions(self, n_var=2, cols=None, inplace=False):
        def create_title(title, count):
            n_title = count[title]
            if n_title != 1:
                return '*%s^%d' % (title, n_title)
            return title
        cols = self._check_columns_index(cols)
        operate_with = SeriesSet(nan=self.nan)

        for combine in combinations(cols, n_var):
            combine.sort()
            count = Counter(combine)
            title = create_title(combine[0], count)
            seq = self._data[combine[0]]
            for col in combine[1:]:
                seq *= self._data[col]
                if col not in title:
                    title += '*' + create_title(col, count)

        if inplace is False:
            return operate_with
        return self.join(operate_with, inplace=True)

    def get_ranks(self, cols=None, duplicate='mean', inplace=False):
        from DaPy.operation import get_ranks
        cols = self._check_columns_index(cols)
        if inplace is False:
            operate_with = SeriesSet(nan=self.nan)
        else:
            operate_with = self

        for col in cols:
            rank_col = get_ranks(self[col], duplicate)
            operate_with.append_col(rank_col, '%s_rank' % col)
        return operate_with

    def items(self):
        for column in self.columns:
            yield column, Series(self._data[column])

    def iter_items(self):
        for column in self.columns:
            yield column, self._data[column]

    def iter_rows(self):
        for row in zip(*(self._data[col] for col in self.columns)):
            yield row

    def iter_values(self):
        for col in self.columns:
            yield self._data[col]

    def _iloc(self, subset, indexs):
        if is_iter(indexs) is False:
            indexs = tuple(indexs)
        for miss, (key, sequence) in zip(self._missing, self.iter_items()):
            seq = sequence[indexs]
            if isinstance(seq, Series) is False:
                seq = Series([seq, ])
            if miss != 0:
                miss = count_nan(self._isnan, seq)
            subset_quickly_append_col(subset, key, seq, miss)
        return subset

    def iloc(self, indexs):
        indexs = self._check_rows_index(indexs)
        return self._iloc(SeriesSet(nan=self.nan), indexs)

    def insert_row(self, index, item):
        '''insert a record to the frame, position in <index>

        Notes
        -----
        1. This function has been added into unit test.
        '''
        item = self._add_row(item)
        for value, seq in zip(item, self.iter_values()):
            seq.insert(index, value)

    def insert_col(self, index, series, variable_name=None):
        '''insert a series of data to the frame, position in <index>

        Notes
        -----
        1. This function has been added into unit test.
        '''
        variable_name = self._check_col_new_name(variable_name)
        miss, series = self._check_sequence(series, self._dim.Ln)

        empty_size = len(series) - self.shape.Ln
        if empty_size > 0:
            empty_seq = [self.nan] * empty_size
            for i, sequence in enumerate(self.iter_values()):
                sequence.extend(empty_seq)
                self._missing[i] += empty_size

        self._columns.insert(index, variable_name)
        self._dim = SHEET_DIM(len(series), self._dim.Col + 1)
        self._missing.insert(index, miss)
        self._data[variable_name] = series

    def join(self, other, inplace=False):
        '''right join another sheet to the current sheet

        Notes
        -----
        1. This function has been added into unit test.
        '''
        error = 'can  not join a value to the dataset.'
        assert is_value(other) is False, error
        error = "can't join empty object, given %s" % other
        assert (hasattr(other, 'shape') and other.shape[0] != 0) or\
                not other, error
        if inplace is False:
            self = SeriesSet(self, nan=self.nan)

        if isinstance(other, SeriesSet):
            bias = other.shape.Ln - self.shape.Ln
            if bias > 0:
                for i, title in enumerate(self._columns):
                    self._data[title].extend(repeat(self.nan, bias))
                    self._missing[i] += bias
            for title, seq in other.iter_items():
                title = self._check_col_new_name(title)
                miss, seq = self._check_sequence(seq, self._dim.Ln)
                self._missing.append(miss)
                self._data[title] = seq
                self._columns.append(title)
            self._dim = SHEET_DIM(self.shape.Ln + bias, 
                                  len(self._columns))

        elif isinstance(other, Frame):
            return self.join(SeriesSet(other), inplace)

        elif all(map(is_iter, other)):
            new_col = [title + '_1' for title in self._columns]
            return self.join(SeriesSet(other, new_col), inplace)

        elif isinstance(other, Series) or all(map(is_value, other)):
            return self.append_col(other, inplace)

        else:
            raise TypeError('could not extend a single value only.')
        return self

    def keys(self):
        return self._data.keys()

    def normalized(self, process='NORMAL', cols=None, inplace=True, **attr):
        assert process in ('NORMAL', 'STANDAR', 'LOG', 'BOX-COX')
        cols = self._check_columns_index(cols)
        if inplace is False:
            self = SeriesSet(self)

        from DaPy import describe, log, boxcox
        for title in cols:
            if process in ('NORMAL', 'STANDAR'):
                if not attr:
                    statis = describe(self._data[title])
                if process == 'NORMAL':
                    center = float(attr.get('min', statis.Min))
                    var = float(attr.get('range', statis.Range))
                elif process == 'STANDAR':
                    center = float(attr.get('mean', statis.Mean))
                    var = float(attr.get('std', statis.Sn))
                assert var != 0, 'range or std of `%s` is 0 ' % title
                self.apply(
                    lambda val: (val - center) / var,
                    col=title,
                    inplace=True,
                    axis=1)

            elif process == 'BOX-COX':
                func = lambda val: boxcox(
                    val, 
                    attr.get('lamda', 1), 
                    attr.get('a', 0),
                    attr.get('k', 1))
                self.apply(func, col=title, inplace=True, axis=1)

            elif process == 'LOG':
                self.apply(
                    lambda val: log(val, attr.get('base', 2.71828183)),
                    col=title,
                    inplace=True,
                    axis=1)
        return self

    def merge(self, other, how='inner', self_on=0, right_on=0):
        '''right join another sheet and automatically arranged by key columns

        Notes
        -----
        1. This function has been added into unit test.
        '''
        assert how in ('inner', 'outer', 'left', 'right')
        if isinstance(other, SeriesSet) is False:
            other = SeriesSet(other)

        self_on = self._check_columns_index(self_on)
        right_on = other._check_columns_index(right_on)
        assert len(self_on) == len(
            right_on) == 1, 'only support 1 matching index'
        self_on, right_on = self_on[0], right_on[0]

        # match the records according to the index
        joined = SeriesSet(nan=self.nan)
        if how == 'left':
            return left_join(self, other, self_on, right_on, joined)

        if how == 'right':
            return left_join(other, self, right_on, self_on, joined)

        if how == 'inner':
            return inner_join(self, other, self_on, right_on, joined)

        # if how == 'outer':
        return outer_join(self, other, self_on, right_on, joined)

    def reverse(self, axis=0):
        assert axis in (0, 1)
        if axis == 1:
            self._columns.reverse()
            self._missing.reverse()
        else:
            for sequence in self._data.values():
                sequence.reverse()

    def update(self, where, set_value=None, **set_values):
        if callable(where) is False:
            where = self._trans_where(where, axis=0)
        if set_value is None:
            set_value = {}
        assert isinstance(set_value, dict)
        set_value.update(set_values)
        assert set_value, '`set_value` are empty'
        for key, exp in set_value.items():
            if callable(exp) is False:
                exp = str(exp)
                if exp.isalpha() is True:
                    exp = '"%s"' % exp
                set_value[key] = self._trans_where(exp, axis=0)

        for index in self._where_by_rows(where, limit='all'):
            row = Row(self, index)
            for key, value in set_value.items():
                row[key] = value(row)

    def shuffle(self):
        new_index = range(self._dim.Ln)
        shuffles(new_index)
        self._arrange_by_index(new_index)

    def values(self):
        for col in self.columns:
            yield self._data[col]


class Frame(BaseSheet):
    '''Maintains the data as records.
    '''

#     def __init__(self, frame=None, columns=None, nan=None):
#         self._data = []
#         BaseSheet.__init__(self, frame, columns, nan)

#     @property
#     def info(self):
#         new_m_v = map(str, self._missing)
#         max_n = len(max(self._columns, key=len))

#         info = ''
#         for i in xrange(self._dim.Col):
#             info += ' ' * 15
#             info += self._columns[i].center(max_n) + '| '
#             info += ' ' + new_m_v[i] + '\n'

#         print('1.  Structure: DaPy.Frame\n' +
#               '2. Dimensions: Ln=%d | Col=%d\n' % self._dim +
#               '3. Miss Value: %d elements\n' % sum(self._missing) +
#               '4.    Columns: ' + 'Title'.center(max_n) + '|' +
#               '  Miss\n' + info)

#     @property
#     def T(self):
#         return Frame(self.iter_values(), None, self.nan)

#     def _init_col(self, obj, columns):
#         if columns is None:
#             columns = copy(obj._columns)
#         self._data = [list(record) for record in zip(*list(obj.values()))]
#         self._missing = copy(obj._missing)
#         self._dim = SHEET_DIM(obj._dim.Ln, obj._dim.Col)
#         self._init_col_name(columns)

#     def _init_frame(self, frame, columns):
#         if columns is None:
#             columns = copy(obj._columns)
#         self._data = deepcopy(frame._data)
#         self._dim = copy(frame._dim)
#         self._init_col_name(columns)
#         self._missing = copy(frame._missing)

#     def _init_dict(self, frame, columns):
#         if columns is None:
#             columns = list(obj.keys())
#         frame = copy(frame)
#         self._dim = SHEET_DIM(max(map(len, frame.values())), len(frame))
#         self._missing = [0] * self._dim.Col
#         self._init_col_name(columns)
#         for i, (title, col) in enumerate(frame.items()):
#             miss, sequence = self._check_sequence(col, self._dim.Ln)
#             frame[title] = sequence
#             self._missing[i] += miss
#         self._data = [list(record) for record in zip(*frame.values())]

#     def _init_like_table(self, frame, columns):
#         self._data = map(list, frame)
#         dim_Col, dim_Ln = len(max(self._data, key=len)), len(frame)
#         self._dim = SHEET_DIM(dim_Ln, dim_Col)
#         self._missing = [0] * self._dim.Col

#         for i, item in enumerate(self._data):
#             if len(item) < dim_Col:
#                 item.extend([self._nan] * (dim_Col - len(item)))
#             for j, value in enumerate(item):
#                 if value == self.nan or value is self.nan:
#                     self._missing[j] = self._missing[j] + 1
#         self._init_col_name(columns)

#     def _init_like_seq(self, frame, columns):
#         self._data = [[value, ] for value in frame]
#         self._dim = SHEET_DIM(len(frame), 1)
#         self._init_col_name(columns)
#         self._missing.append(self._check_sequence(frame, len(frame))[0])

#     def __repr__(self):
#         return self.show(30)

#     def _getslice_col(self, i, j):
#         new_data = [record[i: j + 1] for record in self._data]
#         return Frame(new_data, self._columns[i: j + 1], self._nan)

#     def _getslice_ln(self, i, j, k):
#         return Frame(self._data[i:j:k], self._columns, self._nan)

#     def __getitem__(self, interval):
#         if isinstance(interval, int):
#             return Row(self, interval)

#         elif isinstance(interval, slice):
#             return self.__getslice__(interval)

#         elif is_str(interval):
#             col = self._columns.index(interval)
#             return [item[col] for item in self._data]

#         elif isinstance(interval, (tuple, list)):
#             return_obj = Frame()
#             return self._getitem_by_tuple(interval, return_obj)

#         else:
#             raise TypeError('item must be represented as slice, int, str.')

#     def __iter__(self):
#         for i in xrange(self._dim.Ln):
#             yield Row(self, i)

#     def append_row(self, item):
#         '''append a new record to the Frame tail
#         '''
#         item = self._add_row(item)
#         self._data.append(item)

#     def append_col(self, series, variable_name=None):
#         '''append a new variable to the current records tail
#         '''
#         miss, series = self._check_sequence(series, self._dim.Ln)
#         size = len(series) - self._dim.Ln
#         if size > 0:
#             self._missing = [m + size for m in self._missing]
#             self._data.extend(
#                 [[self._nan] * self._dim.Col for i in xrange(size)])

#         self._missing.append(miss)
#         for record, element in zip(self._data, series):
#             record.append(element)
#         self._columns.append(self._check_col_new_name(variable_name))
#         self._dim = SHEET_DIM(max(self._dim.Ln, len(series)), self._dim.Col + 1)
#         assert len(self._missing) == self._dim.Col == len(self.columns)

#     def count(self, X, point1=None, point2=None):
#         if is_value(X):
#             X = (X,)
#         counter = Counter()
#         L1, C1, L2, C2 = self._check_area(point1, point2)

#         for record in self._data[L1:L2 + 1]:
#             for value in record[C1:C2 + 1]:
#                 if value in X:
#                     counter[value] += 1

#         if len(X) == 1:
#             return counter[X[0]]
#         return dict(counter)

#     def extend(self, other, inplace=False):
#         if isinstance(other, Frame):
#             if inplace is False:
#                 self = SeriesSet(Frame)
#             new_title = 0
#             for title in other._columns:
#                 if title not in self._columns:
#                     self._columns.append(title)
#                     new_title += 1

#             for record in self._data:
#                 record.extend([self._nan] * new_title)

#             extend_part = [[self._nan] * len(self._columns)
#                            for i in xrange(len(other))]
#             new_title_index = [self._columns.index(title)
#                                for title in other._columns]
#             self._dim = SHEET_DIM(len(self) + len(other), len(self._columns))
#             self._missing.extend([self._dim.Ln] * new_title)

#             for i, record in enumerate(other._data):
#                 for j, value in zip(new_title_index, record):
#                     if value == other._nan:
#                         value = self._nan
#                     extend_part[i][j] = value
#             self._data.extend(extend_part)
#             return self

#         elif isinstance(other, SeriesSet):
#             return self.extend(Frame(other), inplace)

#         else:
#             return self.extend(Frame(other, self._columns), inplace)

#     def join(self, other, inplace=False):
#         if isinstance(other, Frame):
#             if inplace is False:
#                 self = Frame(self)
#             for title in other._columns:
#                 self._columns.append(self._check_col_new_name(title))
#             self._missing.extend(other._missing)

#             for i, record in enumerate(other._data):
#                 if i < self._dim.Ln:
#                     current_record = self._data[i]
#                 else:
#                     current_record = [self._nan] * self._dim.Col
#                     self._data.append(current_record)
#                 for value in record:
#                     if value == other.nan:
#                         value = self._nan
#                     current_record.append(value)
#             if i < self._dim.Ln:
#                 for record in self._data[i + 1:]:
#                     record.extend([self._nan] * other.shape.Col)
#             self._dim = SHEET_DIM(len(self._data), len(self._columns))
#             return self

#         else:
#             self.join(Frame(other, nan=self.nan), inplace)

#     def insert_row(self, index, item):
#         '''insert a new record to the frame with position `index`
#         '''
#         item = self._add_row(item)
#         self._data.insert(index, item)

#     def insert_col(self, index, series, variable_name=None):
#         '''insert a new variable to the current records in position `index`
#         '''
#         miss, series = self._check_sequence(series)

#         size = len(series) - self._dim.Ln
#         if size > 0:
#             for i in xrange(self._dim.Col):
#                 self._missing[i] += size
#             self._data.extend([[self._nan] * self._dim.Col
#                                for i in xrange(size)])

#         self._missing.insert(index, miss)
#         for i, element in enumerate(series):
#             self._data[i].insert(index, element)
#         self._columns.insert(index, self._check_col_new_name(variable_name))
#         self._dim = SHEET_DIM(max(self._dim.Ln, size), self._dim.Col + 1)

#     def items(self):
#         for i, sequence in enumerate(zip(*self._data)):
#             yield self._columns[i], list(sequence)

#     def keys(self):
#         return self._columns

#     def pop_row(self, pos=-1):
#         '''pop(remove & return) a record from the Frame
#         '''
#         err = 'an int or ints in list is required.'
#         assert isinstance(pos, (int, list, tuple)), err
#         if isinstance(pos, int):
#             pos = [pos, ]
#         pos = sorted(pos, reverse=True)
#         pop_item = Frame([self._data.pop(pos_)
#                           for pos_ in pos], list(self._columns))
#         self._dim = SHEET_DIM(self._dim.Ln - len(pos), self._dim.Col)
#         self._missing = map(
#             lambda x, y: x - y,
#             self._missing,
#             pop_item._missing)
#         return pop_item

#     def pop_col(self, pos=-1):
#         '''pop(remove & return) a series from the Frame
#         '''
#         pop_name = self._check_columns_index(pos)
#         for name in pop_name:
#             index = self._columns.index(name)
#             self._columns.pop(index)
#             self._missing.pop(index)

#         pop_data = [[] for i in xrange(len(pop_name))]
#         new_data = [0] * self._dim.Ln
#         for j, record in enumerate(self._data):
#             line = []
#             for i, value in enumerate(record):
#                 if i in pop_name:
#                     pop_data[pop_name.index(i)].append(value)
#                 else:
#                     line.append(value)
#             new_data[j] = line

#         self._dim = SHEET_DIM(self._dim.Ln, self._dim.Col - len(pos))
#         self._data = new_data
#         return SeriesSet(dict(zip(pop_name, pop_data)))

#     def dropna(self, axis='LINE'):
#         '''pop all records that maintains miss value while axis is `LINE` or
#         pop all variables that maintains miss value while axis is `COL`
#         '''
#         pops = []
#         if str(axis).upper() in ('0', 'LINE'):
#             for i, record in enumerate(self._data):
#                 if self._nan in record:
#                     pops.append(i)

#         if str(axis).upper() in ('1', 'COL'):
#             for i, sequence in enumerate(zip(*self._data)):
#                 if self._nan in sequence:
#                     pops.append(self._columns[i])

#         if len(pops) != 0:
#             self.__delitem__(pops)

#     def from_file(self, addr, **kwrd):
#         '''read dataset from csv or txt file.
#         '''
#         f = open(addr, 'r')
#         freader, col_types, nan, prefer = self._check_read_text(f, **kwrd)
#         self._data = [0] * self.shape.Ln
#         try:
#             for line, record in enumerate(freader):
#                 line = tuple(
#                     self._trans_str2val(
#                         i,
#                         v,
#                         col_types,
#                         nan,
#                         prefer) for i,
#                     v in enumerate(record))
#                 if len(line) != self._dim.Col:
#                     line = list(chain(line, (self._nan) *
#                                       (self._dim.Col - len(line))))
#                 self._data[line] = line
#         except MemoryError:
#             self._dim = SHEET_DIM(len(self._data), self._dim.Col)
#             warn('since the limitation of memory, DaPy can not read the' +
#                  ' whole file.')
#         finally:
#             f.close()

#     def reverse(self):
#         self._data.reverse()

#     def shuffle(self):
#         shuffles(self._data)

#     def _values(self):
#         for sequence in zip(*self._data._data):
#             yield list(sequence)

#     def values(self):
#         for sequence in zip(*self._data):
#             yield Series(sequence)
