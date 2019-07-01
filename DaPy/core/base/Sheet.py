from collections import Counter, OrderedDict, namedtuple, deque
from copy import copy, deepcopy
from datetime import datetime
from random import shuffle as shuffles
from re import compile, search, sub, findall
from itertools import groupby as Groupby, repeat, chain
from operator import itemgetter, attrgetter
from array import array

from .constant import VALUE_TYPE, STR_TYPE, MATH_TYPE, SEQ_TYPE, DUPLICATE_KEEP
from .constant import pickle, nan, inf, PYTHON3, PYTHON2, SHEET_DIM
from .IndexArray import SortedIndex
from .Row import Row
from .Series import Series
from .utils import is_seq, is_iter, is_math, is_value, is_empty, isnan, is_str
from .utils import pickle, split, strip, xrange, zip_longest
from .utils import auto_plus_one, argsort, fast_str2value, auto_str2value, count_nan
from .utils.utils_join_table import left_join, outer_join, inner_join

__all__ = ['SeriesSet', 'Frame']
dims = SHEET_DIM
PATTERN_AND_OR = compile(r'\sand\s|\sor\s')
PATTERN_COMBINE = compile(r'[(](.*?)(\sand\s|\sor\s)(.*?)[)]')
PATTERN_RECOMBINE = compile(r'[)](\sand\s|\sor\s)[(]')
PATTERN_COLUMN = r'[(|\s]{0,1}%s[\s|)]{0,1}'

PATTERN_EQUAL = compile(r'(.*?)(!=|==)(.*?)')
PATTERN_LESS = compile(r'(.*?)(<=|<)(.*?)')
PATTERN_GREAT = compile(r'(.*?)(>=|>)(.*?)')
PATTERN_BETWEEN1 = compile(r'(.+?)(>=|>)(.+?)(>=|>)(.+?)')
PATTERN_BETWEEN2 = compile(r'(.+?)(>=|>)(.+?)(>=|>)(.+?)')

class BaseSheet(object):
    '''
    Attributes

    shape : sheet(Ln, Col)
        a two dimensional span of this sheet.

    nan : value (default=Nan)
        the symbol represented miss value in current seriesset.

    columns : str in list
        names for each feature

    data : dict / list in list
        the object contains all the data by columns or row.

    missing : int in list
        the number of missing value in each column.
    '''

    def __init__(self, obj=None, columns=None, nan=nan):
        self._missing = []
        self._nan = nan
        self._isnan = self._init_nan_func()
        self._sorted_index = {}

        if isinstance(obj, SeriesSet):
            if columns is None:
                columns = copy(obj._columns)
            self._init_col(obj, columns)

        elif isinstance(obj, Frame):
            if columns is None:
                columns = copy(obj._columns)
            self._init_frame(obj, columns)
            
        elif obj is None or is_empty(obj):
            self._dim, self._columns = dims(0, 0), []
            if columns is not None:
                if is_str(columns):
                    columns = [columns,]
                for name in columns:
                    self.append_col([], name)
                    
        elif isinstance(obj, (dict, OrderedDict)) or hasattr(obj, 'items'):
            if columns is None:
                columns = list(obj.keys())
            self._init_dict(obj, columns)

        elif isinstance(obj, Series) or (is_seq(obj) and all(map(is_value, obj))):
            self._init_like_seq(obj, columns)

        elif is_seq(obj) and all(map(is_iter, obj)):
            self._init_like_table(obj, columns)

        elif hasattr(obj, '__iter__'):
            self._init_like_table(list(obj), columns)

        else:
            raise TypeError("sheet structure does not support %s." % type(obj))
        
    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._dim

    @property
    def columns(self):
        return copy(self._columns)

    @columns.setter
    def columns(self, item):
        old_col = self.columns
        if old_col == [] and item != []:
            self._dim = dims(0, len(item))
            self._missing = [0] * len(item)
            old_col = item
        self._init_col_name(item)
        if isinstance(self._data, dict):
            new_data, old_data = dict(), self._data
            for old, new in zip(old_col, self.columns):
                new_data[new] = old_data.get(old, [])
            self._data = new_data

    @property
    def T(self):
        if isinstance(self.data, dict):
            return SeriesSet(self.iter_values(), None, self.nan)
        return Frame(self.iter_values(), None, self.nan)

    @property
    def nan(self):
        return self._nan

    @nan.setter
    def nan(self, item):
        assert is_value(item), 'sheet.nan should be a value type'
        if self.missing != 0:
            for missing, sequence in zip(self._missing, self.iter_values()):
                if missing == 0:
                    continue
                for i, value in enumerate(sequence):
                    if value == nan:
                        sequence[i] = item
        self._nan = item
        self._isnan = self._init_nan_func()
        
    @property
    def missing(self):
        return SeriesSet(OrderedDict(zip(self.columns, self._missing)))[0]

    def __getattr__(self, name):
        if name in self._columns:
            return self.__getitem__(name)
        raise AttributeError("Sheet object has no attribute '%s'" % name)

    def __len__(self):
        return self._dim.Ln

    def __contains__(self, cmp_):
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
        if is_value(other):
            temp = SeriesSet()
            for title, sequence in self.iter_items():
                temp[title] = [True if v == other else False for v in sequence]
            return temp

    def __setitem__(self, key, value):
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
            
        else:
            raise TypeError('only can set one record or one column each time.')

    def _getitem_by_tuple(self, interval, obj):
        ERROR = "don't get subset by columns and index at the "+\
                "same time. Try: sheet['A':'B'][3:10] or sheet[3:10]['A':'B']"
        if is_seq(interval) and len(interval) == 2 and isinstance(interval[0], slice):
            t1, t2 = interval
            if t1.start is None and t1.stop is None:
                if isinstance(t2, slice) and \
                   isinstance(t2.start, int) and isinstance(t2.stop, int):
                    t1_ = (self.columns[t1.start], self.columns[t1.stop])
                    raise SyntaxError('do you mean sheet["%s":"%s"]' % t1_ +
                                      ' or sheet[%s:%s]?' % (t1.start, t1.stop))
                if isinstance(t2, int):
                    raise SyntaxError('do you mean sheet["%s"] ' % self.columns[t2]+
                                      'or sheet[%s]' % t2)

        args, int_args, slc_args = [], [], []
        for arg in interval:
            if isinstance(arg, slice):
                slc_args.append(arg)
                for value in (arg.start, arg.stop):
                    if value is not None:
                        args.append(value)
                
            else:
                args.append(arg)
                if isinstance(arg, int):
                    int_args.append(arg)

        subcol = all(map(lambda x: is_str(x), args))
        subrow = all(map(lambda x: isinstance(x, int), args))
        assert subcol or subrow, ERROR
        
        if subcol is True:
            for arg in interval:
                if is_str(arg):
                    seq = self.data[arg]
                    miss = self._missing[self._columns.index(arg)]
                    obj = self._subset_quickly_append_col(obj, arg, seq, miss)

                elif isinstance(arg, slice):
                    start, stop = arg.start, arg.stop
                    if start is None:
                        start = self._columns[0]
                    if stop is None:
                        stop = self._columns[-1]
                    assert start in self.columns, "'%s' is not a sheet column name" % start
                    assert stop in self.columns, "'%s' is not a sheet column name" % stop
                    start, stop = self._columns.index(start), self._columns.index(stop)
                    for arg in self._columns[start:stop + 1]:
                        if arg not in obj.columns:
                            index = self._columns.index(arg)
                            miss = self._missing[index]
                            seq = self.data[arg]
                            obj = self._subset_quickly_append_col(obj, arg, seq, miss)
                else:
                    raise TypeError('bad statement as sheet[]' % arg)

        if subrow is True:
            subset = self._iloc(obj, int_args)
            for arg in slc_args:
                start, stop = arg.start, arg.stop
                subset.extend(self.__getslice__(start, stop))
        return obj

    def __delitem__(self, key):
        assert isinstance(key, tuple([STR_TYPE] + [int, list, tuple, slice]))
        if isinstance(key, int):
            self.drop_row(key)

        if is_str(key):
            self.drop_col(key)

        if isinstance(key, (list, tuple)):
            int_keys = list(filter(is_math, key))
            str_keys = list(filter(lambda x: isinstance(x, str), key))
            if str_keys != []:
                self.drop_col(str_keys)
            if int_keys != []:
                self.drop_row(int_keys)

    def __getslice__(self, start, stop, step=1):
        if start in self._columns or stop in self._columns:
            return self._getslice_col(*self._check_slice_pos_col(start, stop))
        if isinstance(start, int) or isinstance(stop, int):
            return self._getslice_ln(start, stop, step)
        raise TypeError('bad expression as [%s:%s]' % (start, stop))

    def __getstate__(self):
        instance = self.__dict__.copy()
        instance['_dim'] = tuple(self._dim)
        instance['_data'] = self._data
        return instance

    def __setstate__(self, dict):
        '''load this object from a stream file'''
        self._dim = dims(*dict['_dim'])
        self._columns = dict['_columns']
        self._missing = dict['_missing']
        self._nan = dict['_nan']
        self._data = dict['_data']

    def _init_col_name(self, columns):
        if is_str(columns) and self._dim.Col == 1:
            self._columns = [columns,]
        elif is_str(columns) and self._dim.Col != 1:
            self._columns = [columns + '_%d' % i for i in xrange(self._dim.Col)]
        elif columns is None or str(columns).strip() == '':
            self._columns = ['C_%d' % i for i in xrange(self._dim.Col)]
        elif is_iter(columns) is True:
            self._columns, columns = [], list(columns)
            columns.extend(['C_%d' % i for i in xrange(self._dim.Col - len(columns))])
            for col in columns[:self._dim.Col]:
                self._columns.append(self._check_col_new_name(col))
            for i in xrange(self._dim.Col - len(self._columns)):
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
                    where = where[: index] + ' ___x___' + where[index: ]
                    index += bias
            where = 'lambda ___x___: ___x___ ' + where

        return eval(where)

    def _add_row(self, item):
        if is_value(item):
            item = [item] * self._dim.Col
            
        lenth_bias = len(item) - self._dim.Col
        if lenth_bias > 0:
            if self.shape.Ln == 0:
                for i in xrange(lenth_bias):
                    self.append_col([])
            else:
                self.join([[self.nan] * lenth_bias for i in xrange(self.shape.Ln)], inplace=True)

        mv, item = self._check_sequence_type(item, self.shape.Col)
        if mv != 0:
            for i, value in enumerate(item):
                if self._isnan(value):
                    self._missing[i] += 1
        self._dim = dims(self._dim.Ln + 1, max(self._dim.Col, len(item)))
        return item

    def _check_sequence_type(self, series, size):
        '''check the shape of the sequence and fill nan if not long enough
        
        if sequence is record, size = self.shape.Ln;
        else size = self.shape.Col
        '''
        if is_value(series):
            return 0, [series] * size

        assert is_iter(series), "append item should be an iterable object"
        if is_seq(series) is False:
            series = list(series)
        if len(series) < size:
            series = chain(series,  [self._nan] * (size - len(series)))
        series = Series(series)   
        return count_nan(self._isnan, series), series

    def _check_remove_index(self, index):
        assert isinstance(index, (int, list, tuple, slice)), 'an int or ints in list is required.'
        if isinstance(index, int):
            index = [index,]
        return sorted(set(index), reverse=True)

    def _check_replace_condition(self, col, condition, new_value):
        col == self._check_columns_index(col)[0]
        if not is_str(condition) and not callable(condition):
            raise TypeError('condition should be python statement or callable object')

        if not is_value(new_value):
            raise TypeError('SeriesSet does not support %s ' % type(new_value),
                            'as a value type.')
        return col, condition, new_value

    def _check_col_new_name(self, new_name):
        if new_name is None:
            return self._check_col_new_name('C_%d' % len(self._columns))
        
        new_name = str(new_name)
        if is_str(new_name) and new_name not in self._columns:
            return new_name
        return auto_plus_one(self._columns, new_name)

    def _check_slice_pos_col(self, i, j):
        if i in self._columns:
            i = self._columns.index(i)
        elif i is None:
            i = 0
        else:
            raise ValueError('%s is not a title in current dataset'%i)

        if j in self._columns:
            j =  self._columns.index(j)
        elif j is None:
            j = self._dim.Col - 1
        else:
            raise ValueError('%s is not a title in current dataset'%j)
        return (i, j)

    def _check_operation_key(self, keys):
        '''transfer the string key name into itemgetter object'''
        return itemgetter(*tuple(map(self.columns.index, keys)))
    
    def _check_area(self, point1, point2):
        def _check_pos(point, left=True):
            if left is True and point is None:
                x, y = 0, 0
            elif left is False and point is None:
                x, y = None, None
            else:
                x, y = point
                
            if isinstance(y, str) and y in self._columns:
                y = self._columns.index(y)
            elif y is None and left is True:
                y = 0
            elif y is None and left is False:
                y = self._dim.Col
            elif not(isinstance(y, int) and abs(y) < self._dim):
                raise IndexError('your request col=`%s` is out of dataset.' % y)

            if x is None and left is True:
                x = 0
            elif x is None and left is False:
                x = self._dim.Ln
            elif not (isinstance(x, int) and abs(x) < self._dim.Ln):
                raise IndexError('your request ln=`%s` is out of dataset.' % x)
            return x, y

        L1, C1 = _check_pos(point1, left=True)
        L2, C2 = _check_pos(point2, left=False)

        if C2 < C1 or L2 < L1:
            raise ValueError('point2 should be larger than point1.')
        return L1, C1, L2, C2

    def _init_nan_func(self):
        if isnan(self._nan) is True:
            return isnan
        return lambda value: value == self._nan

    def _check_nan(self, nan):
        if is_value(nan):
            return set(nan, self._nan)
        return set(nan)

    def _check_columns_index(self, col):
        if col is None:
            return tuple(self._columns)

        if is_str(col):
            if col in self._columns:
                return (col,)
            raise ValueError('%s is not a title in current dataset' % col)

        if isinstance(col, int):
            assert abs(col) < self.shape.Col, 'title index is out of range'
            return (self._columns[col],)

        if is_seq(col):
            return tuple(self._check_columns_index(col_)[0] for col_ in col)

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
        if isinstance(self.data, dict):
            return SeriesSet(self)
        return Frame(self)
        
    def replace(self, where, value, col=None, regex=False):
        col = self._check_columns_index(col)
        if regex is True:
            assert is_str(where), '`where` must be a str when regex=True in this version'
            _cond = compile(where)
            def where(x):
                if _cond.match(x) is None:
                    return x
                return value

        if is_value(where):
            where, set_value = [where], [value]
            
        if is_seq(where) and is_seq(value):
            assert len(where) == len(value), 'length of where unequals to length of value'
            _cond = dict(zip(where, value))
            where = lambda x: _cond.get(x, x)
        assert callable(where), '`where` must be a callable object'

        for column in col:
            sequence = self._data[column]
            for i, value_ in enumerate(sequence):
                sequence[i] = where(value_)

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
        function, has a high efficiency with O(logN) speed. On the other
        hand, normal linear comparing select, implemented like `where`
        function, has a linear efficiency of O(N) speed.

        Parameters
        ----------
        expression : str
            the statement you want to use to select data,
            you can write it like python condition syntax.

        col : None, str or list (default=None)
            which columns you want to select

        limit : int, 'all' (default=1000)
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
        >>> data.query('5.5 >= sepal length > 5 and sepal width > 4').show()
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
        assert len(useful_col) != 0, 'can not match any column from the `expression`' 
        if limit == 'all':
            limit = self.shape.Ln
            
        if all([col in self._sorted_index for col in useful_col]) is False:
            subset = self[useful_col]
            where = subset._trans_where(expression, axis=0)
            sub_index = subset._where_by_rows(where, limit)
        else:
            sub_index = sorted(self._where_by_index(expression))[:limit]
        try:
            return self._iloc(SeriesSet(nan=self.nan), sub_index)[select_col]
        except TypeError: # there is no row match the query
            return SeriesSet(None, select_col, nan=self.nan)

    def _where_by_index(self, substring):
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
        combines = PATTERN_COMBINE.findall(substring)
        if combines:
            combines_symbol = [each[1:-1].strip() for each in PATTERN_RECOMBINE.findall(substring)]
            rows = [self._where_by_index(''.join(subpattern)) for subpattern in combines]
            final_rows = set(rows[0])
            for rows, comb in zip(rows[1:], combines_symbol):
                if comb == 'and':
                    final_rows = final_rows & rows
                else:
                    final_rows = final_rows | rows
            return final_rows

        subcombine = PATTERN_AND_OR.split(substring)
        if len(subcombine) > 1:
            subcombine_symbol = PATTERN_AND_OR.findall(substring)
            rows = [self._where_by_index(subsubpattern) for subsubpattern in subcombine]
            final_rows = set(rows[0])
            for rows, comb in zip(rows[1:], subcombine_symbol):
                if comb.strip() == 'and':
                    final_rows = final_rows & rows
                else:
                    final_rows = final_rows | rows
            return final_rows

        column = self._match_column_from_str(substring)
        assert len(column) <= 1, 'Index select can only be used by calculating single column'
        assert len(column) == 1, 'can not match the column from `%s`' % substring
        assert column[0] in self._sorted_index, 'column name `%s` is not in the statement `%s`' % (column[0], substring)
        sorted_index = self._sorted_index[column[0]]
        equal_pattern = ['!=', '<=', '>=']
        operater = (sorted_index.equal, sorted_index.lower, sorted_index.upper)
        patterns = [match_pattern.split(substring) for match_pattern in (PATTERN_EQUAL,
                                                                         PATTERN_LESS,
                                                                         PATTERN_GREAT,)]
        for i, (pattern, symbol, func) in enumerate(zip(patterns,
                                                        equal_pattern,
                                                        operater)):
            
            pattern = [pat for pat in pattern if pat.strip()]
            if len(pattern) == 3:
                value = auto_str2value(pattern[2].strip().replace('"', '').replace("'", ''))
                if pattern[1] == symbol:
                    if i == 0:
                        return set(sorted_index.unequal(value))
                    return set(func(value))
                try:
                    return set(func(value, False))
                except TypeError:
                    return set(func(value))

        pattern = [pat for pat in PATTERN_BETWEEN1.split(substring) if pat.strip()]
        lvalue = auto_str2value(pattern[4].strip().replace('"', '').replace("'", ''))
        hvalue = auto_str2value(pattern[0].strip().replace('"', '').replace("'", ''))
        left, right = True, True
        for subpattern, boundary in zip([pattern[1], pattern[3]], [left, right]):
            if subpattern == '>':
                boundary = False
        return set(sorted_index.between(lvalue, hvalue, (right, left)))

    def _where_by_rows(self, where, limit):
        assert isinstance(limit, int)
        assert callable(where), '`where` must be a callable object, try: Sheet.query(where)'
        rows = self.iter_rows()
        
        try:
            where(tuple())
        except AttributeError:
            rows = self.__iter__()
        except:
            pass
        
        selected = 0
        for i, row in enumerate(rows):
            if where(row):
                selected += 1
                yield i
                if selected == limit:
                    break

    def reshape(self, nshape):
        assert isinstance(nshape, (tuple, list)), '`new_shape` must contain in a tuple'
        assert len(nshape) == 2, 'supports 2D sheet only'
        assert isinstance(nshape[0], int), isinstance(nshape[1], int)
        assert nshape[0] * nshape[1] == self.shape.Ln * self.shape.Col, "can't reshape sheet of size %d into shape %s" % (size, new_shape)

        shape_ln = nshape[0]
        sheet, row = [], []
        for i, value in enumerate(chain(*self.iter_rows()), 1):
            row.append(value)
            if i % shape_ln == 0:
                sheet.append(row)
                row = []

        if isinstance(self.data, dict):
            return SeriesSet(sheet)
        return Frame(sheet)
                
    def select(self, where, col=None, limit=1000):
        '''sheet.select(lambda x: x.A_col != 1, col='B_col', limit=3000) -> SeriesSet
        SQL Statement: SELECT B_col FROM sheet WHERE A_col != 1 LIMIT 3000
        '''
        if limit == 'all':
            limit = self.shape.Ln
        col = self._check_columns_index(col)
        sub_index = self._where_by_rows(where, limit)
        return self._iloc(SeriesSet(nan=self.nan), sub_index)[col]

    def get(self, key, default=None):
        if key in self.columns:
            return self[key]
        return default

    def groupby(self, keys, func=None, apply_col=None, unapply_col=None):
        if func is not None:
            result = tuple(self.iter_groupby(keys, func, apply_col, unapply_col))
            return SeriesSet(result, self.columns)
        
        result = {}
        for key, subset in self.iter_groupby(keys, func, None, None):
            result[key] = subset
        return result

    def iter_groupby(self, keys, func=None, apply_col=None, unapply_col=None):
        from time import clock
        def operate_subset(subset, key):
            ret = subset.apply(func, col=apply_col, axis=1)
            for key_value, key_name in zip(key, keys):
                if key_name not in ret.columns:
                    pos = self.columns.index(key_name)
                    ret.insert_col(pos, key_value, key_name)
            return ret[0]
            
        keys = self._check_columns_index(keys)
        assert len(keys) != 0, 'must give at least 1 key column to group by'
        if len(keys) == 1 and keys[0] in self._sorted_index:
            subsets = {}
            index = self._sorted_index[keys[0]]
            for group_value in set(self._data[keys[0]]):
                subsets[(group_value,)] = index.equal(group_value)
        else:
            subsets = self._group_index_by_column_value(keys)
            
        if func is not None:
            if unapply_col is None:
                unapply_col = []
            apply_col = set(self._check_columns_index(apply_col))
            unapply_col = set(self._check_columns_index(unapply_col)) & set(keys)
            apply_col = ((col, self.columns.index(col)) for col in apply_col - unapply_col)
            apply_col = list(map(itemgetter(0), sorted(apply_col, key=itemgetter(1))))
            for keyword, rows in subsets.items():
                subset = self._iloc(SeriesSet(nan=self._nan), rows)
                yield operate_subset(subset, keyword)
        else:
            for keyword, rows in subsets.items():
                yield keyword, self._iloc(SeriesSet(nan=self._nan), rows)

    def show(self, lines='all'):
        if len(self._columns) == 0:
            return 'empty sheet instant'

        if not isinstance(lines, int) and not lines == 'all':
            raise TypeError('`lines` must be an int() or keyword `all`.')

        if lines == 'all' or 2 * lines >= self._dim.Ln:
            lines, omit = -1, 0
            temporary_data = self
        elif lines <= 0:
            raise ValueError('`lines` must be greater than 0.')
        else:
            omit = self._dim.Ln - 2 * lines
            temporary_data = self[:lines]
            temporary_data.extend(self[-lines:], inplace=True)
        temporary_series = [[title,] for title in self._columns]
        for i, col in enumerate(temporary_data.values()):
            temporary_series[i].extend(map(str, col))

        column_size = [len(max(col, key=len)) for col in temporary_series]
        frame = ' ' + ' | '.join([title.center(column_size[i]) for i, title in \
                         enumerate(self._columns)]) + '\n'
        frame += '+'.join(['-' * (size + 2) for size in column_size]) + '\n'

        for i, item in enumerate(temporary_data.iter_rows()):
            if i == lines:
                frame += ('.. Omit %d Ln ..'%omit).center(len(line)) + '\n'
            line = ''
            for i, value in enumerate(item):
                line += ' ' + str(value).center(column_size[i]) + ' |'
            frame += line[:-1] + '\n'
        return frame

    def sort(self, *orderby):
        '''S.sort(('A_col', 'DESC'), ('B_col', 'ASC')) --> Return sorted object
        '''
        ERROR = "orderby must be a sequence of conditions like ('A_col', 'DESC')"
        assert all(map(lambda x: (is_seq(x) and len(x) == 2) or is_str(x), orderby)), ERROR
        compare_symbol = ['ASC' if is_str(order) else str(order[1]) for order in orderby]
        if is_str(orderby[0]):
            orderby = [(order,) for order in orderby]
        compare_key_list = self._check_columns_index([order[0] for order in orderby])
        compare_pos_list = tuple(map(self.columns.index, compare_key_list))
        assert all(map(lambda x: x.upper() in ('DESC', 'ASC'), compare_symbol)), ERROR
        assert len(compare_key_list) == len(compare_symbol), ERROR
        size_orders = len(compare_key_list) - 1

        def hash_sort(datas_, i=0):
            # initialize values
            index = compare_pos_list[i]
            inside_data, HashTable = list(), dict()

            # create the diction
            for item in datas_:
                HashTable.setdefault(item[index], []).append(item)

            # sorted the values
            sequence = sorted(HashTable)

            # transform the record into Frame
            for each in sequence:
                items = HashTable[each]
                if i < size_orders:
                    items = hash_sort(items, i+1)
                inside_data.extend(items)

            # finally, reversed the list if necessary.
            if i != 0 and compare_symbol[i] != compare_symbol[i-1]:
                inside_data.reverse()
            return inside_data

        if len(compare_symbol) == 1 and isinstance(self._data, dict):
            reverse = False
            if compare_symbol[0] == 'DESC':
                reverse = True
                
            new_index = argsort(self._data[compare_key_list[0]], reverse=reverse)
            getter = tuple(new_index)
            sort_subset = SeriesSet(nan=self.nan)
            for i, (key, seq) in enumerate(self.iter_items()):
                self._subset_quickly_append_col(sort_subset, key,
                                                seq[getter],
                                                self._missing[i])
            return sort_subset

        temp = hash_sort(self.iter_rows())
        if compare_symbol[0] == 'DESC':
            temp.reverse()
        if isinstance(self._data, dict):
            return SeriesSet(temp, self.columns)
        return Frame(temp, self.columns)


class SeriesSet(BaseSheet):
    '''Variable stores in sequenes.
    '''
    def __init__(self, series=None, columns=None, nan=None):
        self._data = dict()
        self._isMutable = True
        BaseSheet.__init__(self, series, columns, nan)

    @property
    def info(self):
        self.describe(level=0, show=True)

    @property
    def IsMutable(self):
        return self._isMutable
        
    def _init_col(self, series, columns):
        '''initialzie from a SeriesSet

        Notes
        -----
        1. This function has been added into unit test.
        '''
        self._dim = copy(series._dim)
        self._init_col_name(columns)
        for col, seq in zip(self.columns, series.values()):
            self._data[col] = copy(seq)
        self._missing = copy(series._missing)
        self._nan = copy(series._nan)

    def _init_dict(self, series, columns):
        '''initialize from a dict object

        Notes
        -----
        1. This function has been added into unit test.
        '''
        for key, value in series.items():
            if is_value(value) is True:
                series[key] = [value]
        max_Ln = max(map(len, series.values()))
        self._dim = dims(max_Ln, len(series))
        self._init_col_name(columns)
        for column in self.columns:
            mv, sequence = self._check_sequence_type(series[column], self._dim.Ln)
            self._missing.append(mv)
            self._data[column] = sequence

    def _init_frame(self, series, columns):
        '''initialize from a Frame

        Notes
        -----
        1. This function has been added into unit test.
        '''
        self._dim = dims(series._dim.Ln, series._dim.Col)
        self._missing = copy(series._missing)
        self._init_col_name(columns)
        for sequence, title in zip(zip(*series), self._columns):
            self._data[title] = Series(sequence)

        if self._nan != series._nan:
            nan, self._nan = self.nan, series.nan
            self.nan = nan

    def _init_like_seq(self, series, columns):
        '''initialize from a single sequence

        Notes
        -----
        1. This function has been added into unit test.
        '''
        self._dim = dims(len(series), 1)
        self._init_col_name(columns)
        mv, series = self._check_sequence_type(series, self._dim.Ln)
        self._missing = [mv,]
        self._data[self._columns[0]] = series

    def _init_like_table(self, series, columns):
        '''initialize from an array-like object

        Notes
        -----
        1. This function has been added into unit test.
        '''
        lenth_Col = len(max(series, key=len))
        self._dim = dims(len(series), lenth_Col)
        self._init_col_name(columns)
        self._missing = [0] * self._dim.Col
        for j, sequence in enumerate(zip_longest(fillvalue=self.nan, *series)):
            mv, series = self._check_sequence_type(sequence, self._dim.Ln)
            self._missing[j] += mv
            self._data[self._columns[j]] = series

    def __repr__(self):
        if self._dim.Ln > 10:
            def write_Ln(i, title, blank):
                item = self._data[title]
                msg = ' ' * blank + title + ': <'
                msg += ', '.join([str(value) for value in item[:5]])
                msg += ', ... ,'
                msg += ', '.join([str(value) for value in item[-5:]])
                msg +=  '>\n'
                return msg
        elif self._dim.Ln != 0:
            def write_Ln(i, title, blank):
                item = self._data[title]
                msg = ' ' * blank + title + ': <'
                msg += ', '.join([str(value) for value in item])
                msg += '>\n'
                return msg
        else:
            return 'empty SeriesSet instant'

        msg = str()
        size = len(max(self._columns, key=len))
        for i, title in enumerate(self._columns):
            msg += write_Ln(i, title, size - len(title))
        return msg[:-1]

    def _getslice_col(self, i, j):
        subset = SeriesSet(None, None, self.nan)
        for i, col in enumerate(self._columns[i: j+1], i):
            sequence = self._data[col]
            subset._data[col] = sequence
            subset._missing.append(self._missing[i])
            subset._columns.append(col)
        subset._dim = dims(len(sequence), len(subset._columns))
        return subset

    def _getslice_ln(self, i, j, k):
        subset = SeriesSet(None, None, self.nan)
        for miss, col in zip(self._missing, self._columns):
            sequence = self._data[col][i:j:k]
            subset._data[col] = sequence
            if miss != 0:
                miss = count_nan(self._isnan, sequence)
            subset._missing.append(miss)
            subset._columns.append(col)
        subset._dim = dims(len(sequence), len(subset._columns))
        return subset

    def __getitem__(self, interval):
        if isinstance(interval, int):
            return Row(self, interval)

        elif isinstance(interval, Series):
            assert len(interval) == self.shape.Ln
            return self.iloc([i for i, val in enumerate(interval) if val])

        elif isinstance(interval, (tuple, list)):
            return self._getitem_by_tuple(interval, SeriesSet(nan=self._nan))

        elif isinstance(interval, slice):
            return self.__getslice__(interval.start, interval.stop)

        elif is_str(interval):
            return self._data[interval]

        else:
            raise TypeError('SeriesSet index must be int, str and slice, '+\
                            'not %s' % str(type(interval)).split("'")[1])

    def __iter__(self):
        for i in xrange(self._dim.Ln):
            yield Row(self, i)

    def __reversed__(self):
        for i in xrange(self._dim.Ln-1, -1, -1):
            yield Row(self, i)

    def _arrange_by_index(self, self_new_index=None, other_new_index=None):
        assert (self_new_index, other_new_index).count(None) == 1
        if self_new_index:
            getter = self_new_index
        else:
            getter = argsort(other_new_index)
            
        for title, sequence in self._data.items():
            self._data[title] = sequence[getter]

    def _replace_typical(self, col, cond, new):
        col, cond, new = self._check_replace_condition(col, cond, new)
        if is_str(cond):
            cond = self._trans_where(cond, axis=1)
        self._data[col] = [new if cond(value) else value\
                           for value in self._data[col]]

    def _subset_quickly_append_col(self, subset, arg, seq, miss):
        subset._data[arg] = seq
        subset._columns.append(arg)
        subset._missing.append(miss)
        subset._dim = dims(min(self.shape.Ln, len(seq)), subset._dim.Col + 1)
        return subset

    def append_row(self, item):
        '''append a series data as a row into the tail of the sheet

        Notes
        -----
        1. This function has been added into unit test.
        '''
        item = self._add_row(item)
        for value, seq in zip(item, self.iter_values()):
            seq.append(value)

    def apply(self, func, col=None, inplace=False, axis=0):
        assert inplace in (True, False), '`inplace` must be True or False'
        assert axis in (0, 1), '`axis` must be 0 or 1'
        assert callable(func), '`func` parameter should be a callable object'
        cols = self._check_columns_index(col)

        if inplace is False:
            if axis == 0:
                return SeriesSet(map(func, self[cols]), columns=cols, nan=self.nan)

            ret = SeriesSet(columns=self.columns, nan=self.nan)
            ret.append_row([func(self._data[col]) if col in cols else self.nan for col in self.columns])
            return ret[cols]
        
        if axis == 1:
            for name in cols:
                self[name] = map(func, self[name])
                
        if axis == 0:
            new_col = self._check_col_new_name(None)
            new_val = [func(row) for row in self[cols]]
            self.append_col(new_val, new_col)

    def append_col(self, series, variable_name=None):
        '''append a series data to the seriesset last

        Notes
        -----
        1. This function has been added into unit test.
        '''
        variable_name = self._check_col_new_name(variable_name)
        mv, series = self._check_sequence_type(series, self._dim.Ln)
        size = len(series)
        if size > self._dim.Ln:
            bias = size - self._dim.Ln
            for i, title in enumerate(self._columns):
                self._missing[i] += bias
                self._data[title].extend([self._nan] * bias)

        self._columns.append(variable_name)
        self._missing.append(mv)
        self._dim = dims(size, self._dim.Col+1)
        self._data[variable_name] = series

    def create_index(self, column):
        for column in self._check_columns_index(column):
            assert column not in self._sorted_index, '%s column has been an index already' % column
            self._sorted_index[column] = SortedIndex(self._data[column])
        self._isMutable = False

    def corr(self, method='pearson'):
        '''correlation between variables in data -> Frame object
        '''
        from DaPy import corr as corr_
        frame = [[1.0] * self.shape.Col for i in xrange(self.shape.Col)]
        for i, (title, sequence) in enumerate(self.iter_items()):
            for j, (next_title, next_sequence) in enumerate(self[:title].items()):
                if title != next_title:
                    coef = corr_(sequence, next_sequence, method)
                    frame[i][j] = coef
                    frame[j][i] = coef
        new_ = SeriesSet(frame, self._columns, nan='')
        new_.insert_col(0, self._columns, '')
        return new_

    def count(self, X, point1=None, point2=None):
        '''count the frequency of X in area (point1, point2)-> Counter object
        '''
        if is_value(X):
            X = (X,)
        L1, C1, L2, C2 = self._check_area(point1, point2)
        
        counter = Counter()
        for title in self._columns[C1 : C2+1]:
            sequence = self._data[title]
            for value in sequence[L1 : L2+1]:
                if value in X:
                    counter[value] += 1
        if len(X) == 1:
            return counter[X[0]]
        return counter

    def count_values(self, col=None):
        '''SeriesSet.count_values(col=None) -> Counter

        Parameters
        ----------
        col : None, str or str in list (default=None)
            the column you expected to analysis
            None -> means all columns
        '''
        col = self._check_columns_index(col)
        counter = Counter()
        for title in col:
            counter.update(Counter(self._data[title]))
        return counter

    def describe(self, level=0, show=True):
        assert level in (0, 1, 2)
        from DaPy import describe
        mins, maxs, avgs, stds, skes, kurs = [], [], [], [], [], []
        for sequence in self.iter_values():
            d = describe(sequence)
            for ls, value in zip([mins, maxs, avgs, stds, skes, kurs],
                                 [d.Min, d.Max, d.Mean, d.S, d.Skew, d.Kurt]):
                if value == None:
                    ls.append('-')
                elif isinstance(value, float):
                    ls.append('%.2f' % value)
                else:
                    ls.append(str(value))

        miss = list(map(str, self._missing))
        blank_size = [max(len(max(self._columns, key=len)), 5) + 2,
                      max(len(max(miss, key=len)), 4) + 2,
                      max(len(max(mins, key=len)), 3) + 2,
                      max(len(max(maxs, key=len)), 3) + 2,
                      max(len(max(avgs, key=len)), 4) + 2,
                      max(len(max(stds, key=len)), 3) + 2,]
        if level >= 1:
            blank_size.extend([
                      max(len(max(skes, key=len)), 4) + 2,
                      max(len(max(kurs, key=len)), 4) + 1])

        # Draw the title line of description
        title_line = '|'.join([
                     'Title'.center(blank_size[0]),
                     'Miss'.center(blank_size[1]),
                     'Min'.center(blank_size[2]),
                     'Max'.center(blank_size[3]),
                     'Mean'.center(blank_size[4]),
                     'Std'.center(blank_size[5])])
        if level >= 1:
            title_line += '|' + '|'.join([
                     'Skew'.center(blank_size[6]),
                     'Kurt'.center(blank_size[7])])
        title_line += '\n%s\n' % '+'.join(map(lambda x: '-' * x, blank_size))

        # Draw the main table of description
        info = ''
        for i, title in enumerate(self._columns):
            info += title.center(blank_size[0]) + '|'
            info += miss[i].center(blank_size[1]) + '|'
            info += mins[i].center(blank_size[2]) + '|'
            info += maxs[i].center(blank_size[3]) + '|'
            info += avgs[i].center(blank_size[4]) + '|'
            info += stds[i].center(blank_size[5])
            if level >= 1:
                info += '|' + skes[i].center(blank_size[6]) + '|'
                info += kurs[i].center(blank_size[7])
            info += '\n'

        lenth = 5 + 2 * level + sum(blank_size)
        line = '=' * lenth
        whole = '1.  Structure: DaPy.SeriesSet\n' +\
                '2. Dimensions: Lines=%d | Variables=%d\n'%self._dim +\
                '3. Miss Value: %d elements\n' % sum(self.missing) +\
                'Descriptive Statistics'.center(lenth) + '\n' +\
              line + '\n' + title_line + info + line
        if show is False:
            return whole
        print(whole)


    def pop(self, index=-1, axis=0):
        assert axis in (0, 1)
        if axis == 0:
            return self.pop_row(index)
        return self.pop_col(index)
        
    def pop_row(self, index=-1):
        '''pop(remove & return) record(s) from the sheet
        '''
        pop_index = set(self._check_remove_index(index))
        get_index = [i for i in xrange(self.shape.Ln) if i not in pop_index]
        pop_item = dict()
        for i, (title, seq) in enumerate(self.iter_items()):
            pop_item[title] = seq[get_index]
            if self._missing[i] != 0:
                self._missing[i] -= pop_item[title].count(self._nan)
        self._dim = dims(self._dim.Ln - len(index), self._dim.Col)
        return SeriesSet(pop_item, nan=self._nan)

    def pop_col(self, col=-1):
        '''pop(remove & return) record(s) from the sheet
        '''
        pop_name = self._check_columns_index(col)
        pop_data = dict()
        for title in pop_name:
            pos = self._columns.index(title)
            pop_data[title] = self._data.pop(title)
            self._columns.pop(pos)
            self._missing.pop(pos)
        self._dim = dims(self._dim.Ln, self._dim.Col-len(pop_name))
        return SeriesSet(pop_data, None, self._nan)
    
    def drop(self, index=-1, axis=0):
        '''drop rows or columns by specifying label names corresponding axis

        Parameters
        ----------
        index : single value or list-like (default=-1)
            Index or column name or index of column

        axis : 0 or 1
            drop index along the row (axis=0) or column (axis=1)

        Return
        ------
        None

        Example
        -------
        >>> data = dp.SeriesSet(range(5))
        >>> data.drop(0, axis=0)
        >>> data
        C_0 : <1, 2, 3, 4>
        >>> data.drop('C_0', axis=1)
        empty SeriesSet instant

        Notes
        -----
        1. This function has been added into unit test.
        '''
        assert axis in (0, 1)
        if axis == 0:
            return self.drop_row(index)
        return self.drop_col(index)

    def drop_col(self, index=-1):
        pop_name = self._check_columns_index(index)
        ln, col = self.shape
        for title in pop_name:
            pos = self._columns.index(title)
            del self._data[title], self._missing[pos], self._columns[pos]
        col -= len(pop_name)
        if col == 0:
            ln = 0
        self._dim = dims(ln, col)

    def drop_row(self, index=-1, inplace=True):
        if inplace is False:
            self = copy(self)
            
        index = self._check_remove_index(index) # O(klnk)
        for i, seq in enumerate(self.iter_values()):
            for j in index:
                del seq[j]
            self._missing[i] = count_nan(self._isnan, seq) # O(n - k)
            
        self._dim = dims(self._dim.Ln - len(index), self._dim.Col)
        return self

    def dropna(self, axis=1, how='any', inplace=True):
        assert axis in (0, 1), 'axis must be 1 or 0.'
        assert how in ('any', 'all'), 'how must be "any" or "all"'

        if axis == 1:
            pops = []
            for i, value in enumerate(self._missing):
                if how == 'any' and value > 0:
                    pops.append(self._columns[i])
                if how == 'all' and value == df.shape.Ln:
                    pops.append(self._columns[i])

        if axis == 0:
            pops = Counter()
            for sequence in self._data.values():
                for i, value in enumerate(sequence):
                    if self._isnan(value):
                        pops.update((i,))
            if how == 'all':
                pops = dict([(key, 1) for key,value in pops.items() if value == self.shape.Ln])
            pops = list(pops.keys())

        if inplace is True:
            if len(pops) != 0 and axis == 0:
                self.drop_row(pops)
            if len(pops) != 0 and axis == 1:
                self.drop_col(list(set(pops)))

        if inplace is False:
            _ = SeriesSet(self)
            if len(pops) != 0 and axis == 0:
                _.drop_row(pops)
            if len(pops) != 0 and axis == 1:
                _.drop_col(list(set(pops)))
            return _

    def drop_duplicates(self, col=None, keep='first', inplace=False):
        '''SeriesSet.drop_duplicates(col=None, keep='first') -> SeriesSet
        '''
        assert keep in ('first', 'last', False)
        pop_col = self._check_columns_index(col)
        drop_index, drop_symbol = [], DUPLICATE_KEEP[keep]
        
        droped_table = self._group_index_by_column_value(pop_col) # O(n)
        for values in droped_table.values(): # O(n)
            if len(values) != 1:
                droped_list.extend(values[drop])

        if inplace is False:
            self = copy(self)
        return self.drop_row(index=droped_list) # O(k*lnk + n)

    def extend(self, item, inplace=False):
        '''extend the current SeriesSet with records in set.

        Notes
        -----
        1. This function has been added into unit test.
        '''
        if isinstance(item, SeriesSet):
            if inplace is False:
                self = SeriesSet(self)

            for title, sequence in item.iter_items():
                mv = item._missing[item._columns.index(title)]
                if title not in self._columns:
                    self._columns.append(self._check_col_new_name(title))
                    self._missing.append(self._dim.Ln + mv)
                    To = [self._nan] * self._dim.Ln
                else:
                    self._missing[self._columns.index(title)] += mv
                    To = self._data[title]
                To.extend(sequence)
                self._data[title] = To
            self._dim = dims(self._dim.Ln+item._dim.Ln, len(self._columns))

            for i, sequence in enumerate(self._data.values()):
                if len(sequence) != self._dim.Ln:
                    add_miss_size = self._dim.Ln - len(sequence)
                    sequence.extend([self._nan] * add_miss_size)
                    self._missing[i] += add_miss_size
            return self
        
        elif isinstance(item, Frame):
            return self.extend(SeriesSet(item), inplace)

        elif all(filter(is_iter, item)):
            return self.extend(SeriesSet(item, self._columns), inplace)

        else:
            raise TypeError('could not extend a single value only.')

    def fillna(self, fill_with=None, col=None, method=None, limit=None):
        '''fill nan in the dataset

        Parameters
        ----------
        fill_with : value, dict in valu (default=None)
            the value used to fill with

        cols : str (default=None)
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
        _isnan = self._isnan
        if limit is None:
            limit = self.shape.Ln
        assert limit >= 1, 'fill with at least 1 missing value, not limit=%s' % limit
        assert method in ('linear', 'polynomial', 'quadratic', None)
        if method is None:
            self._fillna_value(fill_with, col, _isnan, limit)
        else:
            self._fillna_simple_function(col, isnan, limit, None)
        
    def _fillna_value(self, values, col, _isnan, all_limit):
        assert isinstance(values, (dict,) + VALUE_TYPE), 'fill_with must be a value or dict'
        if isinstance(values, dict) is False:
            values = dict(zip(col, repeat(values)))

        for key, value in values.items():
            limit = all_limit
            key_index = self.columns.index(key)
            if key in col and self._missing[key_index] != 0:
                sequence = self._data[key]
                for i, value in enumerate(sequence): 
                    if _isnan(value) is True:
                        sequence[i] = values[key]
                        self._missing[key_index] -= 1
                        limit -= 1
                        if limit == 0:
                            break

    def _fillna_simple_function(self, col, _isnan, all_limit, func):
        '''establish a linear model to predict the missing value

        This function will predict the missing value with a linear model,
        which is established by the arounding records.
        '''
        def simple_linear_reg(x, y):
            x_bar, y_bar = sum(x) / len(x), sum(y) / len(y)
            l_xx = sum(map(lambda x: (x - x_bar) ** 2, x))
            l_xy = sum(map(lambda x,y: (x - x_bar) * (y - y_bar), x, y))
            k = l_xy / float(l_xx)
            return k, y_bar - k * x_bar
            
        for key in col:
            limit = all_limit
            key_index = self.columns.index(key)
            if self._missing[key_index] == 0:
                continue
            
            sequence = self._data[key]
            
            for i, value in enumerate(sequence):
                if _isnan(value) is False and _isnan(sequence[i+1]) is False:
                    break
                
            if i != 0:
                x = []
                for value in sequence[i:2 * i + 1]:
                    if _isnan(value) is True:
                        break
                    x.append(value)
                k, b = simple_linear_reg(x, range(i, 2 * i + 1))
                for transfer_index in xrange(0, i):
                    sequence[transfer_index] = k * transfer_index + b
                    self._missing[key_index] -= 1
                    limit -= 1
                    if limit == 0:
                        break

            start = None
            for stop, value in enumerate(sequence):
                if limit == 0:
                    break
                
                if _isnan(value) is True:
                    if start is None:
                        start = stop
                    
                elif start is not None:
                    empty_length = stop - start
                    back = max(start - empty_length, 0)
                    fore = min(stop + empty_length, len(sequence))
                    left_length = start - back
                    y = sequence[back:start] + sequence[stop:fore]
                    x = range(1, left_length + 1) + range(1 + left_length + empty_length,
                                                           len(y) + empty_length + 1)
                    data = [(_x, _y) for _x, _y in zip(x, y) if _isnan(_y) is False]
                    k, b = simple_linear_reg(*zip(*data))
                    for transfer_index, x in enumerate(xrange(2 + left_length,
                                                              2 + left_length + empty_length),
                                                       start):
                        sequence[transfer_index] = k * x + b
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

        types : Type name or dict of columns (default=None):
            use one type to 
        '''
        sheet = cls()
        first_line = kwrd.get('first_line', 1)
        title_line = kwrd.get('title_line', 0)
        columns = kwrd.get('columns', [])
        nan = sheet._check_nan(kwrd.get('nan', ('nan', '?', '??', '', ' ', 'NA', 'None')))
        sep = kwrd.get('sep', ',')
        prefer_type = kwrd.get('prefer_type', None)
        col_types = kwrd.get('types', [])
        sheet._missing, temp_data = [], []
        param = {'mode': 'rU'}
        if PYTHON3:
            param['encoding'] = kwrd.get('encoding', None)
            param['file'] = addr
            param['newline'] = kwrd.get('newline', None)
        if PYTHON2:
            param['name'] = addr
        
        assert first_line > title_line, 'first line should be greater than title line'
        assert isinstance(columns, list), 'column name must be stored with a list'
        assert all(map(lambda x: is_str(x), columns)), 'column name must `str`'
            
        with open(**param) as f:
            # skip the rows which are unexpected to read
            for i in xrange(first_line):
                line = f.readline()
                if i == title_line:
                    # setup the title line
                    columns = tuple(map(str.strip, split(line, sep)))

            # begin to read the data
            for line in f: # iter row
                for i, (transfer, value) in enumerate(zip_longest(col_types, split(line.strip(), sep))): # iter value
                    try:
                        # the normal way
                        if value in nan:
                            temp_data[i].append(self._nan)
                            sheet._missing[i] += 1
                        else:
                            temp_data[i].append(transfer(value))
                        
                    except ValueError:
                        # different types of data in the same column
                        temp_data[i].append(auto_str2value(value))
                        
                    except IndexError:
                        transed = auto_str2value(value, prefer_type)
                        if transed in nan:
                            transed = self._nan
                            sheet._missing.append(1)
                        else:
                            sheet._missing.append(0)
                            transfer_name = str(transed.__class__).split()[1][1:-2].split('.')[0]
                            col_types.append(fast_str2value[transfer_name])
                        
                        if len(temp_data) == 0:
                            temp_data.append(Series([transed]))
                        else:
                            missed  =  len(temp_data[0]) - 1
                            sheet._missing[-1] += missed
                            temp_data.append(Series(chain([sheet._nan] * missed, [transed])))
                
        sheet._dim = dims(len(temp_data[0]), len(temp_data))
        sheet._init_col_name(columns)
        for i, (seq, col) in enumerate(zip(temp_data, sheet._columns)):
            add_space = sheet._dim.Ln - len(seq)
            seq.extend([sheet._nan] * add_space)
            sheet._missing[i] += add_space
            sheet._data[col] = seq
        return sheet

    def get_categories(self, col, cut_points, group_name, boundary=(False, True), inplace=False):
        '''transfer numerical variables into categorical variable'''
        from DaPy.operation import get_categories
        cols = self._check_columns_index(col)
        if inplace is False:
            self = SeriesSet(nan)

        for col in cols:
            categories = get_categories(self[col], cut_points, group_name, boundary)
            self.append_col(categories, '%s_category' % col)

    def get_date_label(self, col, daytime=True, weekend=True, season=True, inplace=False):
        '''transfer a datetime object into categorical variable'''
        cols = self._check_columns_index(col)
        if inplace is False:
            operate_with = SeriesSet(nan=self.nan)
        else:
            operate_with = self

        def dummy_date(col_name):
            sequence = copy(self._data[col_name])
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
                sub = self._check_col_new_name('%s_daytime' % col)
                self[sub] = date._data['hour']
                self.update('%s in set([23] + range(7))' % sub, {sub: 'latenight'})
                self.update('%s in set(range(7, 11))' % sub, {sub: 'morning'})
                self.update('%s in set(range(11, 14))' % sub, {sub: 'noon'})
                self.update('%s in set(range(14, 18))' % sub, {sub: 'afternoon'})
                self.update('%s in set(range(18, 23))' % sub, {sub: 'night'})

            if weekend is True:
                sub = self._check_col_new_name('%s_weekend' % col)
                self[sub] = date._data['week']
                self.replace([0, 1, 2, 3, 4, 5, 6], [1, 0, 0, 0 ,0 , 0, 1], col=sub)

            if season is True:
                sub = self._check_col_new_name('%s_season' % col)
                self[sub] = date._data['month']
                self.update('%s in set([3, 4, 5])' % sub, {sub: 'spring'})
                self.update('%s in set([6, 7, 8])' % sub, {sub: 'summer'})
                self.update('%s in set([9, 10, 11])' % sub, {sub: 'autumn'})
                self.update('%s in set([12, 1, 2])' % sub, {sub: 'winter'})
        return self

    def get_dummies(self, col=None, value=1, inplace=False):
        from DaPy import get_dummies
        if inplace is False:
            operate_with = SeriesSet(nan=self.nan)
        else:
            operate_with = self
            
        cols = self._check_columns_index(col)
        for title in cols:
            dummies = get_dummies(self._data[title], value, 'set')
            dummies.columns = [title+'='+title_ for title_ in dummies.columns]
            operate_with.join(dummies, inplace=True)
        return operate_with

    def get_interactions(self, col=None, inplace=False):
        col = self._check_columns_index(col)
        if inplace is False:
            operate_with = SeriesSet(nan=self.nan)
        else:
            operate_with = self

        for i, lcol in enumerate(self.columns):
            seq = self[lcol]
            for rcol in self.columns[i:]:
                if lcol == rcol:
                    title = '%s^2' % (lcol)
                    rseq = seq
                else:
                    title = '%s_%s' % (lcol, rcol)
                    rseq = self[rseq]
                operate_with.append_col(title, seq * rseq)
        return operate_with

    def get_ranks(self, col=None, duplicate='mean', inplace=False):
        from DaPy.operation import get_ranks
        cols = self._check_columns_index(col)
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
        getter = tuple(indexs)
        for miss, (key, sequence) in zip(self._missing, self.iter_items()):
            seq = sequence[getter]
            if isinstance(seq, Series) is False:
                seq = Series([seq,])
            if miss != 0:
                miss = count_nan(self._isnan, seq)
            subset = self._subset_quickly_append_col(subset, key, seq, miss)
        return subset

    def iloc(self, indexs):
        assert isinstance(indexs, (slice, int)) or is_iter(indexs)
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
        mv, series = self._check_sequence_type(series, self._dim.Ln)

        empty_size = len(series) - self.shape.Ln
        if empty_size > 0:
            empty_seq =  [self.nan] * empty_size
            for i, sequence in enumerate(self.iter_values()):
                sequence.extend(empty_seq)
                self._missing[i] += empty_size
        
        self._columns.insert(index, variable_name)
        self._dim = dims(len(series), self._dim.Col + 1)
        self._missing.insert(index, mv)
        self._data[variable_name] = series

    def join(self, other, inplace=False):
        '''right join another sheet to the current sheet

        Notes
        -----
        1. This function has been added into unit test.
        '''
        error = "can't join empty object, given %s"  % other
        assert is_value(other) is False, 'can  not join a value to the dataset.'
        assert (hasattr(other, 'shape') and other.shape[0] !=  0) or len(other) != 0, error
        if inplace is False:
            self = SeriesSet(self)

        if isinstance(other, SeriesSet):
            lenth = max(self._dim.Ln, other._dim.Ln)
            for title, sequence in other.iter_items():
                title = self._check_col_new_name(title)
                mv, sequence = self._check_sequence_type(sequence, self._dim.Ln)
                self._missing.append(mv)
                self._data[title] = sequence
                self._columns.append(title)
            self._dim = dims(lenth, len(self._columns))

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

    def normalized(self, process='NORMAL', col=None, inplace=True, **attr):
        process = str(process).upper()
        assert process in ('NORMAL', 'STANDAR', 'LOG', 'BOX-COX')
        col = self._check_columns_index(col)
        if inplace is False:
            self = SeriesSet(self)

        from DaPy import describe, log, boxcox
        for title in col:
            if title is None:
                continue

            if process in ('NORMAL', 'STANDAR'):
                if attr == {}:
                    statis = describe(self._data[title])
                if process == 'NORMAL':
                    A = float(attr.get('min', statis.Min))
                    B = float(attr.get('range', statis.Range))
                elif process == 'STANDAR':
                    A = float(attr.get('mean', statis.Mean))
                    B = float(attr.get('std', statis.Sn))
                assert B != 0, '`%s` can not have 0 range or std' % title
                self.apply(lambda x: (x - A) / B, col=title, inplace=True, axis=1)
                
            elif process == 'BOX-COX':
                lamda = attr.get('lamda', 1)
                a = attr.get('a', 0)
                k = attr.get('k', 1)
                self.apply(lambda x: boxcox(x, lamba, a, k), col=title, inplace=True, axis=1)
                
            elif process == 'LOG':
                base = attr.get('base', 2.71828183)
                self.apply(lambda x: log(x, base), col=title, inplace=True, axis=1)
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
        assert len(self_on) == len(right_on) == 1, 'only support 1 matching index'
        self_on, right_on = self_on[0], right_on[0]
        new_other_key = self._check_col_new_name(right_on)

        # match the records according to the index
        joined = SeriesSet(nan=self.nan)
        if how == 'left':
            return left_join(self, other, self_on, right_on, joined)
        
        if how == 'right':
            return left_join(other, self, right_on, self_on, joined)
        
        if how == 'inner':
            return inner_join(self, other, self_on, right_on, joined)
        
        if how == 'outer':
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
        assert len(set_value) != 0, '`set_value` are empty'
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

    def __init__(self, frame=None, columns=None, nan=None):
        self._data = []
        BaseSheet.__init__(self, frame, columns, nan)

    @property
    def info(self):
        new_m_v = map(str, self._missing)
        max_n = len(max(self._columns, key=len))

        info = ''
        for i in xrange(self._dim.Col):
            info += ' '*15
            info += self._columns[i].center(max_n) + '| '
            info += ' ' + new_m_v[i] + '\n'

        print('1.  Structure: DaPy.Frame\n' +\
              '2. Dimensions: Ln=%d | Col=%d\n' % self._dim +\
              '3. Miss Value: %d elements\n' % sum(self._missing) +\
              '4.    Columns: ' + 'Title'.center(max_n) + '|'+\
                             '  Miss\n'+ info)

    def _init_col(self, obj, columns):
        self._data = [list(record) for record in zip(*list(obj.values()))]
        self._missing = copy(obj._missing)
        self._dim = dims(obj._dim.Ln, obj._dim.Col)
        self._init_col_name(columns)

    def _init_frame(self, frame, columns):
        self._data = deepcopy(frame._data)
        self._dim = copy(frame._dim)
        self._init_col_name(columns)
        self._missing = copy(frame._missing)

    def _init_dict(self, frame, columns):
        frame = copy(frame)
        self._dim = dims(max(map(len, frame.values())), len(frame))
        self._missing = [0] * self._dim.Col
        self._init_col_name(columns)
        for i, (title, col) in enumerate(frame.items()):
            mv, sequence = self._check_sequence_type(col, self._dim.Ln)
            frame[title] = sequence
            self._missing[i] += mv
        self._data = [list(record) for record in zip(*frame.values())]

    def _init_like_table(self, frame, columns):
        self._data = map(list, frame)
        dim_Col, dim_Ln = len(max(self._data, key=len)), len(frame)
        self._dim = dims(dim_Ln, dim_Col)
        self._missing = [0] * self._dim.Col

        for i, item in enumerate(self._data):
            if len(item) < dim_Col:
                item.extend([self._nan] * (dim_Col - len(item)))
            for j, value in enumerate(item):
                if value == self.nan or value is self.nan:
                    self._missing[j] = self._missing[j] + 1
        self._init_col_name(columns)

    def _init_like_seq(self, frame, columns):
        self._data = [[value,] for value in frame]
        self._dim = dims(len(frame), 1)
        self._init_col_name(columns)
        self._missing.append(self._check_sequence_type(frame, len(frame))[0])

    def __repr__(self):
        return self.show(30)

    def _getslice_col(self, i, j):
        new_data = [record[i : j+1] for record in self._data]
        return Frame(new_data, self._columns[i : j+1], self._nan)

    def _getslice_ln(self, i, j, k):
        return Frame(self._data[i:j:k], self._columns, self._nan)

    def __getitem__(self, interval):
        if isinstance(interval, int):
            return Row(self, interval)

        elif isinstance(interval, slice):
            return self.__getslice__(interval.start, interval.stop)

        elif is_str(interval):
            col = self._columns.index(interval)
            return [item[col] for item in self._data]

        elif isinstance(interval, (tuple, list)):
            return_obj = Frame()
            return self._getitem_by_tuple(interval, return_obj)

        else:
            raise TypeError('item must be represented as slice, int, str.')

    def __iter__(self):
        for i in xrange(self._dim.Ln):
            yield Row(self, i)

    def _replace_typical(self, col, cond, new):
        col, cond, new = self._check_replace_condition(col, cond, new)
        if is_str(cond):
            cond = self._trans_where(cond, axis=1)
        for line in self._data:
            for j, value in enumerate(line):
                if cond(value) is True:
                    line[j] = new

    def append_row(self, item):
        '''append a new record to the Frame tail
        '''
        item = self._add_row(item)
        self._data.append(item)

    def append_col(self, series, variable_name=None):
        '''append a new variable to the current records tail
        '''
        mv, series = self._check_sequence_type(series, self._dim.Ln)
        size = len(series) - self._dim.Ln
        if size > 0:
            self._missing = [m+size for m in self._missing]
            self._data.extend([[self._nan] * self._dim.Col for i in xrange(size)])

        self._missing.append(mv)
        for record, element in zip(self._data, series):
            record.append(element)
        self._columns.append(self._check_col_new_name(variable_name))
        self._dim = dims(max(self._dim.Ln, len(series)), self._dim.Col+1)
        assert len(self._missing) == self._dim.Col == len(self.columns)

    def count(self, X, point1=None, point2=None):
        if is_value(X):
            X = (X,)
        counter = Counter()
        L1, C1, L2, C2 = self._check_area(point1, point2)

        for record in self._data[L1:L2 + 1]:
            for value in record[C1:C2 + 1]:
                if value in X:
                    counter[value] += 1

        if len(X) == 1:
            return counter[X[0]]
        return dict(counter)

    def extend(self, other, inplace=False):
        if isinstance(other, Frame):
            if inplace is False:
                self = SeriesSet(Frame)
            new_title = 0
            for title in other._columns:
                if title not in self._columns:
                    self._columns.append(title)
                    new_title += 1

            for record in self._data:
                record.extend([self._nan] * new_title)

            extend_part = [[self._nan] * len(self._columns)\
                           for i in xrange(len(other))]
            new_title_index = [self._columns.index(title)
                               for title in other._columns]
            self._dim = dims(len(self) + len(other), len(self._columns))
            self._missing.extend([self._dim.Ln] * new_title)

            for i, record in enumerate(other._data):
                for j, value in zip(new_title_index, record):
                    if value == other._nan:
                        value = self._nan
                    extend_part[i][j] = value
            self._data.extend(extend_part)
            return self

        elif isinstance(other, SeriesSet):
            return self.extend(Frame(other), inplace)

        else:
            return self.extend(Frame(other, self._columns), inplace)

    def join(self, other, inplace=False):
        if isinstance(other, Frame):
            if inplace is False:
                self = Frame(self)
            for title in other._columns:
                self._columns.append(self._check_col_new_name(title))
            self._missing.extend(other._missing)

            for i, record in enumerate(other._data):
                if i < self._dim.Ln:
                    current_record = self._data[i]
                else:
                    current_record = [self._nan] * self._dim.Col
                    self._data.append(current_record)
                for value in record:
                    if value == other.nan:
                        value = self._nan
                    current_record.append(value)
            if i < self._dim.Ln:
                for record in self._data[i+1:]:
                    record.extend([self._nan] * other.shape.Col)
            self._dim = dims(len(self._data), len(self._columns))
            return self

        else:
            self.join(Frame(other, nan=self.nan), inplace)

    def insert_row(self, index, item):
        '''insert a new record to the frame with position `index`
        '''
        item = self._add_row(item)
        self._data.insert(index, item)

    def insert_col(self, index, series, variable_name=None):
        '''insert a new variable to the current records in position `index`
        '''
        mv, series = self._check_sequence_type(series)

        size = len(series) - self._dim.Ln
        if size > 0:
            for i in xrange(self._dim.Col):
                self._missing[i] += size
            self._data.extend([[self._nan] * self._dim.Col\
                                for i in xrange(size)])

        self._missing.insert(index, mv)
        for i, element in enumerate(series):
            self._data[i].insert(index, element)
        self._columns.insert(index, self._check_col_new_name(variable_name))
        self._dim = dims(max(self._dim.Ln, size), self._dim.Col+1)

    def items(self):
        for i, sequence in enumerate(zip(*self._data)):
            yield self._columns[i], list(sequence)

    def keys(self):
        return self._columns

    def pop_row(self, pos=-1):
        '''pop(remove & return) a record from the Frame
        '''
        assert isinstance(pos, (int, list, tuple)), 'an int or ints in list is required.'
        if isinstance(pos, int):
            pos = [pos,]
        pos = sorted(pos, reverse=True)
        pop_item = Frame([self._data.pop(pos_) for pos_ in pos], list(self._columns))
        self._dim = dims(self._dim.Ln - len(pos), self._dim.Col)
        self._missing = map(lambda x, y: x - y, self._missing, pop_item._missing)
        return pop_item

    def pop_col(self, pos=-1):
        '''pop(remove & return) a series from the Frame
        '''
        pop_name = self._check_columns_index(pos)
        for name in pop_name:
            index = self._columns.index(name)
            self._columns.pop(index)
            self._missing.pop(index)

        pop_data = [[] for i in xrange(len(pop_name))]
        new_data = [0] * self._dim.Ln
        for j, record in enumerate(self._data):
            line = []
            for i, value in enumerate(record):
                if i in pop_name:
                    pop_data[pop_name.index(i)].append(value)
                else:
                    line.append(value)
            new_data[j] = line

        self._dim = dims(self._dim.Ln, self._dim.Col-len(pos))
        self._data = new_data
        return SeriesSet(dict(zip(pop_name, pop_data)))

    def dropna(self, axis='LINE'):
        '''pop all records that maintains miss value while axis is `LINE` or
        pop all variables that maintains miss value while axis is `COL`
        '''
        pops = []
        if str(axis).upper() in ('0', 'LINE'):
            for i, record in enumerate(self._data):
                if self._nan in record:
                    pops.append(i)

        if str(axis).upper() in ('1', 'COL'):
            for i, sequence in enumerate(zip(*self._data)):
                if self._nan in sequence:
                    pops.append(self._columns[i])

        if len(pops) != 0:
            self.__delitem__(pops)

    def from_file(self, addr, **kwrd):
        '''read dataset from csv or txt file.
        '''
        f = open(addr, 'r')
        freader, col_types, nan, prefer = self._check_read_text(f, **kwrd)
        self._data = [0] * self.shape.Ln
        try:
            for line, record in enumerate(freader):
                line = tuple(self._trans_str2val(i, v, col_types, nan, prefer) for i, v in enumerate(record))
                if len(line) != self._dim.Col:
                    line = list(chain(line,  (self._nan) * (self._dim.Col - len(line))))
                self._data[line] = line
        except MemoryError:
            self._dim = dims(len(self._data), self._dim.Col)
            warn('since the limitation of memory, DaPy can not read the'+\
                 ' whole file.')
        finally:
            f.close()

    def reverse(self):
        self._data.reverse()

    def shuffle(self):
        shuffles(self._data)

    def _values(self):
        for sequence in zip(*self._data._data):
            yield list(sequence)

    def values(self):
        for sequence in zip(*self._data):
            yield Series(sequence)

