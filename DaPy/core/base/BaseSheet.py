from collections import Counter, defaultdict
from copy import copy
from itertools import chain, combinations, repeat
from operator import eq, ge, gt, itemgetter, le, lt
from random import shuffle as shuffles
from re import compile as re_compile
from re import findall, sub

from .constant import (DUPLICATE_KEEP, PYTHON2, PYTHON3, SHEET_DIM, STR_TYPE,
                       VALUE_TYPE)
from .constant import nan as NaN
from .DapyObject import Object
from .IndexArray import SortedIndex
from .Row import Row
from .Series import Series
from .utils import (argsort, auto_plus_one, auto_str2value, count_nan,
                    fast_str2value, hash_sort, is_dict, is_empty, is_iter,
                    is_math, is_seq, is_str, is_value, isnan, range, split,
                    str2date, strip, xrange, zip_longest)
from .utils.utils_str_patterns import *
from .utils.utils_join_table import inner_join, left_join, outer_join
from .utils.utils_regression import simple_linear_reg

LOCK_ERROR = 'sheet is locked by indexes, See drop_index()'

def where_by_index_combine(rows, symbols):
    '''put all rows together'''
    final_rows = set(rows[0])
    for row, comb in zip(rows[1:], symbols):
        if comb.strip() == 'and':
            final_rows = final_rows & row
        else:
            final_rows = final_rows | row
    return final_rows


class BaseSheet(Object):
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
        Object.__init__(self)
        self._missing = []
        self.nan = nan
        self._sorted_index = {}

        if hasattr(obj, 'values') and not callable(obj.values):
            # Pandas DataFrame -> Numpy array
            columns = obj.columns
            obj = obj.values 

        if isinstance(obj, BaseSheet): 
            if is_dict(self.data):
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
                    self._append_col([], name)

        elif is_dict(obj): 
            # initialize from a dict
            self._init_dict(obj, columns)

        elif isinstance(obj, Series) or \
            (is_seq(obj) and all(map(is_value, obj))):
             # initialize from a single series
            self._init_like_seq(obj, columns)

        elif is_seq(obj) and all(map(is_iter, obj)):
            # initialize from array-like object
            self._init_like_table(obj, columns)

        elif is_iter(obj): 
            # initialie from an iterator object
            self._init_like_iter(obj, columns)

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
    def locked(self):
        '''self.is_mutable -> bool
        check whether sheet is clock by sorted indexes or not'''
        if not self._sorted_index:
            return True
        return False

    def __repr__(self):
        if self._dim.Ln > 10:
            def write_line(title, blank):
                item = self[title]
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

    def __getattr__(self, name):
        '''self.A -> return column A'''
        if name in self._columns:
            return self.__getitem__(name)
        raise AttributeError("Sheet object has no attribute '%s'" % name)

    def __len__(self):
        '''len(sheet) -> number of rows'''
        return self._dim.Ln

    def __compare_value__(self, val, empty, symbol):
        assert is_value(val), 'compare elements must be given a value'
        for title, seq in self.iter_items():
            empty._quickly_append_col(title, symbol(seq, val), 0)
        return empty

    def __contains__(self, cmp_):
        '''3 in sheet -> True / False'''
        if is_str(cmp_):
            return cmp_ in self._data

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
        del instance['_isnan']
        return instance

    def __setstate__(self, read_dict):
        '''load this object from a stream file'''
        self._dim = SHEET_DIM(*read_dict['_dim'])
        self._columns = read_dict['_columns']
        self._missing = read_dict['_missing']
        self._nan = read_dict['_nan']
        self._sorted_index = read_dict['_sorted_index']
        self._init_nan_func()
        self._data = dict((key, Series(val)) for key, val in read_dict['_data'].items())

    def __getitem__(self, interval):
        if isinstance(interval, int):
            return Row(self, interval)

        if isinstance(interval, Series):
            assert len(interval) == self.shape.Ln
            return self._iloc([i for i, val in enumerate(interval) if val is True])

        if isinstance(interval, (tuple, list)):
            return self._getitem_by_tuple(interval, type(self)(nan=self._nan))

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

    def _add_row(self, row):
        # when user just input a single value as a row
        if is_value(row):
            assert self._dim.Col != 0, 'Adding a single value into an empty sheet is illegal.'
            if self._isnan(row) is True:
                self._missing = [_ + 1 for _ in self._missing]
            self._dim = SHEET_DIM(self._dim.Ln + 1, self._dim.Col)
            return [row] * self._dim.Col

        # when user input a dict as a row
        row = row._asdict() if hasattr(row, '_asdict') else row
        if is_dict(row):
            row, dic_row = [row.get(col, self.nan) for col in self.columns], row
            for key, value in dic_row.items():
                if key not in self._data:
                    row.append(value)
                    seq = Series(repeat(self.nan, self.shape.Ln))
                    self._quickly_append_col(key, seq, self.shape.Ln)

        # in the normal way, we first calculate the bias of length
        lenth_bias = len(row) - self._dim.Col
        if lenth_bias > 0 and self.shape.Ln == 0:
            for i in xrange(lenth_bias):
                self._append_col(Series())
        elif lenth_bias > 0:
            for _ in xrange(lenth_bias):
                series = Series(repeat(self._nan, self.shape.Ln))
                self._quickly_append_col(None, series, self.shape.Ln)

        miss, row = self._check_sequence(row, self.shape.Col)
        self._dim = SHEET_DIM(self._dim.Ln + 1, max(self._dim.Col, len(row)))
        if miss != 0:
            for i, value in enumerate(row):
                if self._isnan(value):
                    self._missing[i] += 1
        return row

    def _append_row(self, row):
        row = self._add_row(row)
        for val, col, seq in zip(row, self.columns, self.iter_values()):
            seq.append(val)
            if col in self._sorted_index:
                self._sorted_index[col].append(val)

    def _append_col(self, series, variable_name):
        miss, series = self._check_sequence(series, self._dim.Ln)
        size = len(series)
        if size > self._dim.Ln:
            bias = size - self._dim.Ln
            for i, title in enumerate(self._columns):
                self._missing[i] += bias
                self._data[title].extend([self._nan] * bias)
        self._quickly_append_col(variable_name, series, miss)

    def _apply_inplace(self, func, cols, axis):
        if axis == 1:
            err = 'Your are operating columns which are Index. '
            err += 'Please delete that column at first!'
            assert all(map(lambda x: x not in self._sorted_index, cols)), err
            for name in self._check_columns_index(cols):
                seq = Series(map(func, self.data[name]))
                ind = self.columns.index(seq)
                self._missing[ind] = count_nan(self._isnan, seq)
                self.data[name] = seq
        if axis == 0:
            new_col = self._check_col_new_name(None)
            subset = self[cols]
            try:
                func(subset[0].tolist())
                subset = subset.iter_values()
            except Exception as e:
                pass
            new_seq = (func(row) for row in subset)
            self._append_col(new_seq, new_col)
        return self

    def _arrange_by_index(self, self_new_index=None):
        for title, sequence in self._data.items():
            self._data[title] = sequence[self_new_index]
        return self

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

    def _check_sequence(self, series, size):
        '''check the shape of the sequence and fill nan if not long enough

        if sequence is record, size = self.shape.Ln;
        else size = self.shape.Col
        '''
        if is_value(series):
            return 0, Series(repeat(series, size))

        assert is_iter(series), "append item should be an iterable object"
        series = Series(series)
        if len(series) < size:
            series = chain(series, [self._nan] * (size - len(series)))
            series = Series(series)
        return count_nan(self._isnan, series), series

    def _check_col_new_name(self, new_name):
        if new_name is None:
            return self._check_col_new_name('C_%d' % len(self._columns))

        new_name = PATTERN_CHANGE_LINE.sub('', str(new_name))
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

        if stop is None or stop > lenth:
            stop = lenth
        elif stop < 0:
            stop += lenth
        error = 'Index out of range'
        assert 0 <= start <= lenth and 0 <= stop <= lenth, error
        return start, stop

    def _check_slice_col(self, start, stop):
        if start in self._columns:
            start = self._columns.index(start)
        elif start is None:
            start = 0
        else:
            raise ValueError('`%s` is not a title in this sheet' % start)

        if stop in self._columns:
            stop = self._columns.index(stop)
        elif stop is None:
            stop = self._dim.Col - 1
        else:
            raise ValueError('`%s` is not a title in this sheet' % stop)
        return start, stop

    def _check_operation_key(self, keys):
        '''transfer the string key name into itemgetter object'''
        return itemgetter(*tuple(map(self.columns.index, keys)))

    def _check_columns_index(self, col):
        if col is None:
            return tuple(self._columns)

        if is_str(col):
            error = '%s is not a title in current dataset' % col
            assert col in self._columns, error
            return (col,)

        if isinstance(col, int):
            if col < 0:
                col += self.shape.Col
            assert col < self.shape.Col, 'title index is out of range'
            return (self._columns[col],)

        if is_seq(col):
            return tuple(self._check_columns_index(_)[0] for _ in col)

        if isinstance(col, slice):
            start, stop, axis = self._check_slice(col)
            assert axis == 1, "don't put a row index in here"
        return self.columns[start, stop]
    
    def _check_rows_index(self, row):
        assert is_str(row) is False, 'row index must not be a str'
        if row is None:
            return range(self.shape.Ln)
        
        if isinstance(row, int):
            if row < 0:
                row += self.shape.Ln
            assert row < self.shape.Ln, 'row index is out of range'
            return (row,)

        if is_seq(row):
            return tuple(self._check_rows_index(_)[0] for _ in row)
        
        # isinstance(row, slice):
        start, stop, axis = self._check_slice(row)
        assert axis == 0, "don't put a column index in here"
        return range(start, stop)

    
    def _drop_col(self, index):
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

    
    def _drop_row(self, index):
        assert self.locked, LOCK_ERROR
        index = self._check_rows_index(index)
        for i, seq in enumerate(self.iter_values()):
            del seq[index]
            self._missing[i] = count_nan(self._isnan, seq)
        self._dim = SHEET_DIM(self._dim.Ln - len(index), self._dim.Col)
        return self

    
    def _extend(self, item):
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

    
    def _fillna(self, fill_with=None, col=None, method=None, limit=None):
        cols = self._check_columns_index(col)
        isnan_fun = self._isnan
        if limit is None:
            limit = self.shape.Ln
        assert limit >= 1, 'fill with at least 1 missing value, not limit=%s' % limit
        assert method in ('linear', 'polynomial', 'quadratic', None, 'mean', 'mode')
        if method is None:
            self._fillna_value(fill_with, cols, isnan_fun, limit)
        elif method == 'mean':
            for col in cols:
                mean = Series(_ for _ in self[col] if not isnan_fun(_)).mean()
                self._fillna_value(mean, [col], isnan_fun, limit)
        elif method == 'mode':
            for col in cols:
                seq = Series(_ for _ in self[col] if not isnan_fun(_))
                mode = Counter(seq).most_common()[0][0]
                self._fillna_value(mode, [col], isnan_fun, limit)
        else:
            func = simple_linear_reg
            self._fillna_simple_function(cols, isnan_fun, limit, func)
        return self

    def _fillna_value(self, fill_with, col, _isnan, all_limit):
        err = '`fill_with` must be a value'
        assert isinstance(fill_with, (dict,) + VALUE_TYPE), err
        if isinstance(fill_with, dict) is False:
            fill_with = dict(zip(col, repeat(fill_with)))

        for key, fill_val in fill_with.items():
            limit = all_limit
            key_index = self.columns.index(key)
            if key in col and self._missing[key_index] != 0:
                sequence = self[key]
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
    
    def _getitem_by_tuple_subcol(self, key, subset):
        '''given columns, get subset'''
        for arg in key:
            if is_str(arg):
                seq = self.data[arg]
                miss = self._missing[self._columns.index(arg)]
                subset._quickly_append_col(arg, seq, miss)

            elif isinstance(arg, slice):
                start, stop = self._check_slice_col(arg.start, arg.stop)
                for col in self._columns[start:stop + 1]:
                    miss = self._missing[self._columns.index(col)]
                    seq = self.data[col]
                    subset._quickly_append_col(col, seq, miss)
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

    
    def _group_index_by_column_value(self, columns, engine=list):
        subset = defaultdict(engine)
        for i, row in enumerate(zip(*(self._data[col] for col in columns))):
            subset[row].append(i)
        return subset

    def _getslice_col(self, i, j):
        subset = type(self)(nan=self.nan)
        for ind, col in enumerate(self._columns[i: j + 1], i):
            subset._quickly_append_col(col, self._data[col], self._missing[ind])
        return subset

    def _getslice_ln(self, i, j, k):
        subset = type(self)(nan=self.nan)
        for miss, col in zip(self._missing, self._columns):
            seq = self._data[col][i:j:k]
            if miss != 0:
                miss = count_nan(self._isnan, seq)
            subset._quickly_append_col(col, seq, miss)
        return subset

    def _get(self, key, default=None):
        if is_str(key) and key in self._columns:
            return self[key]
        if isinstance(key, int):
            if key < 0:
                key += self.shape.Ln
            if key < self.shape.Ln:
                return self[key]
        return default

    
    def _get_best_features(self, method, X, Y, top_k):
        cols = self._check_columns_index(X)
        assert method in ('variance', 'anova')
        if isinstance(top_k, float):
            top_k = int(top_k * self.shape.Col)
        assert isinstance(top_k, int) and top_k >= 0, '`top_k` must be greater than 0'
        
        if method == 'variance':
            feature_importance = [(self[_].cv(), _) for _ in cols]

        if method == 'anova':
            assert Y in self.data, 'Y must be a column in this dataset'
            from DaPy.methods import ANOVA
            feature_importance = [(ANOVA(self[_, Y], Y).F, _) for _ in cols]

        feature_importance = filter(lambda val: not self._isnan(val[0]), feature_importance)
        feature_importance = sorted(feature_importance, reverse=True)[:top_k]
        return self[tuple(_[1] for _ in feature_importance)]

    def _get_categories(self, cols, cut_points, group_name, boundary):
        from DaPy.operation import get_categories
        assert any(map(self._isnan, group_name)), '%s can not be a group name' % self.nan
        cols = self._check_columns_index(cols)
        for i, (key, seq) in enumerate(self.items):
            if key in cols:
                cate = get_categories(seq, cut_points, group_name, boundary)
                self._quickly_append_col('%s_category' % col, cate, self._missing[i])
        return self

    def _get_date_label(self, date, col, day, weekend, season):
        if day is True:
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

    def _get_dummies(self, cols, value=1):
        from DaPy import get_dummies
        cols = self._check_columns_index(cols)
        for title in cols:
            dummies = get_dummies(self._data[title], value, 'set')
            dummies.columns = [title + '=' + _ for _ in dummies.columns]
            self.join(dummies, inplace=True)
        return self

    
    def _get_interactions(self, new_features, n_power, cols):
        def combinations(arr, n):
            if n == 1:
                return [(val,) for val in arr]

            tup = set()
            for exist in combinations(arr, n - 1):
                for val in arr:
                    tup.add(tuple(sorted(exist + (val,))))
            return tup

        def create_title(title, count):
            n_title = count[title]
            if n_title != 1:
                return '%s^%d' % (title, n_title)
            return title

        cols = self._check_columns_index(cols)
        assert isinstance(n_power, int) and n_power > 1, '`n_power` must be an integer and greater than 1'
        pairs = combinations(cols, n_power)

        for combine in pairs:
            count = Counter(combine)
            title, seq = create_title(combine[0], count), self[combine[0]]
            miss_values = 0
            for col in combine[1:]:
                seq *= self._data[col]
                miss_values += self[col]
                if col not in title:
                    title += '*' + create_title(col, count)
            if miss_values > 0:
                miss_values = count_nan(self._isnan, seq)
            new_features._quickly_append_col(title, seq, miss_values)
        return new_features
    
    def _get_nan_instrument(self, instruments, cols):
        check_nan = self._isnan
        for col in self._check_columns_index(cols):
            seq = self[col].apply(lambda val: 1 if check_nan(val) else 0)
            instruments._quickly_append_col(col + '=NaN', seq, 0, pos=None)
        return instruments            

    def _get_ranks(self, ranks, cols, duplicate):
        from DaPy.operation import get_ranks
        cols = self._check_columns_index(cols)
        for col in cols:
            rank_col = get_ranks(self[col], duplicate)
            ranks._quickly_append_col(rank_col, '%s_rank' % col, 0)
        return ranks

    
    def _get_numeric_label(self, to_return, cols):
        for col in self._check_columns_index(cols):
            label = {}
            seq = self[col].apply(lambda val: label.setdefault(val, len(label)))
            to_return[col] = seq
        return labels

    
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

    def _init_nan_func(self):
        if isnan(self._nan) is True:
            self._isnan = isnan
        else:
            self._isnan = lambda val: val == self._nan

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

    def _iloc(self, subset, indexs):
        if is_seq(indexs) is False:
            indexs = tuple(indexs)
        for miss, (key, sequence) in zip(self._missing, self.iter_items()):
            seq = sequence[indexs]
            if isinstance(seq, Series) is False:
                seq = Series([seq])
            if miss != 0:
                miss = count_nan(self._isnan, seq)
            subset._quickly_append_col(key, seq, miss)
        return subset

    def _iter_groupby(self, keys, func=None, apply_col=None):
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
                subset = self.iloc(rows)
                yield operate_subset(subset, keyword)
        else:
            for keyword, rows in subsets.items():
                yield keyword, self.iloc(rows)

    
    def _insert_row(self, index, new_row):
        new_row = self._add_row(new_row)
        for value, seq in zip(new_row, self.iter_values()):
            seq.insert(index, value)

    
    def _insert_col(self, index, new_series, new_name):
        miss, new_series = self._check_sequence(new_series, self._dim.Ln)
        empty_size = len(new_series) - self.shape.Ln
        if empty_size > 0:
            for i, sequence in enumerate(self.iter_values()):
                sequence.extend(repeat(self.nan, empty_size))
                self._missing[i] += empty_size
        self._quickly_append_col(new_name, new_series, miss, index)

    
    def _join(self, other):
        assert is_value(other) is False, 'cannot join a value to the dataset.'
        error = "can't join empty object, given %s" % other
        assert (hasattr(other, 'shape') and other.shape[0] != 0) or other, error
        bias = other.shape.Ln - self.shape.Ln
        if bias > 0:
            for i, title in enumerate(self._columns):
                self._data[title].extend(repeat(self.nan, bias))
                self._missing[i] += bias
        for (title, seq), miss in zip(other.iter_items(), other._missing):
            if miss != 0:
                miss, seq = self._check_sequence(seq, self._dim.Ln)
            self._quickly_append_col(title, seq, miss)
        self._dim = SHEET_DIM(self.shape.Ln + bias, len(self._columns))
        return self

    def _match_column_from_str(self, statement):
        sorted_column = sorted(self.columns, key=len, reverse=True)
        pattern = '|'.join([PATTERN_COLUMN % x for x in sorted_column])
        useful_col = [col.strip() for col in findall(pattern, statement)]
        return [col.replace('(', '').replace(')', '') for col in useful_col]

    
    def _normalized(self, process, cols):
        assert process in ('NORMAL', 'STANDAR', 'LOG', 'BOX-COX')
        cols = self._check_columns_index(cols)
        from DaPy import describe, log, boxcox
        for title in cols:
            if process in ('NORMAL', 'STANDAR'):
                statis = describe(self._data[title])
                if None in statis:
                    continue

                if process == 'NORMAL':
                    center = float(statis.Min)
                    var = float(statis.Range)
                elif process == 'STANDAR':
                    center = float(statis.Mean)
                    var = float(statis.Sn)

                assert var != 0, 'range or std of `%s` is 0 ' % title
                self._apply_inplace(
                    lambda val: (val - center) / var,
                    cols=title,
                    axis=1)

            elif process == 'BOX-COX':
                func = lambda val: boxcox(
                    val, 
                    kwrds.get('lamda', 1), 
                    kwrds.get('a', 0),
                    kwrds.get('k', 1))
                self._apply_inplace(func, col=title, axis=1)

            elif process == 'LOG':
                self._apply_inplace(
                    lambda val: log(val, kwrds.get('base', 2.71828183)),
                    cols=title,
                    axis=1)
        return self

    
    def _pop_row(self, pop_data, index=-1):
        assert self.locked, LOCK_ERROR
        pop_index = self._check_rows_index(index)
        for i, (title, seq) in enumerate(self.iter_items()):
            pop_data[title] = seq.pop(pop_index)
            if self._missing[i] != 0:
                self._missing[i] -= pop_data[title].count(self._nan)
        self._dim = SHEET_DIM(self._dim.Ln - len(index), self._dim.Col)
        return pop_data

      
    def _pop_col(self, pop_data, col):
        pop_name = self._check_columns_index(col)
        for title in pop_name:
            pos = self._columns.index(title)
            pop_data._quickly_append_col(
                col=self._columns.pop(pos), 
                seq=self._data.pop(title), 
                miss=self._missing.pop(pos)
            )
            if title in self._sorted_index:
                pop_ind = self._sorted_index.pop(title)
                pop_data._sorted_index[title] = pop_ind
        self._dim = SHEET_DIM(self._dim.Ln, self._dim.Col - len(pop_name))
        return pop_data

    def _query(self, expression, col, limit):
        assert is_str(expression), '`expression` should be a python statement'
        select_col = self._check_columns_index(col)
        useful_col = self._match_column_from_str(expression)
        assert useful_col, "can't match any column from `expression`"
        if limit is None:
            limit = self.shape.Ln

        if all([col in self._sorted_index for col in useful_col]) is False:
            subset = self[useful_col]
            where = subset._trans_where(expression, axis=0)
            return subset._where_by_rows(where, limit), select_col
        return sorted(self._where_by_index(expression))[:limit], select_col

    def _quickly_append_col(self, col, seq, miss, pos=None):
        '''append a new column to the sheet without checking'''
        col = self._check_col_new_name(col)
        if pos is None:
            pos = len(self.columns)
        self.data[col] = seq
        self._columns.insert(pos, col)
        self._missing.insert(pos, miss)
        self._dim = SHEET_DIM(len(seq), self._dim.Col + 1)
        return self

    def _replace(self, old, new, col, regex):
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
    
    def _reverse(self, axis=0):
        assert axis in (0, 1)
        if axis == 1:
            self._columns.reverse()
            self._missing.reverse()
        else:
            for sequence in self._data.values():
                sequence.reverse()
        return self

    
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

    
    def _shuffle(self):
        new_index = range(self._dim.Ln)
        shuffles(new_index)
        return self._arrange_by_index(new_index)

    def _show(self, lines=None):
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

    def _sort(self, subset, *orderby):
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
            return type(self)(temp, self.columns)

        reverse = False
        if symbols[0] == 'DESC':
            reverse = True

        new_index = argsort(list(self.data[keys[0]]), reverse=reverse)        
        for i, (key, seq) in enumerate(self.iter_items()):
            subset._quickly_append_col(key,
                                      seq[new_index],
                                      self._missing[i])
        return subset

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

    
    def _update(self, where, **set_values):
        if callable(where) is False:
            where = self._trans_where(where, axis=0)
        assert set_values, '`set_value` are empty'
        for key, exp in set_values.items():
            if callable(exp) is False:
                exp = str(exp)
                if exp.isalpha() is True:
                    exp = '"%s"' % exp
                set_value[key] = self._trans_where(exp, axis=0)

        for index in self._where_by_rows(where, limit=self.shape.Ln):
            row = Row(self, index)
            for key, value in set_values.items():
                row[key] = value(row)
    
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
                    pass
                try:
                    return set(func(val))
                except TypeError:
                    return set()

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
        
        rows = self.iter_rows
        try:
            where(tuple())
        except AttributeError:
            rows = self.__iter__
        except TypeError:
            rows = self.__iter__
        except:
            pass 

        selected = 0
        for i, row in enumerate(rows()):
            if where(row):
                selected += 1
                yield i
                if selected == limit:
                    break
    
    def todict(self):
        return dict(self.data)

    def tolist(self):
        '''return the data as lists in list'''
        return list(map(list, self.iter_rows()))

    def toarray(self):
        '''return the data as a numpy.array object'''
        try:
            from numpy import array
        except ImportError:
            raise ImportError("can't find numpy library")
        return array(tuple(self.iter_rows()))
