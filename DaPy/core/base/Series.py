from copy import copy
from datetime import datetime, timedelta

try:
    from numpy import darray
except ImportError:
    darray = list

from .constant import STR_TYPE, VALUE_TYPE, SEQ_TYPE
from .utils import filter, map, range, xrange, zip
from .utils import is_iter, is_math, is_seq, is_value, auto_plus_one


class Series(object):
    def __init__(self, data=None, index=None, **kwrds):
        self._column = kwrds.get('column', 'Series')
        self._miss_symbol = kwrds.get('nan', None)
        assert isinstance(self._column, STR_TYPE), "name should be a str"
        assert is_value(self._miss_symbol), "missing value should be a value"
        assert is_iter(index) or index is None, "index object is not iterable"
        assert is_iter(data) or data is None, 'data object is not iterable'

        self._index = []
        self._data = self._check_data(data)
        if index is None and hasattr(data, '_index'):
            index = data._index
        self._index = self._check_index(index)
        self._miss_value = self.data.count(self._miss_symbol)
        if len(self._index) != len(self._data):
            raise ValueError("Length of series uneuqals to index")

    @property
    def column(self):
        return self._column

    @column.setter
    def column(self, value):
        assert isinstance(value, STR_TYPE)
        self._column = value

    @property
    def data(self):
        return self._data
            
    @property
    def index_(self):
        return copy(self._index)

    @index_.setter
    def index_(self, value):
        self._index = self._check_index(value)

    def __repr__(self):
        return self._data.__repr__()

    def _check_data(self, value):
        if value is None:
            return list()
        if isinstance(value, list) is False:
            return list(value)
        return value

    def _check_index(self, value):
        if value is None:
            value= range(len(self._data))
        return [self._autoindex(value) for value in value]

    def _autoindex(self, index):
        if index is None:
            return self._autoindex(len(self))
        
        if hasattr(self, 'index') and index not in self.index_:
            return index

        if isinstance(index, int):
            return self._autoindex(index + 1)

        if isinstance(index, datetime):
            return self._autoindex(index + timedelta(days=1))
        return auto_plus_one(self._index, index)
        

    def __contains__(self, value):
        return value in self._data

    def __delitem__(self, index):
        self._data.__delitem__(self._index.index(index))
        self._index.__delitem__(self._index.index(index))

    def __eq__(self, other):
        return Series(map(lambda x: x == other, self), self._index)

    def __getitem__(self, index):
        if isinstance(index, Series) and len(index) == len(self):
            indexs = [i for i, value in zip(index._index, index) if value is True]
            return Series(self.__getitem__(indexs), indexs)
        
        if isinstance(index, (list, tuple)):
            index = list(map(self.index_.index, index))
            sequence = map(self._data.__getitem__, index)
            return Series(sequence, index)
        if index in set(self.index_):
            return self._data.__getitem__(self.index_.index(index))
        if isinstance(index, int):
            return self._data.__getitem__(index)
        raise ValueError('"%s" is not a index in the series' % index)

    def __getslice__(self, start, stop, step=None):
        assert isinstance(step, int) or step is None, 'step should be an int or None'
        if isinstance(start, STR_TYPE):
            start = self._index.index(start)
        if isinstance(start, STR_TYPE):
            stop = self._index.index(stop)
        return Series(self._data[start:stop:step])

    def __iter__(self):
        for value in self._data:
            yield value

    def __len__(self):
        return self._data.__len__()

    def __setitem__(self, index, value):
        self._data[self._index.index(index)] = value
        self._miss_value = self._data.count(self._miss_symbol)

    def __setslice__(self, start, stop, value):
        i = self._index.index(start)
        j = self._index.index(stop)
        self._data.__setslice__(i, j, value)
        self._miss_value = self._data.count(self._miss_symbol)

    def __add__(self, right):
        '''[1, 2, 3] + 3 -> [4, 5, 6]
        '''
        if is_iter(right) is True:
            return Series([x + v for x, v in zip(self, right)], self._index)
        return Series(map(lambda x: x + right, self), self._index)

    def __radd__(self, left):
        '''3 + [1, 2, 3] -> [4, 5, 6]
        '''
        if is_iter(left) is True:
            return Series([v + x for x, v in zip(self, left)], self._index)
        return Series(map(lambda x: left + x, self), self._index)

    def __sub__(self, right):
        '''[1, 2, 3] - 3 -> [-2, -1 ,0]
        '''
        if is_iter(right) is True:
            return Series([x - v for x, v in zip(self, right)], self._index)
        return Series(map(lambda x: x - right, self), self._index)
        
    def __isub__(self, left):
        '''3 - [1, 2, 3] -> [2, 1, 0]
        '''
        if is_iter(left) is True:
            return Series([v - x for x, v in zip(self, left)], self._index)
        return Series(map(lambda x: left - x, self), self._index)

    def __mul__(self, right):
        '''[1, 2, 3] * 3 -> [3, 6, 9]
        '''
        if is_iter(right) is True:
            return Series([x * v for x, v in zip(self, right)], self._index)
        return Series(map(lambda x: x * right, self), self._index)

    def __imul__(self, left):
        '''3 * [1, 2, 3] -> [3, 6, 9]
        '''
        if is_iter(left) is True:
            return Series([v * x for x, v in zip(self, left)], self._index)
        return Series(map(lambda x: left * x, self), self._index)

    def __div__(self, right):
        '''[1, 2, 3] / 2 -> [0.5, 1, 1.5]
        '''
        if is_iter(right) is True:
            return Series([x / v for x, v in zip(self, right)], self._index)
        return Series(map(lambda x: x / right, self), self._index)

    def __idiv__(self, left):
        '''3 / [1, 2, 3] -> [3, 1.5, 1]
        '''
        if is_iter(left) is True:
            return Series([v / x for x, v in zip(self, left)], self._index)
        return Series(map(lambda x: left / x, self), self._index)

    def __mod__(self, right):
        '''[1, 2, 3] % 3 -> [0, 0, 1]
        '''
        if is_iter(right) is True:
            return Series([x % v for x, v in zip(self, right)], self._index)
        return Series(map(lambda x: x % right, self), self._index)

    def __imod__(self, left):
        '''3 % [1, 2, 3] -> [3, 1, 1]
        '''
        if is_iter(left) is True:
            return Series([v % x for x, v in zip(self, left)], self._index)
        return Series(map(lambda x: left % x, self), self._index)

    def __pow__(self, right):
        '''[1, 2, 3] ** 2 -> [1, 4, 9]
        '''
        if is_iter(right) is True:
            return Series([x ** v for x, v in zip(self, right)], self._index)
        return Series(map(lambda x: x ** right, self), self._index)

    def __float__(self):
        '''float([1, 2, 3]) -> [1.0, 2.0, 3.0]
        '''
        return Series(map(float, self), self._index)

    def __abs__(self):
        '''abs([-1, 2, -3]) -> [1, 2, 3]
        '''
        return Series(map(abs, self), self._index)

    def append(self, value, index=None):
        self.index_.append(self._autoindex(index))
        self._data.append(value)

    def abs(self):
        return self.__abs__()

    def insert(self, position, value, index=None):
        if index is None and all(map(lambda x: isinstance(x, int), self.index_)):
            self.index_.insert(position, position)
        else:
            self.index_.insert(position, self._autoindex(index))
        self._data.insert(position, value)

    def index(self, value):
        return self._data.index(value)

    def extend(self, other):
        assert hasattr(other, '__iter__')
        other = Series(other)
        self._index.extend(other.index_)
        self._data.extend(other)

    def pop(self, index):
        index = self.index_.index(index)
        self.index_.pop(index)
        return self._data.pop(index)

    def remove(self, value):
        index = self._data.index(value)
        del self.index_[index], self._data[index]

    def count(self, value):
        if isinstance(value, VALUE_TYPE):
            try:
                return self._data.count(value)
            except ValueError:
                return 0
        return [self.count(value) for value in value]

    def apply(self, func, inplace=False):
        assert inplace in (True, False)
        assert callable(func), 'func expects callable object'

        if inplace is False:
            return Series(map(func, self._data), self._index)
        self.data = map(func, self._data)

    def idxmax(self):
        pass

    def idxmin(self, skipna=False):
        pass

    def sort(self, key=None, reverse=False):
        pass

    def between(self, left, right):
        pass

    def count(self, value, range='all'):
        pass

    def count_element(self):
        pass

    def drop(self, label):
        pass

    def drop_duplicates(self, keep=['first', 'last', 'False'], inplace=False):
        pass

    def dropna(self, inplace=False):
        pass

    def normalize(self):
        pass

    def isna(self):
        pass

    def keys(self):
        pass

    def map(self, func):
        pass

    def max(self):
        pass

    def min(self):
        pass

    def mean(self):
        pass

    def quantile(self, q):
        pass
    
    def replace(self):
        pass

    def select(self):
        pass

    def to_list(self):
        return self._data

    def to_array(self):
        try:
            from numpy import array
        except ImportError:
            raise ImportError("can't find numpy")
        else:
            return array(self._data)

SEQ_TYPE += (Series,)
