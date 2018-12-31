from .tools import is_value, is_math, is_iter, is_seq, _str_types
from .tools import auto_plus_one
from .tools import range, filter_ as filter, map_ as map, zip_ as zip
from datetime import datetime, timedelta

class BaseSeries:
    def __init__(self, data=None, index=None, **kwrds):
        self._column = kwrds.get('column', 'Series')
        self._miss_symbol = kwrds.get('na', None)
        assert isinstance(self._column, _str_types), "name should be a str"
        assert is_value(self._miss_symbol), "missing value should be a value"
        assert is_iter(index) or index is None, "index object is not iterable"
        assert is_iter(data) or data is None, 'data object is not iterable'

        self._index = []
        self._data = self._check_data(data)
        self._index = self._check_index(index)
        self._miss_value = self.data.count(self._miss_symbol)
        if len(self._index) != len(self._data):
            raise ValueError("Length of series uneuqals to index")

    @property
    def column(self):
        return self._column

    @column.setter
    def column(self, value):
        assert isinstance(value, _str_types)
        self._column = value

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = self._check_data(value)
            
    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = self._check_index(value)

    def __repr__(self):
        return self._data.__repr__()

    def _check_data(self, value):
        if value is None:
            return list()
        return list(value)

    def _check_index(self, value):
        if value is None:
            return list(range(len(self._data)))
        return map(self._autoindex, value)

    def _autoindex(self, index):
        if index is None:
            return self._autoindex(len(self))

        if hasattr(self, 'index') and index not in self.index:
            return index

        if isinstance(index, int):
            return self._autoindex(index + 1)

        if isinstance(index, datetime):
            return _autoindex(index + timedelta(days=1))
        
        return auto_plus_one(self._index, index)
        

    def __contains__(self, value):
        return value in self._data

    def __delitem__(self, index):
        self._data.__delitem__(self._index.index(index))
        self._index.__delitem__(self._index.index(index))

    def __eq__(self, other):
        return BaseSeries(map(lambda x: x == other, self), self._index)

    def __getitem__(self, index):
        if isinstance(index, BaseSeries) and len(index) == len(self):
            indexs = [i for i, value in zip(index._index, index) if value is True]
            return BaseSeries(self.__getitem__(indexs), indexs)
        
        if isinstance(index, (list, tuple)):
            return BaseSeries([self[i] for i in index], index)
        
        return self._data.__getitem__(self._index.index(index))

    def __getslice__(self, start, stop, step):
        assert isinstance(step, int), 'step should be a integer'
        i = self._index.index(start)
        j = self._index.index(stop)
        return self._data.__getslice__(i, j, step)

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
            return BaseSeries([x + v for x, v in zip(self, right)], self._index)
        return BaseSeries(map(lambda x: x + right, self), self._index)

    def __radd__(self, left):
        '''3 + [1, 2, 3] -> [4, 5, 6]
        '''
        if is_iter(left) is True:
            return BaseSeries([v + x for x, v in zip(self, left)], self._index)
        return BaseSeries(map(lambda x: left + x, self), self._index)

    def __sub__(self, right):
        '''[1, 2, 3] - 3 -> [-2, -1 ,0]
        '''
        if is_iter(right) is True:
            return BaseSeries([x - v for x, v in zip(self, right)], self._index)
        return BaseSeries(map(lambda x: x - right, self), self._index)
        
    def __isub__(self, left):
        '''3 - [1, 2, 3] -> [2, 1, 0]
        '''
        if is_iter(left) is True:
            return BaseSeries([v - x for x, v in zip(self, left)], self._index)
        return BaseSeries(map(lambda x: left - x, self), self._index)

    def __mul__(self, right):
        '''[1, 2, 3] * 3 -> [3, 6, 9]
        '''
        if is_iter(right) is True:
            return BaseSeries([x * v for x, v in zip(self, right)], self._index)
        return BaseSeries(map(lambda x: x * right, self), self._index)

    def __imul__(self, left):
        '''3 * [1, 2, 3] -> [3, 6, 9]
        '''
        if is_iter(left) is True:
            return BaseSeries([v * x for x, v in zip(self, left)], self._index)
        return BaseSeries(map(lambda x: left * x, self), self._index)

    def __div__(self, right):
        '''[1, 2, 3] / 2 -> [0.5, 1, 1.5]
        '''
        if is_iter(right) is True:
            return BaseSeries([x / v for x, v in zip(self, right)], self._index)
        return BaseSeries(map(lambda x: x / right, self), self._index)

    def __idiv__(self, left):
        '''3 / [1, 2, 3] -> [3, 1.5, 1]
        '''
        if is_iter(left) is True:
            return BaseSeries([v / x for x, v in zip(self, left)], self._index)
        return BaseSeries(map(lambda x: left / x, self), self._index)

    def __mod__(self, right):
        '''[1, 2, 3] % 3 -> [0, 0, 1]
        '''
        if is_iter(right) is True:
            return BaseSeries([x % v for x, v in zip(self, right)], self._index)
        return BaseSeries(map(lambda x: x % right, self), self._index)

    def __imod__(self, left):
        '''3 % [1, 2, 3] -> [3, 1, 1]
        '''
        if is_iter(left) is True:
            return BaseSeries([v % x for x, v in zip(self, left)], self._index)
        return BaseSeries(map(lambda x: left % x, self), self._index)

    def __pow__(self, right):
        '''[1, 2, 3] ** 2 -> [1, 4, 9]
        '''
        if is_iter(right) is True:
            return BaseSeries([x ** v for x, v in zip(self, right)], self._index)
        return BaseSeries(map(lambda x: x ** right, self), self._index)

    def __float__(self):
        '''float([1, 2, 3]) -> [1.0, 2.0, 3.0]
        '''
        return BaseSeries(map(float, self), self._index)

    def __abs__(self):
        '''abs([-1, 2, -3]) -> [1, 2, 3]
        '''
        return BaseSeries(map(abs, self), self._index)

    def append(self, value, index=None):
        pass

    def apply(self, func, args=(), **kwrds):
        pass

    def idxmax(self):
        pass

    def idxmin(self, skipna=False):
        pass

    def sort(self, **kwrds):
        pass

    def between(self, left, right):
        pass

    def corr(self, other, method='pearson'):
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

    def items(self):
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


if __name__ == '__main__':
    s = BaseSeries()
