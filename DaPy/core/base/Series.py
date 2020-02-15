from copy import copy
from collections import Counter
from itertools import repeat, compress, accumulate
from datetime import datetime, timedelta
from operator import add, sub, mul, mod, pow
from operator import eq, gt, ge, lt, le
from operator import itemgetter
from math import sqrt
from heapq import nlargest, nsmallest
from time import clock

try:
    from numpy import darray
except ImportError:
    darray = list

from .constant import STR_TYPE, VALUE_TYPE, SEQ_TYPE, DUPLICATE_KEEP, PYTHON3, nan
from .utils import filter, map, range, xrange, zip, zip_longest
from .utils import is_iter, is_math, is_seq, is_value, isnan, auto_plus_one
from .utils.utils_isfunc import SET_SEQ_TYPE

if PYTHON3:
    from operator import truediv as div, itemgetter
else:
    from operator import div, itemgetter
    
SHAPE_UNEQUAL_WARNING = "can't broadcast together with lenth %d and %d"

def quickly_apply(operation, left, right):
    assert callable(operation) is True
    return Series(map(operation, left, right))

getter1, getter0 = itemgetter(1), itemgetter(0)

class Series(list):
    def __init__(self, array=[]):
        if is_iter(array) is False:
            array = (array,)
        list.__init__(self, array)

    @property
    def data(self):
        return self

    @property
    def shape(self):
        return (len(self), 1)

    def __repr__(self):
        if len(self) > 10:
            head = ','.join(map(str, self[:5]))
            tail = ','.join(map(str, self[-5:]))
            return 'Sereis([%s, ..., %s])' % (head, tail)
        return 'Series(%s)' % list.__repr__(self)

    def __eq__(self, other):
        other = self._check_operate_value(other)
        return quickly_apply(eq, self, other)

    def __gt__(self, other):
        other = self._check_operate_value(other)
        return quickly_apply(gt, self, other)

    def __ge__(self, other):
        other = self._check_operate_value(other)
        return quickly_apply(ge, self, other)

    def __lt__(self, other):
        other = self._check_operate_value(other)
        return quickly_apply(lt, self, other)

    def __le__(self, other):
        other = self._check_operate_value(other)
        return quickly_apply(le, self, other)

    def __setitem__(self, key, val):
        '''refresh data from current series

        Parameters
        ----------
        key : slice, int, same-size series and tuple

        val : single value or iterable container

        Return
        ------
        None
        '''
        setitem = list.__setitem__
        if isinstance(key, int):
            setitem(self, key, val)

        if isinstance(key, Series):
            err = 'Index should be same size with current series'
            assert len(key) == len(self), err
            for i, key in enumerate(key):
                if key is True:
                    setitem(self, key, val)

        if is_seq(key):
            for key in key:
                setitem(self, key, val)

        if isinstance(key, slice):
            start = 0 if key.start is None else key.start
            stop = len(self) if key.stop is None else key.stop
            step = 1 if key.step is None else key.step
            start = start if start > 0 else start + len(self)
            stop = start if stop > 0 else stop + len(self)
            val = repeat(val, int((stop - start) / 2)) if is_value(val) else val
            setitem(self, key, val)                        
    
    def __getitem__(self, key):
        '''get data from current series

        Parameters
        ----------
        key : slice, int, same-size series and tuple

        Return
        ------
        number or numbers in Series

        Example
        -------
        >>> ser = Series(range(2, 10))
        >>> ser[2:5] # get elements by slice
        [4, 5, 6]
        >>> ser[-1] # get element by index
        9
        >>> ser[ser > 4] # get elements by sequence of bool
        [5, 6, 7, 8, 9]
        >>> ser[2, 4, 2, 3] # get elements by multiple index
        [4, 6, 4, 5]
        '''
        if isinstance(key, int):
            return list.__getitem__(self, key)
        
        if isinstance(key, Series):
            assert len(key) == len(self)
            return Series(compress(self, key))

        if is_seq(key):
            if len(key) == 1:
                return Series([list.__getitem__(self, key[0])])
            
            if len(key) < len(self) * 0.1:
                return Series(map(list.__getitem__, repeat(self, len(key)), key))
        
        if is_iter(key):
            try:
                key = itemgetter(*key)
            except TypeError:
                return Series()
            
        if isinstance(key, itemgetter):
            list_self = list(self)
            return Series(key(list_self)) 
        
        if isinstance(key, slice):
            return Series(list.__getitem__(self, key))

    def __delitem__(self, key):
        '''delete data from current series

        Parameters
        ----------
        key : slice, int, same-size series and tuple

        Return
        ------
        number or numbers in Series

        Example
        -------
        >>> ser = Series(range(2, 10))
        >>> ser[2:5] # get elements by slice
        [4, 5, 6]
        >>> ser[-1] # get element by index
        9
        >>> ser[ser > 4] # get elements by sequence of bool
        [5, 6, 7, 8, 9]
        >>> ser[2, 4, 2, 3] # get elements by multiple index
        [4, 6, 4, 5]
        '''
        func = list.__delitem__
        if isinstance(key, int):
            return func(self, key)
        
        if isinstance(key, Series):
            assert len(key) == len(self)
            for ind, val in enumerate(key):
                if val:
                    func(self, ind)

        if isinstance(key, (tuple, list)):
            key = sorted(set(key), reverse=True)
            for ind in key:
                func(self, ind)

        if isinstance(key, slice):
            func(self, key)

    def __getslice__(self, start, stop):
        return Series(list.__getslice__(self, start, stop))

    def _check_operate_value(self, value):
        lself = len(self)
        if is_value(value):
            return repeat(value, lself)
        
        if hasattr(value, 'len') is False:
            value = list(value)
            
        rl = len(value)
        assert lself == rl, SHAPE_UNEQUAL_WARNING % (lself, rl)
        return value

    def __add__(self, right):
        '''[1, 2, 3] + 3 -> [4, 5, 6]
        [1, 2, 3] + [4, 5, 6] -> [5, 7, 9]
        '''
        right = self._check_operate_value(right)
        return quickly_apply(add, self, right)

    def __radd__(self, left):
        '''3 + [1, 2, 3] -> [4, 5, 6]
        '''
        left = self._check_operate_value(left)
        return quickly_apply(add, left, self)

    def __sub__(self, right):
        '''[1, 2, 3] - 3 -> [-2, -1 ,0]
        '''
        value = self._check_operate_value(right)
        return quickly_apply(sub, self, value)
    
    def __rsub__(self, left):
        '''3 - [1, 2, 3] -> [2, 1, 0]
        '''
        value = self._check_operate_value(left)
        return quickly_apply(sub, value, self)

    def __mul__(self, right):
        '''[1, 2, 3] * 3 -> [3, 6, 9]
        '''
        value = self._check_operate_value(right)
        return quickly_apply(mul, self, value)

    def __rmul__(self, left):
        '''3 * [1, 2, 3] -> [3, 6, 9]
        '''
        value = self._check_operate_value(left)
        return quickly_apply(mul, value, self)

    def __div__(self, right):
        '''[1, 2, 3] / 2 -> [0.5, 1, 1.5]
        '''
        value = self._check_operate_value(right)
        return quickly_apply(div, self, value)

    def __truediv__(self, right):
        return self.__div__(right)

    def __rdiv__(self, left):
        '''3 / [1, 2, 3] -> [3, 1.5, 1]
        '''
        value = self._check_operate_value(left)
        return quickly_apply(div, value, self)

    def __mod__(self, right):
        '''[1, 2, 3] % 3 -> [0, 0, 1]
        '''
        value = self._check_operate_value(right)
        return quickly_apply(mod, self, value)

    def __rmod__(self, left):
        '''3 % [1, 2, 3] -> [3, 1, 1]
        '''
        value = self._check_operate_value(left)
        return quickly_apply(mod, value, self)

    def __pow__(self, right):
        '''[1, 2, 3] ** 2 -> [1, 4, 9]
        '''
        value = self._check_operate_value(right)
        return quickly_apply(pow, self, value)

    def __float__(self):
        '''float([1, 2, 3]) -> [1.0, 2.0, 3.0]
        '''
        return Series(map(float, self))

    def __abs__(self):
        '''abs([-1, 2, -3]) -> [1, 2, 3]
        '''
        return Series(map(abs, self))

    def abs(self):
        return self.__abs__()

    def accumulate(self, func=None, skipna=True):
        '''return accumulate for each item in the series'''
        assert skipna in (True, False), '`skipna` must be True or False'
        values = Series(self) if skipna else self
        if skipna:
            index_nan = [i for i, val in enumerate(self) if isnan(val)]
            values[index_nan] = 0.0
        return Series(accumulate(values, func))

    def apply(self, func, *args, **kwrds):
        return Series(func(val, *args, **kwrds) for val in self)

    def argmax(self):
        max_val, max_ind = - float('inf'), None
        for ind, val in enumerate(self):
            if val > max_val:
                max_val, max_ind = val, ind
        return max_ind

    def argmin(self):
        max_val, max_ind = float('inf'), None
        for ind, val in enumerate(self):
            if val < max_val:
                max_val, max_ind = val, ind
        return max_ind

    def argsort(self, key=None, reverse=False):
        '''return the indices that would sort an array

        Parameters
        ----------
        key : function or None (default=None)

        reverse : True or False (default=False)

        Return
        ------
        Series : index of original data

        Example
        -------
        >>> Series([5, 2, 1, 10]).argsort()
        Series([2, 1, 0, 3])
        '''
        return Series(map(getter0, sorted(enumerate(self), key=getter1, reverse=reverse)))

    def between(self, left, right, boundary='both'):
        '''select the values which fall between `left` and `right`

        this function quickly select the values which are larger
        than `left` as well as smaller than `right`

        Parameters
        ----------
        left : val
            to select values which are all larger than `left`

        right : val
            to select values which are all less than `right`

        boundary : 'both', False, 'left', 'right (default='both')
        '''
        assert boundary in ('both', False, 'left', 'right')
        bound_left, bound_right = ge, ge
        if boundary in (False, 'right'):
            bound_left = gt
        if boundary in (False, 'left'):
            bound_right = gt
        def func(x):
            bound_left(left, x) and bound_rgiht(right, x)
        return Series(map(func, self))

    def cv(self):
        Ex, Ex2, length = 0, 0, float(len(self))
        if length <= 1:
            return 0
        
        for val in self:
            Ex += val
            Ex2 += pow(val, 2)
        if Ex == 0:
            return sqrt((Ex2 - Ex ** 2 / length) / (length - 1.0))
        return sqrt((Ex2 - Ex ** 2 / length) / (length - 1.0)) / (Ex / length)

    def count_values(self):
        '''return a counter object that contains frequency of values'''
        return Counter(self)

    def diff(self, lag):
        '''return a differential series that has only len(arr) - lag elements'''
        getter = list.__getitem__
        return Series(getter(self, i) - getter(self, i - lag) for i in range(lag, len(self)))

    def drop(self, todrop):
        '''remove values that matches `label` from the series'''
        if is_seq(todrop) is False:
            todrop = (todrop,)
        todrop = set(todrop)
        return Series(filter(lambda val: val not in todrop, self))

    def drop_duplicates(self, keep=['first', 'last', False]):
        assert keep in ('first', 'last', False)
        
        # find the all ununiqual values: O(n)
        val_ind = {}
        for i, value in enumerate(self):
            val_ind.setdefault(value, []).append(i)

        # get index from the quickly preview table: O(k)
        to_drop_index, keep_arg = set(), DUPLICATE_KEEP[keep]
        for value, index in val_ind.items():
            if len(index) != 1:
                to_drop_index.update(index[keep_arg])

        # drop out these index: O(n)
        return Series((val for i, val in enumerate(self) if i not in to_drop_index))

    def dropna(self):
        return Series(filter(lambda val: not isnan(val), self))
    
    def get(self, index, default=None):
        try:
            return list.__getitem__(self, index)
        except Exception:
            return default

    def has_duplicates(self):
        return len(self) != len(set(self))

    def normalize(self):
        mini, maxm = float(min(self)), max(self)
        rang = maxm - mini
        return Series(map(lambda x: (x - mini) / rang, self))
        
    def isnan(self):
        return Series(map(isnan, self))

    def map(self, func):
        '''given a map, return values that are tranformed by map

        Parameters
        ----------
        func : callable-object or dict

        Return
        ------
        Series : mapped values

        Examples
        --------
        >>> arr = dp.Series([3, 5, 7, 1])
        >>> arr.map(lambda val: val + 1)
        Series([4, 6, 8, 2])
        >>> arr.map({3: 'C'})
        Series(['C', 5, 7, 1])
        '''
        if hasattr(func, '__getitem__'):
            func = lambda val: obj.__getitem__(val) if val in obj else val
        assert callable(func), '`func` expects a callable object or dict-like object'
        return Series(map(func, self))

    def max(self, axis=0):
        return max(self)

    def max_n(self, n=1):
        return Series(nlargest(n, self))

    def min(self, axis=0):
        return min(self)

    def min_n(self, n=1):
        return Series(nsmallest(1, self))

    def mean(self, axis=0):
        return sum(self, 0.0) / len(self)

    def percenttile(self, q):
        return sorted(self)[int(q * len(self))]

    def pop(self, ind):
        if isinstance(ind, int):
            return list.pop(self, ind)
        
        if isinstance(ind, slice):
            start, stop = ind.start, ind.stop
            return Series(list.pop(self, i) for i in xrange(start, stop))
        
        if is_seq(ind):
            ind = sorted(set(ind), reverse=True)
            to_ret = Series(list.pop(self, i) for i in ind)
            to_ret.reverse()
            return to_ret
    
    def replace(self, old, new):
        return Series(new if _ == old else _ for _ in self)

    def sum(self):
        return sum(self, 0.0)

    def std(self):
        Ex, Ex2, length = 0, 0, float(len(self))
        if length <= 1:
            return 0
        
        for val in self:
            Ex += val
            Ex2 += pow(val, 2)
        return sqrt((Ex2 - Ex ** 2 / length) / (length - 1.0))

    def tolist(self):
        return list(self)

    def toarray(self):
        try:
            from numpy import array
            return array(self)
        except ImportError:
            raise ImportError("can't find numpy")

    def unique(self):
        '''return unique items in the series'''
        uniq_vals, temp_vals = Series(), set()
        additem, appitem = set.add, list.append
        for i, val in enumerate(self):
            if val not in uniq_vals:
                additem(temp_vals, val)
                appitem(uniq_vals, val)
        return uniq_vals


SET_SEQ_TYPE.add(Series)

if __name__ == '__main__':
    init = Series(xrange(2, 8))
    assert all(init == [2, 3, 4, 5, 6, 7])
    assert len(init) == 6
    assert str(init) == 'Series([2, 3, 4, 5, 6, 7])'
    assert all(init[2:5] == [4, 5, 6])
    assert init[-1] == 7
    assert all(init[init >= 4] == [4, 5, 6, 7])
    
    assert all(init + 1 == [3, 4, 5, 6, 7, 8])
    assert all(init + init == [4, 6, 8, 10, 12, 14])
    assert all(init - 1 == [1, 2, 3, 4, 5, 6])
    assert all(init - init == [0, 0, 0, 0, 0, 0])
    assert all(init * 1 == [2, 3, 4, 5, 6, 7])
    assert all(init * init == [4, 9, 16, 25, 36, 49])
    assert all(init / 2.0 == [1, 1.5, 2, 2.5, 3, 3.5])
    assert all(init / init == [1, 1, 1, 1, 1, 1])

    assert all(1.0 + init == [3, 4, 5, 6, 7, 8])
    assert all(1 - init == [-1, -2, -3, -4, -5, -6])
    assert all(1 * init == [2, 3, 4, 5, 6, 7])
    assert all(10.0 / init == [5.0, 3.3333333333333335, 2.5, 2.0, 1.6666666666666667, 1.4285714285714286])
