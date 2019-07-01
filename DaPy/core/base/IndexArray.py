from collections import Sequence
from itertools import chain
from bisect import bisect_left, bisect_right
from operator import itemgetter
from .utils import is_iter, is_value, argsort


class SortedIndex(Sequence):
    def __init__(self, array=(), index=()):
        assert is_iter(array), '`array` in the index must be iterable'
        assert is_iter(index), '`index` in the index must be iterable'
        
        if not index:
            self._val = sorted(array)
            self._ind = argsort(array)
        else:
            val_ind = sorted(zip(array, index), key=itemgetter(0))
            self._val, self._ind = tuple(zip(*val_ind))
            self._val, self._ind = list(self._val), list(self._ind)
        
        
    def __getitem__(self, indices):
        if isinstance(indices, (slice, int)):
            return self._ind[indices], self._val[indices]
        assert IndexError('can not operate like SortedIndex[%s]' % indices)

    def __len__(self):
        return len(self._val)

    def __repr__(self):
        if len(self) > 10:
            values = '[%s, ..., %s]' % (', '.join(map(str, chain(self._val[:5]))),
                                    ', '.join(map(str, chain(self._val[-5:]))))                                     
        else:
            values = str(self._val)
        return 'SortedIndex(%s)' % values

    def append(self, value):
        '''insert a single value to the exists list

        This function uses binary select to locate the position,
        than insert it into the series.
        '''
        self.insert(value, len(self._ind))

    def between(self, low, high, boundary=(True, True), return_value=False):
        '''select values which are between [low, high] and return their index

        This function uses binary select to find the values which are between
        [low, high], then it will return the index of these values for you.

        Parameters
        ----------
        low : value
            the lower boundary

        high : value
            the upper boundary

        boundary : bool or bools in tuple (default=(True, True))
            (True, True) -> values belong to [low, high]
            (True, False) -> values belong to [low, high)
            (False, True) -> values belong to (low, high]
            (False, False) -> values belong to (low, high)
        
        Returns
        -------
        index_list : the values index of original data

        Examples
        --------
        >>> original = [4, 23, 31, 33, 34, 34, 21, 23, 33]
        >>> index = SortedIndex(original)
        >>> index.between(23, 33, (True, True), True)
        [23, 23, 31, 33, 33]
        >>> index.between(23, 33, (False, True), True)
        [31, 33, 33]
        >>> index.between(23, 33, (True, False), True)
        [23, 23, 31]
        >>> index.between(23, 33, (False, False), True)
        [31]
        '''        
        if boundary[0] is False:
            low_ind = bisect_right(self._val, low)
        else:
            low_ind = bisect_left(self._val, low)
        if boundary[1] is False:
            hih_ind = bisect_left(self._val, high)
        else:
            hih_ind = bisect_right(self._val, high)
        if return_value is True:
            return self._val[low_ind:hih_ind]
        return self._ind[low_ind:hih_ind]

    def count(self, value):
        '''count how many items euqal to `value`'''
        return len(self.equal(value))

    def equal(self, value):
        '''select the value which is best match given value and return the index'''
        try:
            return self._ind[self._get_item_index(value)]
        except ValueError:
            return []
    
    def insert(self, value, index):
        '''the index of new value in original sequence is `index`'''
        item_index = bisect_left(self._val, value)
        self._val.insert(item_index, value)
        self._ind.insert(item_index, index)

    def _get_item_index(self, value):
        '''find the index of the item of the current series'''
        item_index_l = bisect_left(self._val, value)
        item_index_r = bisect_right(self._val, value)
        if item_index_l == item_index_r:
            raise ValueError('`%s` is not in this SortedIndex' % value)

        if item_index_l != item_index_r - 1:
            return slice(item_index_l, item_index_r)
        return item_index_l


    def index(self, value):
        '''find the index of an exacte matched value

        This function uses binary select to locate the position
        of the matched value in the source index.

        Parameters
        ----------
        value : anything
            something you want to match with

        original_index : bool (default=False)
            True: return the index of the value in this row
            False: return the index of the value in the source sequence

        Returns
        -------
        int : the index you want to find

        Examples
        --------
        >>> original = [1, 1, 2, 2, 2, 3, 3]
        >>> si = SortedIndex(original)
        >>> si.index(1) # index of sorted object
        [0, 1]
        >>> si.index(2)
        [2, 3, 4]
        >>> si.index(3)
        [5, 6]
        >>> si.index(5)
        ValueError(`5` is not in the current container)
        '''
        index = self._get_item_index(value)
        if isinstance(index, slice):
            return [ind for ind in self._ind[index]]
        return [self._ind[index]]
        
    def lower(self, value, include_equal=True):
        '''find `x` which is lower than `value`, x <= value

        This function uses binary select to locate the positions of values which
        are less than `value`, and return their indexes of source sequence

        Parameters
        ----------
        value : any
            the cut point you want to match

        include_equal : bool (default=True)
            True -> return x <= value
            False -> return x < value

        Returns
        -------
        list : index in list
        '''
        if include_equal is True:
            return self._ind[:bisect_right(self._val, value)]
        return self._ind[:bisect_left(self._val, value)]

    def remove(self, value):
        '''remove the value from the Index

        It will remove all items which match the value from the Index,
        then it will change all indexes which have impacts.

        Parameters
        ----------
        value : any
            the value you want to remove

        Returns
        -------
        None

        Example
        -------
        >>> source = [3, 4, 2, 4, 5]
        >>> index = SortedIndex(source)
        >>> index.equal(5)
        4
        >>> index.remove(4)
        >>> index
        SortedIndex([2, 3, 5])
        >>> index.equal(5)
        2
        
'''
        remove_index = self._get_item_index(value)
        value_index = sorted(self._ind[remove_index])
        del self._ind[remove_index], self._val[remove_index]
        self._ind = [ind - bisect_left(value_index, ind) for ind in self._ind]
            
    def upper(self, value, include_equal=True):
        '''find `x` which is greater than `value`, x >= value

        This function uses binary select to locate the positions of values which
        are less than `value`, and return their indexes of source sequence

        Parameters
        ----------
        value : any
            the cut point you want to match

        include_equal : bool (default=True)
            True -> return x >= value
            False -> return x > value

        Returns
        -------
        list : index in list
        '''
        if include_equal is True:
            return self._ind[bisect_left(self._val, value):]
        return self._ind[bisect_right(self._val, value):]

    def unequal(self, value):
        '''select the items which are not equal to `value`'''
        equal_index = set(self.equal(value))
        return list(set(self._ind) - equal_index)

if __name__ == '__main__':
##    from time import clock
####    from cProfile import run
####    run("SortedIndex(xrange(4300000))")
##    t0 = clock()
##    index = SortedIndex(xrange(4300000))
##    print clock() - t0
    original = [4, 23, 31, 33, 34, 34, 21, 23, 33]
    index = SortedIndex(original)
    assert index.between(23, 33, (True, True), True) == [23, 23, 31, 33, 33]
    assert index.between(23, 33, (False, True), True) == [31, 33, 33]
    assert index.between(23, 33, (True, False), True) == [23, 23, 31]
    assert index.between(23, 33, (False, False), True) == [31]
    assert index.index(23) == [1, 7]
    index.append(9)
    index.append(9)
    original.extend([9, 9])
    assert str(index), 'SortedIndex([4, 9, 9, 21, 23, ..., 31, 33, 33, 34, 34])'
    assert itemgetter(*index.lower(9, True))(original) == (4, 9, 9)
    assert itemgetter(*index.lower(9, False))(original) == 4
    assert index.equal(23) == [1, 7]

    index = SortedIndex(['Jackson', 'Martin', 'John', 'Alice', 'Bob', 'Baker'])
    assert index.equal('Bob') == 4
