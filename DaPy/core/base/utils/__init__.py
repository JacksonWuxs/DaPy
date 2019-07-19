from math import isnan as _isnan
from re import compile as _compile

from DaPy.core.base.constant import PYTHON2, PYTHON3, STR_TYPE

from .utils_2to3 import (filter, map, pickle, range, split,
                         strip, xrange, zip, zip_longest)
from .utils_isfunc import is_empty, is_iter, is_math, is_seq, is_str, is_value
from .utils_str_transfer import _str2bool, _str2date, _str2percent

__all__ = ['str2value', 'argsort', 'hash_sort',
           'is_value', 'is_math', 'is_iter', 'is_seq', 
           'range', 'xrange', 'map', 'zip', 'filter']

# since these functions are commonly used,
# we use saching mechanisms to optimize them.
if PYTHON3 is True:
    from functools import lru_cache
else:
    from string import atof as str2float, atoi as str2int
    from repoze.lru import lru_cache

str2float = float

@lru_cache(maxsize=None)
def str2int(value):
    return int(value)

@lru_cache(maxsize=None)
def str2bool(value):
    return _str2bool(value)

@lru_cache(maxsize=None)
def str2date(value):
    return _str2date(value)

str2percent = _str2percent

def isnan(value):
    try:
        return _isnan(value)
    except TypeError:
        return False

# following masks are used to recognize string patterns
FLOAT_MASK = _compile(r'^[-+]?[0-9]\d*\.\d*$|[-+]?\.?[0-9]\d*$')
PERCENT_MASK = _compile(r'^[-+]?[0-9]\d*\.\d*%$|[-+]?\.?[0-9]\d*%$')
INT_MASK = _compile(r'^[-+]?[-0-9]\d*$')
DATE_MASK = _compile('^(?:(?!0000)[0-9]{4}([-/.]?)(?:(?:0?[1-9]|1[0-2])([-/.]?)(?:0?[1-9]|1[0-9]|2[0-8])|(?:0?[13-9]|1[0-2])([-/.]?)(?:29|30)|(?:0?[13578]|1[02])([-/.]?)31)|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:0[48]|[2468][048]|[13579][26])00)([-/.]?)0?2([-/.]?)29)$')
BOOL_MASK = _compile('^(true)|(false)|(yes)|(no)|(\u662f)|(\u5426)|(on)|(off)$')

def auto_str2value(value, dtype=None):
    '''using preview masks to auto transfer a string to matchest date type

    Parameters
    ----------
    value : str 
        the string object that you expect to transfer

    dtype : str (default=None)
        "float" -> transfer value into float type
        "int" -> transfer value into int type
        "bool" -> transfer value into bool type
        "datetime" -> transfer value into datetime type
        "percent" -> str2value.auto("3.3", 'percent') -> 0.033
        "str" -> drop out all blanks in the both sides

    Examples
    --------
    >>>> str2value.auto('3')
    3
    >>> str2value.auto('3.3')
    3.3
    >>> str2value.auto(' 3.3.')
    '3.3.'
    >>> str2value.auto('2019-3-23')
    datetime.datetime(2019, 3, 23, 0, 0)
    >>> str2value.auto('3.3%')
    0.033
    >>> str2value.auto('Yes')
    True
    '''
    if dtype is not None:
        assert isinstance(dtype, STR_TYPE), 'prefer_type should be a string'
        assert dtype.lower() in ('float', 'int', 'bool', 'datetime', 'str')
        return fast_str2value[dtype](value)

    if INT_MASK.match(value):
        return str2int(value)
    
    elif FLOAT_MASK.match(value):
        return str2float(value)

    elif PERCENT_MASK.match(value):
        return str2percent(value)

    elif DATE_MASK.match(value):
        return str2date(value)

    elif BOOL_MASK.match(value.lower()):
        return str2bool(value)

    else:
        return value

# this parser offers a higher speed than `if else` statement
fast_str2value = {'float': str2float,
            'int': str2int,
            'bool': str2bool,
            'datetime': str2date,
            'percent': str2percent,
            'str': lambda x: x}

def argsort(seq, key=None, reverse=False):
    '''sort a sequence than return the index of sequence

    Parameters
    ----------
    See: sorted

    Return
    ------
    list : index of original data

    Example
    -------
    >>> argsort([5, 2, 1, 10])
    [2, 1, 0, 3]
    '''
    if hasattr(seq, '__getitem__'):
        return sorted(xrange(len(seq)), key=seq.__getitem__, reverse=reverse)
    return list(map(itemgetter(0), sorted(enumerate(seq), key=itemgetter(1), reverse=reverse)))

def hash_sort(records, *orders):
    assert all(map(lambda x: isinstance(x[0], int), orders)), 'keyword should be int'
    assert all(map(lambda x: x[1] in ('ASC', 'DESC'), orders)), 'orders symbol should be "ASC" or "DESC"'

    compare_pos = [x[0] for x in orders]
    compare_sym = [x[1] for x in orders]
    size_orders = len(compare_pos) - 1
    
    def _hash_sort(datas_, i=0):
        # initialize values
        index = compare_pos[i]
        inside_data, HashTable = list(), dict()

        # create the diction
        for item in datas_:
            key = item[index]
            if key in HashTable:
                HashTable[key].append(item)
            else:
                HashTable[key] = [item]

        # sorted the values
        sequence = sorted(HashTable)

        # transform the record into Frame
        for each in sequence:
            items = HashTable[each]
            if i < size_orders:
                items = _hash_sort(items, i+1)
            inside_data.extend(items)

        # finally, reversed the list if necessary.
        if i != 0 and compare_sym[i] != compare_sym[i-1]:
            inside_data.reverse()
        return inside_data
    
    output = _hash_sort(records)
    if compare_sym[0] == 'DESC':
        output.reverse()
    return output

def auto_plus_one(exists, item, start=1):
    exists = set(map(str, exists))
    while '%s_%d' % (item, start) in exists:
        start += 1
    return '%s_%d' % (item, start)

def count_nan(nan_func, series):
    return [nan_func(v) for v in series].count(True)
