from re import compile, search
from sys import version_info
from collections import Iterable, deque, Counter, OrderedDict
from .constant import MATH_TYPE, SEQ_TYPE, STR_TYPE, VALUE_TYPE

__all__ = ['str2value', 'get_sorted_index', 'hash_sort',
           'is_value', 'is_math', 'is_iter', 'is_seq', 
           'range', 'xrange', 'map', 'zip', 'filter']

if version_info.major == 2:
    from itertools import izip, imap, ifilter
    from string import split
    range, xrange, map, zip, filter = range, xrange, imap, izip, ifilter

if version_info.major == 3:
    xrange = range
    range = lambda x: list(xrange(x))
    split, map, filter, zip = str.split, map, filter, zip

try:
    import cPickle as pickle
except ImportError:
    import pickle

# basic function that transfer a string to float, int or striped string
from distutils.util import strtobool as _strtobool
try:
    from string import atof as str2float, atoi as str2int, strip
except ImportError:
    str2float, str2int, strip = float, int, str.strip
    
try:
    from dateutil.parser import parse as _str2date
except ImportError:
    from datetime import datetime
    def _str2date(value, day='1900-1-1', time='0:0:0'):
        if ' ' in value:
            day, time = value.split(' ')
        elif ':' in value:
            time = value
        elif '-' in value:
            day = value
        day, time = map(int, day.split('-')), map(int, time.split(':'))
        return datetime(day[0], day[1], day[2], time[0], time[1], time[2])

def str2date(value):
    try:
        return _str2date(value)
    except ValueError:
        return value

def str2bool(value):
    try:
        if value == u'\u662f' or _strtobool(value) == 1:
            return True
    except ValueError:
        pass
    return False

def str2percent(value):
    return str2float(value.replace('%', '')) / 100.0

# following masks are used to recognize string patterns
FLOAT_MASK = compile(r'^[-+]?[0-9]\d*\.\d*$|[-+]?\.?[0-9]\d*$')
PERCENT_MASK = compile(r'^[-+]?[0-9]\d*\.\d*%$|[-+]?\.?[0-9]\d*%$')
INT_MASK = compile(r'^[-+]?[-0-9]\d*$')
DATE_MASK = compile('^(?:(?!0000)[0-9]{4}([-/.]?)(?:(?:0?[1-9]|1[0-2])([-/.]?)(?:0?[1-9]|1[0-9]|2[0-8])|(?:0?[13-9]|1[0-2])([-/.]?)(?:29|30)|(?:0?[13578]|1[02])([-/.]?)31)|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:0[48]|[2468][048]|[13579][26])00)([-/.]?)0?2([-/.]?)29)$')
BOOL_MASK = compile('^(true)|(false)|(yes)|(no)|(\u662f)|(\u5426)|(on)|(off)$')

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

def get_sorted_index(seq, index=None, key=None, reverse=False):
    if index is None:
        index = xrange(len(seq))
    return [value[0] for value in sorted(enumerate(seq), key=lambda x: x[1], reverse=reverse)]

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

def is_value(n):
    '''Determine that if a variable is a value

    Return
    ------
    Bool : the result of evaluation.
        True - input is a value
        False - input is not a value
    '''
    if isinstance(n, VALUE_TYPE):
        return True
    return False

def is_math(n):
    '''Determine that if a variable is a number

    Return
    ------
    Bool : the result of evaluation.
        True - 'n' is a number
        False - 'n' is not a number
    '''
    if isinstance(n, MATH_TYPE):
        return True
    return False

def is_iter(obj):
    '''Determine that if a variable is a iterable
    '''
    try:
        if isinstance(obj, Iterable):
            return True
    except TypeError:
        return False
    else:
        return False

def isnan(value):
    if isinstance(value, float):
        if value > 0 or value < 0 or value == 0:
            return False
        return True
    return False

def is_seq(obj):
    ''' Determine that if a variable is a sequence object
    '''
    if isinstance(obj, SEQ_TYPE):
        return True
    return False

def is_empty(obj):
    '''determine whether a object is empty'''
    if hasattr(obj, 'empty'):
        return obj.empty
    
    if isinstance(obj, (dict, list, deque, Counter, OrderedDict, set)):
        if obj in (dict(), list(), deque(), Counter(), OrderedDict(), set()):
            return True
        return False
    
    if hasattr(obj, '__len__'):
        if len(obj) == 0:
            return True
        return False
    
    if is_iter(obj):
        return False
    raise TypeError('can not determine whether %s is empty' % type(obj))

def auto_plus_one(exists, item, start=1):
    exists = set(map(str, exists))
    while '%s_%d' % (item, start) in exists:
        start += 1
    return '%s_%d' % (item, start)
