from collections import namedtuple, deque, Iterable, deque
from datetime import datetime
from time import struct_time
from array import array
from string import atof, atoi, strip
from distutils.util import strtobool

__all__ = ['str2value', 'get_sorted_index',
           'is_value', 'is_math', 'is_iter', 'is_seq']

_value_types = (type(None), int, float, str, long, complex,
                unicode, datetime, struct_time, bool)
_math_types = (int, float, long, complex)
_sequence_types = [list, tuple, deque, array, set, frozenset]

try:
    from numpy import ndarray
except ImportError:
    pass
else:
    _sequence_types.append(ndarray)

try:
    from pandas import Series
except ImportError:
    pass
else:
    _sequence_types.append(Series)

_sequence_types = tuple(_sequence_types)

transfer_funcs = {float: atof,
                  int: atoi,
                  bool: strtobool,
                  str: strip}

def str2value(value, prefer_type=None):
    if prefer_type is str:
        return value
    elif value.isdigit() or value[1:].isdigit():
        if prefer_type is float:
            return atof(value.replace(',', ''))
        return atoi(value)
    elif value.count('.') == 1:
        return atof(value.replace(',', ''))
    elif prefer_type is bool:
        try:
            return strtobool(value)
        except ValueError:
            pass
    return strip(value)
    
def get_sorted_index(seq, cmp=None, key=None, reverse=False):
    index_dict = dict()
    for i, value in enumerate(seq):
        if value in index_dict:
            index_dict[value].append(i)
        else:
            index_dict[value] = [i, ]

    new_value = sorted(index_dict.keys(), cmp, key, reverse)
    new_index = list()
    for value in new_value:
        new_index.extend(index_dict[value])
    return new_index

def is_value(n):
    '''Determine that if a variable is a value

    Return
    ------
    Bool : the result of evaluation.
        True - input is a value
        False - input is not a value
    '''
    if isinstance(n, _value_types):
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
    if isinstance(n, _math_types):
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

def is_seq(obj):
    ''' Determin that if a variable is a sequence object
    '''
    if isinstance(obj, _sequence_types):
        return True
    return False


