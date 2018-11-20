from collections import namedtuple, deque, Iterable, deque
from datetime import datetime
from time import struct_time
from array import array
from string import atof as str2float, atoi as str2int
from distutils.util import strtobool as str2bool
from ciso8601 import parse_datetime as str2date
from warnings import warn
from re import compile

__all__ = ['str2value', 'get_sorted_index',
           'is_value', 'is_math', 'is_iter', 'is_seq']

_value_types = (type(None), int, float, str, long, complex,
                unicode, datetime, struct_time, bool)
_math_types = (int, float, long, complex)
_sequence_types = [list, tuple, deque, array, set, frozenset]
_float_mask = compile(r'^[-+]?[0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
_int_mask = compile(r'^[-+]?[-0-9]\d*$')

try:
    from numpy import ndarray
    _sequence_types.append(ndarray)
    from pandas import Series
    _sequence_types.append(Series)
except ImportError:
    pass

_sequence_types = tuple(_sequence_types)
transfer_funcs = {float: str2float,
                  int: str2int,
                  bool: str2bool,
                  datetime: str2date}

def str2str(value):
    return value  
    
def str2value(value, prefer_type=None):
    if prefer_type is not None:
        try:
            return transfer_funcs[prefer_type]
        except ValueError:
            warn('cannot transform "%s" into %s' % (value, prefer_type))
        except KeyError:
            warn('unsupported prefer type as "%s", use float, int' % prefer_type+\
                 ', bool, str, or datetime.')
            
    if _int_mask.match(value):
        return str2int(value)
    
    elif _float_mask.match(value):
        return str2float(value)

    return str2str(value)
    
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


