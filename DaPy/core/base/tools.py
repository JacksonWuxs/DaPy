from collections import namedtuple, deque, Iterable, deque
from datetime import datetime
from time import struct_time
from array import array
from distutils.util import strtobool
from warnings import warn
from re import compile, search
from sys import version_info

try:
    from string import atof as str2float, atoi as str2int, strip
except ImportError:
    str2float, str2int, strip = float, int, lambda x: x.strip()

try:
    from ciso8601 import parse_datetime as str2date
except ImportError:
    try:
        from dateutil.parser import parse as str2date
        warn('`ciso8601` not found, uses `dateutil` instead for parsing datetime.')
    except ImportError:
        warn('`dateutil` not found, DaPy can not auto-parse date type from text.')
        str2date = strip

__all__ = ['str2value', 'get_sorted_index',
           'is_value', 'is_math', 'is_iter', 'is_seq']


_value_types = [type(None), int, float, str, complex,
                datetime, struct_time, bool]
_str_types = [str]
_math_types = [int, float, complex]
_sequence_types = [list, tuple, deque, array, set, frozenset]
_float_mask = compile(r'^[-+]?[0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
_int_mask = compile(r'^[-+]?[-0-9]\d*$')
_date_mask = compile('^(?:(?!0000)[0-9]{4}([-/.]?)(?:(?:0?[1-9]|1[0-2])([-/.]?)(?:0?[1-9]|1[0-9]|2[0-8])|(?:0?[13-9]|1[0-2])([-/.]?)(?:29|30)|(?:0?[13578]|1[02])([-/.]?)31)|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:0[48]|[2468][048]|[13579][26])00)([-/.]?)0?2([-/.]?)29)$')
_bool_mask = compile('(true)|(false)|(yes)|(no)|(\u662f)|(\u5426)')

try:
    from numpy import ndarray
    _sequence_types.append(ndarray)
    from pandas import Series
    _sequence_types.append(Series)
except ImportError:
    pass

if version_info.major == 2:
    _value_types.extend([unicode, long])
    _math_types.append(long)
    _str_types.append(unicode)
    range, filter_, map_ = xrange, filter, map

if version_info.major == 3:
    filter_ = lambda fun, seq: list(filter(fun, seq))
    range = range
    def map_(fun, *sequence):
        return list(map(fun, *sequence))

try:
    import cPickle as pickle
except ImportError:
    import pickle

_value_types, _math_types, _sequence_types, _str_types = map(tuple,
                                                        [_value_types,
                                                         _math_types,
                                                         _sequence_types,
                                                         _str_types])

def str2str(value):
    return strip(value)

def str2bool(value):
    if strtobool(value) == 1:
        return True
    return False

transfer_funcs = {float: str2float,
                  int: str2int,
                  bool: str2bool,
                  datetime: str2date}
    
def str2value(value, prefer_type=None):
    if prefer_type != None:
        try:
            return transfer_funcs[prefer_type](value)
        except ValueError:
            warn('cannot transform "%s" into %s' % (value, prefer_type))
        except KeyError:
            warn('unsupported prefer type as "%s", use float, int' % prefer_type+\
                 ', bool, str, or datetime.')
    
    if _int_mask.match(value):
        return str2int(value)
    
    elif _float_mask.match(value):
        return str2float(value)

    elif _date_mask.match(value):
        return str2date(value)

    elif _bool_mask.match(value.lower()):
        try:
            return str2bool(value)
        except ValueError:
            if value == '\u662f':
                return 1
            return 0
    else:
        return str2str(value)
    
def get_sorted_index(seq, cmp=None, key=None, reverse=False):
    if all(map(is_value, seq)):
        return [value[0] for value in\
                sorted(enumerate(seq), cmp=cmp, key=lambda x: x[1], reverse=reverse)]
    
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

def auto_plus_one(exists, item, start=1):
    exists = '|'.join(map(str, exists)) + '|'
    while True:
        if not search('%s_%d|' % (item, start), exists):
            return '%s_%d' % (item, start)
        start += 1


