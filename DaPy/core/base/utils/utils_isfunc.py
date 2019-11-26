from operator import itemgetter
from collections import Iterable, OrderedDict
from DaPy.core.base.constant import PYTHON3, PYTHON2
from DaPy.core.base.constant import MATH_TYPE, SEQ_TYPE, STR_TYPE, VALUE_TYPE

SET_VALUE_TYPE = set(VALUE_TYPE)
SET_STR_TYPE = set(STR_TYPE)
SET_MATH_TYPE = set(MATH_TYPE)
SET_SEQ_TYPE = set(SEQ_TYPE)

DICT_TYPE = (dict, OrderedDict)
def is_dict(val):
    return isinstance(val, DICT_TYPE) or hasattr(val, 'items')

def is_value(n):
    '''Determine that if a value is a value

    Return
    ------
    Bool : the result of evaluation.
        True - input is a value
        False - input is not a value
    '''
    if type(n) in SET_VALUE_TYPE:
        return True
    return False

def is_math(n):
    '''Determine that if a value is a number

    Return
    ------
    Bool : the result of evaluation.
        True - 'n' is a number
        False - 'n' is not a number
    '''
    if type(n) in SET_MATH_TYPE:
        return True
    return False

def is_str(value):
    '''Determine that if a value is a string'''
    if type(value) in SET_STR_TYPE:
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
    ''' Determine that if a variable is a sequence object
    '''
    if type(obj) in SET_SEQ_TYPE:
        return True
    return False

def is_empty(obj):
    '''determine whether a object is empty'''
    if hasattr(obj, 'empty'):
        return obj.empty
    
    if hasattr(obj, '__len__'):
        if len(obj) == 0:
            return True
        return False
    
    if is_iter(obj):
        return False
    raise TypeError('can not determine whether %s is empty' % type(obj))
