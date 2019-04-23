from array import array
from collections import Iterable, deque, namedtuple
from datetime import datetime
from sys import version_info
from time import struct_time
from logging import (info as LogInfo, warning as LogWarn,
                     error as LogErr, basicConfig, INFO)

basicConfig(level=INFO, format=' - %(message)s')

nan = float('nan')
inf = float('inf')   
VALUE_TYPE = [type(None), int, float, str, complex,
                datetime, struct_time, bool]
STR_TYPE = [str]
MATH_TYPE = [int, float, complex]
SEQ_TYPE = [list, tuple, deque, array, set, frozenset]

try:
    import cPickle as pickle
except ImportError:
    import pickle


try:
    from numpy import ndarray
    from numpy.matrixlib.defmatrix import matrix
    SEQ_TYPE.extend([ndarray, matrix])
    from pandas import Series, DataFrame, Index
    SEQ_TYPE.extend([Series, DataFrame, Index])
except ImportError:
    pass

if version_info.major == 2:
    VALUE_TYPE.extend([unicode, long])
    MATH_TYPE.append(long)
    STR_TYPE.append(unicode)

try:
    import cPickle as pickle
except ImportError:
    import pickle

VALUE_TYPE = tuple(VALUE_TYPE)
STR_TYPE = tuple(STR_TYPE)
MATH_TYPE = tuple(MATH_TYPE)
SEQ_TYPE = tuple(SEQ_TYPE)
