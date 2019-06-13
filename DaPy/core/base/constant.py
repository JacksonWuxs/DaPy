from array import array
from collections import Iterable, deque, namedtuple
from datetime import datetime
from sys import version_info
from time import struct_time
from logging import (info as LogInfo, warning as LogWarn,
                     error as LogErr, basicConfig, INFO)

basicConfig(level=INFO, format=' - %(message)s')
sysVersion = version_info.major

nan = float('nan')
inf = float('inf')
STR_TYPE = [str, bytes]
MATH_TYPE = [int, float, complex]
VALUE_TYPE = [bool, type(None), datetime] 
SEQ_TYPE = [list, tuple, deque, array, set, frozenset, bytearray]

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
    PYTHON3, PYTHON2 = False, True
else:
   PYTHON2, PYTHON3 = False, True

try:
    import cPickle as pickle
except ImportError:
    import pickle

VALUE_TYPE = tuple(VALUE_TYPE + STR_TYPE + MATH_TYPE)
STR_TYPE = tuple(STR_TYPE)
MATH_TYPE = tuple(MATH_TYPE)
SEQ_TYPE = tuple(SEQ_TYPE)

