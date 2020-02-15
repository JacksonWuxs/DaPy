from cython import boundscheck, wraparound
from libc.stdlib cimport atoll, atof
from datetime import datetime, date
from re import compile as _compile
from cpython.array cimport array

@boundscheck(False)
@wraparound(False)
cpdef long long str2int(char *string):
    return atoll(string)

@boundscheck(False)
@wraparound(False)
cpdef double str2float(char *string):
    return atof(string)

@boundscheck(False)
@wraparound(False)
cpdef double str2pct(char *string):
    return atof(string[:-1]) / 100.0

cdef  array hash_true, hash_false
cdef long long hash_val, hash_label
hash_true = array('q', [hash(_) for _ in ['True', 'true', 'TRUE', 'YES', 'yes', 'Yes']])
hash_false = array('q', [hash(_) for _ in ['False', 'false', 'FALSE', 'NO', 'no', 'No']])

@boundscheck(False)
@wraparound(False)
cpdef bint str2bool(char *string):
    hash_val = hash(string)
    for hash_label in hash_true:
        if hash_val == hash_label:
            return True
    for hash_label in hash_false:
        if hash_val == hash_label:
            return False
    raise ValueError('cannot transfer "%s" into bool' % string)

cdef int year, month, day
dsep = u'-'.encode('utf-8')
@boundscheck(False)
@wraparound(False)
cpdef object str2date(char *string):
    year, month, day = string.split(dsep)
    return date(atoll(year), atoll(month), atoll(day))

cdef int hour, minu, sec
tsep = u':'.encode('utf-8')
@boundscheck(False)
@wraparound(False)
cpdef object str2datetime(char *string):
    date, time = string.split()
    year, month, day = date.split(dsep)
    hour, minu, sec = time.split(tsep)
    return datetime(atoll(year), atoll(month), atoll(day),
                    atoll(hour), atoll(minu), atoll(sec))

FLOAT_MASK = _compile('^[-+]?[0-9]\d*\.\d*$|[-+]?\.?[0-9]\d*$'.encode('utf-8'))
PERCENT_MASK = _compile(r'^[-+]?[0-9]\d*\.\d*%$|[-+]?\.?[0-9]\d*%$'.encode('utf-8'))
INT_MASK = _compile('^[-+]?[-0-9]\d*$'.encode('utf-8'))
DATE_MASK = _compile('^(?:(?!0000)[0-9]{4}([-/.]?)(?:(?:0?[1-9]|1[0-2])([-/.]?)(?:0?[1-9]|1[0-9]|2[0-8])|(?:0?[13-9]|1[0-2])([-/.]?)(?:29|30)|(?:0?[13578]|1[02])([-/.]?)31)|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:0[48]|[2468][048]|[13579][26])00)([-/.]?)0?2([-/.]?)29)$'.encode('utf-8'))
BOOL_MASK = _compile('^(true)|(false)|(yes)|(no)|(\u662f)|(\u5426)|(on)|(off)$'.encode('utf-8'))

cpdef analyze_str_type(char *string):
    if INT_MASK.match(string):
        return str2int    

    elif FLOAT_MASK.match(string):
        return str2float

    elif PERCENT_MASK.match(string):
        return str2pct

    elif DATE_MASK.match(string):
        return str2date

    elif BOOL_MASK.match(string.lower()):
        return str2bool

    else:
        return string