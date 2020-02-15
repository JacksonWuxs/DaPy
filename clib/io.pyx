# -*- coding: utf-8 -*-
"""
data_mining.pyx
~~~~~~~~~~~~~~~
This module is a cython pyx file that is used to mine text efficiently
from the various support file formats.
"""
# -- python imports
import os
import sys
import time

from cython import boundscheck, wraparound
from libc.stdlib cimport atoll, atof
from datetime import datetime, date
from re import compile as _compile

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

cdef dict BOOL_SYMBOL = {u'TRUE'.encode('utf-8'): True, u'FALSE'.encode('utf-8'): False,
                         u'是'.encode('utf-8'): True, u'否'.encode('utf-8'): False,
                         u'True'.encode('utf-8'): True, u'False'.encode('utf-8'): False,}
@boundscheck(False)
@wraparound(False)
cpdef bint str2bool(char *string):
    return BOOL_SYMBOL[string]

cdef char *year
cdef char *month
cdef char *day
cdef char *dsep1 = '-'
cdef char *dsep2 = '/'
@boundscheck(False)
@wraparound(False)
cdef object str2date(char *string):
    year = strtok(string, dsep2)
    month = strtok(NULL, dsep2)
    day = strtok(NULL, dsep2)
    return date(atoll(year), atoll(month), atoll(day))

cdef char *hour
cdef char *minu
cdef char *sec
cdef char *tsep = ':'
@boundscheck(False)
@wraparound(False)
cdef object str2datetime(char *string):
    date = strtok(string, ' ')
    time = strtok(NULL, ' ')
    year = strtok(date, dsep2)
    month = strtok(NULL, dsep2)
    day = strtok(NULL, dsep2)
    hour = strtok(time, tsep)
    minu = strtok(NULL, tsep)
    sec = strtok(NULL, tsep)
    return datetime(atoll(year), atoll(month), atoll(day),
                    atoll(hour), atoll(minu), atoll(sec))

FLOAT_MASK = _compile('^[-+]?[0-9]\d*\.\d*$|[-+]?\.?[0-9]\d*$'.encode('utf-8'))
PERCENT_MASK = _compile(r'^[-+]?[0-9]\d*\.\d*%$|[-+]?\.?[0-9]\d*%$'.encode('utf-8'))
INT_MASK = _compile('^[-+]?[-0-9]\d*$'.encode('utf-8'))
DATE_MASK = _compile('^(?:(?!0000)[0-9]{4}([-/.]?)(?:(?:0?[1-9]|1[0-2])([-/.]?)(?:0?[1-9]|1[0-9]|2[0-8])|(?:0?[13-9]|1[0-2])([-/.]?)(?:29|30)|(?:0?[13578]|1[02])([-/.]?)31)|(?:[0-9]{2}(?:0[48]|[2468][048]|[13579][26])|(?:0[48]|[2468][048]|[13579][26])00)([-/.]?)0?2([-/.]?)29)$'.encode('utf-8'))
BOOL_MASK = _compile('^(true)|(false)|(yes)|(no)|(\u662f)|(\u5426)|(on)|(off)$'.encode('utf-8'))
@boundscheck(False)
@wraparound(False)
cdef object analyze_str_type(char *string):
    if INT_MASK.match(string):
        return atoll   

    elif FLOAT_MASK.match(string):
        return atof

    elif PERCENT_MASK.match(string):
        return str2pct

    elif DATE_MASK.match(string):
        return str2date

    elif BOOL_MASK.match(string.lower()):
        return BOOL_SYMBOL.__getitem__

    else:
        return str

# -- cython c imports
from libc.stdlib cimport malloc, realloc, free
from libc.stdlib cimport atoll, atof
from libc.stdio cimport fopen, fclose, FILE
from libc.stdio cimport fgets
from libc.string cimport strtok, strsep
from cython.parallel import prange, parallel, threadid

# preprocessor directive
DEF BUFFER_SIZE = 4096
cdef FILE *fp
cdef char buff[BUFFER_SIZE]

cdef list column_dtype, data

@boundscheck(False)
@wraparound(False)
cpdef list read_csv(char *addr, const char *sep=',', int skip_rows=1, char *nan=''):
    """Read the file contents."""
    fp = fopen(addr, "r")
    if fp == NULL:
        raise FileNotFoundError(2, "No such file: '%s'" % addr)
    
    column_dtype = []
    data = []
    cdef double NaN = float('nan')
    cdef char *row
    cdef int index, max_col
    cdef char *val
    max_col = 0
    while fgets(buff, BUFFER_SIZE, fp) != NULL:
        if skip_rows > 0:
            skip_rows -= 1
            continue

        row = strtok(buff, '\n')
        index = 0
        val = strtok(row, sep)
        while val != NULL:
            if index == max_col:
                column_dtype.append(analyze_str_type(val))
                data.append([])
                max_col += 1
            if nan == val:
                data[index].append(NaN)
            else:
                data[index].append(column_dtype[index](val))

            # move to next value
            val = strtok(NULL, sep)
            index += 1
    fclose(fp)   
    return data