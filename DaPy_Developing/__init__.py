
'''
DaPy module is a useful tool, which help you readily process and analysis data.

Copyright (C) 2018  Xuansheng Wu

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see https:\\www.gnu.org\licenses.
'''

from __version__ import *

__all__ = [ 'datasets', 'Frame', 'SeriesSet', 'machine_learn',
            'mathematical_statistics',
           'DataSet', 'Table', 'Matrix', 'cov', 'corr', 'frequency',
           'quantiles', 'sum', 'is_iter', 'read', 'encode',
           'distribution','describe', 'mean', 'exp', 'dot', 'is_math']

from os.path import dirname, join
from core import (cov, corr, distribution, describe, sum,
                  Frame, SeriesSet, DataSet, Matrix, Table,
                  frequency, quantiles, mean, is_math, is_iter)
from matlib import exp, dot
import machine_learn 
import mathematical_statistics

try:
    module_path = dirname(__file__)
    with open(join(module_path, 'README.md'),'r') as f:
        __doc__ = f.read()
except IOError:
    __doc__ = 'DaPy - A readily data processing and analysing library for Python.\n' +\
           '===============================================================\n'+\
           '\n-----------------------------\n'+\
           'Name: DaPy\n'+\
           'Author: Jackson Woo\n'+\
           'Version: %s'%__version__ +\
           '\nUpdata: Jun. 13th, 2018\n'+\
           'E-Mail:Wuxsmail@163.com\n'+\
           '-----------------------------\n\n' 


def read(addr, dtype='col', sheet_name=None, miss_symbol='', miss_value=None, sep=None,
            first_line=1, title_line=0, type_float=False, type_str=False):
    data = DataSet()
    data.read(addr, dtype, sheet_name, miss_symbol, miss_value, sep, first_line,
              title_line, type_float, type_str)
    return data

def encode(code='cp936'):
    import sys
    stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
    reload(sys)
    sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde
    sys.setdefaultencoding(code)
    return
