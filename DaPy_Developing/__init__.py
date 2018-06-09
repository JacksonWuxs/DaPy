
'''
DaPy module is a useful tool, which try to help you process and analysis data easily.

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



__author__ = 'Xuansheng Wu'

__version__ = '1.3.3.1'

__license__ = 'DaPy  Copyright (C) 2018  Jackson Woo'+\
              '''This program comes with ABSOLUTELY NO WARRANTY; '''+\
              '''for details type `show w'.This is free software,'''+\
              '''and you are welcome to redistribute it under certain'''+\
              '''conditions; type `show c' for details.'''

__all__ = ['MLP', 'datasets', 'Frame', 'SeriesSet',
           'DataSet', 'Table', 'Matrix', 'cov', 'corr', 'frequency',
           'quantiles', 'sum', 'is_iter', 'read', 'set_encode',
           'distribution','describe', 'mean', 'exp', 'dot', 'is_math']

from os.path import dirname, join
from core import (cov, corr, distribution, describe, sum,
                  Frame, SeriesSet, DataSet, Matrix, Table,
                  frequency, quantiles, mean, is_math, is_iter)
from matlib import exp, dot
import multilayer_perceptron
import inference_statistic

try:
    module_path = dirname(__file__)
    with open(join(module_path, 'README.md'),'r') as f:
        __doc__ = f.read()
except IOError:
    __doc__ = 'DaPy\n' +\
          '===============================================================\n'+\
          'A light data processing and analysing library for Python.\n'+\
          '\n-----------------------------\nName: DaPy\nAuthor: Jackson Woo\n'+\
           'Version: %s\nUpdata: May. 9th, 2018\nE-Mail:Wuxsmail@163.com\n'%__version__+\
           '-----------------------------\n\n' 


def read(addr, dtype='col', miss_symbol='', miss_value=None, sep=None,
            first_line=1, title_line=0, type_float=False, type_str=False):
    data = DataSet()
    data.read(addr, dtype, miss_symbol, miss_value, sep, first_line,
              title_line, type_float, type_str)
    return data

def set_encode(code='cp936'):
    import sys
    stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
    reload(sys)
    sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde
    sys.setdefaultencoding(code)
    return
