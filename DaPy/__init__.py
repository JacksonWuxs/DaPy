
'''
DaPy module is a fundemantal data process tool, which help you readily process and analysis data.

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

from version import __version__, __author__, __copyright__

__all__ = [ 'datasets', 'Frame', 'SeriesSet', 'machine_learn',
            'statis', 'ones', 'zeros', 
           'DataSet', 'Table', 'Matrix', 'cov', 'corr', 'frequency',
           'quantiles', 'sum', 'is_iter', 'read', 'encode',
           'distribution','describe', 'mean', 'exp', 'dot', 'is_math']

from os.path import dirname, join
from core import (#cov, corr, distribution, describe, sum,
                  Frame, SeriesSet, DataSet, )
                 # frequency, quantiles, mean, is_math, is_iter)
from core import Matrix as mat
from matlib import exp, dot, multiply
from io import read, encode
from copy import deepcopy

try:
    module_path = dirname(__file__)
    with open(join(module_path, 'doc/README.md'),'r') as f:
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


def ones(shape):
    matrix = mat()
    matrix.make(shape[0], shape[1], 1)
    return matrix

def zeros(shape):
    matrix = mat()
    matrix.make(shape[0], shape[1], 0)
    return matrix

def column_stack(lis):
    if not isinstance(lis[0], mat):
        lis[0] = mat(deepcopy(lis[0]), False)
        return column_stack(lis)
    
    new = lis[0]
    for i, line in enumerate(new):
        for current in lis[1:]:
            line.extend(list(current[1]))
    return mat(new)

def delete(data, index, axis=0):
    if isinstance(index, int):
        index = [index, ]
        
    if not isinstance(data, mat):
        return delete(mat(deepcopy(data)), index, axis)
    
    if axis == 1:
        new = [0] * len(data)
        for i, line in enumerate(data):
            for i in index:
                del line[i]
            new[i] = line
        return mat(new, False)
    
    if axis == 0:
        index = sorted(index, reverse=True)
        new = list()
        for i, line in data:
            if i not in index:
                new.append(line)
        return mat(new, False)
        
    

