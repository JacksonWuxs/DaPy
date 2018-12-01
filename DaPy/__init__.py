# user/bin/python
#########################################
# Author         : Xuansheng Wu           
# Email          : wuxsmail@163.com 
# created        : 2017-11-01 00:00 
# Last modified  : 2018-11-14 11:09
# Filename       : DaPy.__init__.py
# Description    : initial file for DaPy                     
#########################################
'''
Data Analysis Library for Human.

DaPy module is a fundemantal data processing tool, which helps you
readily process and analysis data. DaPy offers a series of humane data
structures, including but not limit in SeriesSet, Frame and DataSet. Moreover,
it implements some basic data analysis algorithms, such as Multilayer
Perceptrons, One way ANOVA and Linear Regression. With DaPy help,
data scientists can handle their data and complete the analysis task easily.

Enjoy the tour in data mining!

:Copyright (C) 2018  Xuansheng Wu.
:License: GNU 3.0, see LICENSE for more details.
'''

__all__ = [ 'datasets', 'Frame', 'SeriesSet', 'methods',
            'ones', 'zeros', 'delete', 'column_stack',
           'DataSet', 'Table', 'Matrix', 'cov', 'corr', 'frequency',
           'quantiles', 'sum', 'read', 'encode', 'save',
           'distribution','describe', 'mean', 'exp', 'dot', 'is_math']

from __version__ import __version__, __author__, __copyright__
from core import Frame, SeriesSet, DataSet, Matrix as mat
from matlib import exp, dot, multiply, zeros, ones, C, P, add, diag, log
from matlib import cov, corr, frequency, quantiles, _sum as sum
from matlib import distribution, describe, mean
from io import read, encode, save
from operation import delete, column_stack, merge

        
    

