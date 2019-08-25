# user/bin/python
#########################################
# Author         : Xuansheng Wu           
# Email          : wuxsmail@163.com 
# created        : 2017-11-01 00:00 
# Last modified  : 2019-08-21 15:32
# Filename       : DaPy.__init__.py
# Description    : initial file for DaPy                     
#########################################
'''
Data Analysis Library for Human.

DaPy module is a fundemantal data processing tool, which helps you
readily process and analysis data. DaPy offers a series of humane data
structures, including but not limiting in SeriesSet, Frame and DataSet. Moreover,
it implements some basic data analysis algorithms, such as Multilayer
Perceptrons, One way ANOVA and Linear Regression. With DaPy help,
data scientists can handle their data and complete the analysis task easily.

Enjoy the tour in data mining!

:Copyright (C) 2018 - 2019  Xuansheng Wu.
:License: GNU 3.0, see LICENSE for more details.
'''

__all__ = [ 'Frame', 'SeriesSet', 'mat', 'DataSet', 'datasets', 'methods',
            'exp', 'dot', 'multiply', 'zeros', 'ones', 'C', 'P', 'add',
            'diag', 'log', 'boxcox', 'cov', 'corr', 'frequency', 'quantiles',
            'distribution', 'describe', 'mean', 'abs', 'max', 'nan', 'inf', 
            'sum', 'diff', 'read', 'encode', 'save', 'delete', 'column_stack',
            'merge', 'row_stack', 'boxcox', 'show_time', 'get_dummies']

from .core import Frame, SeriesSet, DataSet, Matrix as mat, Series
from .core import nan, inf, LogInfo, LogWarn, LogErr
from .core import argsort
from .matlib import exp, dot, multiply, zeros, ones, C, P, add, diag, log, boxcox
from .matlib import cov, corr, frequency, quantiles, _sum as sum, diff, cumsum
from .matlib import distribution, describe, mean, _abs as abs, _max as max
from .io import read, encode, save
from .operation import delete, column_stack, row_stack, merge, concatenate
from .operation import get_dummies, get_ranks
from warnings import warn      
from datetime import datetime

__title__ = 'DaPy'
__description__ = 'Enjoy the tour in data mining !'
__url__ = 'http://dapy.kitgram.cn'
__version__ = '1.10.5'
__build__ = 0x201908
__author__ = 'Xuansheng Wu (wuxsmail@163.com)'
__license__ = '''DaPy  Copyright (C) 2018 - 2019 WU Xuansheng'+\
              This program comes with ABSOLUTELY NO WARRANTY;
              for details type `show w'.This is free software,
              and you are welcome to redistribute it under certain
              conditions; type `show c' for details.'''
__copyright__ = 'Copyright 2018-2019 Xuansheng Wu.'
__date__ = datetime(2019, 8, 24)

def _unittests():
    from unittest import TestSuite, defaultTestLoader, TextTestRunner
    _tests = TestSuite()
    for case in defaultTestLoader.discover('.', 'test_*.py'):
        _tests.addTests(case)
    tester = TextTestRunner()
    tester.run(_tests)

if 'Alpha' in __version__:
    LogInfo('In developing edition of DaPy-%s' % __version__)
