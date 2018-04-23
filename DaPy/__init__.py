# python2

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

__version__ = '1.3.2'

__license__ = 'DaPy  Copyright (C) 2018  Jackson Woo'+\
              '''This program comes with ABSOLUTELY NO WARRANTY; '''+\
              '''for details type `show w'.This is free software,'''+\
              '''and you are welcome to redistribute it under certain'''+\
              '''conditions; type `show c' for details.'''

__all__ = ['cov', 'MLP',
           'corr', 'CountFrequency', 'CountQuantiles', 'Statistic',
           'CountDistribution', 'dot', 'means', 'exp',
           'Frame', 'SeriesSet', 'DataSet', 'Table', 'Matrix']


from formulas import (cov, corr, CountFrequency, dot, mean, exp,
                      CountQuantiles, CountDistribution,Statistic)
from structure import Frame, SeriesSet, DataSet, Matrix, Table
from multilayer_perceptron import MLP

try:
    with open('README.rst','r') as f:
        __doc__ = f.read()
except:
    __doc__ = 'DaPy\n' +\
          '===============================================================\n'+\
          'A light data processing and analysing library for Python.\n'+\
          '\n-----------------------------\nName: DaPy\nAuthor: Jackson Woo\n'+\
           'Version: %s\nUpdata: Apr. 9th, 2018\nE-Mail:Wuxsmail@163.com\n'%__version__+\
           '-----------------------------\n\n' 




                        
