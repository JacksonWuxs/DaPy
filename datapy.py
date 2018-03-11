# python2
#-*- coding:utf-8 -*-

'''
datapy module is a useful tool, which try to help you load and analysis data easily.

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

import collections
import math

__version__ = '1.2.4'

__license__ = 'Datapy  Copyright (C) 2018  Jackson Woo'+\
              '''This program comes with ABSOLUTELY NO WARRANTY; '''+\
              '''for details type `show w'.This is free software,'''+\
              '''and you are welcome to redistribute it under certain'''+\
              '''conditions; type `show c' for details.'''

__doc__ = 'DataPy - a light data processing library for Python.\n' +\
          '====================================================\n'+\
          'There is only one class called'+\
          "'dbsystem' with following objects:\n 'readall','readcol'"+\
          '\n\n-----------------------------\nName: Datapy\nAuthor: Jackson Woo\n'+\
           'Version: 1.2.3\nUpdata Time: Feb. 10th, 2018\nE-Mail:Wuxsmail@163.com\n'+\
           '-----------------------------\n\n' 
    
class DataSet():
    def __init__(self, addr='data.csv', title=True, split='AUTO', db=None,
                 name='Data', firstline=1, miss_value=None):
        # We set some basic variables as follow.
        self.NoneVariable = ('',' ','False','FALSE','NA')
        split_dic = {'csv':',','txt':' ','xls':'\t'}
        self.addr = addr
        self.header = title
        self.data = db
        self.name = name
        self.first = firstline
        self.miss_value = miss_value
        self.titles = None
        
        if split == 'AUTO':
            self.sep = split_dic.setdefault(addr.split('.')[-1], False)
            if self.sep is False:
                raise TypeError('Auto setting for splitting symbal failed.')
        else:
            self.sep = split
            
        if self.data is None:
            self.size = 0
            
        else:
            self.size = len(self.data)
    
    def __len__(self):
        return self.size
        
    def __getitem__(self,pos):
        return self.data[pos]

    def __cmp__(self, number):
        if self.size > number:
            return True
        return False
        
    def __str__(self):
        return self.name

    def __add__(self, other):
        if type(self.data) == type(list()) and type(other) == type(list()):
            self.data = self.data.extend(other)
            self.size = len(self.data)
        else:
            raise TypeError("Cannot concatenate %s and %s objects."%\
                            (type(self.data),type(other)))

    def __sub__(self, other):
        if type(self.data) == type(list()) and type(other) is int:
            if self.size >= other:
                self.data = self.data[:self.size-other]
                self.size = len(self.data)
            else:
                raise ValueError('The size of data set is less than %d'%other)
        else:
            raise TypeError("Cannot reduce %s and %s objects."%(type(self.data)\
                                                                ,type(other)))
        
    def readtable(self,col=all):
        '''
        This function could be used in loading data from a file in a simple way.
        That will be faster than <readframe> function but couldn't give too much
        information about the variables.
        '''
        with open(self.addr,'r') as f:
            lines = f.readlines()
        self.titles = lines[self.first-1][:-1].split(self.sep)
        if col == all:
            col = range(len(self.titles))
        data = list()
        for i in xrange(self.first,len(lines)):
            every = lines[i][:-1].split(self.sep)
            line = list()
            for j in col:
                data_ = every[j]
                if '.' in data_:
                    try:
                        line.append(float(data_))
                    except:
                        line.append(data_)
                elif data_.isdigit():
                    line.append(int(data_))
                elif data_ in self.NoneVariable:
                    line.append(self.miss_value)
                else:
                    line.append(data_)
            data.append(line)    
        self.data = data
        self.size = len(self.data)
        del data

    def readframe(self,col=all):
        '''
        This function is the most useful function in this class which supports
        users to load data from a document easily and returns some core
        information of the data.
        '''
        # Load Files...
        with open(self.addr,'r') as f:
            lines = f.readlines()
            
        data = list()
        # Set titles
        self.titles = lines[self.first-1][:-1].split(self.sep)
        # Set column numbers
        if col == all:
            col = range(len(self.titles))
        # Set data structure by titles
        if self.header:
            self.data_structure = collections.namedtuple(self.name,
                                                         [self.titles[i] for i in col])
        else:                   
            self.data_structure = collections.namedtuple(self.name,
                                                         ['col'+str(i) for i in col])
        # Reload data into data structure and transfrom the type
        for i in xrange(self.first,len(lines)):
            every = lines[i][:-1].split(self.sep)
            line = list()
            for j in col:
                data_ = every[j]
                if '.' in data_:
                    try:
                        line.append(float(data_))
                    except:
                        line.append(data_)
                elif data_.isdigit():
                    line.append(int(data_))
                elif data_ in self.NoneVariable:
                    line.append(self.miss_value)
                else:
                    line.append(data_)
            data.append(self.data_structure._make(line)) 
        # Reset Local DB     
        self.data = data
        self.size = len(self.data)
        # Release RAM
        del data, lines, self.header, line

    def readcol(self,col=all):
        '''
        <readcol> is another usefule function in datapy because it supports users
        to load data from the document by columen. In another word, you can
        easily pick out the data by each columen.
        '''
        # Load Files
        with open(self.addr,'r') as f:
            lines=f.readlines()
        # Set titles
        self.titles = lines[self.first-1][:-1].split(self.sep)
        if col == all:
            col = range(len(self.titles))
        title = [self.titles[i] for i in col]
        
        data = [list() for i in range(len(col))]
        # Exchange types
        for t in xrange(len(lines)):
            line = lines[t][:-1].split(self.sep)
            if t < self.first:
                continue
            for i in col:
                data_ = line[i]
                if '.' in data_:
                    try:
                        line.append(float(data_))
                    except:
                        line.append(data_)
                elif data_.isdigit():
                    data[i].append(int(data_))
                elif data_ in self.NoneVariable:
                    data[i].append(self.miss_value)
                else:
                    data[i].append(data_)
                    
        self.data = dict(zip(title,data))
        self.size = len(self.data)
        del data


def CountFrequency(data,cut=0.5):
    Group1, Group2 = 0,0
    for each in data:
        if each < cut:
            Group1 += 1
        else:
            Group2 += 1
    return Group1/float(len(data))


def CountDistribution(data,shapes=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]):
    data = sorted(data)
    groups = []
    mindata = data[0]
    lenth = float(len(data))
    t = 1
    i = 0
    for every in data:
        if t/lenth > shapes[i]:
            groups.append(every)
            i += 1
            if i == len(shapes):
                break
        t += 1
    return groups


def Statistic(data):
    '''
    This function will help you calculate some basic describe statistic indexes.'
    '''
    statistic = collections.namedtuple('STAT', ['Mean','Std','CV','Min','Max','Range'])
    size = len(data)
    if size == 0:
        return statistic(None,None,None)

    avg = sum(data)/float(size)
    error = [(float(data[i])-avg)**2 for i in xrange(size)]
    std = math.sqrt(sum(error)/(size-1))
    
    if avg == 0:
        return statistic(avg,std,None,min(data),max(data),max(data)-min(data))
    return statistic(avg,std,std/avg,min(data),max(data),max(data)-min(data))
                        
