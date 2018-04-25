# python2
#-*- coding:utf-8 -*-

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

import collections

__version__ = '1.2.5'

__license__ = 'DaPy  Copyright (C) 2018  Jackson Woo'+\
              '''This program comes with ABSOLUTELY NO WARRANTY; '''+\
              '''for details type `show w'.This is free software,'''+\
              '''and you are welcome to redistribute it under certain'''+\
              '''conditions; type `show c' for details.'''

try:
    with open('README.rst','r') as f:
        __doc__ = f.read()
except:
    __doc__ = 'DaPy\n' +\
          '===============================================================\n'+\
          'A light data processing and analysinglibrary for Python.\n'+\
          '\n-----------------------------\nName: DaPy\nAuthor: Jackson Woo\n'+\
           'Version: %s\nUpdata: Mar. 15th, 2018\nE-Mail:Wuxsmail@163.com\n'%__version__+\
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
        # Exchange types
        data = [list() for i in range(len(col))]
        for t in xrange(len(lines)):
            line = lines[t][:-1].split(self.sep)
            if t < self.first:
                continue
            for i in col:
                data_ = line[i]
                if '.' in data_:
                    try:
                        data[i].append(float(data_))
                    except:
                        data[i].append(data_)
                elif data_.isdigit():
                    data[i].append(int(data_))
                elif data_ in self.NoneVariable:
                    data[i].append(self.miss_value)
                else:
                    data[i].append(data_)
                    
        self.data = dict(zip(title,data))
        self.size = len(data[0])
        del data

    def tocsv(self, addr):
        # open a file
        with open(addr, 'w') as f:
            
            # write title
            f.write(','.join(self.titles))
            f.write('\n')
            
            # write records
            if type(self.data) == type(dict()):
                for i in xrange(self.size):
                    f.write(','.join([str(self.data[title][i]) for title in self.titles]))
                    f.write('\n')
            else:
                for record in self.data:
                    f.write(','.join([str(value) for value in record]))
                    f.write('\n')

def cov(data_1, data_2):
    '''
    formula:  cov(x,y) = E(xy) - E(x)E(y) 
    '''
    # Judgement the dim of two series
    if len(data_1) != len(data_2):
        raise ValueError("Can not calculate covariance with different dimensionalities")
    
    # Some intermediate variables
    size = float(len(data_1))
    xy = [data_1[i]*data_2[i] for i in range(int(size))]
    Exy = sum(xy)/size
    Ex = sum(data_1)/size
    Ey = sum(data_2)/size

    return Exy - Ex*Ey
     
def cor(data_1, data_2):
    '''
    formula: cor(x,y) = cov(x,y)/(std(x)*std(y))
    '''
    static_1 = Statistic(data_1)
    static_2 = Statistic(data_2)
    covariance = cov(data_1, data_2)

    return covariance/(static_1.Std*static_2.Std)
 
def CountFrequency(data,cut=0.5):
    Group1, Group2 = 0,0
    for each in data:
        if each < cut:
            Group1 += 1
        else:
            Group2 += 1
    return Group1/float(len(data))

def CountQuantiles(data,shapes=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]):
    data = sorted(data)
    groups = list()
    lenth = len(data) + 1
    for point in shapes:
        try:
            groups.append(data[int(lenth*point)])
        except:
            pass
    return groups

def CountDistribution(data, breaks=10):
    groups = [0 for i in range(breaks)]
    maxn, minn = max(data), min(data)
    ranges = maxn - minn
    size = len(data)
    breaks = [minn+i*ranges/breaks for i in range(1,breaks+1)]
    for record in data:
        for i,cut_point in enumerate(breaks):
            if cut_point >= record:
                groups[i] += 1
                break
    return [float(i)/size for i in groups]

def Statistic(data):
    '''
    This function will help you calculate some basic describe statistic indexes.'
    formulas:
        <1> E(x) = sum(x)/n
        <2> D(x) = E(x^2) - E(x)^2
    '''
    statistic = collections.namedtuple('STAT', ['Mean','Std','CV','Min','Max','Range'])
    size = len(data)

    Ex = sum(data)/float(size)
    Ex2 = sum([i**2 for i in data])/float(size)
    
    std = (Ex2 - Ex**2)**0.5
    
    if Ex == 0:
        return statistic(Ex,std,None,min(data),max(data),max(data)-min(data))
    return statistic(Ex,std,std/Ex,min(data),max(data),max(data)-min(data))
                        
