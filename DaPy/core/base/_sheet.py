from array import array
from collections import namedtuple, deque, OrderedDict, Counter, Iterable
from copy import deepcopy
import csv
from datetime import date, datetime
from _numeric import describe, mean, _sum, log
from _function import is_seq, is_iter, is_math, is_value, get_sorted_index
from _numeric import corr as f_c
from DaPy.io import str2value, TRANS_FUN_SET
from warnings import warn
from operator import itemgetter, attrgetter
from os import path
from pprint import pprint
from random import random, randint, shuffle
from string import atof, atoi
from distutils.util import strtobool

__all__ = ['SeriesSet', 'Frame']

class SeriesSet(object):
    '''every variable will be stored by a sequene.

    Attributes
    ----------
    _columns : str in list
        the titles in each columns.

    _dim : namedtuple
        the two dimensional span of this data set.
        
    _miss_value : value
        the symbol represented miss value in current seriesset.

    _m_value : values in list
        the number of miss value in each column.

    _set : OrderedDict
        the OrderedDict object contains all the data by columns.

    _type_column : types in list
        the sequence type of each column.
    '''
    dims = namedtuple('SeriesSet', ['Ln', 'Col'])

    def __init__(self, series=None, columns=None, 
                 miss_symbol=None, miss_value=None):
        
        self._set = OrderedDict()
        self._miss_value = miss_value
        self._m_value = list()
        self._columns = columns

        if series is None and columns is None:
            self.__init_empty()
        elif isinstance(series, SeriesSet):
            self.__init_set(series)
        elif isinstance(series, (dict, OrderedDict)):
            self.__init_dict(series, miss_symbol)
        elif hasattr(series, 'items'):
            self.__init_dict(dict(series.items()))
        elif isinstance(series, (Frame, )):
            self.__init_frame_matrix(series, miss_symbol)
        elif is_seq(series) or is_iter(series):
            self.__init_normal(series, miss_symbol, columns)
        else:
            raise TypeError("SeriesSet unsupport %s given."%type(series))

    @property
    def __dict__(self):
        return ['_set', '_columns', '_dim', '_type_column',
                '_type_value', '_m_value', '_miss_value']
    
    @property
    def data(self):
        return self._set
    
    @property
    def shape(self):
        return self._dim

    @property
    def columns(self):
        return self._columns

    @property
    def miss_value(self):
        return self._miss_value

    @columns.setter
    def columns(self, item):
        if isinstance(item, str):
            new_ = OrderedDict()
            for i, title in enumerate(self._columns):
                new_[item + '_%d'%i] = self._set[title]
            self._set = new_
            self._columns = [item + '_' + str(i) for i in range(self._dim.Col)]

        elif is_iter(item):
            if len(set(item)) == self._dim.Col:
                new_ = OrderedDict()
                for i, title in enumerate(self._columns):
                    new_[item[str(i)]] = self._set[title]
                self._set = new_
                self._columns = [str(i) for i in item]
            else:
                raise ValueError('incorrect dimention of titles.')
        else:
            raise TypeError('unknow title as %s'%item)

    @property
    def info(self):
        # calculate the informations
        mins, maxs, avgs, stds = list(), list(), list(), list()
        for sequence in self._set.values():
            d = describe(sequence)
            mins.append(str(d.Min))
            maxs.append(str(d.Max))
            try:
                avgs.append('%.2f'%d.Mean)
            except TypeError:
                avgs.append(' - ')
            try:
                stds.append('%.2f'%d.S)
            except TypeError:
                stds.append(' - ')
        dtype = [str(t).split("'")[1] for t in self._type_column]
        miss = map(str, self._m_value)

        # calculate the blank size of each subject
        blank_size = [max(len(max(self._columns, key=len)), 5) + 2, 
                      max(len(max(miss, key=len)), 4) + 2,
                      max(len(max(mins, key=len)), 3) + 2,
                      max(len(max(maxs, key=len)), 3) + 2,
                      max(len(max(avgs, key=len)), 4) + 2,
                      max(len(max(stds, key=len)), 3) + 2,
                      max(len(max(dtype, key=len)), 5)]

        # Draw the title line of description
        title_line = 'Title'.center(blank_size[0]) + '|' +\
                     'Miss'.center(blank_size[1]) +'|' +\
                     'Min'.center(blank_size[2]) + '|' +\
                     'Max'.center(blank_size[3]) + '|' +\
                     'Mean'.center(blank_size[4]) + '|' +\
                     'Std'.center(blank_size[5]) + '|' +\
                     'Dtype'.center(blank_size[6]) + '\n'
        for lenth in blank_size:
            title_line += '-'*lenth + '+'

        # Draw the main table of description
        info = str()
        for i, title in enumerate(self._columns):
            info += title.center(blank_size[0]) + '|'
            info += miss[i].center(blank_size[1]) + '|'
            info += mins[i].center(blank_size[2]) + '|'
            info += maxs[i].center(blank_size[3]) + '|'
            info += avgs[i].center(blank_size[4]) + '|'
            info += stds[i].center(blank_size[5]) + '|'
            info += dtype[i].center(blank_size[6]) + '\n'

        print('1.  Structure: DaPy.SeriesSet\n' +\
              '2. Dimensions: Ln=%d | Col=%d\n'%self._dim +\
              '3. Miss Value: %d elements\n'%sum(self._m_value) +\
              '4.   Describe: \n'+\
              title_line[:-1] + '\n' +\
              info + '='*(6 + sum(blank_size)))

    def __init_empty(self):
        self._columns = list()
        self._dim = SeriesSet.dims(0, 0)
        self._type_column = list()
        
    def __init_set(self, series):
        self._set = deepcopy(series)
        self._columns = deepcopy(series._columns)
        self._dim = deepcopy(series._dim)
        self._type_column = deepcopy(series._type_column)
        self._m_value = deepcopy(series._m_value)
        self._miss_value = deepcopy(series.miss_value)
        
    def __init_dict(self, series, miss_symbol):
        if not self._columns:
            self._columns = sorted(series) # Column names
        elif len(set(self._columns)) < len(series):
            self._columns.extend(['C%d'%i for i in range(len(series) - len(set(self._columns)))])
        elif len(set(self._columns)) > len(series):
            raise ValueError('the lenth of titles does not match the data size.')

        self._type_column = list() 

        # MaxSize of this Series Set
        max_Ln = max(map(len, series.itervalues()))
        for title, sequence in series.items():
            size = len(sequence)
            # Check the type of each sequence
            if isinstance(sequence, array):
                self._type_column.append(array)

            elif is_seq(sequence):
                self._type_column.append(type(sequence))
                mv, sequence = self.__check_sequence_type(sequence, miss_symbol)
            else:
                raise TypeError("unsupport type '%s' "%type(sequence)+\
                                "in column '%s'."%title)
            
            if size != max_Ln:
                if isinstance(sequence, array):
                    sequence = list(sequence)
                sequence.extend([self._miss_value] * (max_Ln - size))
                
            self._m_value.append(mv)
            self._set[title] = sequence
        self._dim = SeriesSet.dims(max_Ln, len(self._columns))

    def __init_frame_matrix(self, series, miss_symbol):
        self._dim = SeriesSet.dims(series._dim.Ln, series._dim.Col)         
        self._type_column = [list] * self._dim.Col
        if isinstance(series, Frame):
            self._m_value = deepcopy(series._m_value)
            if not self._columns:
                self._columns = deepcopy(series._columns)
        else:
            if not self._columns:
                self._columns = ['C%d'%i for i in range(self._dim.Col)]
            self._m_value = [0] * self._dim.Col
        
        if len(set(self._columns)) < self._dim.Col:
            for i, value in enumerate(self._columns):
                if self._columns.count(value) != 1 and \
                   self._columns.index(value) != i:
                    self._columns[i] = 'C%d'%i
            self._columns.extend(['C%d'%i for i in \
                                  range(len(self._columns), self._dim.Col)])
        elif len(self._columns) > self._dim.Col:
            self._columns = self._columns[:self._dim.Col]

        for sequence, title in zip(zip(*series), series.columns):
            self._set[title] = sequence

    def __init_normal(self, series, miss_symbol, columns):
        # if the input object is a sequence in sequence structure
        if all(filter(is_seq, series)):
            # extend the empty space
            lenth_Col = len(max(series, key=len))
            for pos, record in enumerate(series):
                if len(record) < lenth_Col:
                    new = list(record)
                    new.extend([self._miss_value] * (lenth_Col - len(record)))
                    series[pos] = new
                    
            # initalized the column titles
            if is_iter(self._columns):
                self._columns = map(str, self._columns)
                if len(set(self._columns)) < lenth_Col:
                    for i, value in enumerate(self._columns):
                        if self._columns.count(value) != 1 and \
                           self._columns.index(value) != i:
                            new_title = 'C%d'%i
                            while new_title in self._columns:
                                new_title += '_%d'%i
                            self._columns[i] = new_title
                    self._columns.extend(['C%d'%i for i in \
                                          range(len(self._columns), lenth_Col)])
                elif len(self._columns) > lenth_Col:
                    self._columns = self._columns[:lenth_Col]
                                          
            else:
                self._columns = ['C%d'%i for i in range(lenth_Col)]
                
            self._dim = SeriesSet.dims(len(series), lenth_Col)
            self._m_value = [0] * self._dim.Col
            self._type_column = [list]*self._dim.Col

            for j, sequence in enumerate(zip(*series)):
                for i, v in enumerate(sequence):
                    if v == miss_symbol or v == self._miss_value:
                        sequence = list(sequence)
                        sequence[i] = self._miss_value
                        self._m_value[j] += 1
                self._set[self._columns[j]] = list(sequence)      
            return

        # if input object is a 1-d sequence structure
        if all(filter(is_value, series)):
            
            # initalized the title names
            if isinstance(columns, list):
                self._columns = [str(columns[0]), ]
            elif isinstance(columns, (str, int)):
                self._columns = [str(columns), ]
            else:
                self._columns = ['C0', ]

            self._dim = SeriesSet.dims(len(series), 1)
            self._type_column = [list]
            self._m_value = [0, ]
                    
            for i, value in enumerate(series):
                if value == miss_symbol or value == self._miss_value:
                    series[i] = self._miss_value
                    self._m_value[0] += 1

            self._set[self._columns[0]] = list(series)
            return
        
        raise TypeError('records of input data do not have the same type, '+\
                            'should be all iterable or value.')
    
    def __arrange_by_index(self, new_index):
        for title, sequence in self._set.items():
            self._set[title] = [sequence[j] for j in new_index]
        self._type_column = [list] * self._dim.Col
        
    def __repr__(self):
        if self._dim.Ln > 10:
            def write_Ln(i, title, blank):
                item = self._set[title]
                msg = ' '*blank + title + ': <'
                msg += ', '.join([str(value) for value in item[:5]])
                msg += ', ... ,'
                msg += ', '.join([str(value) for value in item[-5:]])
                msg +=  '>\n'
                return msg
        else:
            def write_Ln(i, title, blank):
                item = self._set[title]
                msg = ' '*blank + title + ': <'
                msg += ', '.join([str(value) for value in item])
                msg += '>\n'
                return msg

        msg = str()
        size = len(max(self._columns, key=len))
        for i, title in enumerate(self._columns):
            msg += write_Ln(i, title, size - len(title))
        return msg[:-1]

    def __contains__(self, e):
        if isinstance(e, str):
            return e in self._columns

        if isinstance(e, (list, tuple)):
            if len(e) == self._dim.Col:
                for i, sequence in enumerate(self._set.values()):
                    for v in sequence:
                        if e[i] != v:
                            return False
                return True
            elif len(e) == self._dim.Ln:
                for seq in self._set.values():
                    for i in range(self._dim.Ln):
                        if seq[i] != e[i]:
                            break
                    else:
                        return True
                return False
        return False

    def __len__(self):  # OK #
        return self._dim.Ln

    def __eq__(self, other):  # OK #
        if isinstance(other, (Frame, SeriesSet)):
            if self._dim == other._dim:
                return True
        return False

    def __getslice__(self, i, j):
        if i in self._columns or j in self._columns:
            if i in self._columns:
                i =  self._columns.index(i)
            elif i is None:
                i = 0
            else:
                raise ValueError('can not get the title of %s'%j)
            
            if j in self._columns:
                j =  self._columns.index(j)
            elif j is None:
                j = self._dim.Col - 1
            else:
                raise ValueError('can not get the title of %s'%j)
            
            if i > j:
                i, j = j, i

            new_set = OrderedDict()
            for title in self._columns[i:j + 1]:
                new_set[title] = self._set[title]
            return SeriesSet(new_set, miss_value=self._miss_value)

        if type(i) != int and type(j) != int:
            raise ValueError('unrecognized symbol as [%s:%s]'%(i, j))

        if i is None:
            i = 0
        elif i < 0:
            i = 0
        elif i > self._dim.Ln:
            i = self._dim.Ln


        if j is None:
            j = self._dim.Ln
        elif j < 0:
            j = self._dim.Ln + j
        elif j > self._dim.Ln:
            j = self._dim.Ln
            
        if i > j:
            i, j = j, i

        return_list = zip(*[self._set[t][i:j] for t in self._columns])
        return Frame(return_list, self._columns, None, self._miss_value)

    def __getitem__(self, pos):  # OK #
        if isinstance(pos, int):
            return [self._set[title][pos] for title in self._columns]

        elif isinstance(pos, slice):
            return self.__getslice__(pos.__getattribute__('start'),
                                     pos.__getattribute__('stop'))

        elif isinstance(pos, str):
            return self._set[pos]
        
        else:
            raise TypeError('SeriesSet index must be int, str and slice, '+\
                            'not %s'%str(type(pos)).split("'")[1])
        
    def __setitem__(self, key, value):
        if isinstance(key, int):
            if key <= self._dim.Ln:
                self.__delitem__(key)
                self.insert(key, value)
            else:
                self.append(value)

        elif isinstance(key, str):
            if key in self._columns:
                pos = self._columns.insert(key)
                self.__delitem__(key)
                self.insert_col(key, value, pos)
            else:
                self.append_col(key, value)

    def __delitem__(self, key):
        if isinstance(key, int):
            drop_record = self.pop(key)

        elif isinstance(key, str) and key in self._columns:
            drop_record = self.pop_col(key)

        else:
            raise KeyError('unrecognized symbol as %s'%key)
                                
    def __iter__(self):
        for i in range(self._dim.Ln):
            yield [self._set[title][i] for title in self._columns]

    def __reversed__(self):
        self.reverse()

    def __check_sequence_type(self, series, miss_symbol):
        if is_value(series):
            return 0, [series] * self._dim.Ln

        mv = 0
        if not isinstance(series, array):
            for i, element in enumerate(series):
                if element == miss_symbol or element == self._miss_value:
                    series[i] = self._miss_value
                    mv += 1
        return mv, series
    
    def append(self, item, miss_symbol=None):
        if is_value(item):
            item = [item] * self._dim.Col
        else:
            raise TypeError("insert item is not a support type "+\
                            "with `%s`"%type(item))

        if not miss_symbol:
            miss_symbol = self._miss_value

        if len(item) > self._dim.Col:
            item = item[:self._dim.Col]
        elif len(item) < self._dim.Col:
            item.extend([self._miss_value] * (self._dim.Col - len(item)))

        for i, seq in enumerate(self._set.values()):
            element = item[i]
            if element == miss_symbol:
                self._m_value[i] += 1
                seq.append(self._miss_value)
            else:
                seq.append(element)

        self._dim = SeriesSet.dims(self._dim.Ln + 1, self._dim.Col)
        
    def append_col(self, variable_name, series, miss_symbol=None): # OK #
        '''append a series data to the seriesset last
        '''
        if not isinstance(variable_name, str):
            raise ValueError('unsupport type to set as variable name.')
        
        if not is_iter(series):
            raise ValueError("unsupport type to append as values")
        
        if variable_name in self._columns:
            raise ValueError("variable `%s` is already taken."%variable_name)

        if not miss_symbol:
            miss_symbol = self._miss_value

        mv, series = self.__check_sequence_type(series, miss_symbol)

        # check the lenth of data
        size_series = len(series)
        if size_series < self._dim.Ln:
            series = list(series)
            mv_2 = self._dim.Ln - size_series
            series.extend([self._miss_value] * mv_2)
            mv += mv_2
            size_series = self._dim.Ln
            
        elif size_series > self._dim.Ln:
            for title in self._columns:
                self._set[title] = list(self._set[title])
                self._set[title].extend([self._miss_value] *(\
                    size_series - self._dim.Ln))
                              
        self._columns.append(variable_name)
        self._m_value.append(mv)
        self._type_column.append(type(series))
        self._dim = SeriesSet.dims(size_series, self._dim.Col + 1)
        self._set[variable_name] = series

    def corr(self):
        '''correlation between variables in data -> Frame object
        '''
        new_ = Frame([[0] * self._dim.Col for i in range(self._dim.Col)],
                     self._columns)
        for i, sequence in enumerate(self._set.values()):
            for j, next_sequence in enumerate(self._set.values()):
                if i == j:
                    new_[j][i] = 1
                    continue
                r = f_c(sequence, next_sequence)
                new_[i][j], new_[j][i] = r, r
        new_.insert_col('', self._columns, 0)
        return new_

    def count(self, X, *area):
        '''count X in area (point1, point2)-> Counter object
        '''
        counter = Counter()
        if is_value(X):
            X = [X, ]

        if not hasattr(X, '__contains__'):
            raise TypeError('X should be a value or values in list')
        
        if area[0] == all:
            C1, L1 = 0, 0
            L2, C2 = self._dim
                
        elif len(area) == 2 and len(area[0]) == len(area[1]) == 2:
            if area[0][1] == all:
                C1 = 0
            elif isinstance(area[0][1], int) and 0 <= area[0][1] <= self._dim.Col:
                C1 = area[0][1]
            elif area[0][1] in self._columns:
                C1 = self._columns.index(area[0][1])
            else:
                raise TypeError('second position in the first tuple should a int '+\
                                'type emblemed column number or a str emblemed'+\
                                'column name.')

            if area[1][1] == all:
                C2 = self._dim.Col
            elif isinstance(area[1][1], int) and 0 <= area[1][1] <= self._dim.Col:
                C2 = area[1][1]
            elif area[1][1] in self._columns:
                C2 = self._columns.index(area[1][1])
            else:
                raise TypeError('second position in the second tuple should a int '+\
                                'type emblemed column number or a str emblemed '+\
                                "column name. Tip: use all to stand for all.")

            if area[0][0] == all:
                L1 = 0
            elif isinstance(area[0][0], int) and 0 <= area[0][0] <= self._dim.Ln:
                L1 = area[0][0]
            else:
                raise TypeError('first position in the first tuple should be a int '+\
                                'type emblemed start line number. Tip: use all to '+\
                                "stand for all.")

            if area[1][0] == all:
                L2 = self._dim.Ln
            elif isinstance(area[1][0], int) and 0 <= area[1][0] <= self._dim.Ln:
                L2 = area[1][0]
            else:
                raise TypeError('first position in the second tuple should be a int '+\
                                'type emblemed start line number. Tip: use all to '+\
                                "stand for all.")

            if C2 <= C1 or L2 <= L1:
                raise ValueError('the postion in the second tuple should be less than '+\
                                 'The first tuple.')
            
        else:
            raise EOFError("unrecognized expression '%s'"%str(area))

        for title in self._columns[C1:C2 + 1]:
            sequence = self._set[title]
            for value in sequence[L1:L2 + 1]:
                if value in X:
                    counter[value] += 1
        if not counter:
            return None
        return counter

    def count_element(self, col=all):
        if col == all:
            col = self._columns
            
        elif isinstance(col, int):
            if abs(col) < self._dim.Col:
                col = [self._columns[col]]
            else:
                raise TypeError("unrecognized expression '%s'"%str(col))
            
        elif col in self._columns:
            col = [col, ]
            
        elif is_iter(col):
            new = list()
            for each in col:
                if each in self._columns:
                    new.append(each)
                elif isinstance(each, int) and abs(each) < self._dim.Col:
                    new.append(self._columns[each])
                else:
                    raise TypeError("`%s` is not a title name"%str(each))
            col = new
        else:
            raise TypeError("unrecognized expression `%s`"%str(col))

        collect_elements = dict()
        for title in col:
            c = Counter(self._set[title])
            collect_elements[title] = c
        return collect_elements
    
    def extend(self, item):
        '''extend the current SeriesSet with records in set.
        '''
        if isinstance(item, SeriesSet):
            for i, title in enumerate(item.columns):
                sequence = item[title]
                if title not in self._columns:
                    To = [self._miss_value for j in range(self._dim.Ln)]
                    self._m_value.append(self._dim.Ln)
                    self._columns.append(title)
                    self._type_column.append(item._type_column[i])
                else:
                    To = list(self._set[title])

                if self._miss_value != item._miss_value:
                    for value in sequence:
                        if valule == item.miss_value:
                            To.append(self._miss_value)
                        else:
                            To.append(value)
                else:
                    To.extend(sequence)
                self._set[title] = To
                index = self._columns.index(title)
                self._m_value[index] = self._m_value[index] + item._m_value[i]
            self._dim = SeriesSet.dims(self._dim.Ln + item._dim.Ln,
                                       len(self._columns))

            for i, sequence in enumerate(self._set.values()):
                if len(sequence) != self._dim.Ln:
                    add_miss_size = self._dim.Ln - len(sequence)
                    sequence.extend([self._miss_value]*add_miss_size)
                    self._m_value[i] += add_miss_size
                    
        elif isinstance(item, Frame):
            self.extend(SeriesSet(item))
            
        elif all(filter(is_iter, item)):
            self.extend(SeriesSet(item, self._columns))
            
        else:
            raise TypeError('could not extend a single value only.')

    def extend_col(self, other):
        if isinstance(other, SeriesSet):
            lenth = max(self._dim.Ln, other._dim.Ln)
            for i, (title, col) in enumerate(other.items()):
                lenth_bias = self._dim.Ln - lenth
                if title not in self._columns:
                    self._columns.append(title)
                    self._m_value.append(other._m_value[i] + lenth_bias)
                    self._type_column.append(list)
                    col.extend([self._miss_value] * lenth_bias)
                    self._set[title] = col
                else:
                    sequence = self._set[title]
                    if not hasattr(sequence, 'extend'):
                        sequence = list(sequence)
                    sequence.extend(col)
                    self._set[title] = sequence
                    self._m_value[self._columns.index(title)] += \
                                                              other._m_value[i]
            lenth = len(max(self._set.values(), key=len))
            for j, sequence in enumerate(self._set.values()):
                if len(sequence) < lenth:
                    sequence.extend([self._miss_value] * (lenth - len(sequence)))
                    self._m_value[j] += lenth - len(sequence)
                if self._miss_value != other._miss_value:
                    for i, value in enumerate(sequence):
                        if value == self._miss_value:
                            sequence[i] = value

            self._dim = SeriesSet.dims(lenth, len(self._columns))

        elif isinstance(other, Frame):
            self.extend_col(SeriesSet(other))
            
        elif all(filter(is_iter, other)):
            new_col = [title + '_1' for title in self._columns]
            self.extend_col(SeriesSet(other, new_col))

        else:
            raise TypeError('could not extend a single value only.')

    def items(self):
        return self._set.items()
   
    def insert(self, pos, item, miss_symbol=None): # OK #
        '''insert a record to the frame, position in <index>
        '''
        if is_value(item):
            item = [item] * self._dim.Col
        if not (isinstance(pos, int) and is_iter(item)):
            raise TypeError("insert item is not a support"+\
                            "type with %s"%type(item))

        if not miss_symbol:
            miss_symbol = self._miss_value

        for i, title in enumerate(self._columns):
            element = item[i]
            if element == miss_symbol or element == self._miss_value:
                if not isinstance(self._type_column[i], list):
                    self._set[title] = list(self._set[title])
                self._set[title].insert(pos, self._miss_value)
                self._m_value[i] += 1
            else:
                self._set[title].insert(pos, element)
        self._dim = SeriesSet.dims(self._dim.Ln + 1, self._dim.Col)

    def insert_col(self, variable_name, series, index, miss_symbol=None):
        '''insert a series of data to the frame, position in <index>
        '''
        if not isinstance(variable_name, str):
            raise ValueError("unsupport type to append in this frame")
        
        # check the variable name
        if variable_name in self._columns:
            raise ValueError("variable name <'%s'> is already taken."%self.\
                             variable_name)

        if not miss_symbol:
            miss_symbol = self._miss_value

        mv, series = self.__check_sequence_type(series, miss_symbol)
        
        # check the lenth of data
        size_series = len(series)
        if size_series < self._dim.Ln:
            series = list(series)
            mv_2 = self._dim.Ln - size_series 
            series.extend([self._miss_value] * mv_2)
            mv += mv_2
            size_series = self._dim.Ln
        elif size_series > self._dim.Ln:
            for title in self._columns:
                self._set[title] = list(self._set[title])
                self._set[title].extend([self._miss_value for t in range(
                    size_series - self._dim.Ln)])

        self._columns.insert(index, variable_name)
        self._type_column.insert(index, type(series))
        self._dim = SeriesSet.dims(size_series, self._dim.Col + 1)
        self._m_value.insert(index, mv)
        
        new_set = OrderedDict()
        if index >= self._dim.Col:
            self._set[variable_name] = series
            return

        for i, title in enumerate(self._set):
            if i == index:
                new_set[variable_name] = series
            new_set[title] = self._set[title]

        self._set = new_set

    def keys(self):
        return self._set.keys()

    def normalized(self, process='NORMAL', col=all, attr=None, get_attr=None):
        if col is all:
            new_col = self._columns
        else:
            new_col = list()
            for each in col:
                if each in self._columns:
                    new_col.append(each)
                else:
                    new_col.append(self._columns[each])

        attrs_dic = dict()
        if process == 'NORMAL':
            attrs_structure = namedtuple('Nr_attr', ['Min', 'Range'])
        elif process == 'STANDARD':
            attrs_structure = namedtuple('Sd_attr', ['Mean', 'S'])

        if process == 'LOG':
            from math import log
            
        for i, title in enumerate(new_col):
            sequence = self._set[title]
            
            if attr:
                A, B = attr[title]
                
            elif process == 'NORMAL':
                statis = describe(sequence)
                A, B = statis.Min, statis.Range

            elif process == 'STANDARD':
                statis = describe(sequence)
                A, B = statis.Mean, statis.Sn

            elif process != 'LOG':
                raise ValueError("unrecognized symbol '%s',"%str(process)+\
                                 "use 'NORMAL', 'STANDARD' and 'LOG'")

            new = [0] * self._dim.Ln
            if process == 'LOG':
                for i, value in enumerate(sequence):
                    try:
                        new[i] = log(value)
                    except ValueError:
                        continue
            else:
                for i, value in enumerate(sequence):
                    try:
                        new[i] = (value - A) / B
                    except ZeroDivisionError:
                        continue
                    except TypeError:
                        continue
                self._set[title] = new
                
            try:
                attrs_dic[title] = attrs_structure(A, B)
            except UnboundLocalError:
                pass
            
        if get_attr:
            return attrs_dic
        return
        
    def pop_miss_value(self, axis='COL'):
        if str(axis).upper() == 'COL':
            pops = dict()
            for i in range(self._dim.Col-1, -1, -1):
                if self._m_value[i] != 0:
                    pops[self._columns.pop(i)] = self._set.pop(self._columns[i])
                    self._type_column.pop(i)
            self._dim = SeriesSet.dims(self._dim.Ln, len(self._columns))
            self._m_value = [0 for i in xrange(self._dim.Col)]
            return SeriesSet(pops)
            
        elif str(axis).upper() == 'LINE':
            drop_line_index = list()
            for sequence in self._set.values():
                for i, v in enumerate(sequence):
                    if v == self._miss_value:
                        drop_line_index.append(i)
            drop = sorted(set(drop_line_index), reverse=True)
            pops = list()
            for sequence in self._set.values():
                pops.append([list(sequence).pop(index) for index in drop])
            self._dim = SeriesSet.dims(self._dim.Ln - len(drop),
                                       len(self._columns))
            self._m_value = [0 for i in range(self._dim.Col)]
            return Frame(pops, self._columns)
        else:
            raise IndexError("axis should be 'COL' or 'LINE'")
        
    def pick(self, conditions):
        '''pick out records that comply with conditions
        '''
        for i, title in enumerate(self._columns):
            conditions = conditions.replace(title, 'record[%d]'%i)
        conditions = 'check = ' + conditions
        pick_record_line = list()
        
        for i, record in enumerate(self):
            exec(conditions)
            if check:
                pick_record_line.append(i)

        return_dict = OrderedDict()
        for title, sequence in self._set.items():
            return_dict[title] = [sequence[i] for i in pick_record_line]

        return SeriesSet(return_dict, self._columns)

    def pop(self, pos=-1):
        '''pop(remove & return) a record from the Frame
        '''
        if not isinstance(pos, int):
            raise TypeError('an integer is required.')
        
        pop_item = [sequence.pop(pos) for sequence in self._set.values()]
        
        self._dim = SeriesSet.dims(self._dim.Ln - 1, self._dim.Col)
        for i, each in enumerate(pop_item):
            if self._miss_value == each:
                self._m_value[i] -= 1
        return pop_item

    def pop_col(self, *titles): 
        pop_name = list()
        if not titles:
            titles = [self._columns[-1]]
        for item in titles:
            if isinstance(item, str):
                if item in self._columns:
                    pop_name.append(item)
                else:
                    raise ValueError("`%s` is not in columns title"%item)
            elif isinstance(item, int) and item < self._dim.Col:
                pop_name.append(self._columns[item])
            else:
                raise TypeError('an integer or string, which stands for a '+\
                                'column name is required,'+
                                'not {0} !'.format(type(item)))
        pop_data = dict()
        for title in pop_name:
            pos = self._columns.index(title)
            pop_data[title] = self._set.pop(title)
            self._columns.pop(pos)
            self._m_value.pop(pos)
            self._type_column.pop(pos)
        self._dim = SeriesSet.dims(self._dim.Ln, self._dim.Col-len(pop_name))
        return SeriesSet(pop_data, pop_name)

    def reverse(self, axis='COL'):
        if axis.upper() == 'COL':
            self._columns.reverse()
            self._type_column.reverse()
            self._m_value.reverse()
            new_set = OrderedDict()
            for key in reversed(self._set):
                new_set[key] = self._set[key]
            del self._set
            self._set = new_set
        else:
            for sequence in self._set.values():
                sequence.reverse()

    def rename_col(self, *new_names):
        try:
            for names in new_names:
                if names[0] in self._columns:
                    if names[1] not in self._columns:
                        self._columns = names[1]
                    else:
                        raise ValueError("Name '%s' has already exist'"%names[1])
                else:
                    raise ValueError("'%s' is not exist"%names[0])
        except:
            raise TypeError('expects some parameters like 2-dimentions tuple')

    def read_text(self, addr, first_line=1, miss_symbol='NA', 
                  title_line=0, sep=',', prefer_type=None):
        
        def transform(i, item):
            try:
                return _col_types[i](item)
            except ValueError:
                if item == miss_symbol:
                    self._m_value[i] += 1
                    return self._miss_value
                
                warn('column %d has different type.'%i)
                return str2value(item, prefer_type)

        with open(addr, 'r') as f:
            reader = csv.reader(f, delimiter=sep)
            
            col = max([len(line) for l_n, line in enumerate(reader)])
            col = range(col)
            datas = [[0]*l_n for i in col]
            self._m_value = [0 for c in col]
            
            if isinstance(prefer_type, str):
                _col_types = dict(zip(col, [str]*len(col)))
            else:
                _col_types = dict()

            f.seek(0)
            for i, lines in enumerate(reader):
                if not self._columns and i == title_line:
                    self._columns = list()
                    lenth_title_line = len(lines)
                    for k in col:
                        if k < lenth_title_line:
                            new_title = lines[k]
                        else:
                            new_title = 'C%d'%k
                        while new_title in self._columns:
                            new_title += '_%d'%k
    
                        self._columns.append(new_title)
                        continue
                    
                if len(_col_types) == len(col):
                    break
                    
                if i >= first_line:
                    for j, item in enumerate(lines):
                        if j in _col_types or item == miss_symbol or item == self._miss_value:
                            continue
                        _col_types[j] = TRANS_FUN_SET[type(str2value(item))]
                    
            f.seek(0)
            for m, record in enumerate(reader):
                if m >= first_line - 1 :
                    break
                
            for m, record in enumerate(reader):
                for i, v in enumerate(record):
                    try:
                        datas[i][m] = transform(i, v)
                    except KeyError:
                        pass

        l_col, l_title = len(col), len(self._columns)
        if not self._columns:
            self._columns = ['C'+str(c) for c in col]
        elif l_title < l_col:
            self._columns.extend(['C'+str(c) for c in col[l_col:l_title]])
        elif l_title > l_col:
            self._columns = self._columns[:l_col]
        
        self._type_column = [list for c in col]
        self._set = OrderedDict(zip(self._columns, datas))
        self._dim = SeriesSet.dims(m+2-first_line, len(self._columns))

    def values(self):
        return self._set.values()

    def replace(self, col=all, *arg):
        if col == all:
            col = self._columns
        elif isinstance(col, int) and col < self._dim.Col:
            col = (self._columns[col], )
        elif isinstance(col, str) and col in self._columns:
            col = (col, )
        else:
            new_col = list()
            for c in col:
                if isinstance(c, str):
                    new_col.append(c)
                elif isinstance(c, int) and abs(c) < self._dim.Col:
                    new_col.append(self._columns[c])
                else:
                    raise TypeError("unrecognized symbol '%s'."%str(c))
            col = new_col
            
        order = dict(arg)
        for title in col:
            sequence = self._set[title]
            for i, value in enumerate(sequence):
                if value in order:
                    sequence[i] = order[value]
        
    def sort(self, *orders): # OK #
        '''orders as tuple like (column_name, 'DESC')

        See Also: numpy/core/fromnumeric.py Line:756
        See Also: numpy/core/numeric.py Line:508
        '''
        if len(orders) > 1:
            new_ = Frame(self)
            new_.sort(*orders)
            new = SeriesSet(new_)
            self._set = new._set
            return
        
        if isinstance(orders[0][0], int):
            compare_title = self._columns[orders[0][0]]
        elif orders[0][0] in self._columns:
            compare_title = orders[0][0]
        else:
            raise TypeError("'%s' keyword is "%orders[0][0] +\
                            "not in frame's columns.")
        
        if orders[0][1] == 'DESC':
            reverse = True
        elif orders[0][1] == 'ASC':
            reverse = False
        else:
            raise TypeError("`%s` is not a recognized symbol."%compare_symbol)

        new_index = get_sorted_index(self._set[compare_title], reverse=reverse)
        self.__arrange_by_index(new_index)
            
    def shuffles(self):
        new_index = range(self._dim.Ln)
        shuffle(new_index)
        self.__arrange_by_index(new_index)


class Frame(object):
    '''Maintains the data as records.

    Attibutes
    ---------
    _columns : str in list
        the titles in each columns.

    _dim : namedtuple
        the two dimensional span of this data set.

    _Frame : tuples in list
        the list objects contains the records of dataset.
        
    _miss_value : value
        the symbol represented miss value in current seriesset.

    _m_value : values
        the number of miss value totally.
    '''
    
    dims = namedtuple('Frame', ['Ln', 'Col'])

    def __init__(self, frame=None, columns=None,
                 miss_symbol=None, miss_value=None):
        
        self._miss_value = miss_value
        
        if frame is None and columns is None:
            self._Frame = list()
            self._columns = list()
            self._dim =  Frame.dims(0, 0)
            self._m_value = list()
            
        elif isinstance(frame, Frame):
            self._Frame = deepcopy(frame._Frame)
            self._columns = deepcopy(frame._columns)
            self._dim = deepcopy(frame._dim)
            self._m_value = deepcopy(frame._m_value)

        elif isinstance(frame, SeriesSet):
            self._Frame = list()
            for line in zip(*frame._set.values()):
                self._Frame.append(list(line))
            self._m_value = deepcopy(frame._m_value)
            self._columns = deepcopy(frame._columns)
            self._dim = Frame.dims(frame._dim.Ln, frame._dim.Col)

        elif is_iter(frame) and not isinstance(frame, str):
            if all(map(is_iter, frame)):
                self._Frame = map(list, frame)
                dim_Col, dim_Ln = len(max(self._Frame, key=len)), len(frame)
            elif all(map(is_value, frame)):
                self._Frame = [[value,] for value in frame]
                dim_Col, dim_Ln = 1, len(frame)

            self._m_value = [0] * dim_Col
            self._dim = Frame.dims(dim_Ln, dim_Col)

            for i, item in enumerate(self._Frame):
                if len(item) < dim_Col:
                    item.extend([self._miss_value] * (dim_Col - len(item)))
                for j, value in enumerate(item):
                    if value == miss_symbol:
                        self._m_value[j] = self._m_value[j] + 1

            if isinstance(columns, str):
                self._columns = [columns + '_%d'%i for i in range(dim_Col)]
            
            elif is_iter(columns):
                self._columns = [str(i) for i in columns[:dim_Col]]
                if len(set(self._columns)) < dim_Col:
                    for i, value in enumerate(self._columns):
                        if self._columns.count(value) != 1 and\
                           self._columns.index(value) != i:
                            self._columns[i] = value + '_%d'%i
                    self._columns.extend(['C%d'%i\
                                          for i in range(len(self._columns),
                                                                    dim_Col)])
            else:
                self._columns = ['C%d'%i for i in range(dim_Col)]
    
        else:
            raise TypeError('can not transform this object while '+\
                            'DaPy.Frame expects a list or deque contained'+\
                            ' tuple or name tuple.')

    @property
    def data(self):
        return self._Frame
    
    @property
    def shape(self):
        return self._dim

    @property
    def miss_value(self):
        return self._miss_value

    @property
    def columns(self):
        return self._columns
    
    @columns.setter
    def columns(self, item):
        if isinstance(item, str):
            self._columns = [item] * self._dim.Col
        elif is_iter(item):
            self._columns = [str(i) for i in item]
        else:
            raise TypeError('unknow title as %s'%item)
        
    @property
    def info(self):
        new_m_v = map(str, self._m_value)
        max_n = len(max(self._columns, key=len))
        
        info = ''
        for i in range(self._dim.Col):
            info += ' '*15
            info += self._columns[i].center(max_n) + '| '
            info += ' ' + new_m_v[i] + '\n'
                
        print('1.  Structure: DaPy.Frame\n' +\
              '2. Dimensions: Ln=%d | Col=%d\n'%self._dim +\
              '3. Miss Value: %d elements\n'%sum(self._m_value) +\
              '4.    Columns: ' + 'Title'.center(max_n) + '|'+\
                             '  Miss\n'+ info)
        
    def __repr__(self):
        temporary_series = [[title, ] for title in self._columns]
        if self._dim.Ln > 20:
            temporary_Frame = self._Frame[:10]
            temporary_Frame.extend(self._Frame[-10:])
        else:
            temporary_Frame = self._Frame
        for line in temporary_Frame:
            for i, value in enumerate(line):
                temporary_series[i].append(str(value))

        column_size = [len(max(col, key=len)) for col in temporary_series]

        frame = ' ' + ' | '.join([title.center(column_size[i]) for i, title in \
                         enumerate(self._columns)]) + '\n'
        frame += '+'.join(['-' * (size + 2) for size in column_size]) + '\n'

        if self._dim.Ln > 20:
            for item in temporary_Frame[:10]:
                line = ''
                for i, value in enumerate(item):
                    line += ' ' + str(value).center(column_size[i]) + ' ' + '|'
                frame += line[:-1] + '\n'
            frame += ('.. Omit %d Ln ..'%(self._dim.Ln - 20)).center(len(line)) + '\n'
            for item in temporary_Frame[-10:]:
                line = ''
                for i, value in enumerate(item):
                    line += ' ' + str(value).center(column_size[i]) + ' ' + '|'
                frame += line[:-1] + '\n'
                
        else:
            for item in temporary_Frame:
                line = ''
                for i, value in enumerate(item):
                    line += ' ' + str(value).center(column_size[i]) + ' ' + '|'
                frame += line[:-1] + '\n'

        return frame[:-1]

    def __contains__(self, e):
        if isinstance(e, str):
            return e in self._columns

        if isinstance(e, tuple):
            for record in self._Frame:
                if record == e:
                    return True
                
        return False
    
    def __len__(self):
        return self._dim.Ln

    def __eq__(self, other):
        if isinstance(other, (Frame, SeriesSet)):
            if self._dim == other.dim:
                return True
            return False

    def __getslice__(self, i, j):
        if i in self._columns or j in self._columns:
            if i in self._columns:
                i = self._columns.index(i)
            elif i is None:
                i = 0
            else:
                raise ValueError('can not get the title of %s'%i)

            if j in self._columns:
                j = self._columns.index(j)
            elif j is None:
                j = self._dim.Col
            else:
                raise ValueError('can not get the title of %s'%j)

            new_frame = list()
            for record in self._Frame:
                new_frame.append(record[i:j + 1])

            return Frame(new_frame, self._columns[i:j + 1], self._miss_value)

        if type(i) != int and type(j) != int:
            raise ValueError('unrecognized symbol as [%s:%s]'%(i, j))

        if i is None or i < 0:
            i = 0
        elif i > self._dim.Ln:
            i = self._dim.Ln

        if j is None:
            j = 0
        elif j < 0:
            j = self._dim.Ln + j
        elif j >= self._dim.Ln:
            j = self._dim.Ln

        return Frame(self._Frame[i:j], self._columns, self._miss_value)
            
    def __getitem__(self, pos):
        if isinstance(pos, int):
            return self._Frame[pos]
        
        if isinstance(pos, slice):
            return self.__getslice__(pos.__getattribute__('start'),
                                     pos.__getattribute__('stop'))

        if isinstance(pos, str): 
            col = self._columns.index(pos)
            return [item[col] for item in self._Frame]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if key <= self._dim.Ln:
                self.__delitem__(key)
                self.insert(key, value)
            else:
                self.append(value)

        elif isinstance(key, str):
            if key in self._columns:
                pos = self._columns.insert(key)
                self.__delitem__(key)
                self.insert_col(key, value, pos)
            else:
                self.append_col(key, value)
                
        else:
            raise TypeError('only can set one record or one column each time.')

    def __delitem__(self, key):
        if isinstance(key, int):
            drop_record = self.pop(key)

        elif isinstance(key, str) and key in self._columns:
            drop_record = self.pop_col(key)

        else:
            raise KeyError('unrecognized symbol %s'%key)
    
    def __iter__(self):
        for record in self._Frame:
            yield record

    def append(self, item, miss_symbol=None):
        '''append a new record to the Frame tail
        '''
        if is_value(item):
            item = [item] * self._dim.Col
            
        elif not isinstance(item, list):
            try:
                item = list(item)
            except:
                raise TypeError("append item is not a support type "+\
                                "with <'%s'>"%type(item))

        if not miss_symbol:
            miss_symbol = self._miss_value
            
        for i, element in enumerate(item):
            if element == miss_symbol:
                self._m_value[i] += 1

        lenth_bias = len(item) - self._dim.Col
        if lenth_bias < 0:
            item.extend([self._miss_value] * abs(lenth_bias))
        elif lenth_bias > 0:
            for record in self._Frame:
                record.extend([self._miss_value] * lenth_bias)
                    
        self._Frame.append(item)
        self._dim = Frame.dims(self._dim.Ln + 1, max(self._dim.Col, len(item)))
        
    def append_col(self, variable_name, series, miss_symbol=None):
        '''append a new variable to the current records tail
        '''
        if variable_name in self._columns:
            raise ValueError("variable name `%s` is already taken."%self.variable_name)
            
        if not isinstance(variable_name, str):
            raise TypeError("variable name should be a ``str``")

        if not miss_symbol:
            miss_symbol = self._miss_value

        if is_value(series):
            series = [series] * self._dim.Ln

        size = len(series)
        if size < self._dim.Ln:
            series.extend([self._miss_value] * (self._dim.Ln - size))
        elif size > self._dim.Ln:
            self._Frame.extend([[self._miss_value] * self._dim.Col\
                                for i in range(size - self._dim.Ln)])
        
        self._m_value.append(0)
        for i, element in enumerate(series):
            if element == miss_symbol:
                self._m_value[-1] += 1
            self._Frame[i].append(element)

        self._columns.append(variable_name)
        self._dim = Frame.dims(max(self._dim.Ln, size), self._dim.Col+1)
        
    def count(self, X, *area):
        if is_value(X):
            X = [X, ]
        elif not is_iter(X):
            raise TypeError('X should be an iterable'+\
                            " object, such as 'tuple' or 'list'")
        
        counter = Counter()
        if area == all or area[0] == all:
            C1, L1 = 0, 0
            L2, C2 = self._dim
                
        elif not (len(area) == 2 and len(area[0]) == len(area[1]) == 2):
            raise EOFError("unreccognized Expression '%s'"%str(area))

        else:        
            if area[0][1] == all:
                C1 = 0
            elif isinstance(area[0][1], int) and 0 <= area[0][1] <= self._dim.Col:
                C1 = area[0][1]
            elif area[0][1] in self._columns:
                C1 = self._columns.index(area[0][1])
            else:
                raise TypeError('second position in the first tuple should a int '+\
                                'type emblemed column number or a str emblemed'+\
                                'column name.')

            if area[1][1] == all:
                C2 = self._dim.Col
            elif isinstance(area[1][1], int) and 0 <= area[1][1] <= self._dim.Col:
                C2 = area[1][1]
            elif area[1][1] in self._columns:
                C2 = self._columns.index(area[1][1])
            else:
                raise TypeError('second position in the second tuple should a int '+\
                                'type emblemed column number or a str emblemed '+\
                                "column name. Tip: use all to stand for all.")

            if area[0][0] == all:
                L1 = 0
            elif isinstance(area[0][0], int) and 0 <= area[0][0] <= self._dim.Ln:
                L1 = area[0][0]
            else:
                raise TypeError('first position in the first tuple should be a int '+\
                                'type emblemed start line number. Tip: use all to '+\
                                "stand for all.")

            if area[1][0] == all:
                L2 = self._dim.Ln
            elif isinstance(area[1][0], int) and 0 <= area[1][0] <= self._dim.Ln:
                L2 = area[1][0]
            else:
                raise TypeError('first position in the second tuple should be a int '+\
                                'type emblemed start line number. Tip: use all to '+\
                                "stand for all.")
        if C2 < C1 or L2 < L1:
            raise ValueError('the postion in the second tuple should be larger '+\
                             'than first tuple.')

        for record in self._Frame[L1:L2 + 1]:
            for value in record[C1:C2 + 1]:
                if value in X:
                    counter[value] += 1
        return counter

    def extend(self, other):
        if isinstance(other, Frame):
            new_title = 0
            for title in other._columns:
                if title not in self._columns:
                    self._columns.append(title)
                    new_title += 1

            for record in self._Frame:
                record.extend([self._miss_value] * new_title)
                
            extend_part = [[self._miss_value] * len(self._columns)\
                           for i in range(len(other))]
            new_title_index = [self._columns.index(title)
                               for title in other._columns]
            self._dim = Frame.dims(len(self) + len(other), len(self._columns))
            self._m_value.extend([self._dim.Ln] * new_title)

            for i, record in enumerate(other._Frame):
                for j, value in zip(new_title_index, record):
                    if value == other._miss_value:
                        value = self._miss_value
                    extend_part[i][j] = value

            self._Frame.extend(extend_part)

        elif isinstance(other, SeriesSet):
            self.extend(Frame(other))

        elif all(map(is_seq, other)):
            self.extend(Frame(other, self._columns))
            
        else:
            raise TypeError('can not extend the dataset with this object.')

    def extend_col(self, other):
        if isinstance(other, Frame):
            new_title = list()
            for i, title in enumerate(other._columns):
                if title not in self._columns:
                    self._columns.append(title)
                    new_title.append(i)
                    self._m_value.append(0)

            new_title = set(new_title)
            for i, record in enumerate(other._Frame):
                new_record, current_record = list(), self._Frame[i]
                for j, value in enumerate(record):
                    if value == other._miss_value:
                        value = self._miss_value
                        self._m_value[self._columns.index(other._columns[j])] += 1
                        
                    if j in new_title:
                        current_record.append(value)
                    else:
                        j_ = self._columns.index(other._columns[j])
                        new_record.extend([self._miss_value] * (j_ - len(new_record)))
                        new_record.append(value)
                        self._m_value[j_] += len(new_record) - j_
                if new_record:
                    self._Frame.append(new_record)
            self._dim = Frame.dims(len(self._Frame), len(self._columns))

            for record in self._Frame:
                record.extend([self._miss_value] * (self._dim.Col - len(record)))
            
        elif isinstance(other, Frame):
            self.extend_col(SeriesSet(other))
            
        elif all(filter(is_iter, other)):
            new_col = [title + '_1' for title in self._columns]
            self.extend_col(SeriesSet(other, new_col))

        else:
            raise TypeError('could not extend a single value only.')
    def pop_miss_value(self, axis='LINE'):
        '''pop all records that maintains miss value while axis is `LINE` or
        pop all variables that maintains miss value while axis is `COL`
        '''
        if axis.upper() == 'LINE':
            pops = list()
            for i, record in enumerate(self._Frame):
                if self._miss_value in record:
                    pops.append(i)
            pops = [self._Frame.pop(i) for i in sorted(pops, reverse=True)]
            self._dim = Frame.dims(len(pops), self._dim.Col)
            return Frame(pops, self._columns)

        if axis.upper() == 'COL':
            pop_col = list()
            for i, sequence in enumerate(zip(*self._Frame)):
                if self._miss_value in sequence:
                    pop_col.append(i)
                    
            pop_col.reverse()
            new_columns = [self._columns.pop(i) for i in pop_col].reverse()
            
            new_frame = [0] * self._dim.Ln
            for line, record in enumerate(self._Frame):
                new_frame[line] = [record.pop(i) for i in pop_col]
            for record in new_frame:
               record.reverse() 
            self._dim = Frame.dims(self._dim.Ln, len(self._columns))
            return SeriesSet(new_frame, new_columns)

        raise ValueError('axis should be (`LINE` or `COL`) symbols only.')

    def insert(self, pos, item, miss_symbol=None):
        '''insert a new record to the frame with position `index`
        '''
        if is_value(item):
            item = [item] * self._dim.Col
            
        elif not isinstance(item, list):
            try:
                item = list(item)
            except:
                raise TypeError("append item is not a support type "+\
                                "with <'%s'>"%type(item))

        if not miss_symbol:
            miss_symbol = self._miss_value
            
        for i, element in enumerate(item):
            if element == miss_symbol:
                self._m_value[i] += 1

        lenth_bias = len(item) - self._dim.Col
        if lenth_bias < 0:
            item.extend([self._miss_value] * abs(lenth_bias))
        elif lenth_bias > 0:
            for record in self._Frame:
                record.extend([self._miss_value] * lenth_bias)
                    
        self._Frame.insert(pos, index)
        self._dim = Frame.dims(self._dim.Ln + 1, max(self._dim.Col, len(item)))

    def insert_col(self, variable_name, series, index, miss_symbol=None):
        '''insert a new variable to the current records in position `index`
        '''
        if variable_name in self._columns:
            raise ValueError("variable name `%s` is already taken."%self.variable_name)
            
        if not isinstance(variable_name, str):
            raise TypeError("variable name should be a ``str``")

        if not miss_symbol:
            miss_symbol = self._miss_value

        if is_value(series):
            series = [series] * self._dim.Ln

        size = len(series)
        if size < self._dim.Ln:
            series.extend([self._miss_value] * (self._dim.Ln - size))
        elif size > self._dim.Ln:
            self._Frame.extend([[self._miss_value] * self._dim.Col\
                                for i in range(size - self._dim.Ln)])
        
        self._m_value.append(0)
        for i, element in enumerate(series):
            if element == miss_symbol:
                self._m_value[-1] += 1
            self._Frame[i].insert(index, element)

        self._columns.insert(index, variable_name)
        self._dim = Frame.dims(max(self._dim.Ln, size), self._dim.Col+1)
                
    def pick(self, conditions):
        '''pick out records that comply with conditions
        '''
        conditions = str(conditions)
        for i, title in enumerate(self._columns):
            new = 'record[%d]'%i
            conditions = conditions.replace(title, new)
        conditions = 'check = ' + conditions

        return_frame = list()
        for record in self._Frame:
            try:
                exec(conditions)
            except SyntaxError:
                raise TypeError('can not transform some of your variable name.')
            if check:
                return_frame.append(record)

        return Frame(return_frame, self._columns)

    def pop(self, item=-1):
        '''pop(remove & return) a record from the Frame
        '''
        if isinstance(item, int):
            pop_item = self._Frame.pop(item)
            self._dim = Frame.dims(self._dim.Ln - 1, self._dim.Col)
            for i, value in pop_item:
                if value == self._miss_value:
                    self._m_value[i] -= 1
            return pop_item
        
        raise TypeError('an integer is required.')

    def pop_col(self, *titles):
        '''pop(remove & return) a series from the Frame
        '''
        pos = list()
        pop_name = list()
        if not titles:
            titles = [self._columns[-1]]
            
        for item in titles:
            if not isinstance(item, (int, str)):
                raise TypeError('an integer or string, which stands for a '+\
                                'column name, is required, '+
                                'not {0} !'.format(type(item)))
            if isinstance(item, str) and item in self._columns:
                pos.append(self._columns.index(item))
                pop_name.append(item)
            elif isinstance(item, int) and abs(item) < self._dim.Col:
                pos.append(item)
                pop_name.append(self._columns[item])
            else:
                raise IndexError('%s is not a column name or index.'%str(item))
        
        for name in pop_name:
            index = self._columns.index(name)
            self._columns.pop(index)
            self._m_value.pop(index)
            
        pop_data = [list() for i in range(len(pop_name))]

        new_frame = [0] * self._dim.Ln
        for j, record in enumerate(self._Frame):
            line = list()
            for i, value in enumerate(record):
                if i in pos:
                    pop_data[pos.index(i)].append(value)
                else:
                    line.append(value)
            new_frame[j] = line

        self._dim = Frame.dims(self._dim.Ln, self._dim.Col-len(pos))
        self._Frame = new_frame
        return SeriesSet(dict(zip(pop_name, pop_data)))

    def read_text(self, addr, first_line=1, miss_symbol='NA',
                 title_line=0, sep=',', prefer_type=None):
        def transform(i, item):
            try:
                return _col_types[i](item)
            except ValueError:
                if item == miss_symbol:
                    self._m_value[i] += 1
                    return self._miss_value
                return str2value(item, prefer_type)
            
        with open(addr, 'r') as f:
            col = 0
            for i, line in enumerate(f):
                if line.count(sep) > col:
                    col = line.count(sep) + 1
            self._Frame = [0] * i
            self._m_value = [0] * col
            _col_types = [None] * col
    
            f.seek(0)
            for i, lines in enumerate(f):
                if i >= first_line:
                    for j, item in enumerate(lines[:-1].split(sep)):
                        if j in _col_types or item == miss_symbol or item == self._miss_value:
                            continue
                        _col_types[j] = TRANS_FUN_SET[type(str2value(item))]
                if all(_col_types):
                    break

            _col_types = tuple(_col_types)

            f.seek(0)
            for m, record in enumerate(f):
                if m == title_line:
                    self._columns = record[:-1].split(sep)
                if m >= first_line - 1 :
                    break

            for m, record in enumerate(f):
                line = [transform(i, v) for i, v in enumerate(record.split(sep))]        
                line.extend([self._miss_value] * (col - len(line)))
                self._Frame[m] = line

        if not self._columns:
            self._columns = ['C%d'%i for i in range(col)]

        if len(self._columns) != col:
            self._columns.extend(['C%d'%i for i in range(len(self._columns),
                                                            col - len(self._columns))])     
        elif len(set(self._columns)) != col:
            for i, title in enumerate(self._columns):
                if self._columns.count(title) <= 1 or i == self._columns.index(title):
                    continue
                
                for j in range(self._columns.count(title)):
                    if title + '_%d'%j not in self._columns:
                        self._columns[i] = title + '_%d'%j
                 
        self._dim = Frame.dims(m + 2 - first_line, col)

    def reverse(self):
        self._Frame.reverse()

    def rename_col(self, *new_names):
        try:
            for names in new_names:
                if names[0] in self._columns:
                    if names[1] not in self._columns:
                        self._columns = names[1]
                    else:
                        raise ValueError("Name '%s' has already exist'"%names[1])
                else:
                    raise ValueError("'%s' is not exist"%names[0])
        except:
            raise TypeError('This function expects some parameters like 2-dimentions tuple')

    def replace(self, col=all, *arg):
        if col == all:
            col = range(self._dim[1])
        elif isinstance(col, int) and col < self._dim.Col:
            col = (col, )
        elif isinstance(col, str):
            col = (self._columns.index(col), )
        else:
            new_col = list()
            for c in col:
                if isinstance(c, str):
                    new_col.append(self._columns.index(c))
                elif isinstance(c, int) and c < self._dim.Col:
                    new_col.append(c)
                else:
                    raise TypeError("unrecognized symbol '%s'."%str(c))
            col = new_col

        order = dict(arg)

        for i, record in enumerate(self._Frame):
            for j in col:
                if record[j] in order:
                    record = list(record)
                    record[j] = order[record[j]]
            self._Frame[i] = tuple(record)
                    
    def sort(self, *orders):
        '''TODO: sort the dataframe
        See Also: numpy/core/fromnumeric.py Line:756
        See Also: numpy/core/numeric.py Line:508
        '''
        for each in orders:
            if len(each) != 2:
                raise TypeError("Order argument expects some 2-dimensions"+\
                        " tuples which contains 'keyword' in the first"+\
                        ' postion and request in the second place.')
                

        compare_pos = list()
        for order in orders:
            if isinstance(order[0], int):
                compare_pos.append(order[0])
            elif order[0] in self._columns:
                compare_pos.append(self._columns.index(order[0]))
            else:
                raise TypeError("'%s' keyword is "%order[0] +\
                                "not in frame's columns.")
        try:
            compare_symbol = [order[1] for order in orders if order[1]\
                              in ('DESC', 'ASC')]
        except:
            raise TypeError("'%s' is not a recognized symbol."%order[1])

        size_orders = len(compare_pos) - 1

        def hash_sort(datas_, i=0):
            # initialize values
            index = compare_pos[i]
            inside_data = list()
            HashTable = dict()

            # create the diction
            for item in datas_:
                key = item[index]
                if key in HashTable:
                    HashTable[key].append(item)
                else:
                    HashTable[key] = [item]

            # sorted the values
            sequence = sorted(HashTable)

            # transform the record into Frame
            for each in sequence:
                items = HashTable[each]
                
                if i < size_orders:
                    items = hash_sort(items, i+1)

                inside_data.extend(items)
                    
            # finally, reversed the list if necessary.
            if i != 0 and compare_symbol[i] != compare_symbol[i-1]:
                inside_data.reverse()

            return inside_data

        if len(set(compare_symbol)) == 1:
            self._Frame = sorted(self._Frame, key=itemgetter(*compare_pos))
        else:
            self._Frame = hash_sort(self._Frame)

        if compare_symbol[0] == 'DESC':
            self._Frame.reverse()
            
    def shuffles(self):
        shuffle(self._Frame)

