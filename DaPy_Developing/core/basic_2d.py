from array import array
from collections import namedtuple, deque, OrderedDict, Counter, Iterable
from copy import deepcopy
import csv
import cPickle as pkl
from datetime import date, datetime
from numeric import describe, mean, _sum
from basic_fun import is_seq, is_iter, is_math, is_value, get_sorted_index
from numeric import corr as f_c
from warnings import warn
from operator import itemgetter, attrgetter
from os import path
from pprint import pprint
from random import random, randint, shuffle
import string

__all__ = ['SeriesSet', 'Frame', 'Matrix', 'Table']
    
                
class SeriesSet(object):
    '''Maintains the data as sequences.

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

    _type_value : types in list
        the value type of each column.
    '''

    
    transform = {str: 'c',
                 int: 'l',
                 float: 'f',
                 long: 'L',
                 unicode: 'u'}
    dims = namedtuple('SeriesSet', ['Ln', 'Col'])

    def __init__(self, series=None, columns=None, element_type=None,
                 miss_symbol=None, miss_value=None):

        self._miss_value = miss_value
        self._m_value = list()
        self._columns = columns

        if series is None and columns is None:
            self.__init_empty()
        elif isinstance(series, SeriesSet):
            self.__init_set(series)
        elif isinstance(series, (dict, OrderedDict)):
            self.__init_dict(series, miss_symbol)
        elif isinstance(series, (Frame, Matrix)):
            self.__init_frame_matrix(series, miss_symbol)
        elif isinstance(series, Table):
            self.__init_table(series, miss_symbol)
        elif is_seq(series) or is_iter(series):
            self.__init_normal(series, miss_symbol, columns, element_type)
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
    def titles(self): # OK #
        return self._columns

    @titles.setter
    def titles(self, item):
        if isinstance(item, str):
            new_ = OrderedDict()
            for i, title in enumerate(self._columns):
                new_[item + '_%d'%i] = self._set[title]
            self._set = new_
            self._columns = [item + '_' + str(i) for i in range(self._dim.Col)]
        elif is_iter(item):
            if len(item) == self._dim.Col:
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
        dtype = [str(t).split("'")[1] for t in self._type_value]
        miss = [str(v) for v in self._m_value]

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

        print '1.  Structure: DaPy.SeriesSet\n' +\
              '2. Dimensions: Ln=%d | Col=%d\n'%self._dim +\
              '3. Miss Value: %d elements\n'%sum(self._m_value) +\
              '4.   Describe: \n'+\
              title_line[:-1] + '\n' +\
              info + '='*(6 + sum(blank_size))

    def __init_empty(self):
        self._set = OrderedDict()
        self._columns = list()
        self._dim = SeriesSet.dims(0, 0)
        self._type_column = list()
        self._type_value = list()
        
    def __init_set(self, series):
        self._set = deepcopy(series)
        self._columns = deepcopy(series._columns)
        self._dim = deepcopy(series._dim)
        self._type_column = deepcopy(series._type_column)
        self._type_value = deepcopy(series._type_value)
        self._m_value = deepcopy(series._m_value)
        self._miss_value = deepcopy(series.miss_value)
        
    def __init_dict(self, series, miss_symbol):
        self._set = OrderedDict() # SeriesSet
        if not self._columns:
            self._columns = sorted([title for title in series]) # Column names
        elif len(self._columns) != len(series):
            raise ValueError('the lenth of titles does not match the data size.')
        self._type_column = list() #
        self._type_value = list()

        # MaxSize of this Series Set
        max_Ln = max([len(series[title]) for title in self._columns])
        for title, sequence in series.items():
            size = len(sequence)
            m_value_line = 0 # miss value in this column
            
            # Check the type of each sequence
            if isinstance(sequence, array):
                self._type_column.append(array)
                self._type_value.append(sequence.typecode)

            elif is_seq(sequence):
                self._type_column.append(type(sequence))
                check_type = False
                for i, item in enumerate(sequence):
                    # Check miss value
                    if item == miss_symbol or item == self._miss_value:
                        m_value_line += 1
                        sequence[i] = self._miss_value
                        continue

                    # Set the value's type in this column
                    if check_type is False:
                        if is_value(item):
                            self._type_value.append(type(item))
                            check_type = type(item)
                        else:
                            raise TypeError("unsupport type '%s' "%type(item)+\
                                        "in column '%s' and "%title+\
                                        "line '%d'."%i)

                    # check the value's type
                    if isinstance(item, check_type) is False:
                        raise TypeError("unsupport type '%s' "%type(item)+\
                                        "in column '%s' and "%title+\
                                        "line '%d'."%i)
            else:
                raise TypeError("unsupport type '%s' "%type(sequence)+\
                                "in column '%s'."%title)

            # Try to transform to array
            if size != max_Ln:
                if isinstance(sequence, array):
                    sequence = list(sequence)
                sequence.extend([self._miss_value] * (max_Ln - size))
                
            self._m_value.append(m_value_line)
            self._set[title] = sequence
        self._dim = SeriesSet.dims(max_Ln, len(self._columns))

    def __init_frame_matrix(self, series, miss_symbol):
        self._dim = SeriesSet.dims(series._dim.Ln, series._dim.Col)         
        self._type_column = [list] * self._dim.Col
        if not self._columns:
            self._columns = deepcopy(series._columns)
        elif len(self._columns) != series.shape.Col:
            raise ValueError('the lenth of titles does not match the data size.')
        self._m_value = [0] * self._dim.Col
        self._set = OrderedDict()
        for i, title in enumerate(self._columns):
            sequence = series[title]
            self._set[title] = sequence
            self._m_value[i] = sequence.count(miss_symbol) + \
                               sequence.count(self._miss_value)
        if isinstance(series, Matrix):
            self._type_value = [float] * self._dim.Col
        else:
            self._type_value = deepcopy(series._type_value)
            
    def __init_table(self, series, miss_symbol):
        if not self._columns:
            self._columns = deepcopy(series._columns)
        elif len(self._columns) != series.shape.Col:
            raise ValueError('the lenth of titles does not match the data size.')
        self._dim = SeriesSet.dims(series._dim.Ln, series._dim.Col) 
        self._m_value = list()
        self._type_value = [None] * self._dim.Col
        self._type_column = [list] * self._dim.Col
        for record in series:
            for i, value in enumerate(record):
                if value != self._miss_value and value != miss_symbol:
                    self._type_value[i] = type(value)
            if None not in self._type_value:
                break
            
        self._set = OrderedDict()
        for i, title in enumerate(self._columns):
            sequence = series[title]
            col_m_value = 0
            for j, value in enumerate(sequence):
                if value == self._miss_value or value == miss_symbol:
                    col_m_value += 1
                elif not isinstance(value, self._type_value[i]):
                    raise TypeError('value in position <%d, %d> '%(j, i)+\
                                    "doesn't have a correct type "+\
                                    "as %s"%str(type(value)))
            self._set[title] = sequence
            self._m_value.append(col_m_value)
            
    def __init_normal(self, series, miss_symbol, columns, element_type):
        self._set = OrderedDict()
        # initalized an element type
        if element_type is not None:
            if isinstance(element_type, (list, tuple)):
                self._type_value = element_type
            else:
                self._type_value = [element_type, ]
        else:
            self._type_value = None

        if is_seq(series[0]):
            # if the input object is a sequence in sequence structure
            lenth_Col = len(max(series, key=len))
            for pos, each in enumerate(series):
                if len(each) < lenth_Col:
                    new = list(each)
                    new.extend([self._miss_value\
                                for i in range(lenth_Col - len(each))])
                    series[pos] = new
            # initalized the column titles
            if is_iter(self._columns):
                self._columns = [str(v) for v in columns]
                if len(self._columns) < lenth_Col:
                    self._columns.extend(['Col_%d'%i for i in range(lenth_Col - \
                                                         len(self._columns))])
                elif len(self._columns) > lenth_Col:
                    self._columns = self._columns[:lenth_Col]
                                          
            else:
                self._columns = ['Col_%d'%i for i in range(lenth_Col)]
                
            self._dim = SeriesSet.dims(len(series), lenth_Col)
            self._m_value = [0]*self._dim.Col

            if not self._type_value:
                self._type_value = [None] * self._dim.Col
                for record in series:
                    for i, value in enumerate(record):
                        if value not in [self._miss_value, miss_symbol]:
                            self._type_value[i] = type(value)
                    if all(self._type_value):
                        break

            for title in self._columns:
                self._set[title] = [self._miss_value] * self._dim.Ln
                
            for j, record in enumerate(series):
                for i, v in enumerate(record):
                    
                    if v == miss_symbol:
                        self._set[self._columns[i]][j] = self._miss_value
                        self._m_value[i] += 1
                    elif not isinstance(v, self._type_value[i]):
                        print v, self._type_value
                        raise TypeError("'%s' in position <%d, %d> "%(v, j, i)+\
                                    "doesn't have a correct type %s."%self._type_value[i])
                    else:
                        self._set[self._columns[i]][j] = v
            self._type_column = [list]*self._dim.Col
            return
                
        else:
            # if input object is a 1-d sequence structure
            if not self._type_value:
                for value in series:
                    if value != self._miss_value and value != miss_symbol:
                        self._type_value = [type(value)]
                        break

            # initalized the title names
            if isinstance(columns, list):
                self._columns = [str(v) for v in columns]
            elif isinstance(columns, (str, int)):
                self._columns = [columns, ]
            else:
                self._columns = ['Col_0', ]

            if len(self._type_value) != len(self._columns):
                raise ValueError('size of type value does not match.')

            self._dim = SeriesSet.dims(len(series), 1)
            self._type_column = [type(series)]
            self._m_value = [0, ]
                    
            for i, value in enumerate(series):
                if value == miss_symbol or value == self._miss_value:
                    value = self._miss_value
                    self._m_value[0] += 1
                elif not isinstance(value, self._type_value[0]):
                    raise TypeError("'%s' in position <0, %d>"%(v, i)+\
                                    "doesn't have a correct type.")
            self._set[self._columns[0]] = series
            return
        
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

        msg = ''
        size = len(max(self._columns, key=len))
        for i, title in enumerate(self._columns):
            msg += write_Ln(i, title, size - len(title))
        return msg[:-1]

    def __contains__(self, e):
        if isinstance(e, str):
            return e in self._columns

        if isinstance(e, (list, tuple)):
            if len(e) == self._dim.Col:
                p0 = e[0]
                scan = [i for i, v in enumerate(self._set[self._columns[0]])\
                        if v == p0]
                for line in scan:
                    Flag = True
                    for i, title in enumerate(self._columns[1:], 1):
                        if self._set[title][line] != e[i]:
                            Flag = False
                            break
                    if Flag:
                        return True
                    
            elif len(e) == self._dim.Ln:
                for sequence in self._set.values():
                    Flag = True
                    for p, i in enumerate(sequence):
                        if e[p] != i:
                            Flag = False
                            break
                    if Flag:
                        return True
        return False

    def __len__(self):  # OK #
        return self._dim.Ln

    def __eq__(self, other):  # OK #
        if isinstance(other, (Frame, SeriesSet, Matrix)):
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
            j = self._dim.Ln - 1
            
        if i > j:
            i, j = j, i

        return_list = list()
        for pos in range(i, j):
            return_list.append([self._set[t][pos] for t in self._columns])
        return Frame(return_list, self._columns, self._type_value,
                     None, self._miss_value)

    def __getitem__(self, pos):  # OK #
        if isinstance(pos, int):
            return [self._set[title][pos] for title in self._columns]

        elif isinstance(pos, slice):
            return self.__getslice__(pos.__getattribute__('start'),
                            pos.__getattribute__('stop'))

        elif isinstance(pos, str):
            return self._set[pos]
        
        else:
            raise TypeError('SeriesSet indices must be int or str, '+\
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
                self.insert_col(pos, value)
            else:
                self.append_col(value)

    def __delitem__(self, key):
        if isinstance(key, int):
            drop_record = self.pop(key)

        elif isinstance(key, str) and key in self._columns:
            drop_record = self.pop_col(key)

        else:
            raise KeyError('unrecognized symbol %s'%key)
                                
    def __iter__(self):  # OK #
        for i in range(self._dim.Ln):
            yield [self._set[title][i] for title in self._columns]

    def __reversed__(self):
        self.reverse()

    def append(self, item, miss_symbol=None):
        if not is_iter(item):
            if is_value(item):
                item = [item for i in range(self._dim.Col)]
            else:
                raise TypeError("insert item is not a support type "+\
                                "with '%s'"%type(item))

        if not miss_symbol:
            miss_symbol = self._miss_value

        for i, title in enumerate(self._columns):
            element = item[i]
            if element == miss_symbol:
                if not isinstance(self._type_column[i], list):
                    self._set[title] = list(self._set[title])
                self._set[title].append(self._miss_value)
                self._m_value[i] += 1
            elif isinstance(element, self._type_value[i]):
                self._set[title].append(element)
            else:
                raise TypeError('the element in position %d is'%i+
                                ' not a correct type.')

        self._dim = SeriesSet.dims(self._dim.Ln + 1, self._dim.Col)
        
    def append_col(self, variable_name, series, element_type='AUTO',
                   miss_symbol=None): # OK #
        '''append a series data to the seriesset last
        '''
        if is_value(series):
            series = [series for i in range(self._dim.Ln)]
            
        if not isinstance(variable_name, str):
            raise ValueError('unsupport type to set as variable name.')
        
        if not isinstance(series, Iterable):
            raise ValueError("unsupport type to append as values")
        
        # check the variable name
        if variable_name in self._columns:
            raise ValueError("variable <'%s'> is already taken."%variable_name)

        if not miss_symbol:
            miss_symbol = self._miss_value
            
        # check the lenth of data
        size_series = len(series)
        if size_series < self._dim.Ln:
            series = list(series)
            series.extend([self._miss_value for t in range(
                self._dim.Ln-size_series)])
            size_series = self._dim.Ln
        elif size_series > self._dim.Ln:
            for title in self._columns:
                self._set[title] = list(self._set[title])
                self._set[title].extend([self._miss_value for t in range(
                    size_series-self._dim.Ln)])
        # set the elements' type
        if element_type.upper() == 'AUTO':
            if isinstance(series, array):
                element_type = type(series[0])
            else:
                for each in series:
                    if each != miss_symbol:
                        element_type = type(each)
                        break
        mv = 0
        # check the miss value and values' type
        if not isinstance(series, array):
            for i, element in enumerate(series):
                if not isinstance(element, element_type):
                    if element != miss_symbol:
                        raise ValueError("The element in postion %d doesn't"%i+\
                                         "have the correct type.")
                    else:
                        mv += 1
                elif element == self._miss_value:
                    mv += 1
                elif element == miss_symbol:
                    mv += 1
                    series[i] = self._miss_value
        self._columns.append(variable_name)
        self._m_value.append(mv)
        self._type_value.append(element_type)
        self._type_column.append(type(series))
        self._dim = SeriesSet.dims(size_series, self._dim.Col+1)
        self._set[variable_name] = series

    def corr(self):
        matrix = Matrix(columns=self._columns)
        matrix.make(self._dim.Col, self._dim.Col, 0)
        for i, title in enumerate(self._columns):
            if self._type_value[i] not in (int, float, long):
                for j in xrange(self._dim.Col):
                    matrix[i][j] = 0
                    matrix[j][i] = 0
                continue

            sequence = self._set[title]
            if self._m_value[i] != 0:
                sequence = [v for v in sequence if v != self._miss_value]

            for j, next_title in enumerate(self._columns):
                if i == j:
                    matrix[j][i] = 1
                    continue
                if self._type_value[j] not in (int, float, long):
                    matrix[j][i] = 0
                    matrix[i][j] = 0
                    continue
                if self._m_value[j] != 0:
                    next_sequence = [v for v in self._set[next_title]\
                                     if v != self._miss_value]
                else:
                    next_sequence = self._set[next_title]
                try:
                    r = f_c(sequence, next_sequence)
                except ZeroDivisionError:
                    r = None
                matrix[i][j], matrix[j][i] = r, r
        new_ = Frame(matrix, self._columns)
        new_.insert_col('', self._columns, 0, str)
        return new_

    def count(self, X, area=all):
        counter = Counter()
        if area == all or area[0] == all:
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
            raise EOFError("unreccognized expression '%s'"%str(area))

        for title in self._columns[C1:C2]:
            sequence = self._set[title]
            for value in sequence[L1:L2]:
                if value in X:
                    counter[value] += 1
        return counter

    def count_element(self, col=all):
        if col == all:
            col = self._columns
        elif isinstance(col, int):
            if col < self._dim.Col:
                col = [self._columns[col]]
            else:
                raise TypeError("unrecognized expression '%s'"%str(col))
        elif col in self._columns:
            col = [col, ]
        elif isinstance(col, (list, deque, tuple)):
            new = list()
            for each in col:
                if each in self._columns:
                    new.append(each)
                elif isinstance(each, int) and -1 < each < self._dim.Col:
                    new.append(self._columns[each])
                else:
                    raise TypeError("<%s> is not a title name"%str(each))
            col = new
        else:
            raise TypeError("unrecognized expression '%s'"%str(col))

        collect_elements = dict()
        for title in col:
            try:
                c = Counter(self._set[title])
            except KeyError:
                print col, 
            collect_elements[title] = c
        return collect_elements

    def drop_miss_value(self, drop='COL'):
        if str(drop).upper() == 'COL':
            for i in xrange(self._dim.Col-1, -1, -1):
                if self._m_value[i] != 0:
                    self._set.pop(self._columns[i])
                    self._columns.pop(i)
                    self._type_value.pop(i)
                    self._type_column.pop(i)
            self._dim = SeriesSet.dims(self._dim.Ln, len(self._columns))
            self._m_value = [0 for i in xrange(self._dim.Col)]
            
        elif str(drop).upper() == 'LINE':
            drop_line_index = list()
            for title in self._columns:
                sequence = self._set[title]
                for i, v in enumerate(sequence):
                    if v == self._miss_value:
                        drop_line_index.append(i)
            drop = sorted(set(drop_line_index), reverse=True)
            for title in self._columns:
                sequence = self._set[title]
                for index in drop:
                    sequence.pop(index)
            self._dim = SeriesSet.dims(self._dim.Ln - len(drop),
                                       len(self._columns))
            self._m_value = [0 for i in range(self._dim.Col)]
        else:
            raise IndexError("unrecognized expression '%s'"%str(drop))

    def items(self):
        return self._set.items()
   
    def insert(self, pos, item, miss_symbol=None): # OK #
        '''
        TODO: insert a record to the frame, position in <index>
        '''
        if not (isinstance(pos, int) and is_iter(item)):
            raise TypeError("Insert item is not a support"+\
                            "type with <'%s'>"%type(item))

        if not miss_symbol:
            miss_symbol = self._miss_value

        for i, title in enumerate(self._columns):
            element = item[i]
            if element == miss_symbol or element == self._miss_value:
                if not isinstance(self._type_column[i], list):
                    self._set[title] = list(self._set[title])
                self._set[title].insert(pos, self._miss_value)
                self._m_value[i] += 1
            elif isinstance(element, self._type_value[i]):
                self._set[title].insert(pos, element)
            else:
                raise TypeError('the element in position %d is'%i+
                                ' not a correct type.')
        self._dim = SeriesSet.dims(self._dim.Ln+1, self._dim.Col)

    def insert_col(self, variable_name, series, index, element_type='AUTO',
                   miss_symbol=None): # OK #
        '''
        TODO: insert a series of data to the frame, position in <index>
        '''
        # TODO: append a series data to the frame last
        if is_value(series):
            series = [series for i in range(self._dim.Ln)]
            
        if not isinstance(variable_name, str):
            raise ValueError("unsupport type to append in this frame")
        
        # check the variable name
        if variable_name in self._columns:
            raise ValueError("variable name <'%s'> is already taken."%self.\
                             variable_name)

        if not miss_symbol:
            miss_symbol = self._miss_value

        # check the lenth of data
        size_series = len(series)
        if size_series < self._dim.Ln:
            series = list(series)
            series.extend([self._miss_value for t in range(
                self._dim.Ln-size_series)])
            size_series = self._dim.Ln
        elif size_series > self._dim.Ln:
            for title in self._columns:
                self._set[title] = list(self._set[title])
                self._set[title].extend([self._miss_value for t in range(
                    size_series-self._dim.Ln)])

        # set the elements' type
        if element_type.upper() == 'AUTO':
            if isinstance(series, array):
                element_type = type(series[0])
            else:
                for each in series:
                    if each != miss_symbol and each != self._miss_value:
                        element_type = type(each)
                        break
        mv = 0
        # check the miss value and values' type
        if not isinstance(series, array):
            for i, element in enumerate(series):
                if isinstance(element, element_type) is False:
                    if element != self._miss_value and element != miss_symbol:
                        raise ValueError("element in postion %d doesn't"%i+\
                                         "have the correct type.")
                    else:
                        mv += 1
                elif element == self._miss_value:
                    mv += 1
                elif element == miss_symbol:
                    mv += 1
                    series[i] = self._miss_value

        self._columns.insert(index, variable_name)
        self._type_value.insert(index, element_type)
        self._type_column.insert(index, type(series))
        self._dim = SeriesSet.dims(size_series, self._dim.Col+1)
        new_set = OrderedDict()
        self._m_value.insert(index, mv)
        if i >= self._dim.Col:
            self._set[variable_name] = series
            return
        
        for i, title in enumerate(self._set):
            if i == index:
                new_set[variable_name] = series
            new_set[title] = self._set[title]

        self._set = new_set

    def keys(self):
        return self._set.keys()

    def normalized(self, col=all, attr=None, get_attr=None, process='NORMAL'):
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
                test = [v for v in sequence if v != self._miss_value]
                A = float(min(test))
                B = max(test) - A

            elif process == 'STANDARD':
                test = [v for v in sequence if v != self._miss_value]
                statis = describe(test)
                A, B = statis.Mean, statis.Sn

            elif process == 'LOG':
                pass
                
            else:
                raise ValueError("unrecognized symbol '%s',"%str(process)+\
                                 "use 'NORMAL', 'STANDARD' and 'LOG'")
            
            if process == 'LOG':
                sequence =list()
                for value in sequence:
                    try:
                        sequence.append(log(value))
                    except ValueError:
                        pass

            else:
                new = list()
                for value in sequence:
                    try:
                        new.append((value-A)/B)
                    except ZeroDivisionError:
                        new.append(0)
                    except TypeError:
                        new.append(value)
                sequence = new
                del new

            if isinstance(self._type_value[i], array):
                sequence = array('f', sequence)
            
            self._set[title] = sequence
            try:
                attrs_dic[title] = attrs_structure(A, B)
            except UnboundLocalError:
                pass
            self._type_value[i] = float

        if get_attr:
            return attrs_dic
        return
    
    def merge(self, item):
        if isinstance(item, SeriesSet):
            for i, title in enumerate(item.titles):
                sequence = item[title]
                if title not in self._columns:
                    To = [self._miss_value for j in range(self._dim.Ln)]
                    self._m_value.append(self._dim.Ln)
                    self._columns.append(title)
                    self._type_value.append(item._type_value[i])
                    self._type_column.append(item._type_column[i])
                else:
                    To = self._set[title]
                    if self._type_value[self._columns.index(title)] != item._type_value[i]:
                        raise TypeError('<%s> does not have same type'%title +\
                                        'with this SeriesSet.')

                if self._miss_value != item._miss_value:
                    for value in sequence:
                        if valule == item.miss_value:
                            To.append(self._miss_value)
                        else:
                            To.append(value)
                else:
                    To.extend(sequence)
                self._set[title] = To
                _miss = self._m_value[self._columns.index(title)]
                self._m_value[self._columns.index(title)] = _miss + item._m_value[i]
            self._dim = SeriesSet.dims(self._dim.Ln + item._dim.Ln,
                                       len(self._columns))

            for i, sequence in enumerate(self._set.values()):
                if len(sequence) != self._dim.Ln:
                    add_miss_size = self._dim.Ln - len(sequence)
                    sequence.extend([self._miss_value]*add_miss_size)
                    self._m_value[i] += add_miss_size
                    
        elif isinstance(item, (Table, Matrix, Frame)):
            new_titles = list()
            for i, title in enumerate(item.titles):
                if self._type_value[self._columns.index(title)] != \
                   item._type_column[i]:
                    raise TypeError('new data in column <%s>'%title +\
                                    'does not have the same type '+\
                                    'with this SeriesSet')
                if title not in self._columns:
                    new_titles.append(title)
                    
            for title in new_titles:
                self._set[title] = [self._miss_value]*self._dim.Ln
            self._columns.extend(new_titles)

            for title in self._set.keys():
                new_sequence = item[title]
                self._set[title].extend(new_sequence)

        elif all(filter(is_iter, item)):
            new_item = SeriesSet(item)
            self.merge(new_item)

        else:
            raise TypeError('could not merge a single value only.')
        
    def pick(self, conditions):
        '''
        TODO: pick out records that comply with conditions
        '''
        conditions = str(conditions)
        for i, title in enumerate(self._columns):
            new = 'record[%d]'%i
            conditions = conditions.replace(title, new)
        conditions = 'check = ' + conditions
        pick_record_line = list()
        
        for i in range(self._dim.Ln):
            record = [self._set[title][i] for title in self._columns]
            exec(conditions)
            if check:
                pick_record_line.append(i)

        return_dict = OrderedDict()
        for title in self._columns:
            sequence = self._set[title]
            return_dict[title] = [sequence[i] for i in pick_record_line]

        return SeriesSet(return_dict, self._columns)
        
    def pop(self, pos=-1):  # OK #
        '''
        TODO: pop(remove & return) a record from the Frame
        '''
        if not isinstance(pos, int):
            raise TypeError('an integer is required.')
        
        pop_item = [self._set[title].pop(pos) for title in self._columns]
        
        self._dim = SeriesSet.dims(self._dim.Ln - 1, self._dim.Col)
        for i, each in enumerate(pop_item):
            if self._miss_value == each:
                self._m_value[i] -= 1
        return pop_item

    def pop_col(self, *titles):  # OK #
        pop_name = list()
        if not titles:
            titles = [self._columns[-1]]
        for item in titles:
            if isinstance(item, str):
                if item in self._columns:
                    pop_name.append(item)
                else:
                    raise ValueError("'%s' is not in columns title"%item)
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
            self._type_value.pop(pos)
            self._type_column.pop(pos)
        self._dim = SeriesSet.dims(self._dim.Ln, self._dim.Col-len(pop_name))
        return SeriesSet(pop_data, pop_name)

    def reverse(self, axis='COL'): # OK #
        if axis == 'COL':
            self._columns.reverse()
        else:
            for sequence in self._set.values():
                sequence.reverse()

    def rename_col(self, *new_names): # OK #
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
                  title_line=0, sep=',', type_float=False, type_str=False):
        
        def transform(m, i, item):
            try:
                if item == miss_symbol:
                    self._m_value[i] += 1
                    return self._miss_value
                return _col_types[i](item)
            except ValueError:
                self._m_value[i] += 1
                warn('<%s> in line %d and column %d '%(item, m, i)+
                     'has a different type.')
                return self._miss_value

        with open(addr, 'r') as f:
            
            reader = csv.reader(f, delimiter=sep)
            col = 0
            for l_n, line in enumerate(reader):
                if len(line) > col:
                    col = len(line)
            col = range(col)
            datas = [[0]*l_n for c in col]
            self._m_value = [0 for c in col]

            f.seek(0)
            for i, lines in enumerate(reader):
                if not self._columns and i == title_line:
                    self._columns = list()
                    lenth_title_line = len(lines)
                    for k in col:
                        if k < lenth_title_line:
                            new_title = lines[k]
                        else:
                            new_title = 'Col_%d'%k

                        if new_title not in self._columns:
                            self._columns.append(new_title)
                            continue

                        for s in range(k, k + 1000):
                            new_title = 'Col_%d'%s
                            if new_title not in self._columns:
                                self._columns.append(new_title)
                                break
                        else:
                            raise ValueError('repeated title name <%s> '%lines[k] +\
                                         'in column %d'%k)
                        

                if i >= first_line:
                    _col_types = dict()
                    self._type_value = list()
                    for j, item in enumerate(lines):
                        if j in _col_types or item == miss_symbol or item == self._miss_value:
                            continue
                        if type_str:
                            _col_types[j] = str
                            self._type_value.append(str)
                        elif item.isdigit() or item[1:].isdigit():
                            if type_float:
                                _col_types[j] = string.atof
                                self._type_value.append(float)
                            else:
                                _col_types[j] = string.atoi
                                self._type_value.append(int)
                        elif '.' in item:
                            _col_types[j] = string.atof
                            self._type_value.append(float)
                        else:
                            _col_types[j] = str
                            self._type_value.append(str)
                            
                    if len(col) == len(_col_types):
                        break
                    
            f.seek(0)
            for m, record in enumerate(reader):
                if m >= first_line - 1 :
                    break
                
            for m, record in enumerate(reader):
                for i, v in enumerate(record):
                    datas[i][m] = transform(m, i, v)

        l_col, l_title = len(col), len(self._columns)
        if not self._columns:
            self._columns = ['col_'+str(c) for c in col]
        elif l_title < l_col:
            self._columns.extend(['col_'+str(c) for c in col[l_col:l_title]])
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
                elif isinstance(c, int) and c < self._dim.Col:
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
        '''
        TODO: sort the dataframe
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
            raise TypeError("'%s' is not a recognized symbol."%compare_symbol)

        new_index = get_sorted_index(self._set[compare_title], reverse=reverse)
        for k, title in enumerate(self._columns):
            sequence = self._set[title]
            new_sequence = [sequence[i] for i in new_index]
            if isinstance(sequence, array):
                new_sequence = array(SeriesSet.transform[\
                    self._type_value[k]], new_sequence)
            self._set[title] = new_sequence    

    def shuffles(self):
        new_index = range(self._dim.Ln)
        shuffle(new_index)
        for i, title in enumerate(self._columns):
            sequence = self._set[title]
            if isinstance(self._type_column[i], array):
                new_sequence = array(SeriesSet.transform[self._type_value[i]])
            else:
                new_sequence = list()
            for j in new_index:
                new_sequence.append(sequence[j])
            self._set[title] = new_sequence


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

    _type_value : types in list
        the value type of each column.
    '''
    
    dims = namedtuple('Frame', ['Ln', 'Col'])

    def __init__(self, frame=None, columns=None, element_type_=None,
                 miss_symbol=None, miss_value=None):

        self._m_value = 0
        self._miss_value = miss_value
        
        if frame == None and columns == None:
            self._Frame = list()
            self._columns = list()
            self._dim =  Frame.dims(0, 0)
            self._type_value = list()
            
        elif isinstance(frame, Frame):
            self._Frame = deepcopy(frame._Frame)
            self._columns = deepcopy(frame._columns)
            self._dim = deepcopy(frame._dim)
            self._m_value = deepcopy(frame._m_value)
            self._type_value = deepcopy(frame.value)

        # Check the type of frame is legal
        elif is_seq(frame):
            if is_iter(frame[0]):
                self._Frame = [tuple(record) for record in frame]
                dim_Col, dim_Ln = len(frame[0]), len(frame)
            else:
                self._Frame = [(value, ) for value in frame]
                dim_Col, dim_Ln = 1, len(frame) 
            
            # Set the type in each column
            if not element_type_:
                self._type_value = [0 for item in self._Frame[0]]
                for item in self._Frame:
                    for i, value in enumerate(item):
                        if self._type_value[i] != 0:
                            continue
                        if value != miss_value and value != miss_symbol:
                            self._type_value[i] = type(value)               
            elif is_seq(element_type_):
                self._type_value = element_type_
            else:
                raise TypeError('element_type should be a sequence.')

            # Check the element types and lenth
            for i, item in enumerate(self._Frame):
                if len(item) != dim_Col:
                    raise ValueError("The No.{0} record doesn't have".format(i)+\
                                     " the enought variables.")
                for j, each in enumerate(item):
                    if not isinstance(each, self._type_value[j]):
                        if each == miss_value:
                            self._m_value += 1
                        elif each == miss_symbol:
                            self._m_value += 1
                            new_line = list(item)
                            new_line[i] = self._miss_value
                            self._Frame[i] = tuple(new_line)
                        else:
                            raise ValueError("The element in (Ln%d,Col%d)"%(i, j)+\
                                             " doesn't have the correct type.")
            # Check the size and type of frame is legal
            dim_Ln, dim_Col = len(self._Frame), len(self._Frame[0])
            
            if columns:
                self._columns = [str(i) for i in columns]
            else:
                self._columns = ['Col_%d'%i for i in range(dim_Col)]
            
            if len(self._columns) != dim_Col:
                raise ValueError("Data doesn't have the same size of columns'"+\
                                 "list.")
                        
            self._dim = Frame.dims(dim_Ln, dim_Col)
            
        elif isinstance(frame, SeriesSet):
            self._Frame = list()
            for line in frame:
                self._Frame.append(tuple(line))
            self._type_value = deepcopy(frame._type_value)
            self._m_value = sum(frame._m_value)
            self._columns = deepcopy(frame._columns)
            self._dim = Frame.dims(frame._dim.Ln, frame._dim.Col)
            
        elif isinstance(frame, Matrix):
            self._Frame = [tuple(line) for line in frame]
            self._columns = deepcopy(frame._columns)
            self._dim = Frame.dims(frame._dim.Ln, frame._dim.Col)
            self._type_value = [float for i in xrange(self._dim.Col)]

        elif isinstance(frame, Table):
            self._Frame = list()
            self._dim = Frame.dims(frame._dim.Ln, frame._dim.Col)
            self._columns = deepcopy(frame._columns)
            self._type_value = [type(v) for v in frame[0]]
            for i, record in enumerate(frame):
                line = list()
                for j, v in enumerate(record):
                    if v is None:
                        self._m_value += 1
                        line.append(self._miss_value)
                    elif isinstance(v, self._type_value[j]):
                        line.append(v)
                    else:
                        raise TypeError('value in position <%d, %d>'%(i, j)+\
                                        "doesn't have correct type")
                line.extend([None for i in xrange(self._dim.Col-len(line))])
                self._Frame.append(tuple(line))
            
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
    def titles(self):
        return self._columns
    
    @titles.setter
    def titles(self, item):
        if isinstance(item, str):
            self._columns = [item]*self._dim.Col
        elif is_iter(item):
            self._columns = [str(i) for i in item]
        else:
            raise TypeError('unknow title as %s'%item)
        
    @property
    def info(self):
        new_type_v = [str(t).split("'")[1] for t in self._type_value]

        max_n = len(max(self._columns, key=len))
        
        info = ''
        for i in range(self._dim.Col):
            info += ' '*15
            info += str(self._columns[i]).center(len(self._columns[i]) - max_n) + '| '
            info += ' ' + new_type_v[i] + '\n'
                
        print '1.  Structure: DaPy.Frame\n' +\
              '2. Dimensions: Ln=%d | Col=%d\n'%self._dim +\
              '3. Miss Value: %d elements\n'%self._m_value +\
              '4.    Columns: Title'.center(max_n-5) + '|'+\
                             ' <Column Type>\n'+\
                             info
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

        frame = ' '.join([title.center(column_size[i]) for i, title in \
                         enumerate(self._columns)]) + '\n'

        if self._dim.Ln > 20:
            for item in temporary_Frame[:10]:
                line = ''
                for i, value in enumerate(item):
                    line += str(value).center(column_size[i]) + ' '
                frame += line + '\n'
            frame += ('Omit %d Ln'%(self._dim.Ln - 20)).center(len(line)) + '\n'
            for item in temporary_Frame[-10:]:
                for i, value in enumerate(item):
                    frame += str(value).center(column_size[i]) + ' '
                frame += '\n'
                
        else:
            for item in temporary_Frame:
                for i, value in enumerate(item):
                    frame += str(value).center(column_size[i]) + ' '
                frame += '\n'

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
        if isinstance(other, (Frame, SeriesSet, Table)):
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

            if i > j:
                i, j = j, i

            new_frame = list()
            for record in self._Frame:
                new_frame.append(tuple(record[i:j + 1]))

            return Frame(new_frame, self._columns[i:j + 1], self._type_value[i:j + 1],
                         None, self._miss_value)

        if type(i) != int and type(j) != int:
            raise ValueError('unrecognized symbol as [%s:%s]'%(i, j))

        if i is None:
            i = 0
        elif i < 0 :
            i = 0
        elif i >= self._dim.Ln:
            i = self._dim.Ln

        if j is None:
            j = 0
        elif j < 0:
            j = self._dim.Ln + j
        elif j >= self._dim.Ln:
            j = self._dim.Ln

        if i > j :
            i, j = j, i

        return Frame(self._Frame[i:j], self._columns, self._type_value,
                     None, self._miss_value)
            
    def __getitem__(self, pos):
        try:
            return self._Frame[pos]
        except TypeError:
            if isinstance(pos, slice):
                return self.__getslice__(pos.start, pos.stop)
            
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
                self.insert_col(pos, value)
            else:
                self.append_col(value)

    def __delitem__(self, key):
        if isinstance(key, int):
            drop_record = self.pop(key)

        elif isinstance(key, str) and key in self._columns:
            drop_record = self.pop_col(key)

        else:
            raise KeyError('unrecognized symbol %s'%key)
    
    def __iter__(self):
        for i in xrange(self._dim.Ln):
            yield self._Frame[i]

    def __reversed__(self):
        self.reverse()

    def append(self, item, miss_symbol=None):
        # TODO append a record to the frame last
        if not is_seq(item):
            raise TypeError("insert item is not a support type "+\
                            "with <'%s'>"%type(item))

        if not miss_symbol:
            miss_symbol = self._miss_value
            
        for i, element in enumerate(item):
            if isinstance(element, self._type_value[i]) is False and\
               element != miss_symbol:
                raise TypeError('The element in position %d is'%i+
                                ' not a correct type.')
            if element == miss_symbol:
                self._m_value += 1
                    
        self._Frame.append(tuple(item))
        self._dim = Frame.dims(self._dim.Ln+1, self._dim.Col)

        
    def append_col(self, variable_name, series, element_type='AUTO',
                   miss_symbol=None):
        # TODO: append a series data to the frame last
        if is_value(series):
            series = [series for i in range(self._dim.Ln)]
            
        if not isinstance(variable_name, str):
            raise TypeError("variable name should be a ``str``")

        if not miss_symbol:
            miss_symbol = self._miss_value

        size = len(series)
        if size < self._dim.Ln:
            series.extend([self._miss_value for i in xrange(self._dim.Ln-size)])
        elif size > self._dim.Ln:
            raise IndexError('Input series has larger dimension than this Frame.')
        
        if str(element_type).upper() == 'AUTO':
            for each in series:
                if each != miss_symbol:
                    element_type = type(each)

        for i, element in enumerate(series):
            if element == miss_symbol:
                self._m_value += 1
                
            elif not isinstance(element, element_type):
                raise ValueError("The element in postion %d doesn't"%i+\
                                     "have the correct type.")

        if variable_name in self._columns:
            raise ValueError("Variable name <'%s'> is already taken."%self.variable_name)

        self._columns.append(variable_name)
        self._dim = Frame.dims(self._dim.Ln, self._dim.Col+1)
        new_frame = list()
        
        for i, record in enumerate(self._Frame):
            line = list(record)
            line.append(series[i])
            new_frame.append(tuple(line))
        self._Frame = new_frame
        del new_frame

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

    def drop_miss_value(self, drop='LINE'):
        if drop.upper() == 'LINE':
            new_frame = list()
            Ln = self._dim.Ln
            for record in self._Frame:
                if self._miss_value in record:
                    Ln -= 1
                else:
                    new_frame.append(record)
                    
            self._dim = Frame.dims(Ln, self._dim.Col)
            self._Frame = new_frame
            del new_frame
        elif drop.upper() == 'COL':
            save_col = list()
            for i in xrange(self._dim.Col):
                sequence = [record[i] for record in self._Frame]
                if not self._miss_value in sequence:
                    save_col.append(i)

            self._columns = [self._columns[i] for i in save_col]
            if len(self._columns) == 0:
                self._dim = Frame.dims(0, 0)
                self._Frame = list()
                return
            
            new_frame = list()
            for record in self._Frame:
                new_frame.append(tuple([record[i] for i in save_col]))
            self._Frame = new_frame
            self._dim = Frame.dims(self._dim.Ln, len(self._columns))
            del new_frame

    def insert(self, pos, item, miss_symbol=None):
        # TODO: insert a record to the frame, position in <index>
        if not (isinstance(pos, int) or is_iter(item_iter)):
            raise TypeError("insert item is not a support"+\
                            "type with <'%s'>"%type(item))

        if len(item) != self._dim.Col:
            raise IndexError("Input Record doesn't have correct dimension")

        if not miss_symbol:
            miss_symbol = self._miss_value
        
        for i,element in enumerate(item):
            if element == miss_value:
                self._m_value += 1
                
            elif not isinstance(element, self._type_value[i]):
                raise TypeError('The element in position %d is '%i+\
                                'not a correct type.')

        self._Frame.insert(pos, tuple(item))
        self._dim = Frame.dims(self._dim.Ln+1, self._dim.Col)

    def insert_col(self, variable_name, series, index, element_type='AUTO',
                   miss_symbol=None):
        # TODO: insert a series of data to the frame, position in <index>
        if is_value(series):
            series = [series for i in range(self._dim.Ln)]
            
        if not (isinstance(variable_name, str) or isinstance(index, int)):
            raise ValueError("Unsupport type to append in this frame")

        if not miss_symbol:
            miss_symbol = self._miss_value

        size = len(series)
        if size < self._dim.Ln:
            series.extend([self._miss_value for i in xrange(self._dim.Ln-size)])
        elif size > self._dim.Ln:
            raise IndexError('Input series has larger dimension than this Frame.')

        if str(element_type).upper() == 'AUTO':
            for each in series:
                if each != miss_symbol:
                    element_type = type(each)
                    break

        for i, element in enumerate(series):
            if element == miss_symbol:
                self._m_value += 1
                
            elif not isinstance(element, element_type):
                    raise ValueError("The element in postion %d "%i+\
                                     "doesn't have the correct type.")

        if variable_name in self._columns:
            raise ValueError("Variable name <'%s'> is "%variable_name+\
                             "already taken.")
               
        self._dim = Frame.dims(self._dim.Ln, self._dim.Col+1)
        self._type_value.insert(index, element_type)
        self._columns.insert(index, variable_name)

        if index < 0:
            index += self._dim.Col
            
        new_frame = list()
        for i, record in enumerate(self._Frame):
            line = list(record)
            line.insert(index, series[i])
            new_frame.append(tuple(line))

        self._Frame = new_frame
        del new_frame

    def normalized(self, col, attr, get_attr):
        raise TypeError("unsupport normalized data with <DaPy.Frame> structure.")

    def merge(self, other):
        if isinstance(other, Frame):
            new_title
            for title in other.titles:
                if title:
                    pass

    def pick(self, conditions):
        '''
        TODO: pick out records that comply with conditions
        '''
        conditions = str(conditions)
        for i, title in enumerate(self._columns):
            new = 'record[%d]'%i
            conditions = conditions.replace(title, new)
        conditions = 'check = ' + conditions

        return_frame = list()
        for record in self._Frame:
            exec(conditions)
            if check:
                return_frame.append(record)

        return Frame(return_frame, self._columns)

    def pop(self, item=-1):
        # TODO: pop(remove & return) a record from the Frame 
        if isinstance(item, int):
            pop_item = self._Frame.pop(item)
            
            self._dim = Frame.dims(self._dim.Ln-1, self._dim.Col)
            if self._miss_value in pop_item:
                self._m_value -= 1
                
            return pop_item
        
        raise TypeError('an integer is required.')

    def pop_col(self, *titles):
        # TODO: pop(remove & return) a series from the Frame
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
            elif isinstance(item, int) and item < self._dim.Col:
                pos.append(item)
                pop_name.append(self._columns[item])
            else:
                raise ValueError('%s is not a column name.'%str(item))

        for name in pop_name:
            self._columns.pop(self._columns.index(name))
        pop_data = [list() for i in range(len(pop_name))]

        new_frame = list()
        # copy a new Frame
        for record in self._Frame:
            line = list()
            for i, value in enumerate(record):
                if i in pos:
                    pop_data[pos.index(i)].append(value)
                else:
                    line.append(value)
            new_frame.append(tuple(line))

        for columns in pop_data:
            self._m_value -= columns.count(self._miss_value)
        self._dim = Frame.dims(self._dim.Ln, self._dim.Col-len(pos))
        self._Frame = new_frame
        return SeriesSet(dict(zip(pop_name, pop_data)))

    def read_text(self, addr, first_line=1, miss_symbol='NA',
                 title_line=0, sep=',', type_float=False, type_str=False):

        self._m_value = 0
        
        with open(addr, 'r') as f:
            
            reader = csv.reader(f, delimiter=sep)
            col = 0
            for l_n, record in enumerate(reader):
                if len(record) > col:
                    col = len(record)

            col = range(col)
            f.seek(0)
                
            for i, lines in enumerate(reader):
                if not self._columns and i == title_line:
                    self._columns = list()
                    lenth_title_line = len(lines)
                    for k in col:
                        if k < lenth_title_line:
                            self._columns.append(lines[k])
                        else:
                            self._columns.append('Col_%d'%k)
                    
                if (i >= first_line) and\
                   (miss_symbol not in lines):
                    
                    _col_types = dict()
                    self._type_value = list()
                    for j, item in enumerate(lines):
                        if type_str:
                            _col_types[j] = str
                            self._type_value.append(str)
                        elif item.isdigit() or item[1:].isdigit():
                            if type_float:
                                _col_types[j] = string.atof
                                self._type_value.append(float)
                            else:
                                _col_types[j] = string.atoi
                                self._type_value.append(int)
                        elif '.' in item:
                            _col_types[j] = string.atof
                            self._type_value.append(float)
                        else:
                            _col_types[j] = str
                            self._type_value.append(str)
                            
                    if len(col) == len(_col_types):
                        dic_ready = True
                        break
                    
        if not self._Frame:
            self._Frame = [0]*l_n
  
        if not self._columns:
            self._columns = ['col_'+str(i) for i in col]
            
        def transform(i, item):
            try:
                return _col_types[i](item)
            except ValueError:
                self._m_value += 1
                warn('<%s> in line %d and column %d '%(item, m, i)+
                     'has a wrong type and '+\
                     'represented by miss value.')
                return self._miss_value

        # Reload data into data structure and transfrom the type
        with open(addr) as f:
            reader = csv.reader(f, delimiter=sep)
            for m, record in enumerate(reader):
                if m >= first_line - 1 :
                    break
            for m, record in enumerate(reader):
                self._Frame[m] = tuple([transform(i, v) for i, v in enumerate(record)])
                
        self._dim = Frame.dims(m+2-first_line, len(col))

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
        return self

class Matrix(object):

    dims = namedtuple('Matrix', ['Ln', 'Col'])

    def __init__(self, table=None, columns=None, check=True):

        self._miss_value = None

        if is_seq(table):
            self.__init_unknow_type(table, check, columns)
                 
        elif isinstance(table, Matrix):
            self._matrix = deepcopy(table)
            self._columns = deepcopy(table._columns)
            self._dim = deepcopy(table._dim)

        elif isinstance(table, SeriesSet):
            self._matrix = list()
            self._columns = deepcopy(table._columns)
            self._dim = deepcopy(table._dim)
            for line in table:
                self._matrix.append([float(v) for v in line])

        elif isinstance(table, Frame):
            self._matrix = list()
            for line in table:
                self._matrix.append([float(value) for value in line])
            self._columns = deepcopy(table._columns)
            self._dim = deepcopy(table._dim)
            
        elif table is None:
            self._matrix = list()
            if columns:
                self._columns = columns
            else:
                self._columns = list()
            self._dim =  Table.dims(0, 0)
            
        else:
            raise TypeError('can not transform this object to DaPy.Matrix'+\
                            ', which expects a list or deque contained'+\
                            ' tuple or name tuple.')

    @property
    def data(self):
        return self._matrix
    
    @property
    def titles(self):
        return self._columns

    @property
    def shape(self):
        return self._dim

    @property
    def info(self):
        print '1.   Structure: DaPy.Matrix\n' +\
              '2.  Dimensions: Ln=%d | Col=%d'%self._dim

    @property
    def T(self):
        new_ = list()
        columns = list()
        for j in range(self._dim.Col):
            new_.append([record[j] for record in self._matrix])
            columns.append('Col_'+str(j))
        return Matrix(new_, columns, False)
    
    def __repr__(self):
        temporary_series = [list()] * self._dim.Col
        if self._dim.Ln > 20:
            temporary_Frame = self._matrix[:10]
            temporary_Frame.extend(self._matrix[-10:])
        else:
            temporary_Frame = self._matrix
            
        for line in temporary_Frame:
            for i, value in enumerate(line):
                print i, value, line
                temporary_series[i].append(str(value))
        column_size = [len(max(col, key=len)) for col in temporary_series]

        frame = 'matrix(['

        if self._dim.Ln > 20:
            for i, item in enumerate(temporary_Frame[:10]):
                if i != 0:
                    line = '       ['
                for i, value in enumerate(item):
                    line += str(value).center(column_size[i]) + ' '
                frame += line + ']\n'
            frame += ('Omit %d Ln'%(self._dim.Ln - 20)).center(len(line)) + '\n'
            for item in temporary_Frame[-10:]:
                frame += '       ['
                for i, value in enumerate(item):
                    frame += str(value).center(column_size[i]) + ' '
                frame += ']\n'
                
        else:
            for i, item in enumerate(temporary_Frame):
                if i != 0:
                    frame += '       ['
                for i, value in enumerate(item):
                    frame += str(value).center(column_size[i]) + ' '
                frame += ']\n'
        return frame[:-1] +  ')'
    
    def __contains__(self, e):
        if isinstance(e, str):
            return e in self._columns

        if isinstance(e, list):
            for record in self._matrix:
                if record == e:
                    return True
        return False

    def __len__(self):
        return self._dim.Ln

    def __eq__(self, other):
        if isinstance(other, (Frame, SeriesSet, Table, Matrix)):
            if self._dim == other._dim:
                return True
            return False

    def __getitem__(self, pos):
        try:
            return self._matrix[pos]
        except:
            col = self._columns.index(pos)
            return [item[col] for item in self._matrix]
    
    def __iter__(self):
        for i in xrange(self._dim.Ln):
            yield self._matrix[i]

    def __sum__(self, other):
        return sum([sum(record) for record in self._matrix])
    
    def __abs__(self):
        new_ = list()
        for line in self._matrix:
            new_.append([abs(value) for value in line])
        return Matrix(new_, self._columns, False)

    def __add__(self, other):
        new_ = list()
        if is_math(other):
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    line_.append(self._matrix[i][j] + other)
                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)):
            if not self._dim == other._dim:
                raise ValueError('operands could not be broadcast'+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self._dim+\
                                 '(%d,%d)'%other._dim)
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    try:
                        line_.append(self._matrix[i][j] + other[i][j])
                    except IndexError:
                        pass
                new_.append(line_)
        else:
            raise TypeError("'+' operation expects the type of"+\
                            "Matrix, Table, Frame, int or float")

        return Matrix(new_, self._columns)

    def __radd__(self, other):
        return self.__add__(other)
        
    def __sub__(self, other):
        new_ = list()
        if is_math(other):
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    line_.append(self._matrix[i][j] - other)
                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)):
            if not self._dim == other._dim:
                raise ValueError('operands could not be broadcast'+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self._dim+\
                                 '(%d,%d)'%other._dim)
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    try:
                        line_.append(self._matrix[i][j] - other[i][j])
                    except IndexError:
                        pass
                new_.append(line_)
        else:
            raise TypeError("'-' operation expects the type of"+\
                            "Matrix, Table, Frame, int or float")
        return Matrix(new_, self._columns)

    def __rsub__(self, other):
        new_ = list()
        if is_math(other):
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    line_.append(other - self._matrix[i][j])
                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)):
            if not self._dim == other._dim:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self._dim+\
                                 '(%d,%d)'%other._dim)
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    try:
                        line_.append(other[i][j] - self._matrix[i][j])
                    except IndexError:
                        pass
                new_.append(line_)
        else:
            raise TypeError("'-' operation expects the type of"+\
                            "Matrix, Table, Frame, int or float")
        return Matrix(new_, self._columns)
                
    def __mul__(self, other):
        new_ = list()
        if is_math(other):
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    try:
                        line_.append(self._matrix[i][j] * other)
                    except:
                        pass
                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)):
            if not self._dim == other._dim:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d,%d) and '%self._dim+\
                                 '(%d,%d)'%other._dim)
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    line_.append(self._matrix[i][j] * other[i][j])
                new_.append(line_)

        else:
            raise TypeError("'*' operation expects the type of "+\
                            "Matrix, Table, Frame, int or float")
        
        return Matrix(new_, self._columns)

    def __rmul__(self, other):
        return self.__mul__(other)
                
    def __div__(self, other):
        new_ = list()
        if is_math(other):
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    line_.append(self._matrix[i][j] / other)

                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)) and \
                        self._dim == other._dim:
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    line_.append(self._matrix[i][j] / other[i][j])
                new_.append(line_)
        else:
            raise TypeError("'/' operation expects the type of"+\
                            "Matrix, Table, Frame, int or float")
        return Matrix(new_, self._columns)

    def __rdiv__(self, other):
        new_ = list()
        if is_math(other):
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    line_.append(other / self._matrix[i][j])

                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)) and \
                        self._dim == other._dim:
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    line_.append(other[i][j] / self._matrix[i][j])
                new_.append(line_)
        else:
            raise TypeError("'/' operation expects the type of"+\
                            "Matrix, Table, Frame, int or float")
        return Matrix(new_, self._columns)

    def __pow__(self, other):
        new_ = list()
        if is_math(other):
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    line_.append(self._matrix[i][j]**other)

                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)) and \
                        self._dim == other._dim:
            for i in range(self._dim.Ln):
                line_ = list()
                for j in range(self._dim.Col):
                    line_.append(self._matrix[i][j]**other[i][j])
                new_.append(line_)
        else:
            try:
                new_ = list()
                for line in self._matrix:
                    new_.append([value**other for value in line])
            except:
                raise TypeError("Matrix does not unsupport '**' operation"+\
                                "with this object.")
        
        return Matrix(new_, self._columns)

    def dot(self, other):
        if isinstance(other, (Matrix, Frame, Table)):
            if self._dim.Col != other._dim.Ln:
                raise ValueError('shapes (%d, %d)'%self._dim +\
                                 ' and (%d, %d) not aligned.'%other._dim)
            col_size_1 = self._dim.Col
            col_size_2 = other._dim.Col
            columns = other._columns
        elif isinstance(other, (list, tuple, deque)):
            if len(self._matrix[0]) != len(other):
                raise ValueError('shapes (%d, %d)'%(self._dim)+\
                                 ' and (%d, %d) not aligned.'%(len(other[0]),
                                                               len(other)))
            col_size_1 = len(self._matrix[0])
            col_size_2 = len(other[0])
        else:
            raise TypeError('unsupported operation dot, with type'+\
                            '< %s > and '%str(type(self._matrix)) +\
                            '< %s > and '%str(type(other)))
        new_ = [[0]*col_size_2 for k in range(self._dim.Ln)]
        for i in range(self._dim.Ln):
            for pos in range(col_size_2):
                sumup = 0
                for j in range(col_size_1):
                    sumup += self._matrix[i][j]*other[j][pos]
                new_[i][pos] = sumup
        return Matrix(new_, self._columns, False)

    def __init_unknow_type(self, table, check, columns):
        try:
            dim_col = len(table[0])
            if check:
                self._matrix = list()
                for i, record in enumerate(table):
                    new_record = list()
                    if len(record) != dim_col:
                        raise IndexError("the No.%d record doesn't "%i+\
                                         "have enough dimensions.")
                    for j, value in enumerate(record):
                        if not is_math(value):
                            raise ValueError('the value in the No.%d'%i+\
                                            " record and the No.%d "%j +\
                                            "position doesn't have "+\
                                             "correct type")
                        new_record.append(value)
                    self._matrix.append(new_record)
            else:
                self._matrix = table
                
        except TypeError:
            dim_col = 1
            self._matrix = [[float(v)] for v in table]
            
        self._dim = Matrix.dims(len(table), dim_col)

        if columns is None:
            self._columns = ['Col_'+str(i) for i in range(dim_col)]
        elif columns and check:
            self._columns = None
        else:
            self._columns = [str(i) for i in columns[:dim_col]]
            if len(self._columns) < dim_col:
                for i in range(len(self._columns)-dim_col):
                    self._columns.append('Col_'+str(i))

    def normalized(self, avg=None, std=None):
        if not (avg and std):
            avg = mean(self)
            std = (_sum((self - avg)**2/(self._dim.Col*self._dim.Ln)))**0.5
            return (self - avg)/std, (avg, std)
        return (self - avg)/std

    def make(self, Ln, Col, element=0):
        if not (isinstance(Ln, int) and isinstance(Col, int) and\
                isinstance(element, (int, float))):
            raise TypeError("arguments 'Ln' and 'Col' expect <int> type,"+\
                            "'element' expects <int> or <float> types.")
        self._matrix = list()
        element = float(element)
        for i in xrange(Ln):
            self._matrix.append([element for j in xrange(Col)])
        self._colmuns = ['Col_'+str(i) for i in xrange(Col)]
        self._dim = Matrix.dims(Ln, Col)

    def make_random(self, Ln, Col, type_int=False):
        if not (isinstance(Ln, int) and isinstance(Col, int)):
            raise TypeError("arguments 'Ln' and 'Col' expect <int> type,")
        if not isinstance(type_int, (bool, tuple)):
            raise TypeError("argutments 'type_int' expects 'False' symbol"+\
                            " or a 2-D tuple.")

        self._matrix = list()
        if type_int:
            for i in xrange(Ln):
                self._matrix.append([randint(*type_int) for j in xrange(Col)])
        else:
            for i in xrange(Ln):
                self._matrix.append([random() for j in xrange(Col)])
                
        self._colmuns = ['Col_'+str(i) for i in xrange(Col)]
        self._dim = Matrix.dims(Ln, Col)
        
    
    def read_text(self, addr, first_line=1, title_line=0, sep=','):
        # Set data columns, and transform diction
        self._m_value = 0
        col = range(len(lines))
        
        with open(addr, 'r') as f:
            reader = csv.reader(f, delimiter=sep)
            col = range(max(reader, key=len))
            f.seek(0)
            
            for i, lines in enumerate(reader):                    
                if i == title_line:
                    self._columns = [lines[k] for k in col]

                if col and self._columns:
                    break
           
        if not self._columns:
            self._columns = ['Col_'+str(i) for i in col]

        # Reload data into data structure and transfrom the type
        with open(addr) as f:
            reader = csv.reader(f, delimiter=sep)
            for m, record in enumerate(reader):
                if m >= first_line -1 :
                    break
            for m, record in enumerate(reader):
                self._matrix.append([float(record[i]) for i in col])
        
        self._dim = Matrix.dims(m+2-first_line, len(col))

    def replace(self, key, target):
        raise TypeError("matrix does not support operation 'replace'")
                    
    def shuffles(self):
        shuffle(self._matrix)


class Table(object):
    
    dims = namedtuple('Table', ['Ln', 'Col'])

    def __init__(self, table=None, columns=None, miss_value=None):
        self._miss_value = miss_value
        
        if table == None and columns == None:
            self._Table = list()
            self._columns = list()
            self._dim =  Table.dims(0, 0)

        elif isinstance(table, Table):
            self._Table = deepcopy(table)
            self._columns = deepcopy(table._columns)
            self._dim = deepcopy(table._dim)

        elif isinstance(table, (list, deque)):
            dim_col = max([len(record) for record in table])
            if columns is None:
                self._columns = ['Col_'+str(i) for i in xrange(dim_col)]
            else:
                self._columns = [str(i) for i in columns]
                if len(self._columns) < dim_col:
                    for i in range(len(self._columns)-dim_col):
                        self._columns.append('Col_'+str(i))

            self._dim = Table.dims(len(table), dim_col)
            self._Table = table

        elif isinstance(table, (Matrix, SeriesSet, Frame)):
            self._Table = [line for line in table]
            self._columns = deepcopy(table._columns)
            self._dim = deepcopy(table._dim)
                
        else:
            raise TypeError('can not transform this object to DaPy.Frame, '+\
                            'which expects a list or deque contained'+\
                            ' tuples or name-tuples.')

    @property
    def data(self):
        return self._Table
    
    @property
    def shape(self):
        return self._dim

    @property
    def titles(self):
        return self._columns

    @property
    def info(self):
        pprint( { '1.  Structure': ' DaPy.Table',
                 '2. Dimensions': ' Ln=%d | Max Col=%d'%self._dim,
                 '3.    Columns': '|'.join(self._columns)})

    def __contains__(self, e):
        if isinstance(e, str):
            return e in self._columns

        if isinstance(e, list):
            for record in self._Table:
                if record == e:
                    return True
        return False
    
    def __repr__(self):
        temporary_series = [[title, ] for title in self._columns]
        if self._dim.Ln > 20:
            temporary_Frame = self._Table[:10]
            temporary_Frame.extend(self._Table[-10:])
        else:
            temporary_Frame = self._Table
            
        for line in temporary_Frame:
            for i, value in enumerate(line):
                temporary_series[i].append(str(value))
        column_size = [len(max(col, key=len)) for col in temporary_series]

        frame = ' '.join([title.center(column_size[i]) for i, title in \
                         enumerate(self._columns)]) + '\n'

        if self._dim.Ln > 20:
            for item in temporary_Frame[:10]:
                line = ''
                for i, value in enumerate(item):
                    line += str(value).center(column_size[i]) + ' '
                frame += line + '\n'
            frame += ('Omit %d Ln'%(self._dim.Ln - 20)).center(len(line)) + '\n'
            for item in temporary_Frame[-10:]:
                for i, value in enumerate(item):
                    frame += str(value).center(column_size[i]) + ' '
                frame += '\n'
                
        else:
            for item in temporary_Frame:
                for i, value in enumerate(item):
                    frame += str(value).center(column_size[i]) + ' '
                frame += '\n'

        return frame[:-1]

    def __len__(self):
        return self._dim.Ln

    def __eq__(self, other):
        if isinstance(other, (Frame, SeriesSet, Table)):
            if self._dim == other._dim:
                return True
            return False

    def __getitem__(self, pos):
        try:
            return self._Table[pos]
        except TypeError:
            if isinstance(pos, slice):
                return self.__getslice__(pos.start, pos.stop)
                
            col = self._columns.index(pos)
            return [item[col] for item in self._Table]

    def __getslice__(self, i, j):
        if i in self._columns and j in self._columns:
            i, j = self._columns.index(i), self._columns.index(j) + 1
            if i > j:
                i, j = j, i
                
            new_table = list()
            for record in self._Table:
                new_table.append(record[i:j])

            return Table(new_table, self._columns[i:j], self._miss_value)
        else:
            raise TypeError('can not recongnized the expression.')
    
    def __iter__(self):
        for i in xrange(self._dim.Ln):
            yield self._Table[i]

    def __reversed__(self):
        self.reverse()

    def append(self, item):
        '''TODO append a record to the frame last
        '''
        self._Table.append(list(item))
        if len(item) > self._dim.Col:
            self._dim = Table.dims(self._dim.Ln+1, len(item))
        else:
            self._dim = Table.dims(self._dim.Ln+1, self._dim.Col)

        
    def append_col(self, variable_name, series, element_type='AUTO'):
        '''TODO: append a series data to the table last
        '''
        if is_value(series):
            series = [series for i in range(self._dim.Ln)]
            
        if not isinstance(variable_name, str):
            raise ValueError("unsupport type to append in this table")

        if variable_name in self._columns:
            raise ValueError("variable name <'%s'> is already taken."%self.variable_name)

        for i, record in enumerate(self._Table[:len(series)]):    
            if len(record) < self._dim.Col:
                record.extend([self._miss_value for j in range(self._dim.Col - len(record))])
            record.append(series[i])

        for v in series[i+1:]:
            self._Table.append([self._miss_value]*self._dim.Ln + [v])

        self._columns.append(variable_name)
        self._dim = Table.dims(max(self._dim.Ln, len(series)), self._dim.Col+1)

    def count(self, X, *area):
        if not is_iter(X):
            raise TypeError('counting object should be contains in a iterable'+\
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

        if C2 <= C1 or L2 <= L1:
            raise ValueError('the postion in the second tuple should be less '+\
                             'than The first tuple.')


        for record in self._Table[L1:L2]:
            for value in record[C1:C2]:
                for x in list(X):
                    if x == value:
                        counter[x] += 1
        return counter           

    def drop_miss_value(self, drop='LINE'):
        if drop.upper() == 'LINE':
            new_table = list()
            Ln = self._dim.Ln
            for record in self._Table:
                if self._miss_value in record:
                    Ln -= 1
                else:
                    new_table.append(record)
                    
            self._dim = Table.dims(Ln-1, self._dim.Col)
            self._Table = new_table
            del new_table
            
        elif drop.upper() == 'COL':
            save_col = list()
            for i in range(self._dim.Col):
                sequence = [record[i] for record in self._Table]
                if not self._miss_value in sequence:
                    save_col.append(i)

            self._columns = [self._columns[i] for i in save_col]
            if len(self._columns) == 0:
                self._dim = Table.dims(0, 0)
                self._Table = list()
                return
            
            new_frame = list()
            for record in self._Table:
                new_frame.append(tuple([record[i] for i in save_col]))
            self._Table = new_frame
            self._dim = Table.dims(Ln, len(self._columns))
            del new_frame

    def insert(self, pos, item):
        '''TODO: insert a record to the frame, position in <index>
        '''
        self._Table.insert(pos, list(item))
        if len(item) > self._dim.Col:
            self._dim = Table.dims(self._dim.Ln+1, len(item))
        else:
            self._dim = Table.dims(self._dim.Ln+1, self._dim.Col)

    def insert_col(self, variable_name, series, index, element_type='AUTO'):
        '''TODO: insert a series of data to the frame, position in <index>
        '''
        if is_value(series):
            series = [series for i in range(self._dim.Ln)]
            
        if not isinstance(variable_name, str):
            raise ValueError("unsupport type to append in this frame")

        if variable_name in self._columns:
            raise ValueError("variable name <'%s'> is already taken."%self.variable_name)
                              
        for i, record in enumerate(self._Table[:len(series)]):
            if len(record) < index:
                record.extend([self._miss_value for i in range(index - len(record))])
            record.insert(index, series[i])

        for v in series[i+1:]:
            self._Table.append([self._miss_value]*index + [v])

        self._columns.insert(index, variable_name)
        self._dim = Table.dims(self._dim.Ln, self._dim.Col+1)

    def pop(self, item=-1):
        '''TODO: pop(remove & return) a record from the Frame
        '''
        if not isinstance(item, int):
            raise TypeError('an integer is required.')
        
        pop_item = self._Table.pop(item)
        self._dim = Table.dims(self._dim.Ln-1, self._dim.Col)
            
        return pop_item

    def pop_col(self, *titles):
        # TODO: pop(remove & return) a series from the Frame
        pos = list()
        pop_name = list()
        if not titles:
            titles = [self._columns[-1]]
        for item in titles:
            if isinstance(item, str) and item in self._columns:
                pos.append(self._columns.index(item))
                pop_name.append(item)
            elif isinstance(item, int) and item < self._dim.Col:
                pos.append(item)
                pop_name.append(self._columns[item])
            else:
                raise TypeError('an integer or string, which stands for a'+\
                                'column name, is required,'+
                                'not {0} !'.format(type(item)))

        for name in pop_name:
            self._columns.pop(self._columns.index(name))
        pop_data = [list() for i in range(len(pop_name))]

        new_table = list()
        # copy a new Frame
        for record in self._Table:
            line = list()
            for i, value in enumerate(record):
                if i in pos:
                    pop_data[pos.index(i)].append(value)
                else:
                    line.append(value)
            new_table.append(line)

        self._dim = Table.dims(self._dim.Ln, self._dim.Col-len(pos))
        self._Table = new_table
        return dict(zip(pop_name, pop_data))

    def read_text(self, addr, col=all, first_line=1, miss_symbol='NA',
                 title_line=0, sep=',', type_str=False):
        # Set data columns, and transform diction
        dic_ready = False
        
        with open(addr, 'r') as f:
            reader = csv.reader(f, delimiter=sep)
            col = range(len(max(reader, key=len)))
            f.seek(0)
            for i, lines in enumerate(reader):
                if i == title_line:
                    self._columns = [lines[k] for k in col]
                    break

        if not self._columns:
            self._columns = ['Col_'+str(i) for i in col]

        def transform(item):
            if item == miss_symbol:
                return self._miss_value
            if '.' in item:
                try:
                    return float(item)
                except ValueError:
                    return item
            if item.isdigit() or item[1:].isdigit():
                return int(item)
            return item

        # Reload data into data structure and transfrom the type
        with open(addr) as f:
            
            reader = csv.reader(f, delimiter=sep)
            for m, record in enumerate(reader):
                if m >= first_line -1 :
                    break

            if type_str:
                for m, record in enumerate(reader):
                    self._Table.append([record[i] for i in col])
            else:
                for m, record in enumerate(reader):
                    self._Table.append([transform(record[i]) for i in col])
                
        self._dim = Table.dims(m+2-first_line, len(col))

    def reverse(self):
        self._Table.reverse()

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
            col = self._columns
        elif isinstance(col, int) and col < self._dim.Col:
            col = (col, )
        elif isinstance(col, str):
            col = (self._columns.index(col), )
        else:
            new_col = list()
            for c in col:
                if isinstance(c, str):
                    new_col.append(self._column.index(c))
                elif isinstance(c, int) and c < self._dim.Col:
                    new_col.append(c)
                else:
                    raise TypeError("unrecognized symbol '%s'."%str(c))
            col = new_col

        order = dict(arg)
            
        for i, record in enumerate(self._Table):
            for j in col:
                if record[j] in order:
                    self._Table[i][j] = order[record[j]]

    def sort(self, *orders):
        '''TODO: sort the table
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
                                "not in table's columns.")
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
            self._Table = sorted(self._Table, key=itemgetter(*compare_pos))
        else:
            self._Table = hash_sort(self._Table)

        if compare_symbol[0] == 'DESC':
            self._Table.reverse()

    def shuffles(self):
        shuffle(self._Table)


