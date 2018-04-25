from array import array
from collections import namedtuple, deque, OrderedDict, Counter
from copy import deepcopy
import csv
import cPickle as pkl
from datetime import date, datetime
from warnings import warn
from operator import itemgetter, attrgetter
from os import path
from pprint import pprint
from random import random, randint, shuffle
import string

__all__ = ['DataSet', 'Frame', 'SeriesSet', 'Table', 'Matrix']

class Table(object):
    
    dims = namedtuple('Table', ['Ln', 'Col'])

    def __init__(self, table=None, columns=None, name='MyTable'):

        self._name = str(name)
        
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
            raise TypeError('Can not transform this object to DaPy.Frame.'+\
                            'DaPy.Frame expects a list or deque contained'+\
                            ' tuple or name tuple.')
    @property
    def dim(self):
        return self._dim

    @property
    def titles(self):
        return self._columns

    @property
    def info(self):
        pprint( { '1.  Structure': ' DaPy.Table',
                 '2. Table Name': ' %s'%self._name,
                 '3. Dimensions': ' Ln=%d | Max Col=%d'%self._dim,
                 '4.    Columns': self._columns})
    
    def __repr__(self):
        msg = ''
        if self._dim.Ln > 10:
            
            for item in self._Table[:4]:
                msg += str(item) + '\n'
            msg += '  ......\n'
            for item in self._Table[-4:]:
                msg += str(item) + '\n'
        else:
            for item in self._Table:
                msg += str(item) + '\n'

        check = '\nDaPy.Table{%d Records & %d Variables}'%self._dim
        msg += '-'*(len(check)-1)
        msg += check
        return msg

    def __call__(self):
        return self._Table

    def __len__(self):
        return self._dim.Ln

    def __eq__(self, other):
        if isinstance(other, (Frame, SeriesSet, Table)):
            if self._dim == other.dim:
                return True
            return False

    def __getitem__(self, pos):
        try:
            return self._Table[pos]
        except:
            col = self._columns.index(pos)
            return [item[col] for item in self._Table]
    
    def __iter__(self):
        for i in xrange(self._dim.Ln):
            yield self._Table[i]

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
        if not (isinstance(variable_name, str) or isinstance(series,
                                                (array, list, deque, tuple))):
            raise ValueError("Unsupport type to append in this frame")

        size = len(series)
        if size < self._dim.Ln:
            series.extend([None for i in xrange(self._dim.Ln-size)])

        if variable_name in self._columns:
            raise ValueError("Variable name <'%s'> is already taken."%self.variable_name)

        self._columns.append(variable_name)
        self._dim = Table.dims(self._dim.Ln, self._dim.Col+1)
        new_frame = list()
        
        for i, record in enumerate(self._Table):
            if len(record) < self._dim.Col:
                record.extend([None for i in range(len(record)-self._dim.Col)])
            record.append(series[i])
            new_frame.append(record)
        self._Table = new_frame
        del new_frame

    def count(self, X, *area):
        if not isinstance(X, (tuple, list, deque, dict)):
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

    def drop_miss_value(self, drop='LINE', symbol=None):
        if drop.upper() == 'LINE':
            new_table = list()
            Ln = self._dim.Ln
            for record in self._Table:
                if symbol in record:
                    Ln -= 1
                else:
                    new_table.append(record)
                    
            self._dim = Table.dims(Ln-1, self._dim.Col)
            self._Table = new_table
            del new_table
            
        elif drop.upper() == 'COL':
            save_col = list()
            for i in xrange(self._dim.Col):
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
        if not (isinstance(variable_name, str) or isinstance(series,
                                                (array, list, deque, tuple))):
            raise ValueError("unsupport type to append in this frame")

        size = len(series)
        if size < self._dim.Ln:
            series.extend([self._miss_value for i in xrange(self._dim.Ln-size)])

        if variable_name in self._columns:
            raise ValueError("variable name <'%s'> is already taken."%self.variable_name)

        self._columns.insert(index, variable_name)
        self._dim = Table.dims(self._dim.Ln, self._dim.Col+1)
                              
        new_frame = list()
        for i, record in enumerate(self._Table):
            if len(record) < index:
                record.extend([None for i in range(len(record)-index)])
            record.insert(index, series[i])
            new_frame.append(record)
        self._Table = new_frame
        del new_frame

    def pop(self, item):
        '''TODO: pop(remove & return) a record from the Frame
        '''
        if not isinstance(item, int):
            raise TypeError('An integer is required.')
        
        pop_item = self._Table.pop(item)
        
        self._dim = Table.dims(self._dim.Ln-1, self._dim.Col)
        self._m_value -= pop_item.count(self._m_symbol)
            
        return pop_item

    def pop_col(self, *titles):
        # TODO: pop(remove & return) a series from the Frame
        pos = list()
        pop_name = list()
        for item in titles:
            if isinstance(item, str) and item in self._columns:
                pos.append(self._columns.index(item))
                pop_name.append(item)
            elif isinstance(item, int) and item < self._dim.Col:
                pos.append(item)
                pop_name.append(self._columns[item])
            else:
                raise TypeError('An integer or string, which stands for a'+\
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

##        for columns in pop_data:
##            self._m_value -= columns.count(self._m_symbol)
        self._dim = Table.dims(self._dim.Ln, self._dim.Col-len(pos))
        self._Table = new_table
        return dict(zip(pop_name, pop_data))

    def read_text(self, addr, col=all, first_line=1, miss_symbol='NA',
                 title_line=0, sep=',', miss_value=None, type_float=False):
        # Set data columns, and transform diction
        dic_ready = False
        self._m_value = 0
        
        with open(addr, 'r') as f:
            reader = csv.reader(f, delimiter=sep)
            for i, lines in enumerate(reader):
                if col == all:
                    col = range(len(lines))
                    
                if i == title_line:
                    self._columns = [lines[k] for k in col]
                    
                if dic_ready is False and i >= first_line:
                    _col_types = dict()
                    self._type_column = list()
                    for j, item in enumerate(lines):
                        if item == miss_symbol:
                            break
                        elif item.isdigit() or item[1:].isdigit():
                            if type_float:
                                _col_types[j] = string.atof
                                self._type_column.append(float)
                            else:
                                _col_types[j] = string.atoi
                                self._type_column.append(int)
                        elif '.' in item:
                            _col_types[j] = string.atof
                            self._type_column.append(float)
                        else:
                            _col_types[j] = str
                            self._type_column.append(str)
                            
                    if len(col) == len(_col_types):
                        dic_ready = True
                        
                if col and self._columns and dic_ready:
                    break
           
        if not self._columns:
            self._columns = ['col_'+str(i) for i in col]

        def transform(i, item):
            try:
                if item == miss_symbol:
                    self._m_value += 1
                    return miss_value
                return _col_types[i](item)
            except ValueError:
                return item

        # Reload data into data structure and transfrom the type
        with open(addr) as f:
            
            reader = csv.reader(f, delimiter=sep)
            for m, record in enumerate(reader):
                if m >= first_line -1 :
                    break
                
            for m, record in enumerate(reader):
                self._Table.append([transform(i, record[i]) for i in col])
                
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
            self._Table = sorted(self._Table, key=itemgetter(*compare_pos))
        else:
            self._Table = hash_sort(self._Table)

        if compare_symbol[0] == 'DESC':
            self._Table.reverse()

    def shuffles(self):
        shuffle(self._Table)


class Matrix(object):
    
    dims = namedtuple('Matrix', ['Ln', 'Col'])

    def __init__(self, table=None, columns=None, name='MyMatrix', check=True):

        self._name = str(name)
        if isinstance(table, DataSet):
            table = table.data

        if isinstance(table, list):
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
            try:
                self.__init_unknow_type(table, check, columns)
            except:
                raise TypeError('can not transform this object to DaPy.Matrix'+\
                                ', which expects a list or deque contained'+\
                                ' tuple or name tuple.')
    @property
    def titles(self):
        return self._columns

    @property
    def dim(self):
        return self._dim

    @property
    def info(self):
        print '1.   Structure: DaPy.Matrix\n' +\
              '2. Matrix Name: %s\n'%self._name +\
              '3.  Dimensions: Ln=%d | Col=%d'%self._dim

    @property
    def T(self):
        new_ = list()
        columns = list()
        for j in xrange(self._dim.Col):
            new_.append([record[j] for record in self._matrix])
            columns.append('Col_'+str(j))
        return Matrix(new_, columns)
    
    def __repr__(self):
        msg = ''
        if self._dim.Ln > 10:
            
            for item in self._matrix[:4]:
                msg += str(item) + '\n'
            msg += '  ......\n'
            for item in self._matrix[-4:]:
                msg += str(item) + '\n'
        else:
            for item in self._matrix:
                msg += str(item) + '\n'

        check = '\nDaPy.Matrix{%d Lines & %d Columns}'%self._dim
        msg += '-'*(len(check)-1)
        msg += check
        return msg

    def __len__(self):
        return self._dim.Ln

    def __eq__(self, other):
        if isinstance(other, (Frame, SeriesSet, Table, Matrix)):
            if self._dim == other.dim:
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
        self._matrix = new_
        return self

    def __pow__(self, other):
        new = list()
        for line in self._matrix:
            new.append([value**other for value in line])
        self._matrix = new_
        return self
    
    def __add__(self, other):
        new_ = list()
        if isinstance(other, (int, float)):
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    line_.append(self._matrix[i][j] + other)
                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)):
            if not self._dim == other._dim:
                raise ValueError('operands could not be broadcast'+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self.dim+\
                                 '(%d,%d)'%other.dim)
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    try:
                        line_.append(self._matrix[i][j] + other[i][j])
                    except IndexError:
                        pass
                new_.append(line_)
        else:
            raise TypeError("'+' operation expects the type of"+\
                            "Matrix, Table, Frame, int or float")

        return Matrix(new_, self._columns)
        
    def __sub__(self, other):
        new_ = list()
        if isinstance(other, (int, float)):
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    line_.append(self._matrix[i][j] - other)
                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)):
            if not self._dim == other._dim:
                raise ValueError('operands could not be broadcast'+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self.dim+\
                                 '(%d,%d)'%other.dim)
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
                
    def __mul__(self, other):
        new_ = list()
        if isinstance(other, (int, float)):
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    try:
                        line_.append(self._matrix[i][j] * other)
                    except:
                        pass
                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)):
            if not self._dim == other._dim:
                raise ValueError('operands could not be broadcast'+\
                                 'together with shapes '+\
                                 '(%d,%d) and '%self._dim+\
                                 '(%d,%d)'%other.dim)
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    line_.append(self._matrix[i][j] * other[i][j])
                new_.append(line_)

        else:
            raise TypeError("'*' operation expects the type of "+\
                            "Matrix, Table, Frame, int or float")
        
        return Matrix(new_, self._columns)
                
    def __div__(self, other):
        new_ = list()
        if isinstance(other, (int, float)):
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    line_.append(self._matrix[i][j] / other)

                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)) and \
                        self._dim == other._dim:
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    line_.append(self._matrix[i][j] / other[i][j])
                new_.append(line_)
        else:
            raise TypeError("'/' operation expects the type of"+\
                            "Matrix, Table, Frame, int or float")
        
        return Matrix(new_, self._columns)

    def __pow__(self, other):
        new_ = list()
        if isinstance(other, (int, float)):
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    line_.append(self._matrix[i][j]**other)

                new_.append(line_)

        elif isinstance(other, (Matrix, Table, Frame)) and \
                        self._dim == other._dim:
            for i in xrange(self._dim.Ln):
                line_ = list()
                for j in xrange(self._dim.Col):
                    line_.append(self._matrix[i][j]**other[i][j])
                new_.append(line_)
        else:
            raise TypeError("'**' operation expects the type of"+\
                            "Matrix, Table, Frame, int or float")
        
        return Matrix(new_, self._columns)

    def dot(self, other):
        if isinstance(other, (Matrix, DataSet, Frame, Table)):
            if self.dim.Col != other.dim.Ln:
                raise ValueError('shapes (%d, %d)'%self.dim +\
                                 ' and (%d, %d) not aligned.'%other.dim)
            col_size_1 = self.dim.Col
            col_size_2 = other.dim.Col
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
        new_ = list()
        for i in xrange(self.dim.Ln):
            new_line = list()
            for pos in xrange(col_size_2):
                sumup = 0
                for j in xrange(col_size_1):
                    sumup += self._matrix[i][j]*other[j][pos]
                new_line.append(sumup)
            new_.append(new_line)
        self._matrix = new_
        self._dim = Matrix.dims(self.dim.Ln, col_size_2)
        return self

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
                        if not isinstance(value, (int, float)):
                            raise TypeError('the value in the No.%d'%i+\
                                            " record and the No.%d "%j +\
                                            "position doesn't have "+\
                                             "correct type")
                        new_record.append(value)
                    self._matrix.append(new_record)
            else:
                self._matrix = table


        except TypeError:
            dim_col = len(table)
            self._matrix = [[float(v)] for v in table]
            
        self._dim = Matrix.dims(len(table), dim_col)

        if columns is None:
            self._columns = ['Col_'+str(i) for i in xrange(dim_col)]
        else:
            self._columns = [str(i) for i in columns]
            if len(self._columns) < dim_col:
                for i in range(len(self._columns)-dim_col):
                    self._columns.append('Col_'+str(i))

    def normalized(self, avg=None, std=None):
        if not (avg and std):
            from formulas import mean, Sum
            avg = mean(self)
            std = (Sum((self - avg)**2/(self._dim.Col*self._dim.Ln)))**0.5
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
        
    
    def read_text(self, addr, col=all, first_line=1, title_line=0, sep=','):
        # Set data columns, and transform diction
        dic_ready = False
        self._m_value = 0
        
        with open(addr, 'r') as f:
            reader = csv.reader(f, delimiter=sep)
            for i, lines in enumerate(reader):
                if col == all:
                    col = range(len(lines))
                    
                if i == title_line:
                    self._columns = [lines[k] for k in col]

                if col and self._columns:
                    break
           
        if not self._columns:
            self._columns = ['col_'+str(i) for i in col]

        # Reload data into data structure and transfrom the type
        with open(addr) as f:
            
            reader = csv.reader(f, delimiter=sep)
            for m, record in enumerate(reader):
                if m >= first_line -1 :
                    break
                
            for m, record in enumerate(reader):
                self._matrix.append([float(record[i]) for i in col])
                
        self._dim = DataSet(dataset=Matrix.dims(m+2-first_line, len(col)))

    def replace(self, key, target):
        raise TypeError("matrix does not support operation 'replace'")
                    
    def shuffles(self):
        shuffle(self._matrix)
        

class DataSet(object):
    
    split_dic = {'csv':',','txt':' ','xls':'\t', 'pkl': None}
    
    def __init__(self, addr=None, title_line=0, first_line=1, split='AUTO',
                 miss_value=None, dataset=None, name='Data', titles=None):
        # We set some basic variables as follow.
        if isinstance(dataset, DataSet):
            self._data = deepcopy(dataset._data)
            self._addr = deepcopy(dataset._addr)
            self._title_line = deepcopy(dataset._title_line)
            self._name = deepcopy(dataset.name)
            self._first = deepcopy(dataset._first)
            self._miss_value = deepcopy(dataset._miss_value)
            self._titles = deepcopy(dataset._titles)
            self._size = deepcopy(dataset._size)
            self._type = deepcopy(dataset._type)
            self._sep = deepcopy(dataset._sep)
            
        elif isinstance(dataset, (Matrix, SeriesSet, Table, Frame)):
            self._data = dataset
            self._addr = addr
            self._title_line = title_line
            self._name = dataset._name
            self._first = first_line
            if isinstance(dataset, Matrix):
                self._miss_value = 0
            else:
                self._miss_value = dataset._miss_value
            self._titles = dataset._columns
            self._size = dataset._dim
            if isinstance(dataset, Matrix):
                self._type = 'DaPy.Matrix'
            elif isinstance(dataset, SeriesSet):
                self._type = 'DaPy.SeriesSet'
            elif isinstance(dataset, Table):
                self._type = 'DaPy.Table'
            else:
                self._type = 'DaPy.Frame'
            self._sep = split

        elif dataset is not None:
            self._data = dataset
            self._title_line = title_line
            self._name = name
            self._first = first_line
            self._miss_value = miss_value
            self._titles = titles
            self._size = len(dataset)
            self._type = 'DaPy.DataSet'
            
        else:
            self._data = None
            if not path.isfile(addr):
                raise IOError('{0} is not a effective address.'.format(addr))
            else:
                self._addr = addr
            self._title_line = title_line
            self._name = name
            self._first = first_line
            self._miss_value = miss_value
            self._titles = titles
            self._size = 0
            self._type = 'DaPy.DataSet'
            if split.upper() == 'AUTO':
                self._sep = DataSet.split_dic.setdefault(addr.split('.')[-1],
                                                         False)
                if self._sep is False:
                    warn("DataSet couldn't auto set delimiter.")
            else:
                self._sep = split
        
    @property
    def data(self):
        return self._data

    @property
    def titles(self):
        return self._titles

    @property
    def name(self):
        return self._type

    @property
    def dim(self):
        return self._size

    @property
    def info(self):
        self._data.info

    def __repr__(self):
        return self._data.__repr__()

    def __class__(self):
        return self._type
    
    def __len__(self):
        return self._size.Ln
        
    def __getitem__(self, pos):
        return self._data[pos]

    def __iter__(self):
        for item in self._data:
            yield item

    def __str__(self):
        return self._name

    def append(self, item):
        '''Append a new record 'item' at the tail of the data set.'''
        self._data.append(item)
        self._size = self._data.dim

    def append_col(self, variable_name, series, element_type='AUTO'):
        '''Append a new variable named 'variable_name' with a list of data
        'series' at the tail of data set.
        '''
        self._data.append_col(variable_name, series, element_type)
        self._size = self._data.dim
        self._titles = self._data.titles
        
    def combine(self, other, func, fill_value=None, overwrite=True):
        '''See Also Pandas/frame.py Line:4716
        '''
        return 'unsupported this function in this version.'

    def corr(self):
        '''Calculate the correlation matrix of your data set.
        SeriesSet structure support this operation only.
        '''
        self._data.corr()

    def count(self, x, *area):
        '''Find one or more set of identification data in the specified area.'''
        if isinstance(x, (str, int, float, bool)):
            x = (x,)
        return self._data.count(x, *area)

    def count_element(self, col=all):
        '''Count the frequency of values for each variable.
        You could count only a part of your data set with setting key-word(col)
        as a iterble inluding column number or variable names.
        '''
        return self._data.count_element(col)
    
    def drop_miss_value(self, order='LINE'):
        '''Drop out all the records, which contain miss value, if 'order' is
        'LINE'. Drop out all the variables, which contain miss value,
        if 'order' is 'COL'.
        '''
        self._data.drop_miss_value(order)
        self._size = self._data.dim
        self._titles = self._data.titles

    def insert(self, index, item):
        '''Insert a record 'item' in position 'index'.'''
        self._data.insert(index, item)
        self._size = self._data.dim

    def insert_col(self, variable_name, series, index, element_type='AUTO'):
        '''Insert a new variable named 'variable_name' with a list of data
        'series' in position 'index'.'''
        self._data.insert_col(variable_name, series, index, element_type)
        self._size = self._data.dim
        self._titles = self._data.titles

    def pop(self, index):
        '''Delete and return the record in position 'index'
        '''
        data = self._data.pop(index)
        self._size = self._data.dim
        return data

    def pop_col(self, *variables):
        '''Delete and return all the value of each 'variables'.
        Key-word(item) could assignment as a number or the variable name.
        '''
        data = self._data.pop_col(*variables)
        self._size = self._data.dim
        self._titles = self._data.titles
        return data

    def normalized(self, process='NORMAL', col=all, attr=None, get_attr=None):
        '''Normalized or standardlized your data in each col.
        '''
        return self._data.normalized(col, attr, get_attr, process)
        
    def readtable(self, col=all, miss_symbol='', type_float=False):
        '''
        This function could be used in loading data from a file in a simple way.
        That will be faster than <readframe> function but couldn't give too much
        information about the variables. In contrast to DaPy.Frame, DaPy.Table
        structure has a more relaxed requirements.
        '''
        self._data = Table()
        self._data.read_text(self._addr, col, self._first, miss_symbol,
                             self._title_line, self._sep,
                             self._miss_value, type_float)
        self._size = self._data.dim
        self._titles = self._data.titles
        self._type = 'DaPy.Table'

    def readframe(self, col=all, miss_symbol='', type_float=False):
        '''This function supports users to load data from a text document
        easily and transform the data into DaPy.Frame structure, which
        is one of the safest structure in DaPy. 
        '''
        self._data = Frame(miss_value=self._miss_value)
        self._data.read_text(self._addr, col, self._first, miss_symbol,
                             self._title_line, sep=self._sep,
                             type_float=type_float)
        self._size = self._data.dim
        self._titles = self._data.titles
        self._type = 'DaPy.Frame'

    def readcol(self, col=all, miss_symbol='', type_array=False,
                type_float=False):
        '''It supports users to load data from the document by columns and
        transform it in to DaPy.SeriesSet. It will be more effecient
        and safer than DaPy.Frame structure because of scrupulous design.
        '''
        self._data = SeriesSet(miss_value=self._miss_value)
        self._data.read_text(self._addr, col, self._first, miss_symbol,
                             self._title_line, self._sep,
                             type_array, type_float)
        self._size = self._data.dim
        self._titles = self._data.titles
        self._type = 'DaPy.SeriesSet'

    def readmatrix(self, col=all):
        '''This function will help you load the data and transform it into
        DaPy.Matrix structure. All of values in this structure should be a float
        type.
        '''

        self._data = Matrix()
        self._data.read_text(self._addr, col, self._first, self._title_line,
                             self._sep)
        self._size = self._data.dim
        self._titles = self._data.titles
        self._type = 'DaPy.Matrix'

    def readpkl(self, addr=None):
        '''Load your data from a .pkl file'''
        if not addr:
            addr = self._addr
        with open(addr) as f:
            data = pkl.load(f)

        if data['Type'] == 'SeriesSet':
            self._data = SeriesSet(data['series'], data['columns'],
                                   name=data['name'],
                                   miss_symbol=data['miss_symbol'],
                                   miss_value=data['miss_value'])
            self._type = 'SeriesSet'
            self._titles = self._data._columns
            self._size = self._data.dim
            self._name = data['name']

    def reverse(self):
        '''Reverse your data set.'''
        self._data.reverse()

    def replace(self, col=all, *arg):
        '''Replace the values from 'key' to 'target' in columns 'col'.
        '''
        self._data.replace(col, *arg)

    def shuffles(self):
        ''' Mess up your data'''
        self._data.shuffles()

    def sort(self, *orders):
        '''You could easily sorted your data set with this function.
        You will be asked to offer at least one ordering conditions.
        The parameter should be like a tuple or a list with two elements,
        on behalf of the key value and arrangement condition (key, arrangement).
        e.g. ('D_col', 'ASC') means that ascending ordered the data set
        with A_col.
        '''
        self._data.sort(*orders)

    def tocsv(self, addr):
        '''Save the DataSet to a .csv file.'''
        with open(addr, 'w') as f:
            
            # write title
            f.write(','.join(self._titles))
            f.write('\n')
            
            # write records
            if isinstance(self._data, (Frame, Table, Matrix, SeriesSet)):
                for line in self._data:
                    f.write(','.join(str(value) for value in line))
                    f.write('\n')
            else:
                for record in self._data:
                    f.write(','.join([str(value) for value in record]))
                    f.write('\n')

    def topkl(self, addr):
        '''Save the DataSet to a .pkl file'''
        
        info = {'Type': str(type(self._data)).split("'")[1].split('.')[2]}

        if info['Type'] == 'SeriesSet':
            info['series'] = self._data._set
            info['columns'] = self._data._columns
            info['name'] = self._data._name
            info['miss_symbol'] = self._data._m_symbol
            info['miss_value'] = self._data._m_value
            with open(addr, 'w') as f:
                pkl.dump(info, f)
        elif info['Type'] == 'Frame':
            pass
    def toframe(self):
        '''Transform the data structure to DaPy.Frame'''
        if self._type != 'DaPy.Frame':
            try:
                self._data = Frame(self._data, self._data._columns)
            except AttributeError:
                self._data = Frame(self._data, self._titles)
            self._type = 'DaPy.Frame'

    def tocol(self):
        '''Transform the data structure to DaPy.SeriesSet'''
        if self._type != 'DaPy.SeriesSet':
            try:
                self._data = SeriesSet(self._data, self._data._columns)
            except AttributeError:
                self._data = SeriesSet(self._data, self._titles)
            self._type = 'DaPy.SeriesSet'

    def totable(self):
        '''Transform the data structure to DaPy.Table'''
        if self._type != 'DaPy.Table':
            try:
                self._data = Table(self._data, self._data._columns)
            except AttributeError:
                self._data = Table(self._data, self._titles)
            self._type = 'DaPy.Table'

    def tomatrix(self):
        '''Transform the data structure to DaPy.Matrix'''
        if self._type != 'DaPy.Matrix':
            try:
                self._data = Matrix(self._data, self._data._columns)
            except AttributeError:
                self._data = Matrix(self._data, self._titles)
            self._type = 'DaPy.Matrix'

    def update(self, other, join='left', overwrite=True, filter_func=None,
               raise_conflict=False):
        '''unsupported in this version
        See Also Pandas/frame.py Line:4872'''
        return 'unsupported this function in this version.'
            
class SeriesSet(object):

    python_series_type = (list, array, deque, tuple)
    python_type = (float, bool, int, str, long, unicode, datetime, date)
    c_type = ('c', 'b', 'B', 'u', 'h', 'H', 'i', 'I', 'l', 'L', 'f', 'd')
    transform = {str: 'c',
                 int: 'l',
                 float: 'f',
                 long: 'L',
                 unicode: 'u'}
    dims = namedtuple('SeriesSet', ['Ln', 'Col'])

    def __init__(self, series=None, columns=None, element_type=None,
               name='MySeries', miss_symbol=None, miss_value=None):

        self._name = str(name)
        self._m_symbol = miss_symbol
        self._miss_value = miss_value
        self._m_value = list()

        if series == None and columns == None:
            self._set = OrderedDict()
            self._columns = list()
            self._dim = SeriesSet.dims(0, 0)
            self._type_column = list()
            self._type_value = list()

        elif isinstance(series, SeriesSet):
            # Data of data set
            self._set = deepcopy(series)
            # Variables of data set
            self._columns = deepcopy(series._columns)
            # Dimension of data set
            self._dim = deepcopy(series._dim)
            # Type of each column
            self._type_column = deepcopy(series._type_column)
            # Type of value in each column
            self._type_value = deepcopy(series._type_value)
            # How much miss value in this data set
            self._m_value = deepcopy(series._m_value)
            # What is the simbol of this data set
            self._m_symbol = deepcopy(series._m_symbol)
            # What is the simbol of this data set
            self._miss_value = deepcopy(series.miss_value)  
            
        elif isinstance(series, (dict, OrderedDict)):
            self._set = OrderedDict() # SeriesSet
            self._columns = sorted([title for title in series]) # Column names
            self._type_column = list() #
            self._type_value = list()
            # MaxSize of this Series Set
            max_Ln = max([len(series[title]) for title in self._columns])
            for title in self._columns:
                sequence = series[title]
                size = len(sequence)
                m_value_line = 0 # miss value in this line

                if size != max_Ln:
                    sequence = list(sequence)
                
                # Check the type of each sequence
                if isinstance(sequence, array):
                    self._type_column.append(array)
                    self._type_value.append(sequence.typecode)

                elif isinstance(sequence, SeriesSet.python_series_type):
                    self._type_column.append(type(sequence))
                    check_type = False
                    for i, item in enumerate(sequence):
                        # Check miss value
                        if item == miss_symbol:
                            m_value_line += 1
                            sequence[i] = miss_value
                            continue

                        # Set the value's type in this column
                        if check_type is False:
                            if isinstance(item, SeriesSet.python_type):
                                self._type_value.append(type(item))
                                check_type = type(item)
                            else:
                                raise TypeError("unsupport type '%s' "%type(item)+\
                                            "in column '%s' and "%title+\
                                            "position '%d'"%i)

                        # check the value's type
                        if isinstance(item, check_type) is False:
                            raise TypeError("unsupport type '%s' "%type(item)+\
                                            "in column '%s' and "%title+\
                                            "position '%d'"%i)
                else:
                    raise TypeError("unsupport type '%s' "%type(sequence)+\
                                    "in column '%s'."%title)

                # Try to transform to array
                if size != max_Ln:
                    sequence.extend([miss_symbol for i in size-max_Ln])
                elif m_value_line == 0:
                    try:
                        sequence = array(SeriesSet.transform[\
                            self._type_value[-1]], sequence)
                    except KeyError:
                        pass
                    except TypeError:
                        pass
                self._m_value.append(m_value_line)
                    
                self._set[title] = sequence
            self._dim = SeriesSet.dims(max_Ln, len(self._columns))

        elif isinstance(series, (Frame, Matrix)):
            self._dim = SeriesSet.dims(series._dim.Ln, series._dim.Col)         
            self._type_column = [list for i in xrange(self._dim.Col)]
            self._columns = deepcopy(series._columns)
            self._m_value = [0 for i in xrange(self._dim.Col)]
            self._set = OrderedDict()
            for i, title in enumerate(self._columns):
                sequence = series[title]
                self._set[title] = sequence
                self._m_value[i] = sequence.count(self._m_symbol) + \
                                   sequence.count(self._miss_value)
            if isinstance(series, Matrix):
                self._m_symbol = miss_symbol
                self._type_value = [float for i in xrange(self._dim.Col)]
            else:
                self._m_symbol = deepcopy(series._m_symbol)
                self._type_value = deepcopy(series._type_column)
                
        elif isinstance(series, Table):
            self._columns = deepcopy(series._columns)
            self._dim = SeriesSet.dims(series._dim.Ln, series._dim.Col) 
            self._m_value = list()
            self._m_symbol = miss_symbol
            self._miss_value = miss_value
            self._type_value = [type(v) for v in series[0]]
                
            self._set = OrderedDict()
            for i, title in enumerate(self._columns):
                sequence = series[title]
                col_m_value = 0
                for j, value in enumerate(sequence):
                    if value == self._miss_value:
                        col_m_value += 1
                    elif not isinstance(value, self._type_value[j]):
                        raise TypeError('value in position <%d, %d>'%(i, j)+\
                                        "doesn't have a correct type.")
                self._set[title] = sequence
                self._m_value.append(col_m_value)

        elif isinstance(series, (list, deque, tuple)):
            if isinstance(columns, list):
                self._columns = [str(v) for v in columns]
            elif isinstance(columns, (str, int)):
                self._columns = [columns, ]
            else:
                self._columns = ['Col_0', ]

            self._dim = SeriesSet.dims(len(series), 1)
            self._m_symbol = miss_symbol
            self._miss_value = miss_value
            self._type_column = [type(series)]
            self._set = OrderedDict()
            self._m_value = [0, ]
            
            if element_type is not None:
                self._type_value = [element_type, ]
            else:
                for value in series:
                    if value != self._miss_value:
                        self._type_value = [type(value)]
                        break
                    
            for i, value in enumerate(series):
                if value == self._m_symbol or value == self._miss_value:
                    value = self._miss_value
                    self._m_value[-1] += 1
                elif not isinstance(value, self._type_value[0]):
                    raise TypeError("value in position <0, %d>"%i+\
                                    "doesn't have a correct type.")
            self._set[self._columns[0]] = series
        else:
            raise TypeError("SeriesSet unsupport type '%s' given."%type(series))
        
    @property
    def dim(self): # OK #
        return self._dim

    @property
    def titles(self): # OK #
        return self._columns

    @property
    def info(self): # OK #
        new_type_c = [str(t).split("'")[1].split(".")[0] for t in self._type_column]
        new_type_v = [str(t).split("'")[1] for t in self._type_value]
        new_m_value = [str(v) for v in self._m_value]
        max_n = len(max(self._columns, key=len))
        info = ' '*15 + '-'*max_n + '+' + '-'*14 +'+---------------+-------------\n'
        for i in xrange(self._dim.Col):
            info += ' '*15
            info += self._columns[i].center(max_n) + '| '
            info += new_m_value[i].center(13) + '|'
            info += ('<' + new_type_c[i] + '>').center(15) + '|'
            info += ('<' +new_type_v[i] + '>').center(13) + '\n'
        info += ' '*15 + '-'*max_n + '+' + '-'*14 +'+---------------+-------------'
                
        print '1.  Structure: DaPy.SeriesSet\n' +\
              '2.   Set Name: %s\n'%self._name +\
              '3. Dimensions: Ln=%d | Col=%d\n'%self._dim +\
              '4. Miss Value: %d elements\n'%sum(self._m_value) +\
              '5.    Columns: ' + 'Title'.center(max_n) +  '|'+\
                             '  Miss Value  |' +\
                             '  Column Type  |  Value Type \n'+\
                             info
        return 

    def __repr__(self): # OK #
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

        def write_col():
            msg = ''
            size = len(max(self._columns, key=len))
            for i, title in enumerate(self._columns):
                msg += write_Ln(i, title, size-len(title))
            return msg

        msg = write_col()
        check = '\nSeriesSet{%d Records & %d Variables}'%self._dim
        msg += '-'*(len(check)-1)
        msg += check
        return msg

    def __len__(self):  # OK #
        return self._dim.Ln

    def __eq__(self, other):  # OK #
        if isinstance(other, (Frame, SeriesSet)):
            if self._dim == other.dim:
                return True
        return False

    def __getitem__(self, pos):  # OK #
        if isinstance(pos, int):
            return self._set[self._columns.index(pos[0])]
        
        elif isinstance(pos, slice):
            pos = str(pos)[6:-1].split(',')
            try:
                start = int(pos[0])
            except ValueError:
                start = 0
            try:
                end = int(pos[1])
            except ValueError:
                end = self._dim.Ln
            try:
                step = int(pos[2])
            except ValueError:
                step = 1
            
            return_list = list()
            for pos in xrange(start, end, step):
                return_list.append([self._set[t][pos] for t in self._columns])
            return return_list
                    
        elif isinstance(pos, str):
            return self._set[pos]
        
        else:
            raise TypeError('SeriesSet indices must be int or str, '+\
                            'not %s'%str(type(pos)).split("'")[1])
                                
    def __iter__(self):  # OK #
        for i in xrange(self._dim.Ln):
            yield [self._set[title][i] for title in self._columns]

    def append(self, item): # OK #
        # TODO append a record to the frame last
        if not isinstance(item, (list, tuple)):
            raise TypeError("insert item is not a support type "+\
                            "with <'%s'>"%type(item))

        for i, title in enumerate(self._columns):
            element = item[i]
            if element == self._m_symbol:
                if not isinstance(self._type_column[i], (list, deque)):
                    self._set[title] = list(self._set[title])
                self._set[title].append(self._miss_value)
                self._m_value[i] += 1
            elif isinstance(element, self._type_value[i]):
                self._set[title].append(element)
            else:
                raise TypeError('the element in position %d is'%i+
                                ' not a correct type.')

        self._dim = Frame.dims(self._dim.Ln+1, self._dim.Col)
        
    def append_col(self, variable_name, series, element_type='AUTO'): # OK #
        # TODO: append a series data to the frame last
        if isinstance(variable_name, str) is False or\
           isinstance(series,(array, list, deque, tuple)) is False:
            raise ValueError("unsupport type to append in this frame")
        
        # check the variable name
        if variable_name in self._columns:
            raise ValueError("variable name <'%s'> is already taken."%self.\
                             variable_name)

        # check the lenth of data
        size_series = len(series)
        if size_series < self._dim.Ln:
            series = list(series)
            series.extend([self._miss_value for t in xrange(
                self._dim.Ln-size_series)])
            size_series = self._dim.Ln
        elif size_series > self._dim.Ln:
            for title in self._columns:
                self._set[title] = list(self._set[title])
                self._set[title].extend([self._miss_value for t in xrange(
                    size_series-self._dim.Ln)])
        # set the elements' type
        if element_type.upper() == 'AUTO':
            if isinstance(series, array):
                element_type = type(series[0])
            else:
                for each in series:
                    if each != self._m_symbol:
                        element_type = type(each)
                        break
        mv = 0
        # check the miss value and values' type
        if not isinstance(series, array):
            for i, element in enumerate(series):
                if not isinstance(element, element_type):
                    if element != self._miss_value:
                        raise ValueError("The element in postion %d doesn't"%i+\
                                         "have the correct type.")
                    else:
                        mv += 1
                elif element == self._miss_value:
                    mv += 1
                elif element == self._m_symbol:
                    mv += 1
                    series[i] = self._miss_value
        self._columns.append(variable_name)
        self._m_value.append(mv)
        self._type_value.append(element_type)
        self._type_column.append(type(series))
        self._dim = SeriesSet.dims(size_series, self._dim.Col+1)
        self._set[variable_name] = series

    def corr(self):
        from formulas import corr as f_c
        matrix = Matrix(columns=self._columns)
        matrix.make(self._dim.Col, self._dim.Col, 0)
        for i, title in enumerate(self._columns):
            if self._type_value[i] not in (int, float, long):
                for j in xrange(self._dim.Col):
                    matrix[i][j] = 0
                    matrix[j][i] = 0
                continue
            if self._m_value[i] != 0:
                sequnce = [v for v in sequence if v != self._miss_value]
            else:
                sequence = self._set[title]

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
                r = f_c(sequence, next_sequence)
                matrix[i][j], matrix[j][i] = r, r
        return SeriesSet(Frame(matrix), self._columns)

    def count(self, X, *area):
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
                raise TypeError('Second position in the first tuple should a int '+\
                                'type emblemed column number or a str emblemed'+\
                                'column name.')

            if area[1][1] == all:
                C2 = self._dim.Col
            elif isinstance(area[1][1], int) and 0 <= area[1][1] <= self._dim.Col:
                C2 = area[1][1]
            elif area[1][1] in self._columns:
                C2 = self._columns.index(area[1][1])
            else:
                raise TypeError('Second position in the second tuple should a int '+\
                                'type emblemed column number or a str emblemed '+\
                                "column name. Tip: use all to stand for all.")

            if area[0][0] == all:
                L1 = 0
            elif isinstance(area[0][0], int) and 0 <= area[0][0] <= self._dim.Ln:
                L1 = area[0][0]
            else:
                raise TypeError('First position in the first tuple should be a int '+\
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
            c = Counter(self._set[title])
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
        else:
            raise IndexError("unrecognized expression '%s'"%str(drop))

    def insert(self, pos, item): # OK #
        '''
        TODO: insert a record to the frame, position in <index>
        '''
        if isinstance(pos, int) is False or\
           isinstance(item, (list, tuple)) is False:
            raise TypeError("Insert item is not a support"+\
                            "type with <'%s'>"%type(item))

        for i, title in enumerate(self._columns):
            element = item[i]
            if element == self._m_symbol:
                if not isinstance(self._type_column[i], (list, deque)):
                    self._set[title] = list(self._set[title])
                self._set[title].insert(pos, self._miss_value)
                self._m_value[i] += 1
            elif isinstance(element, self._type_value[i]):
                self._set[title].insert(pos, element)
            else:
                raise TypeError('the element in position %d is'%i+
                                ' not a correct type.')

        self._dim = SeriesSet.dims(self._dim.Ln+1, self._dim.Col)

    def insert_col(self, variable_name, series, index, element_type='AUTO'): # OK #
        '''
        TODO: insert a series of data to the frame, position in <index>
        '''
        # TODO: append a series data to the frame last
        if isinstance(variable_name, str) is False or\
           isinstance(series,(array, list, deque, tuple)) is False:
            raise ValueError("Unsupport type to append in this frame")
        
        # check the variable name
        if variable_name in self._columns:
            raise ValueError("Variable name <'%s'> is already taken."%self.\
                             variable_name)

        # check the lenth of data
        size_series = len(series)
        if size_series < self._dim.Ln:
            series = list(series)
            series.extend([self._miss_value for t in xrange(
                self._dim.Ln-size_series)])
            size_series = self._dim.Ln
        elif size_series > self._dim.Ln:
            for title in self._columns:
                self._set[title] = list(self._set[title])
                self._set[title].extend([self._miss_value for t in xrange(
                    size_series-self._dim.Ln)])

        # set the elements' type
        if element_type.upper() == 'AUTO':
            if isinstance(series, array):
                element_type = type(series[0])
            else:
                for each in series:
                    if each != self._m_symbol:
                        element_type = type(each)
                        break
        mv = 0
        # check the miss value and values' type
        if isinstance(series, array) is False:
            for i, element in enumerate(series):
                if isinstance(element, element_type) is False:
                    if element != self._miss_value:
                        raise ValueError("The element in postion %d doesn't"%i+\
                                         "have the correct type.")
                    else:
                        mv += 1
                elif element == self._miss_value:
                    mv += 1
                elif element == self._m_symbol:
                    mv += 1
                    series[i] = self._miss_value

        self._columns.insert(index, variable_name)
        self._type_value.insert(index, element_type)
        self._type_column.insert(index, type(series))
        self._dim = SeriesSet.dims(size_series, self._dim.Col+1)
        new_set = OrderedDict()
        self._m_value.insert(index, mv)
        for i, title in enumerate(self._set):
            if i == index:
                new_set[variable_name] = series
            new_set[title] = self._set[title]

        del self._set
        self._set = new_set
        del new_set

    def normalized(self, col=all, attr=None, get_attr=None, process='NORMAL'):
        from formulas import Statistic
        if col is all:
            new_col = self._columns
        else:
            new_col = list()
            for each in col:
                if each in self._columns:
                    new_col.append(each)
                else:
                    new_col.append(self._columns[i])

        attrs_dic = dict()
        if process == 'NORMAL':
            attrs_structure = namedtuple('Nr_attr', ['Min', 'Range'])
        elif process == 'STANDARD':
            attrs_structure = namedtuple('Sd_attr', ['Mean', 'S'])

        for i, title in enumerate(new_col):
            sequence = self._set[title]
            
            if attr:
                A, B = attr[title]
            elif process == 'NORMAL':
                A = float(min(sequence))
                B = max(sequence) - A
            elif process == 'STANDARD':
                statis = Statistic(sequence)
                A, B = statis.Mean, statis.Sn
            else:
                raise ValueError("unrecognized symbol '%s',"%str(process)+\
                                 "use 'NORMAL' or 'STANDARD'")

            sequence = [(value-A)/B for value in sequence]
            if isinstance(self._type_value[i], array):
                sequence = array('f', sequence)
            
            self._set[title] = sequence
            attrs_dic[title] = attrs_structure(A, B)
            self._type_value[i] = float

        if get_attr:
            return attrs_dic
        return

    def pop(self, pos):  # OK #
        '''
        TODO: pop(remove & return) a record from the Frame
        '''
        if not isinstance(pos, int):
            raise TypeError('An integer is required.')
        
        pop_item = [self._set[title].pop(pos) for title in self._columns]
        
        self._dim = Frame.dims(self._dim.Ln-1, self._dim.Col)
        for i, each in enumerate(pop_item):
            if self._m_symbol == each:
                self._m_value[i] -= 1
        return pop_item

    def pop_col(self, *title):  # OK #
        pop_name = list()
        for item in title:
            if isinstance(item, str):
                if item in self._columns:
                    pop_name.append(item)
                else:
                    raise ValueError("'%s' is not in columns title")
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
        return SeriesSet(pop_data)

    def reverse(self): # OK #
        self._columns.reverse()

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
            raise TypeError('This function expects some parameters like 2-dimentions tuple')

    def read_text(self, addr, col=all, first_line=1, miss_symbol='NA', 
                  title_line=0, sep=',', type_array=False, type_float=False):

        dic_ready = False
        datas = list()
        
        with open(addr, 'r') as f:
            reader = csv.reader(f, delimiter=sep)
            for i, lines in enumerate(reader):
                if col == all:
                    col = range(len(lines))
                    
                if i == title_line:
                    self._columns = [lines[k] for k in col]
                    
                if dic_ready is False and i >= first_line:
                    _col_types = dict()
                    self._type_column = list()
                    self._type_value = list()
                    for j, item in enumerate(lines):
                        if item == miss_symbol:
                            break
                        elif item.isdigit() or item[1:].isdigit():
                            if type_float:
                                _col_types[j] = string.atof
                                self._type_value.append(float)
                            else:
                                _col_types[j] = string.atoi
                                self._type_value.append(int)
                            if type_array:
                                datas.append(array('l'))
                            else:
                                datas.append(list())
                        elif '.' in item:
                            _col_types[j] = string.atof
                            self._type_value.append(float)
                            if type_array:
                                datas.append(array('d'))
                            else:
                                datas.append(list())
                        else:
                            _col_types[j] = str
                            self._type_value.append(str)
                            if type_array:
                                datas.append(array('c'))
                            else:
                                datas.append(list())
                            
                    if len(col) == len(_col_types):
                        dic_ready = True
                        
                if col and self._columns and dic_ready:
                    break
           
        if not self._columns:
            self._columns = ['col_'+str(i) for i in col]

        def transform(m, i, item):
            try:
                if item == miss_symbol:
                    self._m_value[i] += 1
                    if isinstance(datas[i], array):
                        datas[i] = list(datas[i])
                    return self._miss_value
                return _col_types[i](item)
            except ValueError:
                self._m_value[i] += 1
                warn('<%s> in line %d and column %d '%(item, m, i)+
                     'has a wrong type!'+\
                     'used miss value instead.')
                return self._miss_value
        

        self._m_value = [0 for t in self._columns]
        
        # Reload data into data structure and transfrom the type
        with open(addr) as f:
            
            reader = csv.reader(f, delimiter=sep)
            for m, record in enumerate(reader):
                if m >= first_line - 1 :
                    break

            for m, record in enumerate(reader):
                for i, l in enumerate(col):
                    datas[i].append(transform(m, l, record[l]))
                    
        for col in datas:
            self._type_column.append(type(col))

        self._set = OrderedDict(zip(self._columns, datas))
        self._dim = SeriesSet.dims(m+2-first_line, len(self._columns))

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
            raise TypeError("Order argument expects one 2-dimensions"+\
                            " tuple which contains 'keyword' in the first"+\
                            ' postion and request in the second place.')
        
        compare_title = list()
        for order in orders:
            if isinstance(order[0], int):
                compare_title.append(self._columns[order[0]])
            elif order[0] in self._columns:
                compare_title.append(order[0])
            else:
                raise TypeError("'%s' keyword is "%order[0] +\
                                "not in frame's columns.")

        compare_symbol = [order[1] for order in orders if order[1]\
                              in ('DESC', 'ASC')]
        if len(compare_symbol) != len(compare_title):
            raise TypeError("'%s' is not a recognized symbol."%order[1])

        size_orders = len(compare_symbol) - 1
        def hash_sort(t=0):
            # initialize values
            sequence = self._set[compare_title[t]]
            total_index = list()
            HashTable = dict()

            # create the diction
            for i, value in enumerate(sequence):
                if value in HashTable:
                    HashTable[value].append(i)
                else:
                    HashTable[value] = [i,]

            # sorted the values
            ordered_sequence = sorted(HashTable)
            
            # transform the record into Frame
            for value in ordered_sequence:
                index = HashTable[value]

                if t != 0 and compare_symbol[t] == 'DESC':
                    index.reverse()
                total_index.extend(index)
                
            return total_index

        new_index = hash_sort()
        if compare_symbol[-1] == 'DESC':
            new_index.reverse()
        new_data = OrderedDict()
        for k, title in enumerate(self._columns):
            sequence = self._set[title]
            new_sequence = [sequence[i] for i in new_index]
            if type(sequence) == array:
                new_sequence = array(SeriesSet.transform[\
                    self._type_value[k]], new_sequence)
            new_data[title] = new_sequence    

        self._set = new_data
        del new_data

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
    
    dims = namedtuple('Frame', ['Ln', 'Col'])
    element_type = (float, bool, int, str, long, unicode, datetime, date)

    def __init__(self, frame=None, columns=None, element_type_='AUTO',
                 name='MyDFrame', miss_symbol=None, miss_value=None):

        self._name = str(name)
        self._m_symbol = miss_symbol
        self._m_value = 0
        self._miss_value = miss_value
        
        if frame == None and columns == None:
            self._Frame = list()
            self._columns = list()
            self._dim =  Frame.dims(0, 0)
            self._type_column = list()
            
        
        elif isinstance(frame, Frame):
            self._Frame = deepcopy(frame._Frame)
            self._columns = deepcopy(frame._columns)
            self._dim = deepcopy(frame._dim)
            self._type_column = deepcopy(frame._type_column)

        # Check the type of frame is legal
        elif isinstance(frame, (list, deque, tuple)):
            if isinstance(frame[0], tuple):
                self._Frame = frame
            elif isinstance(frame[0], list):
                self._Frame = [list(record) for record in frame]
            else:
                raise TypeError('can not transform this object to Frame. '+\
                                'DaPy.Frame expects a sequence contained'+\
                                ' tuple or named-tuple.')

            dim_Col, dim_Ln = len(frame[0]), len(frame)
            
            # Set the type in each column
            if element_type_ == 'AUTO':
                self._type_column = [0 for item in self._Frame[0]]
                for item in self._Frame:
                    for i, value in enumerate(item):
                        if self._type_column[i] != 0:
                            continue
                        if value != miss_value and value != miss_symbol:
                            self._type_column[i] = type(value)
                
            elif isinstance(element_type_, (list, tuple)):
                self._type_column = element_type_
            else:
                raise TypeError('Element_type should be a list or tuple.')

            # Check the element types and lenth
            for i, item in enumerate(self._Frame):
                if len(item) != dim_Col:
                    raise ValueError("The No.{0} record doesn't have".format(i)+\
                                     " the enought variables.")
                for j, each in enumerate(item):
                    if isinstance(each, self._type_column[j]) is False:
                        if each == miss_symbol:
                            self._m_value += 1
                            item[j] = self._miss_value
                        elif each == miss_value:
                            self._m_value += 1
                        else:
                            raise ValueError("The element in (Ln%d,Col%d)"%(i, j)+\
                                             " doesn't have the correct type.")

            self._columns = [str(i) for i in columns]
            self._dim = Frame.dims(dim_Ln, dim_Col)
            
            # Check the size and type of frame is legal
            dim_Ln,dim_Col = len(self._Frame), len(self._Frame[0])
            if len(columns) != dim_Col:
                raise ValueError("Data doesn't have the same size of columns'"+\
                                 "list.")
            
        elif isinstance(frame, SeriesSet):
            self._Frame = list()
            for line in frame:
                self._Frame.append(tuple(line))
            self._type_column = deepcopy(frame._type_value)
            self._m_value = sum(frame._m_value)
            self._columns = deepcopy(frame._columns)
            self._dim = Frame.dims(frame.dim.Ln, frame.dim.Col)
            
        elif isinstance(frame, Matrix):
            self._Frame = [tuple(line) for line in frame]
            self._columns = deepcopy(frame._columns)
            self._dim = Frame.dims(frame._dim.Ln, frame.dim.Col)
            self._type_column = [float for i in xrange(self.dim.Col)]

        elif isinstance(frame, Table):
            self._Frame = list()
            self._dim = Frame.dims(frame.dim.Ln, frame.dim.Col)
            self._columns = deepcopy(frame._columns)
            self._type_column = [type(v) for v in frame[0]]
            for i, record in enumerate(frame):
                line = list()
                for j, v in enumerate(record):
                    if v is None:
                        self._m_value += 1
                        line.append(self._miss_value)
                    elif isinstance(v, self._type_columns[j]):
                        line.append(v)
                    else:
                        raise TypeError('value in position <%d, %d>'%(i, j)+\
                                        "doesn't have correct type")
                line.extend([None for i in xrange(self._dim.Col-len(line))])
                self._Frame.append(tuple(line))
            
        else:
            raise TypeError('can not transform this object to DaPy.Frame.'+\
                            'DaPy.Frame expects a list or deque contained'+\
                            ' tuple or name tuple.')
    @property
    def dim(self):
        return self._dim

    @property
    def titles(self):
        return self._columns

    @property
    def info(self):
        new_type_v = [str(t).split("'")[1] for t in self._type_column]

        max_n = len(max(self._columns, key=len))
        
        info = ''
        for i in xrange(self._dim.Col):
            info += ' '*15
            info += str(self._columns[i]) + ' '*(max_n-len(self._columns[i])) + '| '
            info += '<' + new_type_v[i] + '>\n'
                
        print '1.  Structure: DaPy.Frame\n' +\
              '2. Frame Name: %s\n'%self._name +\
              '3. Dimensions: Ln=%d | Col=%d\n'%self._dim +\
              '4. Miss Value: %d elements\n'%self._m_value +\
              '5.    Columns: Title' + ' '*(max_n-5) + '|'+\
                             ' <Column Type>\n'+\
                             info
    def __repr__(self):
        msg = ''
        if self._dim.Ln > 10:
            
            for item in self._Frame[:4]:
                msg += str(item) + '\n'
            msg += '  ......\n'
            for item in self._Frame[-4:]:
                msg += str(item) + '\n'
        else:
            for item in self._Frame:
                msg += str(item) + '\n'

        check = '\nDaPy.Frame{%d Records & %d Variables}'%self._dim 
        msg += '-'*(len(check) - 1)
        msg += check
        return msg

    def __call__(self):
        return self._set

    def __len__(self):
        return self._dim.Ln

    def __eq__(self, other):
        if isinstance(other, (Frame, SeriesSet, Table)):
            if self._dim == other.dim:
                return True
            return False

    def __getitem__(self, pos):
        try:
            return self._Frame[pos]
        except:
            col = self._columns.index(pos)
            return [item[col] for item in self._Frame]
    
    def __iter__(self):
        for i in xrange(self._dim.Ln):
            yield self._Frame[i]

    def append(self, item):
        # TODO append a record to the frame last
        if isinstance(item, tuple):
            for i,element in enumerate(item):
                if isinstance(element, self._type_column[i]) is False and\
                   element != self._m_symbol:
                    raise TypeError('The element in position %d is'%i+
                                    ' not a correct type.')
                if element == self._m_symbol:
                    self._m_value += 1
        else:
            raise TypeError("Insert item is not a support type "+\
                            "with <'%s'>"%type(item))
               
        self._Frame.append(item)
        self._dim = Frame.dims(self._dim.Ln+1, self._dim.Col)

        
    def append_col(self, variable_name, series, element_type='AUTO'):
        # TODO: append a series data to the frame last
        if not (isinstance(variable_name, str) or isinstance(series,
                                                (array, list, deque, tuple))):
            raise ValueError("Unsupport type to append in this frame")

        size = len(series)
        if size < self._dim.Ln:
            series.extend([self._miss_value for i in xrange(self._dim.Ln-size)])
        elif size > self._dim.Ln:
            raise IndexError('Input series has larger dimension than this Frame.')
        
        if element_type.upper() == 'AUTO':
            for each in series:
                if each != self._m_symbol:
                    element_type = type(each)

        for i, element in enumerate(series):
            if element == self._m_symbol:
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
        if isinstance(X, (str, int, float, bool)):
            X = [X, ]
        elif not isinstance(X, (tuple, list, deque, dict)):
            raise TypeError('Counting object should be contains in a iterable'+\
                            " object, such as 'tuple' or 'list'")
        
        counter = Counter()
        if area == all or area[0] == all:
            C1, L1 = 0, 0
            L2, C2 = self._dim
                
        elif not (len(area) == 2 and len(area[0]) == len(area[1]) == 2):
            raise EOFError("Unreccognized Expression '%s'"%str(area))

        else:        
            if area[0][1] == all:
                C1 = 0
            elif isinstance(area[0][1], int) and 0 <= area[0][1] <= self._dim.Col:
                C1 = area[0][1]
            elif area[0][1] in self._columns:
                C1 = self._columns.index(area[0][1])
            else:
                raise TypeError('Second position in the first tuple should a int '+\
                                'type emblemed column number or a str emblemed'+\
                                'column name.')

            if area[1][1] == all:
                C2 = self._dim.Col
            elif isinstance(area[1][1], int) and 0 <= area[1][1] <= self._dim.Col:
                C2 = area[1][1]
            elif area[1][1] in self._columns:
                C2 = self._columns.index(area[1][1])
            else:
                raise TypeError('Second position in the second tuple should a int '+\
                                'type emblemed column number or a str emblemed '+\
                                "column name. Tip: use all to stand for all.")

            if area[0][0] == all:
                L1 = 0
            elif isinstance(area[0][0], int) and 0 <= area[0][0] <= self._dim.Ln:
                L1 = area[0][0]
            else:
                raise TypeError('First position in the first tuple should be a int '+\
                                'type emblemed start line number. Tip: use all to '+\
                                "stand for all.")

            if area[1][0] == all:
                L2 = self._dim.Ln
            elif isinstance(area[1][0], int) and 0 <= area[1][0] <= self._dim.Ln:
                L2 = area[1][0]
            else:
                raise TypeError('First position in the second tuple should be a int '+\
                                'type emblemed start line number. Tip: use all to '+\
                                "stand for all.")

        if C2 <= C1 or L2 <= L1:
            raise ValueError('The postion in the second tuple should be less '+\
                             'than The first tuple.')

        for record in self._Frame[L1:L2]:
            for value in record[C1:C2]:
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
            self._dim = Frame.dims(Ln, len(self._columns))
            del new_frame

    def insert(self, pos, item):
        # TODO: insert a record to the frame, position in <index>
        if not (isinstance(pos, int) or isinstance(item, tuple)):
            raise TypeError("Insert item is not a support"+\
                            "type with <'%s'>"%type(item))

        if len(item) != self._dim.Col:
            raise IndexError("Input Record doesn't have correct dimension")
        
        for i,element in enumerate(item):
            if element == self._m_symbol:
                self._m_value += 1
                
            elif not isinstance(element, self._type_column[i]):
                raise TypeError('The element in position %d is '%i+\
                                'not a correct type.')

        self._Frame.insert(pos, item)
        self._dim = Frame.dims(self._dim.Ln+1, self._dim.Col)

    def insert_col(self, variable_name, series, index, element_type='AUTO'):
        # TODO: insert a series of data to the frame, position in <index>
        if not (isinstance(variable_name, str) or\
           isinstance(series, (array, list, deque, tuple)) or\
           isinstance(index, int)):
            raise ValueError("Unsupport type to append in this frame")

        size = len(series)
        if size < self._dim.Ln:
            series.extend([self._miss_value for i in xrange(self._dim.Ln-size)])
        elif size > self._dim.Ln:
            raise IndexError('Input series has larger dimension than this Frame.')

        if element_type.upper() == 'AUTO':
            for each in series:
                if each != self._m_symbol:
                    element_type = type(each)
                    break

        for i, element in enumerate(series):
            if element == self._m_symbol:
                self._m_value += 1
                
            elif not isinstance(element, element_type):
                    raise ValueError("The element in postion %d "%i+\
                                     "doesn't have the correct type.")

        if variable_name in self._columns:
            raise ValueError("Variable name <'%s'> is "%variable_name+\
                             "already taken.")
               
        self._dim = Frame.dims(self._dim.Ln, self._dim.Col+1)
        self._type_column.insert(index, element_type)
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

    def pop(self, item):
        # TODO: pop(remove & return) a record from the Frame 
        if isinstance(item, int):
            pop_item = self._Frame.pop(item)
            
            self._dim = Frame.dims(self._dim.Ln-1, self._dim.Col)
            if self._m_symbol in pop_item:
                self._m_value -= 1
                
            return pop_item
        
        raise TypeError('An integer is required.')

    def pop_col(self, *titles):
        # TODO: pop(remove & return) a series from the Frame
        pos = list()
        pop_name = list()
        for item in titles:
            if isinstance(item, str) and item in self._columns:
                pos.append(self._columns.index(item))
                pop_name.append(item)
            elif isinstance(item, int) and item < self._dim.Col:
                pos.append(item)
                pop_name.append(self._columns[item])
            else:
                raise TypeError('An integer or string, which stands for a'+\
                                'column name, is required, '+
                                'not {0} !'.format(type(item)))

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
            self._m_value -= columns.count(self._m_symbol)
        self._dim = Frame.dims(self._dim.Ln, self._dim.Col-len(pos))
        self._Frame = new_frame
        return dict(zip(pop_name, pop_data))

    def read_text(self, addr, col=all, first_line=1, miss_symbol='NA',
                 title_line=0, sep=',', type_float=False):
        # Set data columns, and transform diction
        dic_ready = False
        self._m_value = 0
        
        with open(addr, 'r') as f:
            reader = csv.reader(f, delimiter=sep)
            for i, lines in enumerate(reader):
                if col == all:
                    col = range(len(lines))
                    
                if i == title_line:
                    self._columns = [lines[k] for k in col]
                    
                if dic_ready is False and i >= first_line:
                    _col_types = dict()
                    self._type_column = list()
                    for j, item in enumerate(lines):
                        if item == miss_symbol:
                            break
                        elif item.isdigit() or item[1:].isdigit():
                            if type_float:
                                _col_types[j] = string.atof
                                self._type_column.append(float)
                            else:
                                _col_types[j] = string.atoi
                                self._type_column.append(int)
                        elif '.' in item:
                            _col_types[j] = string.atof
                            self._type_column.append(float)
                        else:
                            _col_types[j] = str
                            self._type_column.append(str)
                            
                    if len(col) == len(_col_types):
                        dic_ready = True
                        
                if col and self._columns and dic_ready:
                    break
           
        if not self._columns:
            self._columns = ['col_'+str(i) for i in col]

        def transform(i, item):
            try:
                if item == miss_symbol:
                    self._m_value += 1
                    return self._miss_value
                return _col_types[i](item)
            except ValueError:
                self._m_value += 1
                warn('<%s> in line %d and column %d '%(item, m, i)+
                     'has a wrong type!'+\
                     'used miss value instead.')
                return self._miss_value

        # Reload data into data structure and transfrom the type
        with open(addr) as f:
            
            reader = csv.reader(f, delimiter=sep)
            for m, record in enumerate(reader):
                if m >= first_line -1 :
                    break
                
            for m, record in enumerate(reader):
                self._Frame.append(tuple([transform(i, record[i]) for i in col]))
                
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

