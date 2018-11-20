from collections import namedtuple
from copy import deepcopy, copy
from tools import is_seq, is_math, is_iter
from random import random
import csv

__all__ = ['Matrix']

class Matrix(object):

    dims = namedtuple('Matrix', ['Ln', 'Col'])

    def __init__(self, table=None, check=True):
        
        if is_iter(table) and not isinstance(table, str):
            self.__init_unknow_type(table, check)
                 
        elif isinstance(table, Matrix):
            self._matrix = deepcopy(table)
            self._dim = deepcopy(table._dim)

        elif table is None:
            self._matrix = list()
            self._dim = Matrix.dims(0, 0)
            
        else:
            raise TypeError('can not transform this object to DaPy.Matrix'+\
                            ', which expects iterables in iterable.')

    @property
    def data(self):
        return self._matrix

    @property
    def shape(self):
        return self._dim

    @property
    def T(self):
        new_ = list()
        for j in range(self._dim.Col):
            new_.append([record[j] for record in self._matrix])
        return Matrix(new_, False)

    @property
    def I(self):# inverse
        '''calculating the invert matrix

        Reference
        ---------
        1. Bin Luo. (2015). The Implement of Matrix with Python.
            from http://www.cnblogs.com/hhh5460/p/4314231.html
        '''
        if self._dim.Ln != self._dim.Col:
            raise ValueError('can not invert a non-sqrt matrix.')

        D = float(self.D)
        N = [1] * self._dim.Ln
        for i in range(self._dim.Ln):
            N[i] = [self._get_cofactor(i, j).D / D for j in range(self._dim.Col)]
        return Matrix(N)
        
    @property
    def D(self): # determinant
        if self._dim.Ln != self._dim.Col:
            raise ValueError('can not determinant a non-sqrt matrix.')

        if self._dim.Ln == 2:
            return self._matrix[0][0] * self._matrix[1][1] - \
                   self._matrix[0][1] * self._matrix[1][0]
        total = 0.0
        for c in range(self._dim.Col):
            total += (-1) ** (c + 1) * self._matrix[1][c] * self._get_cofactor(1, c).D
        return total

        start = 0
        for i, value in enumerate(self._matrix[0]):
            new = self.get_cousin(0, i).D
            if i % 2 == 0:
                start += new
            else:
                start -= new
        return start
                
    def __repr__(self):
        temporary_series = [list()] * self._dim.Col
        if self._dim.Ln > 20:
            temporary_Frame = self._matrix[:10]
            temporary_Frame.extend(self._matrix[-10:])
        else:
            temporary_Frame = self._matrix
            
        for line in temporary_Frame:
            for i, value in enumerate(line):
                temporary_series[i].append(str(value))
        column_size = [len(max(col, key=len)) for col in temporary_series]

        frame = 'matrix(' + u'\u250F'.encode('utf-8') + ' ' * (sum(column_size) + len(column_size) - 1) +\
                u'\u2513'.encode('utf-8') + '\n'
        
        if self._dim.Ln > 20:
            for i, item in enumerate(temporary_Frame[1:10]):
                line = '       ' + u'\u2503'.encode('utf-8')
                for i, value in enumerate(item):
                    line += str(value).center(column_size[i]) + ' '
                frame += line[:-1] + u'\u2503'.encode('utf-8') + '\n'
            frame += '       ' + u'\u2503'.encode('utf-8')
            frame += ('Omit %d Ln'%(self._dim.Ln - 20)).center(sum(column_size) + len(column_size) - 1)
            frame += u'\u2503'.encode('utf-8') +'\n'
            for item in temporary_Frame[-10:]:
                line = '       ' + u'\u2503'.encode('utf-8')
                for i, value in enumerate(item):
                    line += str(value).center(column_size[i]) + ' '
                frame += line[:-1] + u'\u2503'.encode('utf-8') + '\n'
                
        else:
            for i, item in enumerate(temporary_Frame):
                line = '       ' + u'\u2503'.encode('utf-8')
                for i, value in enumerate(item):
                    line += str(value).center(column_size[i]) + ' '
                frame += line[:-1] + u'\u2503'.encode('utf-8') + '\n'
        frame += '       ' + u'\u2517'.encode('utf-8') + ' ' * (sum(column_size) + len(column_size) - 1) +\
                 u'\u251B'.encode('utf-8')+ ')'
        return frame

    def __getstate__(self):
        instance = self.__dict__.copy()
        instance['_dim'] = tuple(self._dim)
        return instance

    def __setstate__(self, dict):
        self._matrix = dict['_matrix']
        self._dim = Matrix.dims(*dict['_dim'])
        
    def __contains__(self, e):
        if isinstance(e, list):
            for record in self._matrix:
                if record == e:
                    return True
        return False

    def __len__(self):
        return self._dim.Ln

    def __eq__(self, other):
        if hasattr(other, 'shape') and other.shape != self._dim:
            return False
        
        elif len(other) != self._dim.Ln or len(other[0]) != self._dim.Col:
            return False
        
        try:
            size_col = range(self._dim.Col)
            for i in range(self._dim.Ln):
                for j in size_col:
                    if self._matrix[i][j] != other[i][j]:
                        return False
        except:
            return False
        else:
            return True

    def __getitem__(self, pos):
        if isinstance(pos, (slice, int)):
            return self._matrix[pos]
        
        if isinstance(pos, tuple):
            return self._matrix[pos[0]][pos[1]]

    def __setitem__(self, pos, value):
        if isinstance(pos, tuple):
            if not is_math(value):
                raise TypeError('the value in the matrix should be a number.')
            self._matrix[pos[0]][pos[1] - value]

        if isinstance(pos, int):
            if not all(map(is_math, value)):
                raise TypeError('the value in the matrix should be a number.')

            if len(value) != self._dim.Col:
                raise IndexError("the input set doesn't have enough dimentions.")
            self._matrix[pos] = value

        raise TypeError('only supports to set a record as line or a single value.')
    
    def __iter__(self):
        for line in self._matrix:
            yield line

    def __int__(self):
        pass

    def __ne__(self, other):
        if self.__eq__(other):
            return False
        return True

    def __neg__(self):
        new = [0] * self._dim.Ln
        for i, line in enumerate(self._matrix):
            new[i] = [value.__neg__() for value in line]
        return Matrix(new, False)

    def __pos__(self):
        new = [0] * self._dim.Ln
        for i, line in self._matrix:
            new[i] = [value.__pos__() for value in line]
        return Matrix(new, False)

    def __sum__(self, other):
        return sum([sum(record) for record in self._matrix])
    
    def __abs__(self):
        new_ = [0] * self._dim.Ln
        for i, line in enumerate(self._matrix):
            new_[i] = [abs(value) for value in line]
        return Matrix(new_, False)

    def __add__(self, other):
        new_ = [0] * self._dim.Ln
        if is_math(other):
            for i in range(self._dim.Ln):
                new_[i] = [self._matrix[i][j] + other\
                           for j in range(self._dim.Col)]

        elif hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self._dim+\
                                 '(%d,%d)'%other.shape)
            for i in range(self._dim.Ln):
                new_[i] = [self._matrix[i][j] + other[i][j]\
                           for j in range(self._dim.Col)]
        else:
            raise TypeError("'+' operation expects the type of"+\
                            "number or an array-like object which has "+\
                            "attribute `shape`")

        return Matrix(new_, False)

    def __radd__(self, other):
        return self.__add__(other)
        
    def __sub__(self, other):
        new_ = [0] * self._dim.Ln
        if is_math(other):
            for i in range(self._dim.Ln):
                new_[i] = [self._matrix[i][j] - other\
                           for j in range(self._dim.Col)]

        elif hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self._dim+\
                                 '(%d,%d)'%other.shape)

            for i in range(self._dim.Ln):
                new_[i] = [self._matrix[i][j] - other[i][j]\
                               for j in range(self._dim.Col)]

        else:
            try:
                return self.__sub__(Matrix(other))
            except:
                raise TypeError("'-' operation expects the type of"+\
                                "number or an array-like object which has "+\
                                "attribute `shape`")

        return Matrix(new_, False)

    def __rsub__(self, other):
        new_ = [0] * self._dim.Ln
        if is_math(other):
            for i in range(self._dim.Ln):
                new_[i] = [other - self._matrix[i][j]\
                           for j in range(self._dim.Col)]

        elif hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self._dim+\
                                 '(%d,%d)'%other.shape)
            for i in range(self._dim.Ln):
                new_[i] = [other[i][j] - self._matrix[i][j]\
                           for j in range(self._dim.Col)]
        else:
            try:
                return self.__rsub__(Matrix(other))
            except:
                raise TypeError("'-' operation expects the type of"+\
                                "number or an array-like object which has "+\
                                "attribute `shape`")

        return Matrix(new_, False)
                
    def __mul__(self, other):
        new_ = [0] * self._dim.Ln
        if is_math(other):
            for i in range(self._dim.Ln):
                new_[i] = [other * self._matrix[i][j]\
                           for j in range(self._dim.Col)]

        elif hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self._dim+\
                                 '(%d,%d)'%other.shape)
            for i in range(self._dim.Ln):
                new_[i] = [other[i][j] * self._matrix[i][j]\
                           for j in range(self._dim.Col)]
        else:
            raise TypeError("'*' operation expects the type of"+\
                            "number or an array-like object which has "+\
                            "attribute `shape`")

        return Matrix(new_, False)

    def __rmul__(self, other):
        return self.__mul__(other)
                
    def __div__(self, other):
        new_ = [0] * self._dim.Ln
        if is_math(other):
            for i in range(self._dim.Ln):
                new_[i] = [self._matrix[i][j] / other\
                           for j in range(self._dim.Col)]

        elif hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self._dim+\
                                 '(%d,%d)'%other.shape)
            
            for i in range(self._dim.Ln):
                new_[i] = [self._matrix[i][j] / other[i][j]\
                           for j in range(self._dim.Col)]
        else:
            raise TypeError("'/' operation expects the type of"+\
                            "number or an array-like object which has "+\
                            "attribute `shape`")

        return Matrix(new_, False)

    def __rdiv__(self, other):
        new_ = [0] * self._dim.Ln
        if is_math(other):
            for i in range(self._dim.Ln):
                new_[i] = [other / self._matrix[i][j] for j in range(\
                                                                self._dim.Col)]

        elif hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self._dim+\
                                 '(%d,%d)'%other._dim)
            for i in range(self._dim.Ln):
                new_[i] = [other[i][j] / self._matrix[i][j]\
                           for j in range(self._dim.Col)]
        else:
            try:
                other = Matrix(other)
                return other / self
            except:
                raise TypeError("'/' operation expects the type of"+\
                                "number or an array-like object which has "+\
                                "attribute `shape`")
        return Matrix(new_, False)

    def __pow__(self, other):
        new_ = [0] * self._dim.Ln
        if is_math(other):
            for i in range(self._dim.Ln):
                new_[i] = [self._matrix[i][j] ** other\
                           for j in range(self._dim.Col)]

        elif hasattr(other, 'shape'):
            if self.shape != other.shape:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d,%d) '%self._dim+\
                                 '(%d,%d)'%other._dim)
            for i in range(self._dim.Ln):
                new_[i] = [self._matrix[i][j] ** other[i][j]\
                           for j in range(self._dim.Col)]
        else:
            raise TypeError("'**' operation expects the type of "+\
                            "number or an array-like object which has "+\
                            "attribute `shape`")

        return Matrix(new_, False)

    def dot(self, other):
        if hasattr(other, 'shape'):
            if self.shape[1] != other.shape[0]:
                raise ValueError('shapes (%d, %d)'%self._dim +\
                                 ' and (%d, %d) not aligned.'%other._dim)
            col_size_1 = self._dim.Col
            col_size_2 = other.shape[1]
            
        elif isinstance(other, (list, tuple, deque)):
            other = Matrix(other)
            if self._dim.Col != other.shape:
                raise ValueError('shapes (%d, %d)'%(self._dim)+\
                                 ' and (%d, %d) not aligned.'%other.shape)
            col_size_1 = self._dim.Col
            col_size_2 = other._dim.Col
            
        else:
            raise TypeError('unsupported operation dot, with type '+\
                            '< Matrix > and '%str(type(self._matrix)) +\
                            '%s and '%str(type(other)))
        
        new_ = [[0]*col_size_2 for k in range(self._dim.Ln)]
        for i in range(self._dim.Ln):
            for pos in range(col_size_2):
                sumup = 0
                for j in range(col_size_1):
                    sumup += self._matrix[i][j]*other[j][pos]
                new_[i][pos] = sumup
        return Matrix(new_, False)

    def __init_unknow_type(self, table, check):
        try:
            dim_col, dim_ln = len(table[0]), len(table)
        except TypeError:
            self._dim = Matrix.dims(len(table), 1)
            self._matrix = [[float(v)] for v in table]
            return
            
        if check:
            self._matrix = [0] * dim_ln
            for i, record in enumerate(table):
                if len(record) != dim_col:
                    raise IndexError("the No.%d record doesn't "%i+\
                                     "have same dimensions.")
                for value in record:
                    if not is_math(value):
                        raise ValueError('value %s in the No.%d'%(value, i)+\
                                        " record is not a number")
                self._matrix[i] = list(record)
        else:
            self._matrix = table
        self._dim = Matrix.dims(len(table), dim_col)

    def make(self, Ln, Col, element=0):
        if not (isinstance(Ln, int) and isinstance(Col, int) and\
                is_math(element)):
            raise TypeError("arguments 'Ln' and 'Col' expect <int> type,"+\
                            "'element' expects <int> or <float> types.")
        element = float(element)
        self._matrix = [[element] * Col for j in range(Ln)]
        self._dim = Matrix.dims(Ln, Col)

    def make_random(self, Ln, Col, type_int=False):
        if not (isinstance(Ln, int) and isinstance(Col, int)):
            raise TypeError("arguments `Ln` and `Col` expect <int> type,")
        if not isinstance(type_int, (bool, tuple)):
            raise TypeError("argutments `type_int` expects `False` symbol"+\
                            " or a tuple-like.")

        self._matrix = [0] * Ln
        if type_int:
            for i in range(Ln):
                self._matrix[i] = [randint(*type_int)] * Col
        else:
            for i in range(Ln):
                self._matrix[i] = [random()] * Col
        self._dim = Matrix.dims(Ln, Col)

    def make_eye(self, size):
        self._matrix = [[0.0] * size for j in range(size)]
        self._dim = Matrix.dims(size, size)
        for i in range(size):
            self._matrix[i][i] = 1.0

    def _get_cofactor(self, x, y):
        if self._dim.Ln != self._dim.Col:
            raise ValueError('can not operate a non-sqrt matrix.')

        if self._dim.Ln < 2:
            raise ValueError('can not opearte a matrix within 2 records.')

        if abs(x) >= self._dim.Ln or abs(y) >= self._dim.Col:
            raise IndexError('index is out of range.')

        M = deepcopy(self._matrix)
        del M[x]
        for line in M:
            del line[y]
        return Matrix(M, False)
            
    def read_text(self, addr, **kward):
        first_line = kward.get('first_line', 1)
        sep = kward.get('sep', ',')
        
        with open(addr, 'r') as f:
            reader = csv.reader(f, delimiter=sep)
            self._matrix = list()
            for m, record in enumerate(reader):
                if m >= first_line - 1:
                    break
            for m, record in enumerate(reader):
                self._matrix.append(map(float, record))

        col = len(max(self._matrix, key=len))
        def has_size(record):
            if len(record) != col:
                return False
            return True

        if not all(map(has_size, self._matrix)):
            raise ValueError('records in your data do not have the same lenth.')
        self._dim = Matrix.dims(len(self._matrix), col)

    def tolist(self):
        if self._dim.Col == 1:
            return [record[0] for record in self._matrix]
        return self._matrix
