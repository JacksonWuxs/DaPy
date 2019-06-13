from collections import namedtuple, deque
from copy import copy, deepcopy
from array import array
from operator import itemgetter
from .utils import is_seq, is_math, is_iter
from .utils import range, filter, map, zip, str2float, xrange
from .constant import SEQ_TYPE
from random import random
from csv import reader

__all__ = ['Matrix']

class Matrix(object):

    dims = namedtuple('Matrix', ['Ln', 'Col'])

    def __init__(self, table=None):
        
        if is_iter(table) and not isinstance(table, str):
            self._init_unknow_type(table=table)
                 
        elif isinstance(table, Matrix):
            self._matrix = copy(table)
            self._dim = copy(table._dim)

        elif table is None:
            self._matrix = list()
            self._dim = Matrix.dims(0, 0)
            
        else:
            raise TypeError('can not transform %s to DaPy.Matrix' % type(table))
    @property
    def src(self):
        return self._matrix

    @property
    def shape(self):
        return self._dim

    @property
    def T(self):
        new_mat = Matrix()
        new_mat._matrix = [list(line) for line in zip(*self._matrix)]
        new_mat._dim = Matrix.dims(self.shape.Col, self.shape.Ln)
        return new_mat

    @property
    def I(self):
        '''calculating the invert matrix

        Reference
        ---------
        1. Bin Luo. (2015). The Implement of Matrix with Python.
            from http://www.cnblogs.com/hhh5460/p/4314231.html
        '''
        assert self._dim.Ln == self._dim.Col, 'can not invert a non-sqrt matrix.'
        if self._dim.Ln == 1:
            return Matrix(self)
        
        D = self.D
        assert D != 0, 'Singular matrix can not calculating the invert matrix.'
        if self._dim.Ln == 2:
            a, b = self[0]
            c, d = self[1]
            v = float(a * d - b * c)
            return Matrix([[d / v, -b / v], [-c / v, a / v]])
        
        new_mat = Matrix()
        new_mat._dim = self._dim
        new_mat._matrix = [[(-1) ** (i+j+1) * self._get_cofactor(j, i).D / D for j in xrange(self._dim.Col)] for i in xrange(self._dim.Ln)]
        return new_mat

    @property
    def D(self):
        assert self._dim.Ln == self._dim.Col, 'can not determinant a non-sqrt matrix.'
        if self._dim.Ln == 2:
            return self._matrix[0][0] * self._matrix[1][1] - \
                   self._matrix[0][1] * self._matrix[1][0]
        return float(sum([(-1)**(1+j) * self[0][j] * self._get_cofactor(0, j).D for j in xrange(self._dim.Ln)]))

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

        frame = u'\u250F' + u' ' * (sum(column_size) + len(column_size) - 1) +\
                u'\u2513' + u'\n'
        
        if self._dim.Ln > 20:
            for i, item in enumerate(temporary_Frame[1:10]):
                line = u'\u2503'
                for i, value in enumerate(item):
                    line += str(value).center(column_size[i]) + u' '
                frame += line[:-1] + u'\u2503' + u'\n'
            frame += u'\u2503'
            frame += (u'Omit %d Ln' % (self._dim.Ln - 20)).center(sum(column_size) + len(column_size) - 1)
            frame += u'\u2503' + u'\n'
            for item in temporary_Frame[-10:]:
                line = u'\u2503'
                for i, value in enumerate(item):
                    line += str(value).center(column_size[i]) + u' '
                frame += line[:-1] + u'\u2503' + u'\n'
                
        else:
            for i, item in enumerate(temporary_Frame):
                line = u'\u2503'
                for i, value in enumerate(item):
                    line += str(value).center(column_size[i]) + u' '
                frame += line[:-1] + u'\u2503' + u'\n'
        frame += u'\u2517' + u' ' * (sum(column_size) + len(column_size) - 1) +\
                 u'\u251B'
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
            return copy(self._matrix[pos])
        
        if isinstance(pos, tuple):
            assert len(pos) == 2, 'too many indices for array.'
            return Matrix([record[pos[1]] for record in self._matrix[pos[0]]])

    def __setitem__(self, pos, value):
        if isinstance(pos, tuple):
            warn = 'position must be like mat[i, j], not mat[%s]' % (pos,)
            assert isinstance(pos[0], (type(None), int, slice)), warn
            assert isinstance(pos[1], (type(None), int, slice)), warn
            assert is_math(value), 'the value in the matrix must be a number.'
            self._matrix[pos[0]][pos[1]] = value

        elif isinstance(pos, int):
            assert abs(pos) <= self.shape.Ln, 'position of %d is out of range'  % pos
            if is_math(value):
                self._matrix[pos] = [value] * self.shape.Col
            if is_seq(value):
                assert len(value) == self._dim.Col, 'cannot copy sequence with size %d to array axis with dimension %d' % (len(value), self.shape.Col)
                self._matrix[pos] = list(value)

        else:
            raise TypeError('only supports to set a record as line or a single value.')
    
    def __iter__(self):
        for line in self._matrix:
            yield list(line)

    def __int__(self):
        mat = Matrix()
        mat._matrix = [list(map(int, row)) for row in self._matrix]
        mat._dim = self.shape
        return mat

    def __ne__(self, other):
        if self.__eq__(other):
            return False
        return True

    def __neg__(self):
        neg = Matrix()
        neg._dim = Matrix.dims(self._dim.Ln, self._dim.Col)
        neg._matrix = [0] * self._dim.Ln
        for i, line in enumerate(self._matrix):
            neg._matrix[i] = array('f', (value.__neg__() for value in line))
        return neg

    def __pos__(self):
        neg = Matrix()
        neg._dim = Matrix.dims(self._dim.Ln, self._dim.Col)
        neg._matrix = [0] * self._dim.Ln
        for i, line in enumerate(self._matrix):
            neg._matrix[i] = array('f', (value.__pos__() for value in line))
        return new

    def __sum__(self, other):
        return sum(sum(record) for record in self._matrix)
    
    def __abs__(self):
        neg = Matrix()
        neg._dim = Matrix.dims(self._dim.Ln, self._dim.Col)
        neg._matrix = [0] * self._dim.Ln
        for i, line in enumerate(self._matrix):
            neg._matrix[i] = tuple(value.__abs__() for value in line)
        return new

    def __add__(self, other):
        new_ = [0] * self._dim.Ln
        if is_math(other):
            for i in range(self._dim.Ln):
                new_[i] = array('f', (self._matrix[i][j] + other\
                           for j in xrange(self._dim.Col)))

        elif hasattr(other, 'shape'):
            x1, y1 = self.shape
            x2, y2 = other.shape
            if x1 == x2 and y1 == y2:
                for i in range(self._dim.Ln):
                    new_[i] = array('f', (self._matrix[i][j] + other[i][j]\
                               for j in xrange(self._dim.Col)))
            elif x1 == x2 and y1 == 1 or y2 == 1:
                for i in range(self._dim.Ln):
                    new_[i] = array('f', (self._matrix[i][j] + other[i][0]\
                               for j in xrange(self._dim.Col)))

            elif y1 == y2 and x1 == 1 or x2 == 1:
                for i in range(self._dim.Ln):
                    new_[i] = array('f', (self._matrix[i][j] + other[0][j]\
                               for j in xrange(self._dim.Col)))
                
            else:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d, %d) and '%self._dim+\
                                 '(%d, %d)'%other.shape)
        else:
            raise TypeError("'+' operation expects the type of"+\
                            "number or an array-like object which has "+\
                            "attribute `shape`")

        return Matrix(new_)

    def __radd__(self, other):
        return self.__add__(other)
        
    def __sub__(self, other):
        new_ = Matrix()
        mat = new_._matrix
        if is_math(other):
            new_._dim = self.shape
            for row in self._matrix:
                mat.append([x - other for x in row])

        elif hasattr(other, 'shape'):
            x1, y1 = self.shape
            x2, y2 = other.shape
            new_._dim = Matrix.dims(max(x1, x2), max(y1, y2))
            if x1 == x2:
                if y1 == y2:
                    for lrow, rrow in zip(self._matrix, other):
                        mat.append([l - r for l,r in zip(lrow, rrow)])
                    
                if y2 == 1:
                    for lrow, rrow in zip(self._matrix, other):
                        mat.append([l - rrow[0] for l in lrow])

                if y1 == 1:
                    for lrow, rrow in zip(self._matrix, other):
                        mat.append([lrow[0] - r for r in rrow])

            elif y1 == y2:
                if x1 == 1:
                    sub_line = self._matrix[0]
                    for rrow in other:
                        mat.append([l - r for l,r in zip(sub_line, rrow)])
                if x2 == 1:
                    sub_line = other[0]
                    for lrow in self:
                        mat.append([l - r for l,r in zip(lrow, sub_line)])
        
            else:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d, %d) and '%self._dim+\
                                 '(%d, %d)'%other.shape)
        else:
            try:
                return self.__sub__(Matrix(other))
            except:
                raise TypeError("'-' operation expects the type of"+\
                                "number or an array-like object which has "+\
                                "attribute `shape`")
        
        return new_

    def __rsub__(self, other):
        new_ = Matrix()
        mat = new_._matrix
        if is_math(other):
            new_._dim = self.shape
            for row in self._matrix:
                mat.append([other - x for x in row])

        elif hasattr(other, 'shape'):
            x1, y1 = self.shape
            x2, y2 = other.shape
            new_._dim = Matrix.dims(max(x1, x2), max(y1, y2))
            if x1 == x2:
                if y1 == y2:
                    for lrow, rrow in zip(other, self._matrix):
                        mat.append([l - r for l,r in zip(lrow, rrow)])
                    
                if y2 == 1:
                    for lrow, rrow in zip(other, self._matrix):
                        mat.append([l - rrow[0] for l in lrow])

                if y1 == 1:
                    for lrow, rrow in zip(other, self._matrix):
                        mat.append([lrow[0] - r for r in rrow])

            elif y1 == y2:
                if x1 == 1:
                    sub_line = self._matrix[0]
                    for rrow in other:
                        mat.append([l - r for l,r in zip(sub_line, rrow)])
                if x2 == 1:
                    sub_line = other[0]
                    for lrow in self:
                        mat.append([l - r for l,r in zip(lrow, sub_line)])
        
            else:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d, %d) and '%self._dim+\
                                 '(%d, %d)'%other.shape)
        else:
            try:
                return self.__sub__(Matrix(other))
            except:
                raise TypeError("'-' operation expects the type of"+\
                                "number or an array-like object which has "+\
                                "attribute `shape`")
        return new_
                
    def __mul__(self, other):
        new_ = Matrix()
        mat = new_._matrix
        if is_math(other):
            new_._dim = self.shape
            for row in self._matrix:
                mat.append([x * other for x in row])

        elif hasattr(other, 'shape'):
            x1, y1 = self.shape
            x2, y2 = other.shape
            new_._dim = Matrix.dims(max(x1, x2), max(y1, y2))
            if x1 == x2:
                if y1 == y2:
                    for lrow, rrow in zip(self._matrix, other):
                        mat.append([l * r for l,r in zip(lrow, rrow)])
                    
                if y2 == 1:
                    for lrow, rrow in zip(self._matrix, other):
                        mat.append([l * rrow[0] for l in lrow])

                if y1 == 1:
                    for lrow, rrow in zip(self._matrix, other):
                        mat.append([lrow[0] * r for r in rrow])

            elif y1 == y2:
                if x1 == 1:
                    sub_line = self._matrix[0]
                    for rrow in other:
                        mat.append([l * r for l,r in zip(sub_line, rrow)])
                if x2 == 1:
                    sub_line = other[0]
                    for lrow in self:
                        mat.append([l * r for l,r in zip(lrow, sub_line)])
        
            else:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d, %d) and '%self._dim+\
                                 '(%d, %d)'%other.shape)
        else:
            try:
                return self.__mul__(Matrix(other))
            except:
                raise TypeError("'*' operation expects the type of"+\
                                "number or an array-like object which has "+\
                                "attribute `shape`")
        
        return new_       

    def __rmul__(self, other):
        return self.__mul__(other)
                
    def __div__(self, other):
        new_ = Matrix()
        mat = new_._matrix
        if is_math(other):
            new_._dim = self.shape
            for row in self._matrix:
                mat.append([x / other for x in row])

        elif hasattr(other, 'shape'):
            x1, y1 = self.shape
            x2, y2 = other.shape
            if x2 == y2 == 1 and hasattr(other, 'tolist'):
                return self.__div__(other.tolist()[0][0])
            new_._dim = Matrix.dims(max(x1, x2), max(y1, y2))
            if x1 == x2:
                if y1 == y2:
                    for lrow, rrow in zip(self._matrix, other):
                        mat.append([l / r for l,r in zip(lrow, rrow)])
                    
                if y2 == 1:
                    for lrow, rrow in zip(self._matrix, other):
                        mat.append([l / rrow[0] for l in lrow])

                if y1 == 1:
                    for lrow, rrow in zip(self._matrix, other):
                        mat.append([lrow[0] / r for r in rrow])

            elif y1 == y2:
                if x1 == 1:
                    sub_line = self._matrix[0]
                    for rrow in other:
                        mat.append([l / r for l,r in zip(sub_line, rrow)])
                if x2 == 1:
                    sub_line = other[0]
                    for lrow in self:
                        mat.append([l / r for l,r in zip(lrow, sub_line)])
        
            else:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d, %d) and '%self._dim+\
                                 '(%d, %d)'%other.shape)
        else:
            try:
                return self.__div__(Matrix(other))
            except:
                raise TypeError("'/' operation expects the type of"+\
                                "number or an array-like object which has "+\
                                "attribute `shape`")
        
        return new_

    def __truediv__(self, other):
        return self.__div__(other)

    def __rdiv__(self, other):
        new_ = Matrix()
        mat = new_._matrix
        if is_math(other):
            new_._dim = self.shape
            for row in self._matrix:
                mat.append([other / x for x in row])

        elif hasattr(other, 'shape'):
            x1, y1 = self.shape
            x2, y2 = other.shape
            if x2 == y2 == 1 and hasattr(other, 'tolist'):
                return self.__rdiv__(other.tolist()[0][0])
            new_._dim = Matrix.dims(max(x1, x2), max(y1, y2))
            if x1 == x2:
                if y1 == y2:
                    for lrow, rrow in zip(other, self._matrix):
                        mat.append([l / r for l,r in zip(lrow, rrow)])
                    
                if y2 == 1:
                    for lrow, rrow in zip(other, self._matrix):
                        mat.append([l / rrow[0] for l in lrow])

                if y1 == 1:
                    for lrow, rrow in zip(other, self._matrix):
                        mat.append([lrow[0] / r for r in rrow])

            elif y1 == y2:
                if x1 == 1:
                    sub_line = self._matrix[0]
                    for rrow in other:
                        mat.append([l / r for l,r in zip(sub_line, rrow)])
                if x2 == 1:
                    sub_line = other[0]
                    for lrow in self:
                        mat.append([l / r for l,r in zip(lrow, sub_line)])
        
            else:
                raise ValueError('operands could not be broadcast '+\
                                 'together with shapes '+\
                                 '(%d, %d) and '%self._dim+\
                                 '(%d, %d)'%other.shape)
        else:
            try:
                return self.__rdiv__(Matrix(other))
            except:
                raise TypeError("'/' operation expects the type of"+\
                                "number or an array-like object which has "+\
                                "attribute `shape`")
        return new_

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

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

        return Matrix(new_)

    def _get_cofactor(self, i, j):
        mat = self._matrix
        return Matrix([r[:j] + r[j+1:] for r in (mat[:i]+mat[i+1:])])

    def _init_unknow_type(self, table):
        if hasattr(table, 'tolist'):
            table = table.tolist()
        try:
            self._matrix = [array('f', row).tolist() for row in table]
        except TypeError:
            self._matrix = [array('f', table).tolist(),]
        self._dim = self._init_src_shape(self._matrix)

    @classmethod
    def _init_src_shape(cls, src, judge_shape=True):
        lenth = tuple(map(len, src))
        max_lenth = max(lenth)
        assert lenth.count(max_lenth) == len(lenth), 'not uniqual dimension of rows'
        return cls.dims(len(lenth), max_lenth)

    def argmax(self, axis=None):
        '''Indexes of the maxium values along an axis.

        Example
        -------
        >>> x = dp.mat(range(10)).reshape((3, 4))
        >>> x.argmax()
        >>> x.argmax(0)

        >>> x.argmax(1)
        '''
        if axis is None:
            iter_each_values = (value for row in self._matrix for value in row)
            return max(enumerate(iter_each_values), key=itemgetter(1))[0]
        if axis == 0:
            return Matrix(max(enumerate(col), key=itemgetter(1)) for col in zip(*self._matrix))[:, 0].T
        return Matrix(max(enumerate(row), key=itemgetter(1)) for row in self._matrix)[:, 0].T
        
    
    def diagonal(self):
        return [self._matrix[i][i] for i in xrange(min(self.shape))]

    def dot(self, other):
        if hasattr(other, 'shape') is False:
            return self.dot(Matrix(other))
        
        assert len(other.shape) == 2, 'unexpect data shape of %s' % other.shape
        assert self.shape[1] == other.shape[0], 'shapes (%d, %d)' % self._dim +\
                             ' and (%d, %d) not aligned.'%other._dim
        
        col_size = other.shape[1]
        new = Matrix()
        new._dim = Matrix.dims(self.shape.Ln, other.shape[1])
        now = new._matrix
        for lineI in self._matrix:
            now.append([sum(left * right[pos] for left, right in zip(lineI, other)) for pos in xrange(col_size)])
        return new

    @classmethod
    def make(cls, Ln, Col, element=0):
        if not (isinstance(Ln, int) and isinstance(Col, int)):
            raise TypeError("arguments 'Ln' and 'Col' expect <int> type")
        cls = cls()
        cls._matrix = [[element] * Col for j in xrange(Ln)]
        cls._dim = Matrix.dims(Ln, Col)
        return cls

    @classmethod
    def make_random(cls, Ln, Col, type_int=False):
        if not (isinstance(Ln, int) and isinstance(Col, int)):
            raise TypeError("arguments `Ln` and `Col` expect <int> type,")
        if not isinstance(type_int, (bool, tuple)):
            raise TypeError("argutments `type_int` expects `False` symbol"+\
                            " or a tuple-like.")
        cls = cls()
        cls._matrix = [0] * Ln
        if type_int:
            for i in range(Ln):
                self._matrix[i] = array('f', [randint(*type_int)] * Col)
        else:
            for i in range(Ln):
                self._matrix[i] = array('f', [random()] * Col)
        cls._dim = Matrix.dims(Ln, Col)
        return cls

    @classmethod
    def make_eye(cls, size, value=None):
        assert is_math(value) or value is None or hasattr(value, '__iter__'), 'value should be a list of number, number or None.'
        if value is None:
            value == [1.0] * size
        elif is_math(value):
            value = [value] * size

        cls = cls()
        cls._matrix = [array('f', [0.0] * size) for j in range(size)]
        cls._dim = Matrix.dims(size, size)
        for i in xrange(size):
            cls._matrix[i][i] = value[i]
        return cls

    @classmethod   
    def from_text(cls, addr, **kward):
        first_line = kward.get('first_line', 1)
        sep = kward.get('sep', ',')
        cls = cls()
        
        with open(addr, 'r') as f:
            _reader = reader(f, delimiter=sep)
            for m, record in enumerate(_reader):
                if m >= first_line - 1:
                    break
            for record in _reader:
                cls._matrix.append(array('f', map(str2float, record)))

        col = len(max(cls._matrix, key=len))
        for record in cls._matrix:
            if len(record) != col:
                record.extend([0.0] * (col - len(record)))
        cls._dim = Matrix.dims(len(cls._matrix), col)
        return cls

    def reshape(self, new_shape):
        x, y = new_shape
        size = self.shape.Ln * self.shape.Col
        assert x * y == size, "can't reshape matrix of size %d into shape %s" % (size, new_shape)
        iter_each_values = (value for row in self._matrix for value in row)
        new = Matrix()
        new._dim = Matrix.dims(new_shape[0], new_shape[1])
        row = []
        for i, value in enumerate(iter_each_values, 1):
            row.append(value)
            if i % y == 0:
                new._matrix.append(row)
                row = []
        return new
    
    def tolist(self):
        if self._dim.Col == 1:
            return [record[0] for record in self._matrix]
        return copy(self._matrix)
