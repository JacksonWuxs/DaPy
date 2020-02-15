import cython
from cpython.array cimport array
from itertools import chain, repeat
from libc.stdlib cimport malloc

cdef class Matrix:

    cdef readonly:
        int _rows
        int _cols
        array _src

    def __cinit__(self, data, dtype='d'):
        self._rows = len(data)
        if isinstance(data, array):
            self._src = data
            self._cols = 1
        elif isinstance(data[0], float) is False:
            self._cols = len(data[0])
            self._src = array(dtype, chain.from_iterable(data))
        else:
            self._cols = 1
            self._src = array(dtype, data)
    
    @property
    def shape(self):
        return (self._rows, self._cols)
    
    @property
    def src(self):
        return self._src
    
    def __getitem__(self, key):
        return self._src[key]
    
    def __len__(self):
        return len(self._src)
    
    def reshape(self, shape):
        assert len(shape) == 2
        assert shape[0] * shape[1] == len(self._src)
        self._rows, self._cols = shape
        return self
    
    def tolist(self):
        arr, row, col = self._src, self._rows, self._cols
        return list(arr[i*col:(i+1)*col].tolist() for i in range(row))

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dot(Matrix X, Matrix Y):
    cdef int x_row, x_col, y_row, y_col
    x_row, x_col = X.shape
    y_row, y_col = Y.shape
    assert x_col == y_row

    cdef array[double] result = array('d', repeat(0.0, x_row * y_col))
    cdef int i, j, times
    cdef double value

    # we would like to visit MemoryView objects rather than 
    # create a new array object. Thus, we declear that 
    # following variables are MemoryView.
    cdef double[:] x_src, y_src, row, col
    x_src, y_src = X.src, Y.src

    # calculate values
    with nogil:
        for i in range(x_row):
            row = x_src[i * x_col:(1 + i) * x_col]
            for j in range(y_col):
                col = y_src[j::y_col]
                value = 0.0
                for index in range(x_col):
                    value += row[index] * col[index]
                result[i * x_row + j] = value
    return Matrix(result).reshape((x_row, y_col))