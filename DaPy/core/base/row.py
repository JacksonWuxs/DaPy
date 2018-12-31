from copy import copy
from collections import Iterable
from .tools import _str_types

__all__ = ['Row']

class Row:
    def __init__(self, sheet, i, line):
        self._sheet = sheet
        self._columns = copy(sheet.columns)
        self._line = i
        self._data = line

    def __getattr__(self, index):
        if hasattr(self._data, index):
            return getattr(self._data, index)
        return self._data[self._columns.index(index)]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        '''for value in row -> iter values
        '''
        for value in self._data:
            yield value

    def __repr__(self):
        return '%s' % str(self._data)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._data[index]
        
        if isinstance(index, _str_types):
            return self._data[self._columns.index(index)]
        
        if isinstance(index, slice):
            if None == index.start and None == index.stop:
                return self

            if None == index.start:
                if isinstance(index.stop, _str_types):
                    return self._data[:self._columns.index(index.stop)+1]
                return self._data[:index.stop]

            if None == index.stop:
                if isinstance(index.start, _str_types):
                    return self._data[self._columns.index(index.start):]
                return self._data[index.start:]

            if isinstance(index.start, _str_types):
                return self._data[self._columns.index(index.start):
                                  self._columns.index(index.stop)+1]
            return self._data[index]
        raise AttributeError('unknow statement row[%s]' % index)

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise NotImplementedError('unsupported set multiple values at the same time')
        
        elif isinstance(index, int):
            if isinstance(self._sheet, SeriesSet):
                self._sheet[self._columns[index]][self._line] = value
            else:
                self._sheet[self._line][index] = value
            self._data[index] = value
            if value == self._sheet._miss_symbol:
                self._sheet._miss_value[index] += 1            
        else:
            raise ValueError('unknow statement row[%s] = %s' % (index, value))
