from collections import Iterable, OrderedDict
from copy import copy
from .constant import STR_TYPE, VALUE_TYPE, MATH_TYPE, SEQ_TYPE
from .utils import range, xrange, map, zip, filter, is_iter

__all__ = ['Row']

class Row(object):
    def __init__(self, sheet, line):
        self._sheet = sheet
        self._line = line

    @property
    def sheet(self):
        return self._sheet

    @property
    def columns(self):
        return self._sheet.columns

    @property
    def data(self):
        _data = self._sheet.data
        return [_data[title][self._line] for title in self.columns]

    def __iter__(self):
        for seq in self._sheet.values():
            yield seq[self._line]

    def __getattr__(self, index):
        if index in self.columns:
            return self.data[self.columns.index(index)]
        raise AttributeError('has not attribute or column named %s.' % index)

    def __eq__(self, y):
        if is_iter(y) is True and len(y) == len(self):
            for left, right in zip(self.data, y):
                if left != right:
                    return False
            return True
        return False

    def __contains__(self, y):
        return y in self.data

    def __delitem__(self, y):
        if y in self.columns:
            self._sheet.__delitem__(y)
        else:
            self._sheet.__delitem__(self.columns[y])

    def __len__(self):
        return self._sheet.shape.Col

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '%s' % str(self.data)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.data[index]

        if isinstance(index, STR_TYPE):
            return self._sheet._data[index][self._line]
        
        if isinstance(index, slice):
            if None == index.start and None == index.stop:
                return self.data

            if None == index.start:
                if isinstance(index.stop, STR_TYPE):
                    return self.data[:self.columns.index(index.stop)+1]
                return self.data[:index.stop]

            if None == index.stop:
                if isinstance(index.start, STR_TYPE):
                    return self.data[self.columns.index(index.start):]
                return self.data[index.start:]

            if isinstance(index.start, STR_TYPE):
                return self.data[self.columns.index(index.start):
                                  self.columns.index(index.stop)+1]
            return self.data[index]

        if isinstance(index, tuple):
            return [self.__getitem__(subindex) for subindex in index]
        
        raise AttributeError('unknow statement row[%s]' % index)

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise NotImplementedError('unsupported set multiple values at the same time')
        
        elif isinstance(index, int):
            if isinstance(self._sheet.data, dict):
                self._sheet._data[self.columns[index]][self._line] = value
            else:
                self.data[index] = value
            if value == self._sheet.nan:
                self._sheet._missing[index] += 1

        elif isinstance(index, STR_TYPE):
            self._sheet._data[index][self._line] = value
            if self._sheet._isnan(value):
                index = self.columns[index]
                self._sheet._missing[index] += 1
                
        else:
            raise ValueError('unknow statement row[%s] = %s' % (index, value))

    def _get_new_column(self, value):
        col = [self._sheet.miss_symbol] * self._sheet.shape.Ln
        col[self._line] = value
        return col
    
    def append(self, value):
        append_col = self._get_new_column(value)
        self._sheet.append_col(append_col)

    def count(self, value):
        return self.data.count(value)

    def extend(self, iterable):
        extend_col = [[self._sheet.miss_symbol] * len(iterable)\
                      for i in range(self._sheet.shape.Ln)]
        extend_col[self._line] = list(iterable)
        self._sheet.extend_col(extend_col)

    def index(self, value):
        return self.data.index(value)

    def insert(self, index, value):
        append_col = self._get_new_column(value)
        self._sheet.insert_col(index, append_col)
 
    def pop(self, index):
        return self._sheet.pop_col(index)[self._line]

    def remove(self, value):
        index = self.data.index(value)
        self._sheet.pop_col(index)

    def tolist(self):
        return self.data


SEQ_TYPE += (Row,)
