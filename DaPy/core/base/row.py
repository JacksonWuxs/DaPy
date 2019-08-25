from collections import Iterable, OrderedDict, Sequence
from copy import copy
from .constant import STR_TYPE, VALUE_TYPE, MATH_TYPE, SEQ_TYPE
from .utils import range, xrange, map, zip, filter, is_iter

__all__ = ['Row']

class Row(Sequence):
    '''This class is a view of a row of source sheet

    This class can help you quickly get the each row of the source sheet,
    and it will return the current data of that row. Operations to this object
    will be mapped to the source sheet. Also, any difference in the source sheet
    can be shown from here.

    Parameters
    ----------
    sheet : Reference of the source sheet

    line : the row number of this view

    Examples
    --------
    >>> from DaPy import SeriesSet
    >>> sheet = SeriesSet([[0, 0, 0, 0], [1, 1, 1, 1]], ['A', 'B', 'C', 'D'])
    >>> row0 = sheet[0] # class Row
    >>> row0
    [0, 0, 0, 0]
    >>>
    >>> sheet['A'][0] += 1
    >>> row0
    [1, 0, 0, 0]
    >>>
    >>> sheet.shape
    sheet(Ln=2, Col=4)
    >>> row0.append(1)
    >>> row0
    [1, 0, 0, 0, 1]
    >>> print sheet.show()
     A | B | C | D | C_4 
    ---+---+---+---+------
     1 | 0 | 0 | 0 |  1   
     1 | 1 | 1 | 1 | None 
    '''
    
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
        return [_[self._line] for _ in self._sheet.values()]

    def __iter__(self):
        for seq in self._sheet.values():
            yield seq[self._line]

    def __getattr__(self, index):
        if index in self._sheet.data:
            return self.sheet[index][self.columns.index(index)]
        raise AttributeError('Row has not attribute or column named %s.' % index)

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
        col = [self._sheet.nan] * self._sheet.shape.Ln
        col[self._line] = value
        return col
    
    def append(self, value):
        append_col = self._get_new_column(value)
        self._sheet.append_col(append_col)

    def count(self, value):
        return self.data.count(value)

    def extend(self, iterable):
        exist = self.data
        exist.extend(iterable)
        self._sheet[self._line] = exist
    
    def get(self, index, default=None):
        if isinstance(index, int):
            if index < 0:
                index += len(self.columns)
            if index >= len(self.columns):
                return default
            index = self.columns[index]
        return self._sheet.get(index, default)
            
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
