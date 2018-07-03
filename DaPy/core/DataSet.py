from collections import namedtuple, deque, OrderedDict, Counter, Iterable
from copy import deepcopy
try:
    import cPickle as pkl
except IOError:
    import Pickle as pkl
from base import is_value, is_iter, SeriesSet, Frame, Matrix
from DaPy.io import parse_addr
from warnings import warn
from os import path

__all__ = ['DataSet']

class DataSet(object):
    '''A general two-dimentional data structure supports users easily
    to opearte the other basic DaPy 2D data structures. It supports
    users to process any inherence data structure in a same way and use
    the Pythonic Syntax. DataSet is one of the fundamantal data structure in
    DaPy.

    Attrbutes
    ---------
    _data : list
        the list stored all the sheets inside.

    _sheets : listg240
    
        the list stored all the names of each sheet.

    _types : list
        the list stored all the type of each sheet.

    Examples
    --------
    >>> import DaPy as dp
    >>> data = dp.DataSet([[1, 2, 3], [2, 3, 4]])
    >>> data.tocol()
    >>> data
    sheet:sheet0
    ============
    Col_0: <1, 2>
    Col_1: <2, 3>
    Col_2: <3, 4>
    >>> data.info
    sheet:sheet0
    ============
    1.  Structure: DaPy.SeriesSet
    2. Dimensions: Ln=2 | Col=3
    3. Miss Value: 0 elements
    4.   Describe: 
     Title | Miss | Min | Max | Mean | Std  |Dtype
    -------+------+-----+-----+------+------+-----
     Col_0 |  0   |  1  |  2  | 1.50 | 0.71 | list
     Col_1 |  0   |  2  |  3  | 2.50 | 0.71 | list
     Col_2 |  0   |  3  |  4  | 3.50 | 0.71 | list
    ==============================================
    '''
    __all__ = ['data', 'columns', 'sheets','info', 'dim', 'append', 'append_col',
               'count', 'count_element', 'drop_miss_value',
               'extend', 'insert', 'insert_col', 'pick', 'pop', 'pop_col',
               'normalized', 'read', 'reverse', 'replace', 'shuffles','corr',
               'sort', 'save', 'tomatrix', 'toframe', 'tocol']

    def __init__(self, obj=None, sheet='sheet0'):
        '''

        Parameter
        ---------
        obj : array-like (default=None)
            initialized your data from a data structure, such as dict(), list()
            Frame(), SeriesSet(), Matrix(), DataSet().
            
        sheet : str (default='sheet0')
            the name of first sheet inside.
        '''
        if obj is None:
            self._data = []
            self._sheets = []
            self._types = []
            
        elif (not is_iter(obj)) and not isinstance(obj, str):
            raise TypeError('DataSet can not store this object.')

        elif isinstance(obj, DataSet):
            self._data = deepcopy(obj._data)
            self._sheets = deepcopy(obj._sheets)
            self._types = deepcopy(obj._types)
            
        elif isinstance(obj, (Matrix, SeriesSet, Frame)):
            self._data = [obj, ]
            self._sheets = [str(sheet), ]
            self._types = [type(sheet), ]

        elif isinstance(sheet, str):
            self._data = [obj, ]
            self._sheets = [str(sheet), ]
            self._types = [str(type(obj)), ]
            
        else:
            self._data = list(obj)
            self._sheets = map(str, sheet)
            self._types = map(type, self._data)
            if len(set(self._sheets)) != len(self._data):
                raise ValueError("the number of sheets' names do not enough.")
                        
    @property
    def data(self):
        if len(self._data) == 1:
            return self._data[0]
        return self._data

    @property
    def columns(self):
        if len(self._data) > 1:
            new_ = list()
            for i, data in enumerate(self._data):
                if hasattr(data, 'columns'):
                    new_.append([self._sheets[i]] + data.columns)
                else:
                    new_.append([self._sheets[i], None])
            new_title = ['sheet name']
            new_title.extend(['title_%d'%i for i in range(1, len(max(new_, key=len)))])
            return Frame(new_, new_title)
        
        elif len(self._data) == 1:
            if hasattr(self._data[0], 'columns'):
                return self._data[0].columns
        return None

    @property
    def size(self):
        return len(self._data)

    @columns.setter
    def columns(self, value):
        for data in self._data:
            if hasattr(data, 'columns'):
                data.columns = value

    @property
    def sheets(self):
        return self._sheets

    @sheets.setter
    def sheets(self, other):
        if isinstance(other, str):
            self._sheets = [other + '%d'%i for i in range(1, len(self._data) + 1)]

        elif is_iter(other):
            if len(set(other)) == len(self._sheets):
                self._sheets = [str(item) for item in other]
            else:
                raise ValueError('the names size does not match the size of '+\
                                 'sheets inside the DataSet')
        else:
            raise ValueError('unrecognized symbol as %s'%other)
                
    @property
    def shape(self):
        new_ = []
        for data in self._data:
            if hasattr(data, 'shape'):
                new_.append(data.shape)
            else:
                new_.append((len(data), 1))
        return new_
    
    @property
    def info(self):
        for i, data in enumerate(self._data):
            print 'sheet:' + self._sheets[i]
            print '=' * (len(self._sheets[i]) + 6)
            if isinstance(data, (Frame, SeriesSet)):
                data.info
            else:
                print('%s has no info() function'%type(data))
        return None

    def __trans_str(self, sheet):
        if sheet not in self._sheets:
            raise IndexError("'%s' is not a sheet name"%sheet)
        return self._sheets.index(sheet)

    def __trans_int(self, sheet):
        if abs(sheet) < len(self._sheets):
            raise IndexError("index '%s' does not exist."%sheet)

        if sheet < 0:
            return len(self._sheets) + sheet - 1
        return sheets

    def __trans_slice(self, i, j, step):
        if isinstance(i, str) or isinstance(j, str):
            if i is not None:
                i = self.__trans_str(i)
            if j is not None:
                j = self.__trans_str(j)
        else:
            if i is not None:
                i = self.__trans_int(i)
            if j is not None:
                j = self.__trans_int(j)
                
        if not isinstance(step, int) and step is None:
            raise TypeError('step shoud be a integer or None.')
        return range(len(self._sheets))[i:j:step]

    def __transform_sheet(self, sheets):
        '''return a list of sheet indexes
        '''
        if isinstance(sheets, slice):
            return self.__trans_slice(sheets.__getattribute__('start'),
                                      sheets.__getattribute__('stop'),
                                      sheets.__getattribute__('step'))
        
        if sheets is all:
            return range(len(self._data))
        
        if not is_iter(sheets):
            sheets = [sheets, ]

        index_sheet = list()
        for sheet in sheets:
            if isinstance(sheet, str):
                pass

    def __contains__(self, e):
        '''__contains__(e) -> e in DataSet

        Determind that weather the object is a sheet name inside.
        '''
        if isinstance(e, str):
            return e in self._sheets
        return any([e == data for data in self._data])

    def __repr__(self):
        reprs = ''
        for i, data in enumerate(self._data):
            reprs += 'sheet:' + self._sheets[i] + '\n'
            reprs += '=' * (len(self._sheets[i]) + 6) + '\n'
            if not isinstance(self._data[i], (SeriesSet, Frame, Matrix)):
                reprs += 'DataSet(%s)'%self._data[i].__repr__() + '\n\n'
            else:
                reprs += self._data[i].__repr__() + '\n\n'
        return reprs[:-2]
    
    def __len__(self):        
        if len(self._data) == 1:
            if hasattr(self._data[0], 'shape'):
                return self._data[0].shape[0]
            return len(self._data[0])
        return len(self._data)
        
    def __getitem__(self, pos):
        if len(self._data) == 1 and (pos not in self._sheets):
            return_data = self._data[0][pos]
            return return_data
        
        if isinstance(pos, slice):
            return self.__getslice__(pos.__getattribute__('start'),
                            pos.__getattribute__('stop'))

        if self.__contains__(pos):
            return DataSet(self._data[self._sheets.index(pos)],
                           self._sheets[self._sheets.index(pos)])

        if isinstance(pos, int):
            return self._data[pos]

        raise ValueError('unrecognized symbol as %s, use string '%pos +\
                         'represented titles or integer which is less than the ' +\
                         'DataSet size.')

    def __getslice__(self, i, j):
        if len(self._data) == 1:
            return DataSet(self._data[0][i:j],
                           self._sheets[0])
        
        if i in self._sheets:
            i = self._sheets.index(i)
        elif i is None:
            i = 0
        elif isinstance(i, int):
            if i < 0:
                i = 0
        else:
            raise ValueError('cannot get the title of %s'%i)

        if j in self._sheets:
            j = self._sheets.index(j)
            
        elif j is None:
            j = len(self._sheets) - 1
            
        elif isinstance(j, int):
            if j < 0:
                j = len(self._sheets) + j

            if j > len(self._sheets):
                j = len(self._sheets)
        else:
            raise ValueError('cannot get the title of %s'%i)

        if i > j:
            i, j = j, i
        return DataSet(self._data[i: j + 1], self._sheets[i: j + 1])

    def __setitem__(self, key, value):
        if len(self._data) == 1 and key not in self._sheets:
            self._data[0].__setitem__(key, value)
            return
        
        if isinstance(key, str):
            if key not in self._sheets:
                self._data.append(value)
                self._types.append(type(value))
                name = key
                while name in self._sheets:
                    name += '_new'
                self._sheets.append(name)
                return
            key = self._sheets.index(key)
        
        if abs(key) > len(self._data):
            raise IndexError('DataSet assignment index out of range')

        if not is_iter(value):
            raise TypeError('value should be a iterable.')

        self._data[key] = value
        self._types[key] = type(value)

    def __delitem__(self, key):
        if key in self._sheets:
            index = self._sheets.index(key)
            del self._sheets[index], self._data[index], self._types[index]
        else:
            for data in self._data:
                data.__delitem__(key)

    def __iter__(self):
        if len(self._data) == 1:
            for item in self._data[0]:
                yield item
        else:
            for item in self._data:
                yield item
        
    def __reversed__(self):
        if len(self._data) == 1:
            self._data[0].reverse()
        else:
            self._data.reverse()

    def add(self, item, sheet=None):
        ''' add a new sheet to this dataset

        Parameter
        ---------
        item : object
            the new sheet object

        sheet : str or None ( default=None)
            the new sheet name

        Example
        -------
        >>> data2 = dp.DataSet([[1, 1, 1], [1, 1, 1]])
        >>> data2.toframe()
        >>> data2
        sheet:sheet0
        ============
         Col_0 | Col_1 | Col_2
        -------+-------+-------
           1   |   1   |   1   
           1   |   1   |   1   
        >>> data.add(data2)
        >>> data
        sheet:sheet0
        ============
        Col_0: <1, 2>
        Col_1: <2, 3>
        Col_2: <3, 4>

        sheet:sheet0
        ============
         Col_0 | Col_1 | Col_2
        -------+-------+-------
           1   |   1   |   1   
           1   |   1   |   1 
        '''
        if isinstance(item, DataSet):
            self._data.extend(item._data)
            self._sheets.extend(item._sheets)
            self._types.extend(item._types)
            
        elif isinstance(item, (Frame, Matrix, SeriesSet)):
            self._data.append(item)
            self._types.append(type(item))
            if not sheet:
                sheet = 'sheet%d'%len(self._sheets)
            while sheet in self._sheets:
                sheet += '_%d'%len(self._sheets)
            self._sheets.append(sheet)

        else:
            self._data.append(list(item))
            if not sheet:
                sheet = 'sheet%d'%len(self._sheets)
            while sheet in self._sheets:
                sheet += '_%d'%len(self._sheets)
            self._sheets.append(sheet)
            self._types.append(list)

    def append(self, item, miss_symbol=None):
        '''Append a new record ``item`` at the tail of each sheet.

        Parameter
        ---------
        item : iterable or value
            append this item as a new record into the original dataset,
            if item is an iterable object, we will direct append. Otherwise,
            we will build a iterable which contains this value inside for you.

        sheet : int or str (defualt=0)
            the name or index represented the sheet
            which you would like to opearte.

        miss_symbol : value (defualt=None)
            the symbol of missing value in this record. It will be
            transformed to the inside miss value.
        '''
        for i, sheet in enumerate(self._data):
            if hasattr(sheet, 'append'):
                try:
                    try:
                        sheet.append(item, miss_symbol)
                    except TypeError:
                        sheet.append(item)
                except Exception, e:
                    warn('sheet:%s append() new object faild '%self._sheets[i] +\
                         'because %s'%e)


    def append_col(self, variable_name, series, element_type='AUTO',
                   miss_symbol=None):
        '''Append a new variable named ``variable_name`` with a list of data
        ``series`` at the tail of data set.

        Parameter
        ---------
        variable_name : str
            the new variable name for this new column.

        series : value or sequence - like
            the iterable object containing values of this new variable or
            a value you expect to append. If
            the number of value is less than the size of dataset, it will be
            added miss value to expand.

        element_type : type (defualt='AUTO')
            the values' type in this variable, use ``AUTO`` to set by itself.

        miss_symbol : value (defualt=None)
            the miss value represented by symbol in this sequence.
        '''
        for i, sheet in enumerate(self._data):
            if hasattr(sheet, 'append_col'):
                try:
                    sheet.append_col(variable_name, series, miss_symbol)
                except Exception, e:
                    warn('sheet:%s append_col() new object faild '%self._sheets[i] +\
                         'because %s'%e)
    def corr(self):
        '''Calculate the correlation matrix of your data set.
        SeriesSet structure support this operation only.

        Return
        ------
        DataSet : the correlation matrix in this datasets.

        Example
        -------
        >>> data = dp.DataSet([[1, 2, 3, 4],
                               [2, None, 3, 4],
                               [3, 3, None, 5],
                               [7, 8, 9, 10]])
        >>> data.tocol()
        >>> data.corr()
        sheet:sheet0
        ============
               |     Col_0      |     Col_1      |     Col_2      |     Col_3     
        -------+----------------+----------------+----------------+----------------
         Col_0 |       1        | -1.35683396265 | -1.50055768303 | 0.981823365277 
         Col_1 | -1.35683396265 |       1        | 0.987829161147 | -2.02522022549 
         Col_2 | -1.50055768303 | 0.987829161147 |       1        | -2.17938510052 
         Col_3 | 0.981823365277 | -2.02522022549 | -2.17938510052 |       1        
        '''
        
        corrs = list()
        new_title = list()
        for i, data in enumerate(self._data):
            if hasattr(data, 'corr'):
                try:
                    corrs.append(data.corr())
                    new_title.append(self._sheets[i])
                except Exception, e:
                    warn('sheet:%s calculate corr() faild '%self._sheets[i] +\
                         'because %s'%e)
                    
        return DataSet(corrs, new_title)

    def count(self, x, area=all):
        '''Find one or more set of identification data in the specified area.

        Parameter
        ---------
        x : value or iterable object
            the value that you expect to statistic.

        area : two tuples in tuple (default=all)
            the area you expect to statistic. The first tuple in the main tuple
            represents the coordinates of the point in the upper left corner of
            the area. The second tuple represents the coordinates of the point
            in the lower right corner of the area.

        Return
        ------
        Counter : the number of each object in your setting area.

        Examples
        --------
        >>> data = dp.DataSet([[1, 2, 3, 4],
                               [2, None, 3, 4],
                               [3, 3, None, 5],
                               [7, 8, 9, 10]])
        >>> data.toframe()
        >>> data.count(3, all)
        sheet:sheet0
        ============
        DataSet(Counter({3: 4}))  # There are a total of four 3 in the entire data set.
        >>> data.count([3, None], ((0, 1), (2, 2)))
        sheet:sheet0
        ============
        DataSet(Counter({3: 3, None: 2}))
        '''
        if is_value(x):
            x = (x,)

        counter = list()
        counter_sheet = list()
        for i, data in enumerate(self._data):
            if hasattr(data, 'count'):
                try:
                    try:
                        counter.append(data.count(x, *area))
                    except TypeError:
                        counter.append(data.count(x))
                    counter_sheet.append(self._sheets[i])
                except Exception, e:
                    warn('sheet:%s count() object faild '%self._sheets[i] +\
                         'because %s'%e)
        return DataSet(counter, counter_sheet)

    def count_element(self, col=all):
        '''Count the frequency of values for each variable.
        You could count only a part of your data set with setting key-word(col)
        as a iterble inluding column number or variable names.

        Parameter
        ---------
        col : int, str, iter (default=all)
            an iterable object containing the columns' names or
             the number of column. ``all`` represents all columns.

        Return
        ------
        dict : the elements' frequency in each column.

        Examples
        --------
        >>> data = dp.DataSet([[1, 2, 3, 4],
                               [2, None, 3, 4],
                               [3, 3, None, 5],
                               [7, 8, 9, 10]])
        >>> data.tocol()
        >>> data.count_element(all)
        {'Col_2': Counter({3: 2, 9: 1, None: 1}),
        'Col_3': Counter({4: 2, 10: 1, 5: 1}),
        'Col_0': Counter({1: 1, 2: 1, 3: 1, 7: 1}),
        'Col_1': Counter({8: 1, 2: 1, 3: 1, None: 1})}
        '''
        counter = list()
        counter_sheet = list()
        for i, data in enumerate(self._data):
            if hasattr(data, 'count_element'):
                try:
                    counter.append(data.count_element(col))
                    counter_sheet.append(self._sheets[i])
                except Exception, e:
                    warn('sheet:%s count_element faild '%self._sheets[i] +\
                         'because %s'%e)
        return DataSet(counter, counter_sheet)

    def insert(self, index, item, miss_symbol=None):
        '''Insert a new record ``item`` in position ``index``.

        Parameter
        ---------
        index : int
            the position of new record.

        item : value or iter
            an iterable object containing new record.

        miss_symbol : value (default=None)
            the symbol of missing value in this record. It will be
            transformed to the inside miss value.
        '''
        for i, data in enumerate(self._data):
            if hasattr(data, 'insert'):
                try:
                    data.insert(index, item, miss_symbol)
                except Exception, e:
                    warn('sheet:%s insert() new records faild '%self._sheets[i] +\
                         'because %s'%e)

    def insert_col(self, variable_name, series, index, 
                   miss_symbol = None):
        '''Insert a new variable named ``variable_name`` with a sequence of data
        ``series`` in position ``index``.

        Parameter
        ---------
        variable_name : str
            the name of new column.

        series : sequence-like
            a sequence containing new variable values.

        index : int
            the position of new variable at.

        element_type : type (default='AUTO')
            the type of new variables' value.

        miss_symbol : value (default=None)
            the symbol of missing value in this sequence, which would be
            replaced by missing value inside.
        '''
        for i, data in enumerate(self._data):
            if hasattr(data, 'insert_col'):
                try:
                    data.insert_col(variable_name, series, index, miss_symbol)
                except Exception, e:
                    warn('sheet:%s insert_col() new variable faild '%self._sheets[i] +\
                         'because %s'%e)                
    def pop_miss_value(self, order='LINE'):
        '''Drop out all the records, which contain miss value, if ``order`` is
        ``LINE``. Drop out all the variables, which contain miss value,
        if ``order`` is ``COL``.

        Examples
        --------
        >>> import DaPy as dp
        >>> data = dp.DataSet([[1, 2, 3, 4],
                               [2, None, 3, 4],
                               [3, 3, None, 5],
                               [7, 8, 9, 10]])
        >>> data.tocol()
        
        There are two different keywords to use. Using keyword as ``LINE``:
        >>> data.pop_miss_value('line')
        sheet:sheet0
        ============
         Col_0 | Col_1
        -------+-------
           3   |   2   
           3   |  None 
          None |   3   
           5   |   4 

        Using keyword as 'COL':
        >>> data.pop_miss_value('col')
        sheet:sheet0
        ============
        Col_1: <2, None, 3, 8>
        Col_2: <3, 3, None, 9>
        '''
        pops, pops_name = list(), list()
        for i, data in enumerate(self._data):
            if hasattr(data, 'pop_miss_value'):
                try:
                    pops.append(data.pop_miss_value(order))
                    pops_name.append(self._sheets[i])
                except Exception, e:
                    warn('sheet:%s pop_miss_value() faild '%self._sheets[i] +\
                         'because %s'%e)
        return DataSet(pops, pops_name)
                
    def pick(self, conditions):
        '''Pick out the records which obey your conditions.

        Return
        ------
        DataSet : a copy of records which obeies your conditions.

        Examples
        --------
        >>> from DaPy import datasets
        >>> data = datasets.example()
        >>> data
        sheet:sample
        ============
        A_col: <3, 4, 1, 3, 4, ... ,4, 1, 3, 2, 3>
        B_col: <2, 3, 3, 3, 5, ... ,7, 9, 2, 9, 4>
        C_col: <1, 2, 4, 1, 4, ... ,8, 8, 6, 1, 1>
        D_col: <4, 2, 2, 2, 3, ... ,3, 3, 5, 5, 6>
        >>> A_1 = data.pick('A_col == 1')
        >>> large_A_B = data.pick('A_col > 2 and B_col > 3')
        >>>
        >>> A_1.toframe()
        >>> large_A_B.toframe()
        >>>
        >>> A_1 # pick the records which A_col is 1.
        sheet:sample
        ============
         A_col | B_col | C_col | D_col
        -------+-------+-------+-------
           1   |   3   |   4   |   2   
           1   |   9   |   8   |   3   
        >>> large_A_B # pick the records which A_col > 2 and B_col >3.
        sheet:sample
        ============
         A_col | B_col | C_col | D_col
        -------+-------+-------+-------
           4   |   5   |   4   |   3   
           6   |   4   |   3   |   2   
           4   |   7   |   8   |   3   
           3   |   4   |   1   |   6
        '''
        new_data = list()
        new_sheets = list()
        for i, (sheet, data) in enumerate(zip(self._sheets, self._data)):
            if hasattr(data, 'pick'):
                try:
                    new_data.append(data.pick(conditions))
                    new_sheets.append(sheet)
                except Exception, e:
                    warn('sheet:%s pick() faild '%self._sheets[i] +\
                         'because %s'%e)
        return DataSet(new_data, new_sheets)

    def pop(self, index=-1):
        '''Delete and return the record in position ``index``.

        Parameter
        ---------
        index : int
            the position of record you would like to drop out.

        Return
        ------
        record : tuple or list
        '''
        new_data = list()
        new_sheets = list()
        for i, (sheet, data) in enumerate(zip(self._sheets, self._data)):
            if hasattr(data, 'pop'):
                try:
                    new_data.append(data.pop(index))
                    new_sheets.append(sheet)
                except Exception, e:
                    warn('sheet:%s pop() current record faild '%self._sheets[i] +\
                         'because %s'%e)
        return DataSet(new_data, new_sheets)

    def pop_col(self, *variables):
        '''Delete and return all the value of each columns in ``variables``.
        Key-word(item) could assignment as a number or the variable name.

        Parameter
        ---------
        variables : int or str
            the title name or index of the columns you expect to pop.

        Return
        ------
        pop_columns : DataSet
            the DataSet consists of drop out columns.

        Examples
        --------
        >>> import DaPy as dp
        >>> data = dp.DataSet([[1,2,3,4],
                               [2,3,4,5],
                               [3,4,5,6],
                               [4,5,6,7],
                               [5,6,7,8]])
        >>> data.tocol()
        >>> data.pop_col('Col_2', 1)
        sheet:sheet0
        ============
        Col_1: <2, 3, 4, 5, 6>
        Col_2: <3, 4, 5, 6, 7>
        >>> data
        Col_0: <1, 2, 3, 4, 5>
        Col_3: <4, 5, 6, 7, 8>
        '''
        new_data = list()
        new_sheets = list()
        for i, (sheet, data) in enumerate(zip(self._sheets, self._data)):
            if hasattr(data, 'pop_col'):
                try:
                    new_data.append(data.pop_col(*variables))
                    new_sheets.append(sheet)
                except Exception, e:
                    warn('sheet:%s pop_col() current variable faild '%self._sheets[i] +\
                         'because %s'%e)
        return DataSet(new_data, new_sheets)

    def extend(self, other):
        '''extend your data by another object, new data as new records.
        '''
        if isinstance(other, DataSet):
            for extend_sheet in other._data:
                self.extend(extend_sheet)
            return
        
        for i, data in enumerate(self._data):
            if hasattr(data, 'extend'):
                try:
                    data.extend(other)
                except Exception, e:
                    warn('sheet:%s extend() new object faild '%self._sheets[i] +\
                         'because %s'%e)
                    
    def extend_col(self, other):
        '''extend your data by another object, new data as new variables of current records.
        '''
        if isinstance(other, DataSet):
            for extendcol_sheet in other._data:
                self.extend_col(extendcol_sheet)
            return

        for i, data in enumerate(self._data):
            if hasattr(data, 'extend_col'):
                try:
                    data.extend_col(other)
                except Exception, e:
                    warn('sheet:%s extend_col() new object faild '%self._sheets[i] +\
                         'because %s'%e)
                    
    def normalized(self, process='NORMAL', col=all, attr=None, get_attr=None):
        '''Normalized or standardlized your data in each col.

        Examples
        --------
        >>> from DaPy import datasets
        >>> data = datasets.example()
        >>> data.info
        sheet:sample
        ============
        1.  Structure: DaPy.SeriesSet
        2. Dimensions: Ln=12 | Col=4
        3. Miss Value: 0 elements
        4.   Describe: 
         Title | Miss | Min | Max | Mean | Std  |Dtype
        -------+------+-----+-----+------+------+-----
         A_col |  0   |  1  |  6  | 3.00 | 1.41 | list
         B_col |  0   |  1  |  9  | 4.33 | 2.67 | list
         C_col |  0   |  1  |  8  | 3.33 | 2.71 | list
         D_col |  0   |  2  |  6  | 3.50 | 1.45 | list
        ==============================================
        >>> data.normalized()
        >>> data.info
        sheet:sample
        ============
        1.  Structure: DaPy.SeriesSet
        2. Dimensions: Ln=12 | Col=4
        3. Miss Value: 0 elements
        4.   Describe: 
         Title | Miss | Min | Max | Mean | Std  |Dtype
        -------+------+-----+-----+------+------+-----
         A_col |  0   |  0  |  1  | 0.08 | 0.29 | list
         B_col |  0   |  0  |  1  | 0.17 | 0.39 | list
         C_col |  0   |  0  |  1  | 0.17 | 0.39 | list
         D_col |  0   |  0  |  1  | 0.08 | 0.29 | list
        ==============================================
        '''
        for i, data in enumerate(self._data):
            if hasattr(data, 'normalized'):
                try:
                    data.normalized(process, col, attr, get_attr)
                except Exception, e:
                    warn('sheet:%s normalized current data faild '%self._sheets[i] +\
                         'because %s'%e)

    def read(self, addr, dtype='col', sheet_name=None, miss_symbol='',
             miss_value=None, sep=None,
            first_line=1, title_line=0, prefer_type=float):
        '''This function could be used in loading data from a file and
        transform it into one of DaPy data structure.

        Parameters
        ----------
        addr : str
            the address of data file.

        dtype : str (default='col')
            the target data structure you prefer.

        miss_symbol : str (default='')
            the miss value symbol in this data file.

        miss_value : any ( default=None)
            the miss value symbol in your new data set.

        first_line : int (default=1)
            the first line which includes data values in this file.

        title_line : int (default=0)
            the line which includes your data's column names.
            tip: if there is no title in your data, used -1 represented,
              and, it will automatic create it.

        prefer_type : type-object (default=0)

        Examples
        --------
        >>> import DaPy as dp
        >>> data = dp.read('your_data_file.csv')
        '''
        def create_dataset(data, titles, miss_symbol, miss_value):
            if dtype.upper() == 'COL' or dtype.upper() == 'SERIESSET':
                self._data.append(SeriesSet(data, titles,
                                            miss_symbol, miss_value))
                self._types.append(SeriesSet)

            elif dtype.upper() == 'FRAME':
                self._data.append(Frame(data, titles,
                                        miss_symbol, miss_value))
                self._types.append(Frame)
                
            elif dtype.upper() == 'MATRIX':
                self._data.append(Matrix(data, titles))
                self._types.append(Matrix)
            else:
                raise RuntimeError('unrecognized symbol of data type')
            
        fpath, fname, fbase, ftype = parse_addr(addr)
        if ftype == 'db':
            try:
                import sqlite3 as sql3
            except ImportError:
                raise ImportError('DaPy uses `sqlite3` to access a database.')

            if not path.isfile(addr):
                raise IOError('can not find the database %s'%fname)
            
            with sql3.connect(addr) as conn:
                cur = conn.cursor()
                cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
                table_list = cur.fetchall()
                
                for table in table_list:
                    table = str(table[0])
                    cur.execute('SELECT * FROM %s'%table)
                    data = cur.fetchall()

                    cur.execute('PRAGMA table_info(%s)'%table)
                    titles = [title[1] for title in cur.fetchall()]
                    try:
                        create_dataset(data, titles, miss_symbol, miss_value)
                    except ValueError:
                        warn('%s is empty.'%titles)
                    
                    while table in self._sheets:
                        table += '_d'%len(self._sheets)
                    self._sheets.append(table)

        elif ftype == 'sav':
            try:
                import savReaderWriter
            except ImportError:
                raise ImportError('DaPy uses `savReaderWriter` to open a .sav file')

            with savReaderWriter.SavReader(addr) as reader:
                titles = reader.getSavFileInfo()[2]
                data = list(reader)
                create_dataset(data, titles, miss_symbol, miss_value)
                if sheet_name:
                    table = sheet_name
                else:
                    table = fbase
                while table in self._sheets:
                    table += '_%d'%len(self._sheets)
                self._sheets.append(table)
                
        elif ftype == 'xls' or ftype == 'xlsx':
            try:
                import xlrd
            except ImportError:
                raise ImportError('DaPy uses `xlrd` to parse a %s file'%file_type)

            book = xlrd.open_workbook(addr)
            for sheet in book.sheets():
                data = [0] * (sheet.nrows - first_line)
                for index, i in enumerate(range(first_line, sheet.nrows)):
                    data[index] = [cell.value for cell in sheet.row(i)]

                if title_line >= 0:
                    try:
                        titles = [cell.value for cell in sheet.row(title_line)]
                    except IndexError:
                        titles = None
                else:
                    titles = None
                    
                try:
                    create_dataset(data, titles, miss_symbol, miss_value)
                except ValueError:
                    warn('%s is empty.'%sheet)
                
                if sheet_name:
                    table = str(sheet_name) + sheet.name
                else:
                    table = sheet.name
                while table in self._sheets:
                    table += '_%d'%len(self._sheets)
                self._sheets.append(table)

        elif ftype == 'txt' or ftype == 'csv':
            if not isinstance(sep, str):
                split_dic = {'csv':',', 'txt':'\t'}
                sep = split_dic[ftype]
            
            if dtype.upper() == 'COL' or dtype.upper() == 'SERIESSET':
                data = SeriesSet(miss_value=miss_value)
                data.read_text(addr, first_line, miss_symbol,
                                     title_line, sep, prefer_type)
                self._types.append(SeriesSet)
                
            elif dtype.upper() == 'FRAME':
                data = Frame(miss_value=miss_value)
                data.read_text(addr, first_line, miss_symbol,
                                     title_line, sep, prefer_type)
                self._types.append(Frame)

            elif dtype.upper() == 'MATRIX':
                data = Matrix()
                data.read_text(addr, first_line, sep)
                self._types.append(Matrix)

            else:
                raise RuntimeError('unrecognized symbol of data type')
            
            self._data.append(data)
            if sheet_name:
                table = sheet_name
            else:
                table = fbase
            while table in self._sheets:
                table += '_%d'%len(self._sheets)
            self._sheets.append(table)
        else:
            raise ValueError('DaPy singly supports file types as'+\
                                 '(xls, xlsx, csv, txt, pkl, db, sav).')

    def reverse(self, axis='sheet'):
        '''Reverse your data set.

        Example
        -------
        >>> data = dp.DataSet([[1,2,3,4],
                               [2,3,4,5],
                               [3,4,5,6],
                               [4,5,6,7],
                               [5,6,7,8]])
        >>> data.tocol()
        >>> data.reverse()
        '''
        if axis.upper() == 'SHEET':
            self._data.reverse()
            self._sheets.reverse()
            self._types.reverse()
            return

        for data in self._data:
            if hasattr(data, 'reverse'):
                data.reverse(axis)

    def replace(self, col, *arg):
        '''Replace the values from 'key' to 'target' in columns 'col'.

        Examples
        --------
        >>> data = dp.DataSet([['Andy', 'Mary', 'Peter'],
                               ['Henry', 'Char', 'Iris'],
                               ['Peter', 'Mary', 'Andy'],
                               ['Peter', 'Cindy', 'Julia']])
        >>> data.toframe()
        >>> data.replace(0, ('Peter', 'Mary'), ('Andy', 'Mary'))
        >>> data
        sheet:sheet0
        ============
         Col_0 | Col_1 | Col_2
        -------+-------+-------
          Mary |  Mary | Peter 
         Henry |  Char |  Iris 
          Mary |  Mary |  Andy 
          Mary | Cindy | Julia 
        >>> data.replace(['Col_0', 2], ('Mary', 'Peter'))
        >>> data
        sheet:sheet0
        ============
         Col_0 | Col_1 | Col_2
        -------+-------+-------
          Mary |  Mary | Peter 
         Henry |  Char |  Iris 
          Mary |  Mary |  Andy 
          Mary | Cindy | Julia 
        >>> data.replace(all, ('Peter', 'Mary'))
        >>> data
        sheet:sheet0
        ============
         Col_0 | Col_1 | Col_2
        -------+-------+-------
          Mary |  Mary |  Mary 
         Henry |  Char |  Iris 
          Mary |  Mary |  Andy 
          Mary | Cindy | Julia 
        '''
        for data in self._data:
            if hasattr(data, 'replace'):
                data.replace(col, *arg)

    def shuffles(self):
        ''' Mess up your data
        '''
        for data in self._data:
            if hasattr(data, 'shuffles'):
                data.shuffles()
            elif hasattr(data, 'shuffle'):
                data.shuffle()

    def sort(self, *orders):
        '''You could easily sorted your data set with this function.

        You will be asked to offer at least one ordering conditions.
        The parameter should be like a tuple or a list with two elements,
        on behalf of the key value and arrangement condition (key, arrangement).
        e.g. ('D_col', 'ASC') means that ascending ordered the data set
        with A_col.

        Examples
        --------
        >>> from DaPy import datasets
        >>> data = datasets.example()
        >>> data
        A_col: <3, 4, 1, 3, 4, ... ,4, 1, 3, 2, 3>
        B_col: <2, 3, 3, 3, 5, ... ,7, 9, 2, 9, 4>
        C_col: <1, 2, 4, 1, 4, ... ,8, 8, 6, 1, 1>
        D_col: <4, 2, 2, 2, 3, ... ,3, 3, 5, 5, 6>
        >>> data.sort(('B_col', 'DESC'), ('A_col', 'ASC'))
        >>> data
        A_col: <1, 2, 4, 4, 3, ... ,3, 4, 3, 3, 2>
        B_col: <9, 9, 7, 5, 4, ... ,3, 3, 2, 2, 1>
        C_col: <8, 1, 8, 4, 1, ... ,1, 2, 1, 6, 1>
        D_col: <3, 5, 3, 3, 6, ... ,2, 2, 4, 5, 5>
        '''
        for data in self._data:
            if hasattr(data, 'sort'):
                data.sort(*orders)

    def save(self, addr, code='utf-8'):
        '''Save the DataSet to a file.
        '''
        fpath, fname, fbase, ftype = parse_addr(addr)
        if ftype == 'csv' or ftype == 'txt':
            for data, sheet in zip(self._data, self._sheets):
                if len(self._data) > 1:
                    addr_ = fpath + fbase + '_' + sheet + '.' + ftype
                else:
                    addr_ = addr
                    
                with open(addr_, 'w') as f:
                    if hasattr(data, 'titles'):
                        f.write(str(data.titles).encode(code) + u'\n')

                    if isinstance(self._data, (Frame, Matrix, SeriesSet)):
                        for line in data:
                            f.write(','.join(map(str, line)).encode(code))
                            f.write(u'\n')

                    elif is_iter(self._data[0]):
                        for line in data:
                            f.write(','.join(map(str, line)).encode(code) + '\n')
                    else:
                        for record in data:
                            f.write(str(record).encode(code) + u'\n')
                        
        elif ftype == 'xls' or ftype == 'xlsx':
            try:
                import xlwt
            except ImportError:
                raise ImportError('DaPy uses xlwt library to save a `xls` file.')

            workbook = xlwt.Workbook(encoding = encode)
            for sheet, data in zip(self._sheets, self._data):
                worksheet = workbook.add_sheet(sheet)
                if hasattr(data, 'titles'):
                    for i, value in enumerate(data.titles):
                        worksheet.write(0, i, value.encode(code))

                if isinstance(data, (Frame, Matrix, SeriesSet)) or\
                   is_iter(data[0]):
                    for i, row in enumerate(self._data, 1):
                        for j, value in enumerate(row):
                            worksheet.write(i, j, str(value.encode(code)))
                else:
                    for i, value in enumerate(self._data):
                        worksheet.write(i, 1, str(value).encode(code))
            workbook.save(addr)
        else:
            raise ValueError('unrecognized file type')
        
    def toframe(self, miss_value=None):
        '''Transform all of the stored data structure to DaPy.Frame
        '''
        for i, data in enumerate(self._data):
            if isinstance(data, Frame):
                continue
            try:
                if hasattr(data, 'columns'):
                    if hasattr(data, 'miss_value'):
                        self._data[i] = Frame(data, data.columns,
                                           miss_value=data.miss_value)
                    else:
                        self._data[i] = Frame(data, data.columns)
                else:
                    self._data[i] = Frame(data, miss_value=miss_value)
            except:
                warn('sheet:%s can not transform to SeriesSet.'%self._sheets[i])
            self._types[i] = Frame

    def tocol(self, miss_value=None):
        '''Transform all of the stored data structure to DaPy.SeriesSet
        '''
        for i, data in enumerate(self._data):
            if isinstance(data, SeriesSet):
                continue
            try:
                if hasattr(data, 'columns'):
                    if hasattr(data, 'miss_value'):
                        self._data[i] = SeriesSet(data, data.columns,
                                           miss_value=data.miss_value)
                    else:
                        self._data[i] = SeriesSet(data, data.columns)
                else:
                    self._data[i] = SeriesSet(data, miss_value=miss_value)
            except:
                warn('sheet:%s can not transform to SeriesSet.'%self._sheets[i])
            self._types[i] = SeriesSet

    def tomat(self):
        '''Transform all of the stored data structure to DaPy.Matrix
        '''
        for i, data in enumerate(self._data):
            if isinstance(data, Matrix):
                continue

            try:
                self._data[i] = Matrix(data)
            except:
                warn('sheet:%s can not transform to SeriesSet.'%self._sheets[i])
            self._types[i] = Matrix

