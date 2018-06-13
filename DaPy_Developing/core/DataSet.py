from collections import namedtuple, deque, OrderedDict, Counter, Iterable
from copy import deepcopy
try:
    import cPickle as pkl
except IOError:
    import Pickle as pkl
from base import is_value, is_iter, SeriesSet, Frame, Matrix, Table
from warnings import warn
from os import path

__all__ = ['DataSet']

def parse_addr(addr):
    file_path, file_name = path.split(addr)
    if file_name.count('.') > 1:
        file_base = '.'.join(file_name.split('.')[:-1])
        file_type = file_name.split('.')[-1]
    else:
        file_base, file_type = file_name.split('.')
    return file_path, file_name, file_base, file_type

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

    _sheets : list
        the list stored all the names of each sheet.

    _types : list
        the list stored all the type of each sheet.

    _dim : 

    Examples
    --------
    >>> import DaPy as dp
    >>> data = dp.DataSet([[1, 2, 3], [2, 3, 4]])
    >>> data.tocol()
    >>> data
    Col_0: <1, 2>
    Col_1: <2, 3>
    Col_2: <3, 4>
    >>> data.info
    1.  Structure: DaPy.SeriesSet
    2. Dimensions: Ln=2 | Col=3
    3. Miss Value: 0 elements
    4.   Describe: 
     Title | Miss | Min | Max | Mean | Std  |Dtype
    -------+------+-----+-----+------+------+-----
     Col_0 |  0   |  1  |  2  | 1.50 | 0.71 | int 
     Col_1 |  0   |  2  |  3  | 2.50 | 0.71 | int 
     Col_2 |  0   |  3  |  4  | 3.50 | 0.71 | int 
    ==============================================
    '''
    __all__ = ['data', 'titles', 'info', 'dim', 'append', 'append_col',
               'combine', 'corr', 'count', 'count_element', 'drop_miss_value',
               'extend', 'insert', 'insert_col', 'pick', 'pop', 'pop_col',
               'merge', 'normalized', 'readtable', 'readframe', 'readcol',
               'readmatrix', 'readpkl', 'reverse', 'replace', 'shuffles',
               'sort', 'tocsv', 'topkl', 'tomatrix', 'toframe', 'tocol',
               'totable', 'update']

    def __init__(self, obj=None, sheet='sheet0'):
        '''

        Parameter
        ---------
        obj : array-like (default=None)
            initialized your data from a data structure, such as dict(), list()
            Frame(), SeriesSet(), Matrix(), DataSet() or Table().
            
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
            
        elif isinstance(obj, (Matrix, SeriesSet, Table, Frame)):
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
            self._types = [type(d) for d in self._data]
            if len(set(self._sheets)) != len(self._data):
                raise ValueError("the number of sheets' names do not enough.")
                        
    @property
    def data(self):
        if len(self._data) == 1:
            return self._data[0]
        return self._data

    @property
    def titles(self):
        if len(self._data) > 1:
            new_ = list()
            for i, data in enumerate(self._data):
                if hasattr(data, 'titles'):
                    new_.append([self._sheets[i]] + data.titles)
                else:
                    new_.append([self._sheets[i], None])
            return Table(new_, ['sheet name', 'Titles'])
        
        elif len(self._data) == 1:
            if hasattr(self._data[0], 'titles'):
                return self._data[0].titles
        return None

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
            if isinstance(data, (Frame, SeriesSet, Frame, Matrix, Table)):
                data.info
            else:
                warn('%s has no info() function'%type(data))
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
            reprs += 'SHEET:' + self._sheets[i] + '\n'
            reprs += '=' * (len(self._sheets[i]) + 6) + '\n'
            if not isinstance(self._data[i], (SeriesSet, Frame, Matrix, Table)):
                reprs += 'DataSet(%s)'%self._data[i].__repr__() + '\n\n'
            else:
                reprs += self._data[i].__repr__() + '\n\n'
        return reprs[:-2]
    
    def __len__(self):        
        if len(self._data) == 1:
            if hasattr(data, 'shape'):
                return data.shape[0]
            return len(data)
        return len(self._data)
        
    def __getitem__(self, pos):
        if len(self._data) == 1 and not self.__contains__(pos):
            return self._data[0][pos]
        
        if isinstance(pos, slice):
            return self.__getslice__(pos.__getattribute__('start'),
                            pos.__getattribute__('stop'))

        if self.__contains__(pos): 
            return self._data[self._sheets.index(pos)]

        if isinstance(pos, int):
            return self._data[pos]

        raise ValueError('unrecognized symbol as %s, use string '%pos +\
                         'represented titles or integer which is less than the ' +\
                         'DataSet size.')

    def __getslice__(self, i, j):
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
        return DataSet(self._data[i: j + 1], self._sheets[i: j+1])

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if key not in self._sheets:
                self._data.append(value)
                self._types.append(type(value))
                name = 'sheet' + str(len(self._data))
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
            del self._sheets[index], self._data[index], self._type[index]
        else:
            self._data.__delitem__(key)

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

    def append(self, item, miss_symbol=None):
        '''Append a new record ``item`` at the tail of the data set.

        Parameter
        ---------
        item : iterable or value
            append this item as a new record into the origin dataset,
            if item is an iterable object, we will direct append. Otherwise,
            we will build a iterable which contains this value inside for you.

        sheet : int or str (defualt=0)
            the name or index represented the sheet
            which you would like to opearte.

        miss_symbol : value (defualt=None)
            the symbol of missing value in this record. It will be
            transformed to the inside miss value.
        '''
        for sheet in self._data:
            if hasattr(sheet, 'append'):
                try:
                    sheet.append(item, miss_symbol)
                except TypeError:
                    sheet.append(item)

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
        for sheet in self._data:
            if hasattr(sheet, 'append'):
                try:
                    sheet.append_col(variable_name, series,
                                     element_type, miss_symbol)
                except Exception, e:
                    warn(e)
                    

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
d                               [7, 8, 9, 10]])
        >>> data.tocol()
        >>> data.corr()
        Col_0: <1, -1.35683396265, -1.50055768303, 0.981823365277>
        Col_1: <-1.35683396265, 1, 0.987829161147, -2.02522022549>
        Col_2: <-1.50055768303, 0.987829161147, 1, -2.17938510052>
        Col_3: <0.981823365277, -2.02522022549, -2.17938510052, 1>
        
        '''
        
        corrs = list()
        new_title = list()
        for i, data in enumerate(self._data):
            if hasattr(data, 'corr'):
                corrs.append(data.corr())
                new_title.append(self._sheets[i])
                
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
        Counter({3: 4})  # There are a total of four 3 in the entire data set.
        >>> data.count([3, None], (0, 1), (2, 2))
        Counter({3: 3, None: 2})
        '''
        if is_value(x):
            x = (x,)

        counter = list()
        counter_sheet = list()
        for i, data in enumerate(self._data):
            if hasattr(data, 'count'):
                try:
                    counter.append(data.count(x, area))
                except TypeError:
                    counter.append(data.count(x))
                counter_sheet.append(self._sheets[i])
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
                counter.append(data.count_element(col))
                counter_sheet.append(self._sheets[i])
        return DataSet(counter, counter_sheet)
    
    def drop_miss_value(self, order='LINE'):
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
        >>> data.toframe()
        
        There are two different keywords to use. Using keyword as ``LINE``:
        >>> data.drop_miss_value('LINE')
        >>> data
        (1, 2, 3, 4)
        (7, 8, 9, 10)

        Using keyword as 'COL':
        >>> data.drop_miss_value('COL')
        >>> data
        (1, 4)
        (2, 4)
        (3, 5)
        (7, 10)
        '''
        for i, data in enumerate(self._data):
            if hasattr(data, 'drop_miss_value'):
                data.drop_miss_value(order)

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
        for data in self._data:
            data.insert(index, item, miss_symbol)

    def insert_col(self, variable_name, series, index, element_type='AUTO',
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
        for data in self._data:
            data.insert_col(variable_name, series, index, element_type,
                            miss_symbol)

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
        A_col: <3, 4, 1, 3, 4, ... ,4, 1, 3, 2, 3>
        B_col: <2, 3, 3, 3, 5, ... ,7, 9, 2, 9, 4>
        C_col: <1, 2, 4, 1, 4, ... ,8, 8, 6, 1, 1>
        D_col: <4, 2, 2, 2, 3, ... ,3, 3, 5, 5, 6>
        >>> A_1 = data.pick('A_col == 1')
        >>> A_1.info
        1.  Structure: DaPy.SeriesSet
        2. Dimensions: Ln=2 | Col=4
        3. Miss Value: 0 elements
        4.   Describe: 
         Title | Miss | Min | Max | Mean | Std  |Dtype
        -------+------+-----+-----+------+------+-----
         A_col |  0   |  1  |  1  | 1.00 | 0.00 | int 
         B_col |  0   |  3  |  9  | 6.00 | 4.24 | int 
         C_col |  0   |  4  |  8  | 6.00 | 2.83 | int 
         D_col |  0   |  2  |  3  | 2.50 | 0.71 | int 
        ==============================================
        '''
        new_data = list()
        new_sheets = list()
        for sheet, data in zip(self._sheets, self._data):
            if hasattr(data, 'pick'):
                new_data.append(data.pick(conditions))
                new_sheets.append(sheet)
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
        for sheet, data in zip(self._sheets, self._data):
            if hasattr(data, 'pop'):
                new_data.append(data.pop(index))
                new_sheets.append(sheet)
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
        Col_1: <2, 3, 4, 5, 6>
        Col_2: <3, 4, 5, 6, 7>
        >>> data
        Col_0: <1, 2, 3, 4, 5>
        Col_3: <4, 5, 6, 7, 8>
        '''
        data = self._data.pop_col(*variables)
        self._titles = self._data.titles
        return DataSet(data)

    def extend(self, other):
        '''extend your data sheet by another object.
        '''
        for data in self._data:
            if hasattr(data, 'extend'):
                data.extend(other)

    def normalized(self, process='NORMAL', col=all, attr=None, get_attr=None):
        '''Normalized or standardlized your data in each col.

        Examples
        --------
        >>> from DaPy import datasets
        >>> data = datasets.example()
        >>> data.normalized()
        >>> data
        A_col: <0.4, 0.6, 0.0, 0.4, 0.6, ... ,0.6, 0.0, 0.4, 0.2, 0.4>
        B_col: <0.125, 0.25, 0.25, 0.25, 0.5, ... ,0.75, 1.0, 0.125, 1.0, 0.375>
        C_col: <0.0, 0.142857142857, 0.428571428571, 0.0, 0.428571428571, ... ,1.0, 1.0, 0.714285714286, 0.0, 0.0>
        D_col: <0.5, 0.0, 0.0, 0.0, 0.25, ... ,0.25, 0.25, 0.75, 0.75, 1.0>
        '''
        for data in self._data:
            if hasattr(data, 'normalized'):
                data.normalized(col, attr, get_attr, process)

    def read(self, addr, dtype='col', sheet_name=None, miss_symbol='',
             miss_value=None, sep=None,
            first_line=1, title_line=0, type_float=False, type_str=False):
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

        type_float : bool (default=False)
            prefer to transform an integer into float, not int.

        type_str : bool (default=False)
            do nothing about transforming with your values.

        Examples
        --------
        >>> import DaPy as dp
        >>> data = dp.read('your_data_file.csv')
        '''
        fpath, fname, fbase, ftype = parse_addr(addr)

        if ftype == 'pkl':
            with open(addr) as f:
                datas = pkl.load(f)
            
            self._data, self._sheets = data['_data'], data['_sheets']
            self._types = data['_types']

            if data['Type'] == 'SeriesSet':
                self._data.append(SeriesSet(d, c, miss_value=data['miss_value']))
                self._types.append(SeriesSet)

            elif data['Type'] == 'Frame':
                self._data.append(Frame(d, c, miss_value=data['miss_value']))
                self._types.append(Frame)

            elif data['Type'] == 'Table':
                self._data.append(Table(d, c))
                self._types.append(Table)

            elif data['Type'] == 'Matrix':
                self._data.append(Matrix(d, c))
                self._types.append(Matrix)

            else:
                self._data.append(d)
                self._types.append(type(d))
                
        elif ftype == 'xls' or ftype == 'xlsx':
            try:
                import xlrd
            except ImportError:
                ImportError('DaPy uses xlrd to parse a %s file'%file_type)

            book = xlrd.open_workbook(addr)
            for sheet in book.sheets():
                data = [0] * (sheet.nrows - first_line)
                for index, i in enumerate(range(first_line, sheet.nrows)):
                    data[index] = [cell.value for cell in sheet.row(i)]

                if title_line >= 0:
                    titles = [cell.value for cell in sheet.row(title_line)]
                else:
                    titles = None

                if dtype.upper() == 'COL' or dtype.upper() == 'SERIESSET':
                    self._data.append(
                        SeriesSet(data, titles, None, miss_symbol, miss_value))
                    self._types.append(SeriesSet)

                elif dtype.upper() == 'FRAME':
                    self._data.append(
                        Frame(data, titles, None, miss_symbol, miss_value))
                    self._types.append(Frame)
                    
                elif dtype.upper() == 'TABLE':
                    self._data.append(Table(data, titles, miss_value))
                    self._types.append(Table)
                    
                elif dtype.upper() == 'MATRIX':
                    self._data.append(Matrix(data, titles))
                    self._types.append(Matrix)
                    
                else:
                    raise RuntimeError('unrecognized symbol of data type')
                if sheet_name:
                    self._sheets.append(str(sheet_name) + sheet.name)
                else:
                    self._sheets.append(sheet.name)

        elif ftype == 'txt' or ftype == 'csv':
            if not isinstance(sep, str):
                split_dic = {'csv':',', 'txt':'\t'}
                sep = split_dic[ftype]
            
            if dtype.upper() == 'COL' or dtype.upper() == 'SERIESSET':
                data = SeriesSet(miss_value=miss_value)
                data.read_text(addr, first_line, miss_symbol,
                                     title_line, sep, type_float, type_str)
                self._types.append(SeriesSet)
                
            elif dtype.upper() == 'FRAME':
                data = Frame(None, miss_value=miss_value)
                data.read_text(addr, first_line, miss_symbol,
                                     title_line, sep, type_float, type_str)
                self._types.append(Frame)

            elif dtype.upper() == 'TABLE':
                data = Table(miss_value=miss_value)
                data.read_text(addr, first_line, miss_symbol,
                                     title_line, sep, type_str)
                self._types.append(Table)

            elif dtype.upper() == 'MATRIX':
                data = Matrix()
                data.read_text(addr, first_line, title_line, sep)
                self._types.append(Matrix)

            else:
                raise RuntimeError('unrecognized symbol of data type')
            
            self._data.append(data)
            if sheet_name:
                self._sheets.append(sheet_name)
            else:
                self._sheets.append(fbase)
        else:
            raise ValueError('DaPy singly supports file types'+\
                                 '(xls, xlsx, csv, txt, pkl).')

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
        ('Mary', 'Mary', 'Peter')
        ('Henry', 'Char', 'Iris')
        ('Mary', 'Mary', 'Andy')
        ('Mary', 'Cindy', 'Julia')
        >>> data.replace(['Col_0', 2], ('Mary', 'Peter'))
        >>> data
        ('Peter', 'Mary', 'Peter')
        ('Henry', 'Char', 'Iris')
        ('Peter', 'Mary', 'Andy')
        ('Peter', 'Cindy', 'Julia')
        >>> data.replace(all, ('Peter', 'Mary'))
        ('Mary', 'Mary', 'Mary')
        ('Henry', 'Char', 'Iris')
        ('Mary', 'Mary', 'Andy')
        ('Mary', 'Cindy', 'Julia')
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

    def save(self, addr, encode='ascii'):
        '''Save the DataSet to a file.
        '''
        fpath, fname, fbase, ftype = parse_addr(addr)
        if ftype == 'csv' or ftype == 'txt':
            for sheet, data in zip(self._data, self._sheets):
                if len(self._data) > 1:
                    addr_ = fpath + fbase + '_' + sheet + '.' + ftype
                else:
                    addr_ = addr
                    
                with open(addr_, 'w') as f:
                    if hasattr(data, 'titles'):
                        f.write(str(data.titles) + '\n')

                    if isinstance(self._data, (Frame, Table, Matrix, SeriesSet)):
                        for line in data:
                            f.write(','.join(map(str, line)))
                            f.write('\n')

                    elif is_iter(self._data[0]):
                        for line in data:
                            f.write(','.join(map(str, line)) + '\n')
                    else:
                        for record in data:
                            f.write(str(record) + '\n')
                        
        elif ftype == 'pkl':
            ########################
            if isinstance(self._data,
                          (Frame, Table, Matrix, SeriesSet)):
                info = {'Type': self._type.split('.')[1]}
                info['_data'] = self._data
                info['_sheets'] = self._sheets
                if info['Type'] == 'SeriesSet' or info['Type'] == 'Frame':
                    info['miss_value'] = self._data._miss_value
            else:
                info = {'Type': self._type}
                info['data'] = self._data
                info['columns'] = self.titles

            with open(addr, 'wb') as f:
                pkl.dump(info, f)

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
                        worksheet.write(0, i, value)

                if isinstance(self._data, (Frame, Matrix, SeriesSet, Table)) or\
                   is_iter(self._data[0]):
                    for i, row in enumerate(self._data, 1):
                        for j, value in enumerate(row):
                            worksheet.write(i, j, value)
                else:
                    for i, value in enumerate(self._data):
                        worksheet.write(i, 1, value)
            workbook.save(addr)
        else:
            raise ValueError('unrecognized file type')
        
    def toframe(self, miss_value=None):
        '''Transform all of the stored data structure to DaPy.Frame
        '''
        for i, data in enumerate(self._data):
            if isinstance(data, Frame):
                continue

            if hasattr(data, 'titles'):
                if hasattr(data, 'miss_value'):
                    self._data[i] = Frame(data, data.titles,
                                       miss_value=data.miss_value)
                else:
                    self._data[i] = Frame(data, data.titles)
            else:
                self._data[i] = Frame(data, miss_value=miss_value)
            self._types[i] = Frame

    def tocol(self, miss_value=None):
        '''Transform the data structure to DaPy.SeriesSet'''
        for i, data in enumerate(self._data):
            if isinstance(data, SeriesSet):
                continue

            if hasattr(data, 'titles'):
                if hasattr(data, 'miss_value'):
                    self._data[i] = SeriesSet(data, data.titles,
                                       miss_value=data.miss_value)
                else:
                    self._data[i] = SeriesSet(data, data.titles)
            else:
                self._data[i] = SeriesSet(data, miss_value=miss_value)
            self._types[i] = SeriesSet

    def totable(self):
        '''Transform the data structure to DaPy.Table'''
        for i, data in enumerate(self._data):
            if isinstance(data, Table):
                continue

            if hasattr(data, 'titles'):
                self._data[i] = Table(data, data.titles)
            else:
                self._data[i] = Table(data)
            self._types[i] = Table

    def tomatrix(self):
        '''Transform the data structure to DaPy.Matrix'''
        for i, data in enumerate(self._data):
            if isinstance(data, Matrix):
                continue

            if hasattr(data, 'titles'):
                self._data[i] = Matrix(data, data.titles)
            else:
                self._data[i] = Matrix(data)
            self._types[i] = Table

