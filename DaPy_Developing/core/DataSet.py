from collections import namedtuple, deque, OrderedDict, Counter, Iterable
from copy import deepcopy
try:
    import cPickle as pkl
except IOError:
    import Pickle as pkl
from basic_fun import is_value, is_iter
from basic_2d import SeriesSet, Frame, Matrix, Table
import time

__all__ = ['DataSet']

class DataSet(object):
    '''A general two-dimentional data structure supports users easily
    to opearte the other basic DaPy 2D data structures. It supports
    users to process any inherence data structure in a same way and use
    the Pythonic Syntax. DataSet is one of the fundamantal data structure in
    DaPy.

    Parameter
    ---------
    object_ : array-like (default=None)
        initialized your data from a data structure, such as dict(), list()
        Frame(), SeriesSet(), Matrix(), DataSet() or Table().
        
    miss_value : object (default=None)
        the missing value symbol in your DataSet.

    titles : iterable, str (defualt=None)
        the variable names of your DataSet.

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

    __class__ = 'DataSet'
    
    def __init__(self, object_=None, titles=None):

        # We set some basic variables as follow.
        if isinstance(object_, DataSet):
            self._data = deepcopy(object_._data)
            self._titles = deepcopy(object_._titles)
            self._type = deepcopy(object_._type)
            
        elif isinstance(object_, (Matrix, SeriesSet, Table, Frame)):
            self._data = object_
            self._titles = object_._columns
            if isinstance(object_, Matrix):
                self._type = 'DaPy.Matrix'
            elif isinstance(object_, SeriesSet):
                self._type = 'DaPy.SeriesSet'
            elif isinstance(object_, Table):
                self._type = 'DaPy.Table'
            else:
                self._type = 'DaPy.Frame'

        elif object_ is None:
            self._data = object_
            self._titles = titles
            self._type = 'DaPy.DataSet'

        elif not isinstance(object_, (str, int, float, long, bool)):
            self._data = object_
            self._titles = titles
            self._type = str(type(object_))
            
        else:
            raise TypeError('DataSet can not package this object.')
                        
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if isinstance(value, DataSet):
            raise TypeError('can not package another DataSet object.')
        
        if isinstance(value, (Matrix, SeriesSet, Table, Frame)):
            self._data = value

        elif is_iter(value):
            self._data = value

        else:
            raise TypeError('can not package this object into DataSet.')

    @property
    def titles(self):
        return self._titles

    @titles.setter
    def titles(self, value):
        if not is_iter(value):
            value = [value, ]
        if isinstance(self._data, (Frame, SeriesSet, Table, Matrix)):
            if self._data.shape.Col != len(value):
                raise IndexError('the titles do not have enough  dimensions.')
        self._title = value
        self._data.titles = value
        
    @property
    def shape(self):
        if isinstance(self._data, (Frame, SeriesSet, Frame, Matrix, Table)):
            return self._data.shape
        return (len(self._data), 1)

    @property
    def info(self):
        if isinstance(self._data, (Frame, SeriesSet, Frame, Matrix, Table)):
            self._data.info
        return None

    def __contains__(self, e):
        return e in self._data

    def __repr__(self):
        if not isinstance(self._data, (SeriesSet, Frame, Matrix, Table)):
            return 'DataSet(%s)'%self._data.__repr__()
        return self._data.__repr__()
    
    def __len__(self):
        if isinstance(self._data, (Frame, SeriesSet, Matrix, Table)):
            return self._data.shape.Ln
        return len(self._data)
        
    def __getitem__(self, pos):
        return self._data[pos]

    def __setitem__(self, key, value):
        self._data.__setitem(key, value)

    def __delitem__(self, key):
        self._data.__delitem__(key)

    def __iter__(self):
        for item in self._data:
            yield item

    def __reversed__(self):
        self._data.reverse()

    def append(self, item, miss_symbol=None):
        '''Append a new record ``item`` at the tail of the data set.

        Parameter
        ---------
        item : iterable or value
            append this item as a new record into the origin dataset,
            if item is an iterable object, we will direct append. Otherwise,
            we will build a iterable which contains this value inside for you.

        miss_symbol : value
            the symbol of missing value in this record. It will be
            transformed to the inside miss value.
        '''
        self._data.append(item, miss_symbol)

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
        self._data.append_col(variable_name, series, element_type, miss_symbol)
        self._titles = self._data.titles

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
        Col_0: <1, -1.35683396265, -1.50055768303, 0.981823365277>
        Col_1: <-1.35683396265, 1, 0.987829161147, -2.02522022549>
        Col_2: <-1.50055768303, 0.987829161147, 1, -2.17938510052>
        Col_3: <0.981823365277, -2.02522022549, -2.17938510052, 1>
        '''
        return DataSet(self._data.corr())

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
            
        return self._data.count(x, *area)

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
        return self._data.count_element(col)
    
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
        self._data.drop_miss_value(order)
        self._titles = self._data.titles

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
        self._data.insert(index, item, miss_symbol)

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
        self._data.insert_col(variable_name, series, index, element_type, miss_symbol)
        self._titles = self._data.titles

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
        return DataSet(self._data.pick(conditions), self._titles)

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
        data = self._data.pop(index)
        return data

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

    def merge(self, other):
        '''Merge another object with this DataSet.
        '''
        if isinstance(other, DataSet):
            other = other._data
        self._data.merge(other)

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
        return self._data.normalized(col, attr, get_attr, process)

    def read(self, addr, dtype='col', miss_symbol='', miss_value=None, sep=None,
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
        file_type = addr.split('.')[-1]
        if file_type == 'pkl':
            with open(addr) as f:
                data = pkl.load(f)

            d, c = data['data'], data['columns']
            self._titles = c

            if data['Type'] == 'SeriesSet':
                self._data = SeriesSet(d, c, miss_value=data['miss_value'])
                self._type = 'DaPy.SeriesSet'

            elif data['Type'] == 'Frame':
                self._data = Frame(d, c, miss_value=data['miss_value'])
                self._type = 'DaPy.Frame'

            elif data['Type'] == 'Table':
                self._data = Table(d, c)
                self._type = 'DaPy.Table'

            elif data['Type'] == 'Matrix':
                self._data = Matrix(d, c)
                self._type = 'DaPy.Matrix'

            else:
                self._data = d
                self._type = 'DaPy.DataSet'

        elif file_type == 'xls' or file_type == 'xlsx':
            try:
                import xlrd
            except ImportError:
                ImportError('DaPy uses xlrd to parse a %s file'%file_type)

            book = xlrd.open_workbook(addr)
            sheet = book.sheet_by_index(0)

            data = [0] * (sheet.nrows - first_line)
            for index, i in enumerate(range(first_line, sheet.nrows)):
                data[index] = [cell.value for cell in sheet.row(i)]

            if self._titles:
                titles = self._titles
            elif title_line >= 0:
                titles = [cell.value for cell in sheet.row(title_line)]
            else:
                titles = None

            if dtype.upper() == 'COL' or dtype.upper() == 'SERIESSET':
                self._data = SeriesSet(data, titles, None, miss_symbol, miss_value)
            elif dtype.upper() == 'FRAME':
                self._data = Frame(data, titles, None, miss_symbol, miss_value)
            elif dtype.upper() == 'TABLE':
                self._data = Table(data, titles, miss_value)
            elif dtype.upper() == 'MATRIX':
                self._data = Matrix(data, titles)
            else:
                raise RuntimeError('unrecognized symbol of data type')

        elif file_type == 'txt' or file_type == 'csv':
            if not isinstance(sep, str):
                split_dic = {'csv':',', 'txt':'\t'}
                sep = split_dic[addr.split('.')[-1]]
            
            if dtype.upper() == 'COL' or dtype.upper() == 'SERIESSET':
                self._data = SeriesSet(self._titles, miss_value=miss_value)
                self._data.read_text(addr, first_line, miss_symbol,
                                     title_line, sep, type_float, type_str)
                self._titles = self._data.titles
                self._type = 'DaPy.SeriesSet'
                
            elif dtype.upper() == 'FRAME':
                self._data = Frame(self._titles, miss_value=miss_value)
                self._data.read_text(addr, first_line, miss_symbol,
                                     title_line, sep, type_float, type_str)
                self._titles = self._data.titles
                self._type = 'DaPy.Frame'

            elif dtype.upper() == 'TABLE':
                self._data = Table(miss_value=miss_value)
                self._data.read_text(addr, first_line, miss_symbol,
                                     title_line, sep, type_str)
                self._titles = self._data.titles
                self._type = 'DaPy.Table'

            elif dtype.upper() == 'MATRIX':
                self._data = Matrix()
                self._data.read_text(addr, first_line, title_line, sep)
                self._titles = self._data.titles
                self._type = 'DaPy.Matrix'

            else:
                raise RuntimeError('unrecognized symbol of data type')

        else:
            raise ValueError('DaPy singly supports file types'+\
                             '(xls, xlsx, csv, txt, pkl).')

    def reverse(self, axis='COL'):
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
        self._data.reverse(axis)

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
        self._data.replace(col, *arg)

    def shuffles(self):
        ''' Mess up your data
        '''
        self._data.shuffles()

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
        self._data.sort(*orders)

    def save(self, addr, encode='ascii'):
        '''Save the DataSet to a file.
        '''
        ftype = addr.split('.')[-1]
        if ftype == 'csv' or ftype == 'txt':
            with open(addr, 'w') as f:
                if not is_iter(self._titles):
                    f.write(str(self._titles) + '\n')
                else:
                    f.write(','.join(self._titles) + '\n')
                # write records
                if isinstance(self._data, (Frame, Table, Matrix, SeriesSet)):
                    for line in self._data:
                        f.write(','.join(str(value) for value in line))
                        f.write('\n')

                elif is_iter(self._data[0]):
                    for line in self._data:
                        f.write(','.join([str(i) for i in line]) + '\n')
                else:
                    for record in self._data:
                        f.write(str(record) + '\n')
                        
        elif ftype == 'pkl':
            if isinstance(self._data,
                          (Frame, Table, Matrix, SeriesSet)):
                info = {'Type': self._type.split('.')[1]}
                info['data'] = self._data.data
                info['columns'] = self._data.titles
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
            worksheet = workbook.add_sheet(self._type)
            if self._titles:
                for i, value in enumerate(self._titles):
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
        '''Transform the data structure to DaPy.Frame'''
        if self._type == 'DaPy.Frame':
            return 
        if isinstance(self._data, (Matrix, Table, SeriesSet)):
            self._data = Frame(self._data, self._data._columns,
                               miss_value=self._data._miss_value)
        else:
            self._data = Frame(self._data, self._titles,
                               miss_value=miss_value)
        self._type = 'DaPy.Frame'

    def tocol(self, miss_value=None):
        '''Transform the data structure to DaPy.SeriesSet'''
        if self._type == 'DaPy.SeriesSet':
            return

        if isinstance(self._data, (Matrix, Table, Frame)):
            self._data = SeriesSet(self._data, self._data._columns,
                               miss_value=self._data._miss_value)
        else:
            self._data = SeriesSet(self._data, self._titles,
                               miss_value=miss_value)
        self._type = 'DaPy.SeriesSet'

    def totable(self):
        '''Transform the data structure to DaPy.Table'''
        if self._type == 'DaPy.Table':
            return

        if isinstance(self._data, (Matrix, SeriesSet, Frame)):
            self._data = Table(self._data, self._data._columns)
        else:
            self._data = Table(self._data, self._titles)
        self._type = 'DaPy.Table'

    def tomatrix(self):
        '''Transform the data structure to DaPy.Matrix'''
        if self._type == 'DaPy.Matrix':
            return
        if isinstance(self._data, (SeriesSet, Table, Frame)):
            self._data = Matrix(self._data, self._data._columns)
        else:
            self._data = Matrix(self._data, self._titles)
        self._type = 'DaPy.Matrix'

