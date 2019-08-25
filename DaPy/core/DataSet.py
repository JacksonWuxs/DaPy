from collections import Counter, namedtuple, Iterator
from copy import copy
from functools import wraps
from os.path import isfile
from operator import methodcaller
from time import clock
from pprint import pprint

from .base import (Frame, LogErr, LogInfo, LogWarn, Matrix, Series, SeriesSet,
                   auto_plus_one, filter, is_iter, is_seq, is_str, is_value,
                   map, pickle, range, zip, PYTHON3)
from .io import (parse_addr, parse_excel, parse_html, parse_sav, parse_sql, parse_db, parse_mysql_server,
                 write_db, write_html, write_txt, write_xls, write_sql)

__all__ = ['DataSet']


def timer(func):
    @wraps(func)
    def timer_func(self, *args, **kwrds):
        start = clock()
        ret = func(self, *args, **kwrds)
        if self.log is True:
            name, spent = func.__name__, clock() - start
            LogInfo('%s() in %.3fs.' % (name, spent))
        return ret
    return timer_func

def operater(callfunc):
    callfunc = getattr(SeriesSet, callfunc.__name__)

    @wraps(callfunc)
    def operate_func(self, *args, **kwrds):
        ret_set = DataSet()
        for name, sheet in zip(self._sheets, self._data):
            ret = callfunc(sheet, *args, **kwrds)
            if isinstance(ret, (SeriesSet, Series, list, tuple)):
                ret_set._add(ret, name)

            elif isinstance(ret, (dict, Counter)):
                for name_, ret_ in ret.items():
                    ret_set._add(ret_, name_)
            
            # try:
            #     ret = callfunc(sheet, *args, **kwrds)
            #     ret_set._add(ret, name)
            # except Exception as e:
            #     LogErr('sheet:%s.apply() failed, because %s' % (name, e))
        return ret_set
    return operate_func

class DataSet(object):
    '''A easy-to-use functional data structure similar to MySQL
    
    DataSet is one of the fundamantal data structure in DaPy. 
    It supports users easily to opearte any data structure in 
    a same way with Pythonic Syntax. Additionally, it has 
    logging function.

    Attrbutes
    ---------
    data : list
        a list stored all the sheets inside.

    sheets : list
        a list stored all the names of each sheet.

    types : list
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
    __all__ = ['data', 'columns', 'sheets','info', 'add', 'append', 'append_col', 'info',
               'count', 'count_element', 'pop_miss_value', 'size', 'shape',
               'extend', 'insert', 'insert_col', 'pick', 'pop', 'pop_col',
               'normalized', 'read', 'reverse', 'replace', 'shuffles','corr',
               'sort', 'save', 'tomat', 'toframe', 'tocol', 'show', 'log']

    def __init__(self, obj=None, sheet='sheet0', log=True):
        '''
        Parameter
        ---------
        obj : array-like (default=None)
            initialized your data from a data structure, such as dict(), list()
            Frame(), SeriesSet(), Matrix(), DataSet().
            
        sheet : str (default='sheet0')
            the name of first sheet inside.

        log : bool (default=True)
            show the time consuming for each operation
        '''
        self.log = log
        
        if obj is None:
            self._data = []
            self._sheets = []
            self._types = []
            
        elif (not is_iter(obj)) and not isinstance(obj, str):
            raise TypeError('DataSet can not store this object.')

        elif isinstance(obj, DataSet):
            self._data = copy(obj._data)
            self._sheets = copy(obj._sheets)
            self._types = copy(obj._types)
            
        elif isinstance(obj, (Matrix, SeriesSet, Frame)):
            self._data = [obj, ]
            self._sheets = [str(sheet), ]
            self._types = [type(sheet), ]

        elif isinstance(sheet, str):
            self._data = [obj, ]
            self._sheets = [str(sheet), ]
            self._types = [type(obj), ]
            
        else:
            self._data = list(obj)
            self._sheets = list(map(str, sheet))
            self._types = list(map(type, self._data))
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
    def logging(self):
        return self._log

    @logging.setter
    def logging(self, value):
        if value is not True:
            self._log = False
        else:
            self._log = True

    @property
    def level(self):
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
            self._sheets = [self._check_sheet_new_name(other) for i in range(len(self._sheets))]

        elif is_iter(other):
            if len(set(other)) == len(self._sheets):
                self._sheets = []
                self._sheets = [self._check_sheet_new_name(item) for item in other]
            else:
                raise ValueError('the names size does not match the size of '+\
                                 'sheets inside the DataSet')
        else:
            raise ValueError('unrecognized symbol as %s'%other)
                
    @property
    def shape(self):
        temp = SeriesSet(None, ['Level', 'Sheet', 'Ln', 'Col'], nan='-')
        for i, (sheet, data) in enumerate(zip(self._sheets, self._data)):
            if hasattr(data, 'shape'):
                temp.append([i, sheet] + list(data.shape))
            else:
                temp.append((i, sheet, len(data)))
        return temp
    
    @property
    def info(self):
        for i, data in enumerate(self._data):
            print('sheet:' + self._sheets[i])
            print('=' * (len(self._sheets[i]) + 6))
            if isinstance(data, (Frame, SeriesSet)):
                data.info
            else:
                print('%s has no info() function'%type(data))
        return None

    def __getattr__(self, name):
        if name in self._sheets:
            return self.__getitem__(name)

        temp = DataSet()
        for sheet, data in zip(self._sheets, self._data):
            if hasattr(data, name) or\
                 (hasattr(data, 'columns') and name in data.columns):
                attr = methodcaller(name)
                try:
                    temp._add(attr(data), sheet)
                except TypeError:
                    temp._add(getattr(data, name), sheet)

        assert temp.level != 0, "DataSet has no sheet `%s`'" % name
        return temp
    
    def _check_col_ind_str(self, ind):
        assert ind in self._sheets, "'%s' is not a sheet name" % ind
        return self._sheets.index(ind)

    def _check_col_ind_int(self, ind):
        if ind < 0:
            sheet += self.level - 1
        assert 0 <= ind < self.level, "'%s' is not exist." % ind
        return ind

    def _check_sheet_new_name(self, new_name):
        new_name = str(new_name)
        if not new_name:
            return self._check_sheet_new_name('sheet_%d' % len(self._sheets))

        if new_name not in self._sheets:
            return new_name
        return auto_plus_one(self._sheets, new_name)

    def _check_sheet_index_slice(self, i, j):
        if is_str(i) or is_str(j):
            if i is not None:
                i = self._check_col_ind_str(i)
            if j is not None:
                j = self._check_col_ind_int(j)
        i = self._check_col_ind_int(i) + 1
        j = self._check_col_ind_int(j) + 1
        return range(len(self._sheets))[slice(i, j)]

    def _check_sheet_index(self, sheet):
        '''return a list of sheet indexes'''
        if sheet is None:
            return range(len(self._data))

        if is_str(sheet):
            return [self._check_col_ind_str(sheet)]
            
        if isinstance(sheet, slice):
            return self._check_sheet_index_slice(sheet.start, sheet.stop)
        
        if isinstance(sheet, int):
            return [self._check_col_ind_int(sheet)]

        if isinstance(sheet, (list, tuple)):
            return [self._check_sheet_index(_) for _ in sheet]

    def __getstate__(self):
        toreturn = self.__dict__.copy()
        for key in toreturn:
            if key not in ('_data', '_sheets', '_types'):
                del toreturn[key]
        return toreturn

    def __setstate__(self, arg):
        self._data = arg['_data']
        self._sheets = arg['_sheets']
        self._types = arg['_types']

    def __contains__(self, e):
        '''__contains__(e) -> e in DataSet

        Determind that weather the object is a sheet name inside.
        '''
        if isinstance(e, str):
            return e in self._sheets
        return any([e == data for data in self._data])

    def __repr__(self):
        if len(self._data) == 0:
            return 'empty DataSet object'
        
        reprs = ''
        for i, data in enumerate(self._data):
            reprs += 'sheet:' + self._sheets[i] + '\n'
            reprs += '=' * (len(self._sheets[i]) + 6) + '\n'
            reprs += data.__repr__() + '\n\n'
        return reprs[:-2]
    
    def __len__(self):        
        if len(self._data) == 1:
            if hasattr(self._data[0], 'shape'):
                return self._data[0].shape[0]
            return len(self._data[0])
        return len(self._data)
        
    def __getitem__(self, key):
        if len(self._data) == 1 and (key not in self._sheets):
            return self._data[0][key]
        
        if isinstance(key, slice):
            return self.__getslice__(key.start, key.stop)

    def __getslice__(self, i, j):
        return DataSet([_[i:j] for _ in self._data], self._sheets)
        
        start, stop = self._check_sheet_index_slice(pos)
        return DataSet(self._data[start: stop], self._sheets[start: stop])

    def __setitem__(self, key, val):
        if len(self._data) == 1 and key not in self._sheets:
            self._data[0].__setitem__(key, val)
            return
        
        if is_str(key):
            if isinstance(val, DataSet):
                for src, title in zip(val._data, val._sheets):
                    self._data.append(src)
                    self._types.append(type(src))
                    new_key = '%s_%s' % (key, title)
                    self._sheets.append(self._check_sheet_new_name(new_key))
                return 

            if key not in self._sheets:
                self._data.append(val)
                self._types.append(type(val))
                self._sheets.append(self._check_sheet_new_name(key))
                return
            
            if key in self._sheets:
                key = self._sheets.index(key)
                self._data[key] = val
                self._types[key] = val
                return

        if isinstance(key, int):
            assert abs(key) <= len(self._data), 'set index out of range'
            self._data[key] = val
            self._types[key] = type(val)

    def __delslice__(self, start, stop):
        if start not in self._sheets and stop not in self._sheets:
            for data in self._data:
                del data[start: stop]
            return
        
        start, stop = self._slice2int(start, stop)
        del self._data[start: stop + 1]

    def __delitem__(self, key):
        if isinstance(key, slice):
            self.__delslice__(key.start, key.stop)

        elif key in self._sheets:
            index = self._sheets.index(key)
            del self._sheets[index], self._data[index], self._types[index]

        elif isinstance(key, tuple):
            for obj in key:
                self.__delitem__(obj)

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
    
    def _add(self, item, sheet):
        if isinstance(item, DataSet):
            if sheet and 1 == len(item._sheets):
                new_sheets = [self._check_sheet_new_name(sheet)
                              for sheet_name in item.sheets]
            else:
                new_sheets = [self._check_sheet_new_name(sheet_name) \
                          for sheet_name in item._sheets]
            self._data.extend(item._data)
            self._sheets.extend(new_sheets)
            self._types.extend(item._types)
            
        else:
            self._data.append(item)
            self._types.append(type(item))
            self._sheets.append(self._check_sheet_new_name(sheet))

    @timer
    def add(self, item, sheet=None):
        ''' add a new sheet to the current dataset

        Parameter
        ---------
        item : object
            the new sheet object

        sheet : str or None ( default=None)
            the new sheet name

        Example
        -------
        >>> import DaPy as dp
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
        self._add(item, sheet)

    @timer
    @operater
    def apply(self, func, col=None, inplace=False, axis=0):
        pass
        
    @timer
    @operater
    def append_row(self, item):
        pass

    @timer
    @operater
    def append_col(self, series, variable_name=None):
        pass

    @timer   
    @operater
    def corr(self, method='pearson', col=None):
        pass

    @timer
    @operater
    def count(self, value, col=None, row=None):
        pass

    @timer
    @operater
    def copy(self):
        pass

    @timer
    @operater
    def count_values(self, col=None):
        pass
    
    @timer
    @operater
    def create_index(self, column):
        pass
    
    @timer
    @operater
    def get(self, key, default):
        pass

    def get_tables(self, cols=None):
        key = self._check_sheet_index(cols)
        title = [self._sheets[_] for _ in key]
        src = [self._data[_] for _ in key]
        return DataSet(src, title)

    @timer
    @operater
    def get_dummies(self, col=None, value=1):
        pass
    
    @timer
    @operater
    def groupby(self, keys, func=None, apply_col=None, unapply_col=None):
        pass
                                 
    @timer
    @operater
    def insert_row(self, index, item):
        pass

    @timer
    @operater
    def insert_col(self, index, series, variable_name=None):
        pass        

    @timer
    @operater
    def dropna(self, axis=0, how='any', inplace=False):
        pass

    @timer
    @operater
    def select(self, where, col=None, limit=1000):
        pass

    @timer
    @operater
    def pop(self, index=-1, aixs=0):
        pass

    @timer
    @operater
    def pop_row(self, index=-1):
        pass

    @timer
    @operater
    def pop_col(self, col='all'):
        pass

    @timer
    @operater
    def query(self, expression, col=None, limit=1000):
        pass

    @timer
    def extend(self, other):
        '''extend your data by another object, new data as new records.

        Examples
        --------
        >>> import DaPy as dp
        >>> data = dp.DataSet() # Initiallized a empty DataSet object.
        >>> data.add(dp.Frame(
                        [[11, 11],
                        [21, 21],
                        [31, 31],
                        [41, 41]],
                        ['C1', 'C2']), 'Table1')
        >>> data.add(dp.Frame(
                        [[21, 21],
                        [22, 22],
                        [23, 23],
                        [24, 24]],
                        ['C2', 'C3']), 'Table2')
        >>> data['Table1'].extend(data['Table2'])
        >>> data
        sheet:Table1
        ============
         C1  | C2 |  C3 
        -----+----+------
         11  | 11 | None 
         21  | 21 | None 
         31  | 31 | None 
         41  | 41 | None 
        None | 21 |  21  
        None | 22 |  22  
        None | 23 |  23  
        None | 24 |  24  

        sheet:Table2
        ============
        C2 | C3
        ----+----
        21 | 21 
        22 | 22 
        23 | 23 
        24 | 24 
        '''
        if isinstance(other, DataSet):
            map(self.extend, other._data)
            return

        for sheet, data in zip(self._sheets, self._data):
            if hasattr(data, 'extend') is False:
                LogErr('sheet: %s has no attribute extend(), ignored.' % sheet)
                continue
            try:
                data.extend(other)
            except Exception as e:
                LogErr('sheet: %s.extend() failed because %s.'%(sheet, e))

    @timer 
    @operater        
    def join(self, other):
        pass

    @timer  
    @operater   
    def normalized(self, process='NORMAL', col=None, **kwrds):
        pass

    @timer
    @operater
    def merge(self, other, self_key=0, other_key=0, keep_key=True, keep_same=True):
        pass

    @timer
    @operater
    def drop(self, index=-1, axis=0, inplace=False):
        pass

    @timer
    @operater
    def drop_row(self, index=-1, axis=0, inplace=False):
        pass

    @timer
    @operater
    def drop_col(self, index=-1, axis=0, inplace=False):
        pass
        
    @timer
    def read(self, addr, dtype='col', **kwrd):
        '''This function could be used with loading data from a file and
        transform it into one of DaPy data structure.

        Parameters
        ----------
        addr : str
            the address of data file or a statement like: 
            "mysql://[username]:[password]@[server_ip]:[server_port]/[database_name]/[table1]/[table2]..."
            to access a mysql database. Attention, if `table` keyword is missing 
            in this address, all records will be loaded.

        ftype : str (default=None)
            the file type of this address
            `None` -> automtotally analysis the file type
            "web" -> a website address, it will use requests.get to load the website
                     then use bs4.BeautifulSoup to find <table> tag in the file.
            "html" -> a local html file
            "db" -> SQLite3 database file
            "sav" -> SPSS data file
            "xls" -> Excel data file
            "csv" -> Text file with ',' as delimeters
            "txt" -> Text file with ' ' as delimeters
            "pkl" -> Python pickle file
            "sql" -> MySQL database commands file
            "mysql" -> MySQL database Server

        sheet_name : str (default=None)
            the sheet name of new table.

        miss_symbol : str or str in list (default=['?', '??', '', ' ', 'NA', 'None'])
            the miss value symbol in this data file.

        nan : value (default=nan)
            the miss value symbol in your new data set.

        first_line : int (default=1)
            the first line which includes data values in this file.

        title_line : int (default=0)
            the line which includes your data's column names.
            tip: if there is no title in your data, used -1 represented,
              and, it will automatic create it.

        sep : str (default=",")
            the delimiter symbol inside.

        dtypes : type name in str or dict of columns (default=None):
            DaPy autometally transfers the str source text into the most
            suitable data type in efficiency, whereas some of process costs
            long time. For example, "2018-1-1" is a datetime label, however,
            it is spend a lot of to time transfer this label into datetime.
            In some case, you don't need it in datetime, so just set this column
            type into "str" to save time. The supported data types are "int",
            "float", "str", "datetime" and "bool".
            
            use this keyword as following samples
            >>> read("addr.csv", dtypes={'A_col': int, 'B_col': float})
            >>> read("addr.csv", dtypes="float")
            >>> read("addr.csv", dtypes=["float", "int"])

        Examples
        --------
        >>> import DaPy as dp
        >>> data = dp.read('your_data_file.csv')
        >>> data.read('another_data_file.xlsx')
        '''
        nan = kwrd.get('nan', float('nan'))
        sheet_name = kwrd.get('sheet_name', None)
        miss_symbol = kwrd.get('miss_symbol', set(['?', '??', '', ' ', 'NA', 'None']))
        fpath, fname, fbase, ftype = parse_addr(addr)
        ftype = kwrd.get('ftype', ftype)
        assert ftype in ('web', 'html', 'htm', 'db', 'sav', 'xls', 'xlsx', 'csv', 'txt', 'pkl', 'sql', 'mysql')
        if ftype not in ('web', 'html', 'htm', 'mysql') and not isfile(addr):
            raise IOError('can not find the target file or auto analysis data source type failed')
        if sheet_name is None:
            sheet_name = fbase

        if ftype == 'db':
            try:
                import sqlite3 as sql3
            except ImportError:
                raise ImportError('DaPy uses "sqlite3" to access a database local file.')
            
            with sql3.connect(addr) as conn:
                cur = conn.cursor()
                for sheet, name in parse_db(addr, dtype, nan):
                    self._add(cur, sheet, name)

        elif ftype == 'sav':
            try:
                import savReaderWriter
            except ImportError:
                raise ImportError('DaPy uses "savReaderWriter" to open a .sav file, '+\
                                'please try command: pip install savReaderWriter.')
            with savReaderWriter.SavReader(addr) as reader:
                self._add(parse_sav(reader, dtype, nan), sheet_name)
                
        elif ftype == 'xls' or ftype == 'xlsx':
            first_line = kwrd.get('first_line', 1)
            title_line = kwrd.get('title_line', 0)
            for sheet, name in parse_excel(dtype, addr, first_line, title_line, nan):
                self._add(sheet, name)

        elif ftype in ('txt', 'csv'):
            kwrd['sep'] = kwrd.get('sep', {'csv':',', 'txt':'\t'}[ftype])
            dtype_dic = {'COL': SeriesSet, 'SERIESSET': SeriesSet, 
                         'MATRIX': Matrix, 'MAT': Matrix}
            dtype = dtype_dic.get(dtype.upper(), SeriesSet)
            self._data.append(dtype.from_file(addr, **kwrd))
            self._types.append(dtype)
            self._sheets.append(self._check_sheet_new_name(sheet_name))

        elif ftype == 'pkl':
            self._add(pickle.load(open(addr, 'rb')), sheet_name)

        elif ftype in ('html', 'htm', 'web'):
            if ftype == 'web':
                try:
                    from requests import get
                except ImportError:
                    raise ImportError('DaPy uses "reqeusts" to load a website.')
                else:
                    text = get(addr).text
            else:
                with open(addr) as doc:
                    text = doc.read()

            assert '<table' in text, 'there is no tag <table> in the html file.'
            for sheet, name in parse_html(text, dtype, miss_symbol, nan, sheet_name):
                self._add(sheet, name)
            return self
        
        elif ftype == 'mysql':
            user, psd = fpath.split(':')
            host, port = fbase.split(':')
            try:
                import pymysql as sql
            except ImportError:
                try:
                    import MySQLdb as sql
                except ImportError:
                    raise ImportError('DaPy uses "pymysql" or "MySQLdb" libraries to access a database server.')
            
            with sql.connect(host=host, port=int(port), user=user, passwd=psd, db=fname[0], charset='utf8') as cur:
                for sheet, name in parse_mysql_server(cur, fname):
                    self._add(sheet, name)
        
        elif ftype == 'sql':
            with open(addr) as doc:
                for sheet, name in parse_sql(doc, nan):
                    self._add(sheet, name)
            return self

        else:
            raise ValueError('DaPy singly supports file types as'+\
                             '(xls, xlsx, csv, txt, pkl, db, sav, html, htm).')

    @timer
    @operater
    def reshape(self, nshape):
        pass

    @timer
    def reverse(self, axis='sheet'):
        '''Reverse your data set or records.

        Parameters
        ----------
        axis : str (default='sheet')
            settle down reverse sheets or records in each sheet.

        Example
        -------
        >>> import DaPy as dp
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

        if axis.upper() == 'RECORD':
            for data in self._data:
                if hasattr(data, 'reverse'):
                    data.reverse(axis)
            return

        raise AttributeError('axis should be "sheet" or "record"')

    @timer
    @operater
    def replace(self, old, new, col=None, regex=False, sheet=None):
        pass

    @timer
    @operater
    def shuffle(self):
        pass

    @timer
    @operater
    def sort(self, *orders):
        pass

    @timer
    def save(self, addr, **kwrds):
        '''Save the DataSet to a file.

        Parameters
        ----------
        addr : str
            the output file address.

        encode : str (default='utf-8')
            saving the file in such code type

        ftype : str
            the file type you want to save as. Use the file type in
            your address as default. For example, 'data.save("test.csv")'
            means save this object into .csv type. DaPy supports
            following file types since V1.5.1:
            .csv, .txt, .xls, .pkl, .db, .html

        newline : str (default='\n')
            use this simble to mark change line.

        delimiter : str (default=',')
            use this simble to seperate a records.

        if_exists : str (default='fail')
            when saving the data into a exist database file, how to face the
            delimma that the sheet name has been existed in the database.
            'fail' -> raise an error;
            'replace' -> replace the exist table with current data;
            'append' -> append these records to the exist sheet
            '
        '''
        fpath, fname, fbase, ftype = parse_addr(addr)
        encode = kwrds.get('encode', 'utf-8')
        ftype = kwrds.get('ftype', ftype)

        if ftype in ('csv', 'txt'):
            newline = kwrds.get('newline', '\n')
            delimiter = kwrds.get('delimiter', ',')
            para = dict(mode='w', buffering=2048)
            if PYTHON3:
                para['encoding'] = encode
                para['file'] = addr
            else:
                para['name'] = addr

            for data, sheet in zip(self._data, self._sheets):
                if data is None:
                    continue
                if len(self._data) > 1:
                    addr = fpath + fbase + '_' + sheet + '.' + ftype
                f = open(**para)
                try:
                    write_txt(f, data, newline, delimiter)
                finally:
                    f.close()

        elif ftype in ('xls', 'xlsx'):
            try:
                import xlwt
            except ImportError:
                raise ImportError('DaPy uses xlwt library to save a `xls/xlsx` file.')

            workbook = xlwt.Workbook(encoding=encode)
            for sheet, data in zip(self._sheets, self._data):
                if not data:
                    continue
                worksheet = workbook.add_sheet(sheet)
                write_xls(worksheet, data)
            workbook.save(addr)

        elif ftype == 'pkl':
            pickle.dump(self, open(addr, 'wb'))
        
        elif ftype == 'db':
            import sqlite3 as sql
            with sql.connect(addr) as conn:
                for data, sheet in zip(self._data, self._sheets):
                    write_db(conn.cursor(), sheet, data, kwrds.get('if_exists', 'fail'), 'sqlite3')

        elif ftype == 'html':
            with open(addr, 'w') as f:
                for data, sheet in zip(self._data, self._sheets):
                    if not data:
                        continue
                    f.write('<table border="1" class="%s">' % sheet)
                    write_html(f, data)
                    f.write('</table>')
        
        elif ftype == 'sql':
            with open(addr, 'w') as doc:
                for name, sheet in zip(self._sheets, self._data):
                    write_sql(doc, sheet, name)
        
        elif ftype == 'mysql':
            try:
                import pymysql as sql
            except ImportError:
                try:
                    import MySQLdb as sql
                except ImportError:
                    raise ImportError('DaPy uses "pymysql" or "MySQLdb" libraries to access a database server.')
            user, psd = fpath.split(':')
            host, port = fbase.split(':')
            with sql.connect(host=host, port=int(port), user=user, passwd=psd, db=fname[0], charset='utf8') as conn:
                for data, sheet in zip(self._data, self._sheets):
                    write_db(conn, sheet, data, kwrds.get('if_exists', 'fail'), 'mysql')

        else:
            raise ValueError('unrecognized file type')

    @timer
    def toframe(self):
        '''Transform all of the stored data structure to DaPy.Frame
        '''
        for i, data in enumerate(self._data):
            if isinstance(data, Frame):
                continue
            try:
                if hasattr(data, 'columns'):
                    if hasattr(data, 'miss_value'):
                        self._data[i] = Frame(data, data.columns,
                                           miss_value=data.miss_symbol)
                    else:
                        self._data[i] = Frame(data, data.columns)
                else:
                    self._data[i] = Frame(data)
            except:
                LogErr('sheet:%s can not transform to Frame.'%self._sheets[i])
            self._types[i] = Frame

    @timer
    def tocol(self):
        '''Transform all of the stored data structure to DaPy.SeriesSet
        '''
        for i, data in enumerate(self._data):
            if isinstance(data, SeriesSet):
                continue
            try:
                if hasattr(data, 'columns'):
                    if hasattr(data, 'miss_symbol'):
                        self._data[i] = SeriesSet(data, list(data.columns),
                                           miss_value=data.miss_symbol)
                    else:
                        self._data[i] = SeriesSet(data, data.columns)
                else:
                    self._data[i] = SeriesSet(data)
            except Exception as e:
                LogErr('sheet[%s] can not transform to SeriesSet, ' % self._sheets[i] +\
                    'because: %s' % e)
            self._types[i] = SeriesSet

    @timer
    def tomat(self):
        '''Transform all of the stored data structure to DaPy.Matrix
        '''
        for i, data in enumerate(self._data):
            if isinstance(data, Matrix):
                continue

            try:
                self._data[i] = Matrix(data)
            except:
                LogErr('sheet:%s can not transform to Matrix.'%self._sheets[i])
            self._types[i] = Matrix
    
    @timer
    @operater
    def tolist(self):
        pass

    @timer
    @operater
    def toarray(self):
        pass

    def show(self, lines=None):
        '''show(lines=None) -> None

        See Also
        --------
        DaPy.SeriesSet.show
        '''
        for i, data in enumerate(self._data):
            print('sheet:' + self._sheets[i])
            print('=' * (len(self._sheets[i]) + 6))
            if hasattr(data, 'show'):
                data.show(lines)
            else:
                pprint(data.__repr__())

if __name__ == '__main__':
    from doctest import testmod
    testmod()
