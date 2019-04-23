from collections import Counter, OrderedDict, namedtuple
from copy import copy, deepcopy
from random import shuffle as shuffles
from itertools import groupby as Groupby

from .Row import Row
from .Series import Series
from .tools import is_seq, is_iter, is_math, is_value, pickle, split, strip, xrange
from .tools import auto_plus_one, get_sorted_index, fast_str2value, auto_str2value
from .constant import VALUE_TYPE, STR_TYPE, MATH_TYPE, SEQ_TYPE
from .constant import pickle, nan, inf

__all__ = ['SeriesSet', 'Frame']
dims = namedtuple('sheet', ['Ln', 'Col'])


class BaseSheet(object):
    '''
    Attributes
    ----------
    columns : str in list
        the titles in each columns.

    shape : sheet(Ln, Col)
        the two dimensional span of this sheet.

    nan : value (default=Nan)
        the symbol represented miss value in current seriesset.

    data : dict / list in list
        the object contains all the data by columns or row.

    missing : int in list
        the number of missing value in each column.
    '''

    def __init__(self, obj=None, columns=None, nan=nan):
        self._nan = nan
        self._missing = []

        if obj is None or obj == []:
            if columns is None:
                self._columns = []
                self._dim = dims(0, 0)

            if columns is not None:
                if isinstance(columns, STR_TYPE):
                    columns = [columns,]
                self._dim, self._columns = dims(0, 0), []
                for name in columns:
                    self.append_col([], name)

        elif isinstance(obj, SeriesSet):
            if columns is None:
                columns = copy(obj._columns)
            self._init_col(obj, columns)

        elif isinstance(obj, Frame):
            if columns is None:
                columns = copy(obj._columns)
            self._init_frame(obj, columns)

        elif isinstance(obj, (dict, OrderedDict)):
            if columns is None:
                columns = obj.keys()
            self._init_dict(obj, columns)

        elif hasattr(obj, 'items'):
            if columns is None:
                columns = list(obj.keys())
            self._init_dict(dict(obj.items()), columns)

        elif is_seq(obj) and all(map(is_value, obj)):
            self._init_like_seq(obj, columns)

        elif is_seq(obj) and all(map(is_iter, obj)):
            self._init_like_table(obj, columns)

        elif hasattr(obj, '__iter__'):
            self._init_like_table(list(obj), columns)

        else:
            raise TypeError("sheet structure does not support %s." % type(obj))

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._dim

    @property
    def columns(self):
        return copy(self._columns)

    @columns.setter
    def columns(self, item):
        old_col = self.columns
        if old_col == [] and item != []:
            self._dim = dims(0, len(item))
            old_col = item
        self._init_col_name(item)
        if isinstance(self._data, dict):
            new_data, old_data = dict(), self._data
            for old, new in zip(old_col, self.columns):
                new_data[new] = old_data.get(old, [])
            self._data = new_data

    @property
    def nan(self):
        return self._nan

    @nan.setter
    def nan(self, item):
        assert is_value(item), 'sheet.nan should be a value type'
        if self.missing != 0:
            for missing, sequence in zip(self._missing, self._values()):
                if self.missing == 0:
                    continue
                for i, value in enumerate(sequence):
                    if value == nan:
                        sequence[i] = item
        self._nan = item

    @property
    def missing(self):
        return SeriesSet([self._missing], self.columns)[0]

    def __getattr__(self, name):
        if name in self._columns:
            return Series(self.__getitem__(name))
        raise AttributeError("Sheet object has no attribute '%s'" % name)

    def __len__(self):
        return self._dim.Ln

    def __contains__(self, e):
        if isinstance(e, str):
            return e in self._columns

        if is_seq(e):
            if len(e) == self._dim.Col:
                for record in self:
                    if record == e:
                        return True
            elif len(e) == self._dim.Ln:
                for variable in self._values():
                    if variable == e:
                        return True

        if is_value(other):
            for record in self:
                for value in record:
                    if value == other:
                        return True
        return False

    def __eq__(self, other):
        if is_value(other):
            temp = SeriesSet()
            for title, sequence in self.items():
                temp[title] = [True if v == other else False for v in sequence]
            return temp

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if key <= self._dim.Ln:
                self.__delitem__(key)
                self.insert_row(key, value)
            else:
                self.append_row(value)

        elif isinstance(key, STR_TYPE):
            if key in self._columns:
                pos = self._columns.index(key)
                self.__delitem__(key)
                self.insert_col(pos, value, key)
            else:
                self.append_col(value, key)
            
        else:
            raise TypeError('only can set one record or one column each time.')

    def _getitem_by_tuple(self, interval, obj):
        ERROR = "DaPy doesn't support getting data by columns and index at the"+\
                "same time. Try: sheet['A':'B'][3:10] or sheet[3:10]['A':'B']"
        if len(interval) == 2 and isinstance(interval[0], slice):
            t1, t2 = interval
            if t1.start is None and t1.stop is None:
                if isinstance(t2, slice) and \
                   isinstance(t2.start, int) and isinstance(t2.stop, int):
                    t1_ = (self.columns[t1.start], self.columns[t1.stop])
                    raise SyntaxError('do you mean sheet["%s":"%s"]' % t1_ +
                                      ' or sheet[%s:%s]?' % (t1.start, t1.stop))
                if isinstance(t2, int):
                    raise SyntaxError('do you mean sheet["%s"] ' % self.columns[t2]+
                                      'or sheet[%s]' % t2)
            
        for arg in interval:
            if isinstance(arg, STR_TYPE):
                if obj and obj.shape.Ln != self._dim.Ln:
                    raise SyntaxError(ERROR)
                obj.append_col(self[arg], arg)

            elif isinstance(arg, slice):
                start, stop = arg.start, arg.stop
                if isinstance(start, str) or isinstance(stop, str):
                    if obj and obj.shape.Ln != self._dim.Ln:
                        raise SyntaxError(ERROR)
                    extend_data = self.__getslice__(start, stop)
                    for title, sequence in extend_data.items():
                        if title not in obj:
                            obj.append_col(sequence, title)

                elif isinstance(start, int) or isinstance(stop, int):
                    if obj and obj.shape.Col != self._dim.Col:
                        raise SyntaxError(ERROR)
                    obj.extend(self.__getslice__(start, stop))
                else:
                    raise TypeError('bad statement as [%s:%s]' % (start, stop))

            elif isinstance(arg, int):
                if obj and obj.shape.Col != self._dim.Col:
                    raise SyntaxError(ERROR)
                obj.columns = self._columns
                obj.append_row(self.__getitem__(arg))
            else:
                raise TypeError('bad statement as "arg"' % arg)
        return obj

    def __delitem__(self, key):
        assert isinstance(key, tuple([STR_TYPE] + [int, list, tuple, slice]))
        if isinstance(key, int):
            self.remove_row(key)

        if isinstance(key, STR_TYPE):
            self.remove_col(key)

        if isinstance(key, (list, tuple)):
            int_keys = list(filter(is_math, key))
            str_keys = list(filter(lambda x: isinstance(x, str), key))
            if str_keys != []:
                self.remove_col(str_keys)
            if int_keys != []:
                self.remove_row(int_keys)

    def __getslice__(self, start, stop, step=1):
        if start in self._columns or stop in self._columns:
            return self._getslice_col(*self._check_slice_pos_col(start, stop))
        elif not (not isinstance(start, int) and not isinstance(stop, int)):
            return self._getslice_ln(start, stop, step)
        raise TypeError('bad expression as [%s:%s]' % (start, stop))

    def __getstate__(self):
        instance = self.__dict__.copy()
        instance['_dim'] = tuple(self._dim)
        instance['_data'] = self._data
        return instance

    def __setstate__(self, dict):
        '''load this object from a stream file'''
        self._dim = dims(*dict['_dim'])
        self._columns = dict['_columns']
        self._missing = dict['_missing']
        self._nan = dict['_nan']
        self._data = dict['_data']

    def _init_col_name(self, columns):
        if isinstance(columns, STR_TYPE) and self._dim.Col == 1:
            self._columns = [columns,]
        elif isinstance(columns, STR_TYPE) and self._dim.Col != 1:
            self._columns = [columns + '_%d' % i for i in range(self._dim.Col)]
        elif columns is None or str(columns).strip() == '':
            self._columns = ['C_%d' % i for i in range(self._dim.Col)]
        elif is_iter(columns) is True:
            self._columns, columns = [], list(columns)
            columns.extend(['C_%d' % i for i in range(self._dim.Col - len(columns))])
            for col in columns[:self._dim.Col]:
                self._columns.append(self._check_col_new_name(col))
            for i in range(self._dim.Col - len(self._columns)):
                self._columns.append(self._check_col_new_name(None))
        else:
            raise TypeError('column names should be stored in a iterable')

    def _trans_where(self, where, axis=0):
        assert axis in (1, 0), 'axis 1 for value, 0 for sequence'        
        if axis == 0:
            if where is None:
                return lambda x: True
            for i in get_sorted_index(self._columns, key=len, reverse=True):
                where = where.replace(self._columns[i], 'x[%d]'%i)
            return eval('lambda x: ' + where)

        if axis == 1:
            opeartes = {' and': 4, ' or': 3}
            for opearte, bias in opeartes.items():
                counts = where.count(opearte)
                index = 0
                for i in range(counts):
                    index = where.index(opearte, index) + bias
                    where = where[: index] + ' x' + where[index: ]
                    index += bias
            return eval('lambda x: x ' + where)

    def _add_row(self, item):
        if is_value(item):
            item = [item] * self._dim.Col
            
        lenth_bias = len(item) - self._dim.Col
        if lenth_bias > 0:
            self.join([[self.nan] * lenth_bias for i in range(self.shape.Ln)])
            
        mv, item = self._check_sequence_type(item, self.shape.Col)
        if mv != 0:
            for i, value in enumerate(item):
                if value == self.nan or value is self.nan:
                    self._missing[i] += 1
        self._dim = dims(self._dim.Ln + 1, max(self._dim.Col, len(item)))
        return item

    def _check_sequence_type(self, series, size):
        '''if sequence is record, size = self.shape.Ln;
           else size = self.shape.Col
        '''
        if is_value(series):
            return 0, [series] * size

        assert is_iter(series), "append item should be an iterable object"
        series = list(series)
        if len(series) < size:
            series.extend([self._nan] * (size - len(series)))

        if self.nan not in series:
            return 0, series
        return series.count(self.nan), series

    def _check_remove_index(self, index):
        assert isinstance(index, (int, list, tuple)), 'an int or ints in list is required.'
        if isinstance(index, int):
            index = [index,]
        return sorted(set(index), reverse=True)

    def _check_replace_condition(self, col, condition, new_value):
        if col == "all":
            col == self._columns

        if is_seq(col):
            for title in col:
                self._replace_typical(title, condition, new_value)
            return

        elif isinstance(col, str):
            if isinstance(self._data, list):
                col = self._columns.index(col)

        elif isinstance(col, int):
            if isinstance(self._data, OrderedDict):
                col = self._columns[col]

        else:
            raise ValueError('your column name can not be found in dataset.')

        if not isinstance(condition, STR_TYPE) and not callable(condition):
            raise TypeError('condition should be python statement or callable object')

        if not is_value(new_value):
            raise TypeError('SeriesSet does not support %s ' % type(new_value),
                            'as a value type.')
        return col, condition, new_value

    def _check_col_new_name(self, new_name):
        if new_name is None:
            return self._check_col_new_name('C_%d' % len(self._columns))
        new_name = str(new_name)
        if isinstance(new_name, STR_TYPE) and new_name not in self._columns:
            return new_name
        return auto_plus_one(self._columns, new_name)

    def _check_slice_pos_col(self, i, j):
        if i in self._columns:
            i = self._columns.index(i)
        elif i is None:
            i = 0
        else:
            raise ValueError('%s is not a title in current dataset'%i)

        if j in self._columns:
            j =  self._columns.index(j)
        elif j is None:
            j = self._dim.Col - 1
        else:
            raise ValueError('%s is not a title in current dataset'%j)
        return (i, j)

    def _check_area(self, point1, point2):
        def _check_pos(point, left=True):
            if left is True and point is None:
                x, y = 0, 0
            elif left is False and point is None:
                x, y = None, None
            else:
                x, y = point
                
            if isinstance(y, str) and y in self._columns:
                y = self._columns.index(y)
            elif y is None and left is True:
                y = 0
            elif y is None and left is False:
                y = self._dim.Col
            elif not(isinstance(y, int) and abs(y) < self._dim):
                raise IndexError('your request col=`%s` is out of dataset.' % y)

            if x is None and left is True:
                x = 0
            elif x is None and left is False:
                x = self._dim.Ln
            elif not (isinstance(x, int) and abs(x) < self._dim.Ln):
                raise IndexError('your request ln=`%s` is out of dataset.' % x)
            return x, y

        L1, C1 = _check_pos(point1, left=True)
        L2, C2 = _check_pos(point2, left=False)

        if C2 < C1 or L2 < L1:
            raise ValueError('point2 should be larger than point1.')
        return L1, C1, L2, C2

    def _check_nan(self, nan):
        if is_value(nan):
            return (nan, self._nan)
        return tuple(nan)

    def _check_columns_index(self, col):
        if isinstance(col, STR_TYPE):
            if col.lower() == 'all':
                return tuple(self._columns)
            if col in self._columns:
                return (col,)
            raise ValueError('%s is not a title in current dataset' % col)

        if isinstance(col, int):
            assert abs(col) < self.shape.Col, 'title index is out of range'
            return (self._columns[col],)

        if is_seq(col):
            return tuple(self._check_columns_index(col_)[0] for col_ in col)

    def replace(self, *arg):
        if arg == tuple():
            raise KeyError('argument is empty!')

        elif len(arg) == 3 and is_value(arg[-1]):
            self._replace_typical(*arg)

        elif all(map(is_iter, arg)):
            if max(map(len, arg)) > 3 or min(map(len, arg)) < 3:
                raise AttributeError('you should input 3 arguments as a order'+\
                                    " which seem like .replace('C1', '> 20', 3)")
            for obj in arg:
                self._replace_typical(*obj)

        else:
            raise TypeError(\
                'argument should be include as a dict() or multiple'+\
                " tuples, like: replace(('C1', '< 100', 2)," +\
                " ('C2', '> 200', 3)). Use help() "+\
                'for using details.')

    def tolist(self):
        if isinstance(self.data, list):
            return self.data
        return list(map(list, zip(*self._values())))

    def select(self, where, column=None, limit=1000):
        '''sheet.select(lambda x: x['A_col'] != 1)
           sheet.select('A_col != 1')
        '''
        assert isinstance(limit, int)
        assert callable(where) or isinstance(where, STR_TYPE), '`where` should be a python statement or callable object'
        if column is None:
            column = 'all'
        column = self._check_columns_index(column)
        if isinstance(where, STR_TYPE):
            where = self._trans_where(where, axis=0)
        yes_row = []
        for row in self:
            if where(row) is True:
                yes_row.append(row[column])
                if len(yes_row) == limit:
                    break
        return SeriesSet(yes_row, column)

    def groupby(self, func, keys):
        keys = self._check_columns_index(keys)
        gb = []
        conditions = [(keys, 'DESC') for key in keys]
        for key, sub in Groupby(self.sort(*conditions), lambda x: x[keys]):
            gb.append(SeriesSet(sub).apply(func, axis=1, inplace=False)[0])
        return SeriesSet(gb, self.columns)

    def show(self, lines='all'):
        if len(self._columns) == 0:
            return 'empty sheet instant'
        
        if not isinstance(lines, int) and not lines == 'all':
            raise TypeError('`lines` should be an int() or keyword `all`.')

        if lines == 'all' or 2 * lines >= self._dim.Ln:
            lines, omit = -1, 0
            temporary_data = zip(*[sequence for sequence in self._values()])
        elif lines <= 0:
            raise ValueError('`lines` should be greater than 0.')
        else:
            omit = self._dim.Ln - 2 * lines
            temporary_data = self[:lines]
            temporary_data.extend(self[-lines:])
        temporary_series = [[title, ] for title in self._columns]
        for i, col in enumerate(zip(*temporary_data)):
            temporary_series[i].extend(map(str, col))

        column_size = [len(max(col, key=len)) for col in temporary_series]
        frame = ' ' + ' | '.join([title.center(column_size[i]) for i, title in \
                         enumerate(self._columns)]) + '\n'
        frame += '+'.join(['-' * (size + 2) for size in column_size]) + '\n'

        for i, item in enumerate(temporary_data):
            if i == lines:
                frame += ('.. Omit %d Ln ..'%omit).center(len(line)) + '\n'
            line = ''
            for i, value in enumerate(item):
                line += ' ' + str(value).center(column_size[i]) + ' |'
            frame += line[:-1] + '\n'
        return frame[:-1]

    def sort(self, *orderby):
        '''S.sort(('A_col', 'DESC'), ('B_col', 'ASC')) --> sort your records.
        '''
        ERROR = "orderby must be a sequence of conditions like ('A_col', 'DESC')"
        assert all(map(lambda x: (is_seq(x) and len(x) == 2) or isinstance(x, STR_TYPE), orderby)), ERROR
        compare_symbol = ['ASC' if isinstance(order, STR_TYPE) else str(order[1]) for order in orderby]
        if isinstance(orderby[0], STR_TYPE):
            orderby = [(order,) for order in orderby]
        compare_pos = self._check_columns_index([order[0] for order in orderby])
        assert all(map(lambda x: x.upper() in ('DESC', 'ASC'), compare_symbol)), ERROR
        assert len(compare_pos) == len(compare_symbol), ERROR
        size_orders = len(compare_pos) - 1

        def hash_sort(datas_, i=0):
            # initialize values
            index = compare_pos[i]
            inside_data, HashTable = list(), dict()

            # create the diction
            for item in datas_:
                key = item[index]
                if key in HashTable:
                    HashTable[key].append(item)
                else:
                    HashTable[key] = [item]

            # sorted the values
            sequence = sorted(HashTable)

            # transform the record into Frame
            for each in sequence:
                items = HashTable[each]
                if i < size_orders:
                    items = hash_sort(items, i+1)
                inside_data.extend(items)

            # finally, reversed the list if necessary.
            if i != 0 and compare_symbol[i] != compare_symbol[i-1]:
                inside_data.reverse()
            return inside_data

        if len(compare_symbol) == 1 and isinstance(self._data, dict):
            if compare_symbol[0] == 'DESC':
                reverse = True
            else:
                reverse = False
            new_index = get_sorted_index(self._data[compare_pos[0]], reverse=reverse)
            temp = SeriesSet()
            for title, sequence in self._data.items():
                temp.append_col((sequence[index] for index in new_index), title)
            return temp
        else:
            temp = hash_sort(self)
            if compare_symbol[0] == 'DESC':
                temp.reverse()
        if isinstance(self._data, dict):
            return SeriesSet(temp, self.columns)
        return Frame(temp, self.columns)


class SeriesSet(BaseSheet):
    '''Variable stores in sequenes.
    '''
    def __init__(self, series=None, columns=None, nan=None):
        self._data = dict()
        BaseSheet.__init__(self, series, columns, nan)

    @property
    def info(self):
        from DaPy import describe
        mins, maxs, avgs, stds, skes, kurs = [], [], [], [], [], []
        for sequence in self._values():
            d = describe(sequence)
            for ls, value in zip([mins, maxs, avgs, stds, skes, kurs],
                                 [d.Min, d.Max, d.Mean, d.S, d.Skew, d.Kurt]):
                if value == None:
                    ls.append('-')
                elif isinstance(value, float):
                    ls.append('%.2f' % value)
                else:
                    ls.append(str(value))

        miss = map(str, self._missing)
        blank_size = [max(len(max(self._columns, key=len)), 5) + 2,
                      max(len(max(miss, key=len)), 4) + 2,
                      max(len(max(mins, key=len)), 3) + 2,
                      max(len(max(maxs, key=len)), 3) + 2,
                      max(len(max(avgs, key=len)), 4) + 2,
                      max(len(max(stds, key=len)), 3) + 2,
                      max(len(max(skes, key=len)), 4) + 2,
                      max(len(max(kurs, key=len)), 4) + 1]

        # Draw the title line of description
        title_line = '|'.join([
                     'Title'.center(blank_size[0]),
                     'Miss'.center(blank_size[1]),
                     'Min'.center(blank_size[2]),
                     'Max'.center(blank_size[3]),
                     'Mean'.center(blank_size[4]),
                     'Std'.center(blank_size[5]),
                     'Skew'.center(blank_size[6]),
                     'Kurt'.center(blank_size[7])]) + '\n'
        title_line += '+'.join(map(lambda x: '-' * x, blank_size)) + '\n'

        # Draw the main table of description
        info = str()
        for i, title in enumerate(self._columns):
            info += title.center(blank_size[0]) + '|'
            info += miss[i].center(blank_size[1]) + '|'
            info += mins[i].center(blank_size[2]) + '|'
            info += maxs[i].center(blank_size[3]) + '|'
            info += avgs[i].center(blank_size[4]) + '|'
            info += stds[i].center(blank_size[5]) + '|'
            info += skes[i].center(blank_size[6]) + '|'
            info += kurs[i].center(blank_size[7]) + '\n'
            
        lenth = 7 + sum(blank_size)
        line = '=' * lenth
        print('1.  Structure: DaPy.SeriesSet\n' +\
              '2. Dimensions: Lines=%d | Variables=%d\n'%self._dim +\
              '3. Miss Value: %d elements\n'%sum(self._missing) +\
              'Descriptive Statistics'.center(lenth) + '\n' +\
              line + '\n' + title_line + info + line)

    def _init_col(self, series, columns):
        self._data = copy(series._data)
        self._dim = copy(series._dim)
        self._init_col_name(columns)
        self._missing = copy(series._missing)
        self._nan = copy(series._nan)

    def _init_dict(self, series, columns):
        max_Ln = max(map(len, series.values()))
        self._dim = dims(max_Ln, len(series))
        self._init_col_name(columns)
        for column in self.columns:
            mv, sequence = self._check_sequence_type(series[column], self._dim.Ln)
            self._missing.append(mv)
            self._data[column] = sequence

    def _init_frame(self, series, columns):
        self._dim = dims(series._dim.Ln, series._dim.Col)
        self._missing = copy(series._missing)
        self._init_col_name(columns)
        for sequence, title in zip(zip(*series), self._columns):
            self._data[title] = list(sequence)

        if self._nan != series._nan:
            nan, self._nan = self.nan, series.nan
            self.nan = nan

    def _init_like_seq(self, series, columns):
        self._dim = dims(len(series), 1)
        self._init_col_name(columns)
        mv, series = self._check_sequence_type(series, self._dim.Ln)
        self._missing = [mv,]
        self._data[self._columns[0]] = series

    def _init_like_table(self, series, columns):
        lenth_Col = len(max(series, key=len))
        for pos, record in enumerate(series):
            if len(record) < lenth_Col:
                new = list(record)
                new.extend([self._nan] * (lenth_Col - len(record)))
                series[pos] = new
        self._dim = dims(len(series), lenth_Col)
        self._init_col_name(columns)
        self._missing = [0] * self._dim.Col
        for j, sequence in enumerate(zip(*series)):
            mv, series = self._check_sequence_type(sequence, self._dim.Ln)
            self._missing[j] += mv
            self._data[self._columns[j]] = series

    def __repr__(self):
        if self._dim.Ln > 10:
            def write_Ln(i, title, blank):
                item = self._data[title]
                msg = ' ' * blank + title + ': <'
                msg += ', '.join([str(value) for value in item[:5]])
                msg += ', ... ,'
                msg += ', '.join([str(value) for value in item[-5:]])
                msg +=  '>\n'
                return msg
        elif self._dim.Ln != 0:
            def write_Ln(i, title, blank):
                item = self._data[title]
                msg = ' ' * blank + title + ': <'
                msg += ', '.join([str(value) for value in item])
                msg += '>\n'
                return msg
        else:
            return 'empty SeriesSet instant'

        msg = str()
        size = len(max(self._columns, key=len))
        for i, title in enumerate(self._columns):
            msg += write_Ln(i, title, size - len(title))
        return msg[:-1]

    def _getslice_col(self, i, j):
        new_data = OrderedDict()
        for title in self._columns[i: j+1]:
            new_data[title] = self._data[title]
        return SeriesSet(new_data, nan=self._nan)

    def _getslice_ln(self, i, j, k):
        return_list = zip(*[self._data[t][i:j:k] for t in self._columns])
        return SeriesSet(return_list, self._columns, None)

    def __getitem__(self, interval):
        if isinstance(interval, int):
            return Row(self, interval)

        elif isinstance(interval, (tuple, list)):
            return_obj = SeriesSet(nan=self._nan)
            return self._getitem_by_tuple(interval, return_obj)

        elif isinstance(interval, slice):
            return self.__getslice__(interval.start, interval.stop)

        elif isinstance(interval, STR_TYPE):
            return self._data[interval]

        else:
            raise TypeError('SeriesSet index must be int, str and slice, '+\
                            'not %s' % str(type(interval)).split("'")[1])

    def __iter__(self):
        for i in range(self._dim.Ln):
            yield Row(self, i)

    def __reversed__(self):
        for i in range(self._dim.Ln-1, -1, -1):
            yield Row(self, i)

    def _arrange_by_index(self, self_new_index=None, other_new_index=None):
        if self_new_index:
            for title, sequence in self.items():
                self._data[title] = [sequence[j] for j in self_new_index]
        elif other_new_index:
            for title, sequence in self.items():
                new_sequence = [value for _,value in sorted(zip(other_new_index, sequence), key=lambda x: x[0])]
                self._data[title] = new_sequence
        else:
            raise RuntimeError('at least one parameter should be filled in.')

    def _replace_typical(self, col, cond, new):
        col, cond, new = self._check_replace_condition(col, cond, new)
        if isinstance(cond, STR_TYPE):
            cond = self._trans_where(cond, axis=1)
        self._data[col] = [new if cond(value) else value\
                           for value in self._data[col]]

    def append_row(self, item):
        item = self._add_row(item)
        for value, seq in zip(item, self._values()):
            seq.append(value)

    def append_col(self, series, variable_name=None):
        '''append a series data to the seriesset last
        '''
        variable_name = self._check_col_new_name(variable_name)
        mv, series = self._check_sequence_type(series, self._dim.Ln)
        size = len(series)
        if size > self._dim.Ln:
            bias = size - self._dim.Ln
            for i, title in enumerate(self._columns):
                self._missing[i] += bias
                self._data[title].extend([self._nan] * bias)

        self._columns.append(variable_name)
        self._missing.append(mv)
        self._dim = dims(size, self._dim.Col+1)
        self._data[variable_name] = series

    def corr(self, method='pearson'):
        '''correlation between variables in data -> Frame object
        '''
        from DaPy import corr as corr_
        new_ = SeriesSet(None, self._columns, nan='')
        for i, (title, sequence) in enumerate(self._data.items()):
            corr_line = []
            for next_title, next_sequence in self[:title].items():
                if title == next_title:
                    corr_line.append(1.0)
                    continue
                corr_line.append(round(corr_(sequence, next_sequence, method), 4))
            new_.append(corr_line)
        new_.insert_col(0, self._columns)
        return new_

    def count(self, X, point1=None, point2=None):
        '''count the frequency of X in area (point1, point2)-> Counter object
        '''
        counter = Counter()
        if is_value(X):
            X = (X,)
        L1, C1, L2, C2 = self._check_area(point1, point2)
        
        for title in self._columns[C1 : C2+1]:
            sequence = self._data[title]
            for value in sequence[L1 : L2+1]:
                if value in X:
                    counter[value] += 1
        if len(X) == 1:
            return counter[X[0]]
        return counter

    def count_values(self, col='all'):
        col = self._check_columns_index(col)
        counter = Counter()
        for title in col:
            counter.update(self._data[title])
        return counter

    def extend(self, item):
        '''extend the current SeriesSet with records in set.
        '''
        if isinstance(item, SeriesSet):
            for mv, (title, sequence) in zip(item._missing, item.items()):
                if title not in self._columns:
                    self._columns.append(self._check_col_new_name(title))
                    self._missing.append(self._dim.Ln + mv)
                    To = [self._nan] * self._dim.Ln
                else:
                    self._missing[self._columns.index(title)] += mv
                    To = self._data[title] 
                To.extend(sequence)
                self._data[title] = To
            self._dim = dims(self._dim.Ln+item._dim.Ln, len(self._columns))

            for i, sequence in enumerate(self._data.values()):
                if len(sequence) != self._dim.Ln:
                    add_miss_size = self._dim.Ln - len(sequence)
                    sequence.extend([self._nan] * add_miss_size)
                    self._missing[i] += add_miss_size

        elif isinstance(item, Frame):
            self.extend(SeriesSet(item))

        elif all(filter(is_iter, item)):
            self.extend(SeriesSet(item, self._columns))

        else:
            raise TypeError('could not extend a single value only.')

    def join(self, other):
        if isinstance(other, SeriesSet):
            lenth = max(self._dim.Ln, other._dim.Ln)
            for title, sequence in other.items():
                title = self._check_col_new_name(title)
                mv, sequence = self._check_sequence_type(sequence, self._dim.Ln)
                self._missing.append(mv)
                self._data[title] = sequence
                self._columns.append(title)
            self._dim = dims(lenth, len(self._columns))

        elif isinstance(other, Frame):
            self.join(SeriesSet(other))

        elif all(filter(is_iter, other)):
            new_col = [title + '_1' for title in self._columns]
            self.join(SeriesSet(other, new_col))

        else:
            raise TypeError('could not extend a single value only.')

    def get_dummies(self, col='all', value=1):
        cols = self._check_columns_index(col)
        from DaPy import get_dummies
        for title in cols:
            dummies = get_dummies(self._data[title], value, 'set')
            dummies.columns = [title+'_'+title_ for title_ in dummies.columns]
            self.join(dummies)
                    
    def items(self):
        for column in self.columns: 
            yield column, Series(self._data[column])

    def insert_row(self, index, item): 
        '''insert a record to the frame, position in <index>
        '''
        item = self._add_row(item)
        for value, seq in zip(item, self._values()):
            seq.insert(index, value)

    def insert_col(self, index, series, variable_name=None, nan=None):
        '''insert a series of data to the frame, position in <index>
        '''
        variable_name = self._check_col_new_name(variable_name)
        mv, series = self._check_sequence_type(series, self._dim.Ln)

        size_series = len(series)
        if size_series < self._dim.Ln:
            series = list(series)
            mv_2 = self._dim.Ln - size_series
            series.extend([self._nan] * mv_2)
            mv += mv_2
            size_series = self._dim.Ln
        elif size_series > self._dim.Ln:
            temp = [[self._nan] * self.shape.Col for i in range(size_series - self.shape.Ln)]
            self._data[title].extend(temp)

        self._columns.insert(index, variable_name)
        self._dim = dims(size_series, self._dim.Col + 1)
        self._missing.insert(index, mv)

        if index >= self._dim.Col-1:
            self._data[variable_name] = series
            return

        new_data = OrderedDict()
        for i, title in enumerate(self._data):
            if i == index:
                new_data[variable_name] = series
            new_data[title] = self._data[title]
        self._data = new_data

    def keys(self):
        return self._data.keys()

    def normalized(self, process='NORMAL', col='all', **attr):
        assert isinstance(process, STR_TYPE)
        assert isinstance(col, (str, list, tuple))
        assert process.upper() in ('NORMAL', 'STANDAR', 'LOG', 'BOX-COX')
        process = process.upper()
        new_col = self._check_columns_index(col)

        from DaPy import describe, log, boxcox
        for title in new_col:
            if title is None:
                continue

            if process in ('NORMAL', 'STANDAR'):
                if attr == {}:
                    statis = describe(self._data[title])
                if process == 'NORMAL':
                    A = float(attr.get('min', statis.Min))
                    B = float(attr.get('range', statis.Range))
                elif process == 'STANDAR':
                    A = float(attr.get('mean', statis.Mean))
                    B = float(attr.get('std', statis.Sn))
                assert B != 0, '`%s` can not have 0 range or std' % title
                self.apply(lambda x: (x - A) / B, col=title, inplace=True, axis=1)
                
            elif process == 'BOX-COX':
                lamda = attr.get('lamda', 1)
                a = attr.get('a', 0)
                k = attr.get('k', 1)
                self.apply(lambda x: boxcox(x, lamba, a, k), col=title, inplace=True, axis=1)
                
            elif process == 'LOG':
                base = attr.get('base', 2.71828183)
                self.apply(lambda x: log(x, base), col=title, inplace=True, axis=1)
            
    def apply(self, func, col='all', inplace=False, axis=0):
        assert axis in (0, 1), 'axis must be 0 or 1'
        assert callable(func), '`func` parameter should be a callable object'

        map_columns = self._check_columns_index(col)
        if inplace is False:
            if axis == 0:
                return SeriesSet(map(func, self[map_columns]), map_columns, self.nan)
            ret = SeriesSet(nan=self.nan)
            for col in map_columns:
                value = func(self.data[col])
                if hasattr(value, '__iter__'):
                    ret[col] = list(value)
                else:
                    ret[col] = [value]
            return ret
        
        if axis == 1:
            for name in map_columns:
                self[name] = map(func, self[name])

    def merge(self, other, self_key=0, other_key=0, keep_key=True, keep_same=True):
        other = SeriesSet(other)
        def check_key(key, d, str_):
            if isinstance(key, int):
                return d._columns[key]
            assert isinstance(key, STR_TYPE), 'key shoule be a string or int type'
            assert key in d._columns, '`%s` is not a vaiable name ' % key
            return key

        if self.shape.Ln != 0:
            self_key = check_key(self_key, self, 'this')
        other_key = check_key(other_key, other, 'other')
        new_other_key = self._check_col_new_name(other_key)
        
        assert keep_key in (True, False, 'other', 'self')
        assert keep_same in (True, False)
   
        # check new variables
        change_name, original_name = [], []
        for i, col in enumerate(other.columns):
            if col in self._columns and col != other_key and keep_same is False:
                continue
            if col in self._columns and col == other_key and keep_key not in (True, 'other'):
                continue
            original_name.append(col)
            col = self._check_col_new_name(col)
            self._missing.append(other._missing[i])
            self._columns.append(col)
            change_name.append(col)
            
        # match the records
        temp_index, new_index = [], self._dim.Ln
        self_key_seq, other_key_seq = self[self_key], other[other_key]
        for i, value in enumerate(other_key_seq):
            self_keys  = self_key_seq.count(value)
            other_keys = other_key_seq.count(value)
            if self_keys == 0:
                temp_index.append(new_index)
                new_index += 1
            elif self_keys == 1 and other_keys == 1:
                temp_index.append(self_key_seq.index(value))
            elif self_keys == 1 and other_keys > 1:
                if other_key_seq.index(value) == i:
                    temp_index.append(self_key_seq.index(value))
                else:
                    temp_index.append(new_index)
                    new_index += 1
            else: # self_keys > 1
                this_index = other_key_seq.index(value)
                number = 1
                while i != this_index:
                    this_index = other_key_seq.index(value, this_index + 1)
                    number += 1

                this_index = -1
                for i in range(number):
                    try:
                        this_index = self_key_seq.index(value, this_index + 1)
                    except ValueError:
                        temp_index.append(new_index)
                        new_index += 1
                        break
                else:
                    temp_index.append(this_index)

        # extend the empty dataset
        how_many_new_index = new_index - other.shape.Ln
        if how_many_new_index != 0:
            other.extend([[None] * other.shape.Col for i in range(how_many_new_index)])

        hash_temp_index = set(temp_index)
        for i in range(new_index):
            if i not in hash_temp_index:
                temp_index.append(i)
        other._arrange_by_index(None, temp_index)

        if other.nan != self._nan:
            other.symbol = self._nan
        for title, origi in zip(change_name, original_name):
            self._data[title] = other[origi]

        self._dim = dims(new_index, len(self._columns))
        for seq in self._data.values():
            seq.extend([None] * (self._dim.Ln - len(seq)))

        if keep_key == 'other' or keep_key is False:
            self.remove_col(self_key)

    def dropna(self, axis=1):
        pops = []
        if str(axis).upper() in ('COL', '1'):
            start = 0
            for i, value in enumerate(self._missing):
                if value != 0:
                    pops.append(self._columns[i])
            if len(pops) != 0:
                self.remove_col(list(set(pops)))

        if str(axis).upper() in ('LINE', '0', 'ROW'):
            for sequence in self._data.values():
                for i, value in enumerate(sequence):
                    if value == self.nan or value is self.nan:
                        pops.append(i)
            if len(pops) != 0:
                self.remove_row(list(set(pops)))

    DUPLICATE_KEEP = {'first': slice(1, None),
                      'last': slice(0, -1),
                      None: slice(None, None)}
    
    def drop_duplicates(self, col='all', keep='first'):
        assert keep in ('first', 'last', False)
        pop_name = self._check_columns_index(col)
        droped_table = dict()
        for i, row in enumerate(zip(*list(self._values))):
            if row not in droped_table:
                droped_table[row] = [i]
            else:
                droped_table[row].append(i)

        droped_list, drop = [], SeriesSet.DUPLICATE_KEEP[keep]
        for values in droped_table.values():
            if len(values) != 1:
                droped_list.extend(values[drop])
        self.remove_row(index=droped_list)

    def remove(self, index=-1, axis='ROW'):
        assert axis in ('ROW', 'COL')
        if axis == 'ROW':
            return self.remove_row(index)
        return self.remove_col(index)

    def remove_col(self, index=-1):
        pop_name = self._check_columns_index(index)
        for title in pop_name:
            pos = self._columns.index(title)
            del self._data[title], self._missing[pos], self._columns[pos]
        self._dim = dims(self._dim.Ln, self._dim.Col-len(pop_name))

    def remove_row(self, index=-1):
        index = self._check_remove_index(index)
        for i, seq in enumerate(self._values()):
            for j in index:
                del seq[j]
            self._missing[i] = seq.count(self._nan)
        self._dim = dims(self._dim.Ln - len(index), self._dim.Col)

    def pop(self, index=-1, axis='ROW'):
        assert axis in ('ROW', 'COL')
        if axis == 'ROW':
            return self.pop_row(index)
        return self.pop_col(index)
        
    def pop_row(self, index=-1):
        '''pop(remove & return) record(s) from the sheet
        '''
        index = self._check_remove_index(index)
        pop_item = dict()
        for i, (title, seq) in enumerate(self.items()):
            pop_item[title] = [seq.pop(pos_) for pos_ in index]
            self._missing[i] -= pop_item[title].count(self._nan)
        self._dim = dims(self._dim.Ln - len(pos), self._dim.Col)
        return SeriesSet(pop_item, self._columns, self._nan)

    def pop_col(self, col=-1):
        '''pop(remove & return) record(s) from the sheet
        '''
        pop_name = self._check_columns_index(col)
        pop_data = dict()
        for title in pop_name:
            pos = self._columns.index(title)
            pop_data[title] = self._data.pop(title)
            self._columns.pop(pos)
            self._missing.pop(pos)
        self._dim = dims(self._dim.Ln, self._dim.Col-len(pop_name))
        return SeriesSet(pop_data, None, self._nan)

    def reverse(self, axis='COL'):
        if axis.upper() == 'COL':
            self._columns.reverse()
            self._missing.reverse()
        else:
            for sequence in self._data.values():
                sequence.reverse()
    
    def from_file(self, addr, **kwrd):
        '''read dataset from csv or txt file.

        Parameters
        ----------
        addr : str
            address of source file.

        first_line : int (default=1)
            the first line with data.

        nan : str (default="NA")
            the symbol of missing value in csv file.

        title_line : int (default=0)
            the line with title, rules design as follow:
            -1 -> there is no title inside;
            >=0 -> the titleline.

        sep : str (default=",")
            the delimiter symbol inside.

        prefer_type : type-object (default=float):
            int -> transfer any possible values into int
            float -> transfer any possible values into float
            str -> keep all values in str type
            datetime -> transfer any possible values into datetime-object
            bool -> transfer any possible values into bool

        types : Type name or dict of columns (default=None):
            use one type to 
        '''
        first_line = kwrd.get('first_line', 1)
        title_line = kwrd.get('title_line', 0)
        columns = kwrd.get('columns', [])
        nan = self._check_nan(kwrd.get('nan', ('nan', '?', '??', '', ' ', 'NA', 'None')))
        sep = kwrd.get('sep', ',')
        prefer_type = kwrd.get('prefer_type', None)
        col_types = kwrd.get('types', {})
        self._missing, temp_data, nan = [0], dict(), dict.fromkeys(nan)
        
        assert first_line > title_line, 'first line should be greater than title line'
        assert isinstance(columns, list), 'column name must be stored with a list'
        assert all(map(lambda x: isinstance(x, STR_TYPE), columns)), 'column name must `str`'

        with open(addr, 'rU') as f:
            for i in xrange(first_line):
                line = f.readline()
                if i == title_line:
                    columns = map(str.strip, split(line, sep))

            for line in f:
                for i, value in enumerate(split(strip(line), sep)):
                    try:
                        if value in nan:
                            temp_data[i].append(self._nan)
                            self._missing[i] += 1
                            continue
                        temp_data[i].append(fast_str2value[col_types[i]](value))
                    except ValueError:
                        temp_data[i].append(auto_str2value(value))
                    except KeyError as e:
                        self._missing.append(0)
                        transed = auto_str2value(value, prefer_type)
                        col_types[i] = str(transed.__class__).split()[1][1:-2].split('.')[0]
                        if transed in nan:
                            transed = self._nan
                        if len(temp_data.keys()) == 0:
                            temp_data[i] = [transed]
                        else:
                            temp_data[i] = [self._nan] * (len(temp_data[0]) - 1) + [transed]
        self._dim = dims(len(temp_data[0]), len(temp_data))
        self._init_col_name(columns)
        for i, col in enumerate(self._columns):
            seq = temp_data[i]
            seq.extend([self._nan] * (self._dim.Ln - len(seq)))
            self._data[col] = seq

    def shuffle(self):
        new_index = list(range(self._dim.Ln))
        shuffles(new_index)
        self._arrange_by_index(new_index)

    def _values(self):
        for col in self.columns:
            yield self._data[col]

    def values(self):
        for col in self.columns:
            yield Series(self._data[col])
    

class Frame(BaseSheet):
    '''Maintains the data as records.
    '''

    def __init__(self, frame=None, columns=None, nan=None):
        self._data = []
        BaseSheet.__init__(self, frame, columns, nan)

    @property
    def info(self):
        new_m_v = map(str, self._missing)
        max_n = len(max(self._columns, key=len))

        info = ''
        for i in range(self._dim.Col):
            info += ' '*15
            info += self._columns[i].center(max_n) + '| '
            info += ' ' + new_m_v[i] + '\n'

        print('1.  Structure: DaPy.Frame\n' +\
              '2. Dimensions: Ln=%d | Col=%d\n' % self._dim +\
              '3. Miss Value: %d elements\n' % sum(self._missing) +\
              '4.    Columns: ' + 'Title'.center(max_n) + '|'+\
                             '  Miss\n'+ info)

    def _init_col(self, obj, columns):
        self._data = [list(record) for record in zip(*list(obj.values()))]
        self._missing = copy(obj._missing)
        self._dim = dims(obj._dim.Ln, obj._dim.Col)
        self._init_col_name(columns)

    def _init_frame(self, frame, columns):
        self._data = copy(frame._data)
        self._dim = copy(frame._dim)
        self._init_col_name(columns)
        self._missing = copy(frame._missing)

    def _init_dict(self, frame, columns):
        frame = copy(frame)
        self._dim = dims(max(map(len, frame.values())), len(frame))
        self._missing = [0] * self._dim.Col
        self._init_col_name(columns)
        for i, (title, col) in enumerate(frame.items()):
            mv, sequence = self._check_sequence_type(col, self._dim.Ln)
            frame[title] = sequence
            self._missing[i] += mv
        self._data = [list(record) for record in zip(*frame.values())]

    def _init_like_table(self, frame, columns):
        self._data = map(list, frame)
        dim_Col, dim_Ln = len(max(self._data, key=len)), len(frame)
        self._dim = dims(dim_Ln, dim_Col)
        self._missing = [0] * self._dim.Col

        for i, item in enumerate(self._data):
            if len(item) < dim_Col:
                item.extend([self._nan] * (dim_Col - len(item)))
            for j, value in enumerate(item):
                if value == self.nan or value is self.nan:
                    self._missing[j] = self._missing[j] + 1
        self._init_col_name(columns)

    def _init_like_seq(self, frame, columns):
        self._data = [[value,] for value in frame]
        self._dim = dims(len(frame), 1)
        self._init_col_name(columns)
        self._missing.append(self._check_sequence_type(frame, len(frame))[0])

    def __repr__(self):
        return self.show(30)

    def _getslice_col(self, i, j):
        new_data = [record[i : j+1] for record in self._data]
        return Frame(new_data, self._columns[i : j+1], self._nan)

    def _getslice_ln(self, i, j, k):
        return Frame(self._data[i:j:k], self._columns, self._nan)

    def __getitem__(self, interval):
        if isinstance(interval, int):
            return Row(self, interval)

        elif isinstance(interval, slice):
            return self.__getslice__(interval.start, interval.stop)

        elif isinstance(interval, STR_TYPE):
            col = self._columns.index(interval)
            return [item[col] for item in self._data]

        elif isinstance(interval, (tuple, list)):
            return_obj = Frame()
            return self._getitem_by_tuple(interval, return_obj)

        else:
            raise TypeError('item should be represented as `slice`, `int`, `str` or `tuple`.')

    def __iter__(self):
        for i in range(self._dim.Ln):
            yield Row(self, i)

    def _replace_typical(self, col, cond, new):
        col, cond, new = self._check_replace_condition(col, cond, new)
        if isinstance(cond, STR_TYPE):
            cond = self._trans_where(cond, axis=1)
        for line in self._data:
            for j, value in enumerate(line):
                if cond(value) is True:
                    line[j] = new

    def append_row(self, item):
        '''append a new record to the Frame tail
        '''
        item = self._add_row(item)
        self._data.append(item)

    def append_col(self, series, variable_name=None):
        '''append a new variable to the current records tail
        '''
        mv, series = self._check_sequence_type(series, self._dim.Ln)
        size = len(series) - self._dim.Ln
        if size > 0:
            self._missing = [m+size for m in self._missing]
            self._data.extend([[self._nan] * self._dim.Col for i in range(size)])

        self._missing.append(mv)
        for record, element in zip(self._data, series):
            record.append(element)
        self._columns.append(self._check_col_new_name(variable_name))
        self._dim = dims(max(self._dim.Ln, len(series)), self._dim.Col+1)

    def count(self, X, point1=None, point2=None):
        if is_value(X):
            X = (X,)
        counter = Counter()
        L1, C1, L2, C2 = self._check_area(point1, point2)

        for record in self._data[L1:L2 + 1]:
            for value in record[C1:C2 + 1]:
                if value in X:
                    counter[value] += 1

        if len(X) == 1:
            return counter[X[0]]
        return dict(counter)

    def extend(self, other):
        if isinstance(other, Frame):
            new_title = 0
            for title in other._columns:
                if title not in self._columns:
                    self._columns.append(title)
                    new_title += 1

            for record in self._data:
                record.extend([self._nan] * new_title)

            extend_part = [[self._nan] * len(self._columns)\
                           for i in range(len(other))]
            new_title_index = [self._columns.index(title)
                               for title in other._columns]
            self._dim = dims(len(self) + len(other), len(self._columns))
            self._missing.extend([self._dim.Ln] * new_title)

            for i, record in enumerate(other._data):
                for j, value in zip(new_title_index, record):
                    if value == other._nan:
                        value = self._nan
                    extend_part[i][j] = value
            self._data.extend(extend_part)

        elif isinstance(other, SeriesSet):
            self.extend(Frame(other))

        else:
            self.extend(Frame(other, self._columns))

    def join(self, other):
        if isinstance(other, Frame):
            for title in other._columns:
                self._columns.append(self._check_col_new_name(title))
            self._missing.extend(other._missing)

            for i, record in enumerate(other._data):
                if i < self._dim.Ln:
                    current_record = self._data[i]
                else:
                    current_record = [self._nan] * self._dim.Col
                    self._data.append(current_record)
                for value in record:
                    if value == other.nan:
                        value = self._nan
                    current_record.append(value)
            if i < self._dim.Ln:
                for record in self._data[i+1:]:
                    record.extend([self._nan] * other.shape.Col)
            self._dim = dims(len(self._data), len(self._columns))

        else:
            self.join(Frame(other, nan=self.nan))

    def insert_row(self, index, item):
        '''insert a new record to the frame with position `index`
        '''
        item = self._add_row(item)
        self._data.insert(index, item)

    def insert_col(self, index, series, variable_name=None):
        '''insert a new variable to the current records in position `index`
        '''
        mv, series = self._check_sequence_type(series)

        size = len(series) - self._dim.Ln
        if size > 0:
            for i in range(self._dim.Col):
                self._missing[i] += size
            self._data.extend([[self._nan] * self._dim.Col\
                                for i in range(size)])

        self._missing.insert(index, mv)
        for i, element in enumerate(series):
            self._data[i].insert(index, element)

        self._columns.insert(index, self._check_col_new_name(variable_name))
        self._dim = dims(max(self._dim.Ln, size), self._dim.Col+1)

    def items(self):
        for i, sequence in enumerate(zip(*self._data)):
            yield self._columns[i], list(sequence)

    def keys(self):
        return self._columns

    def pop_row(self, pos=-1):
        '''pop(remove & return) a record from the Frame
        '''
        assert isinstance(pos, (int, list, tuple)), 'an int or ints in list is required.'
        if isinstance(pos, int):
            pos = [pos,]
        pos = sorted(pos, reverse=True)
        pop_item = Frame([self._data.pop(pos_) for pos_ in pos], list(self._columns))
        self._dim = dims(self._dim.Ln - len(pos), self._dim.Col)
        self._missing = map(lambda x, y: x - y, self._missing, pop_item._missing)
        return pop_item

    def pop_col(self, pos=-1):
        '''pop(remove & return) a series from the Frame
        '''
        pop_name = self._check_columns_index(pos)
        for name in pop_name:
            index = self._columns.index(name)
            self._columns.pop(index)
            self._missing.pop(index)

        pop_data = [[] for i in range(len(pop_name))]
        new_data = [0] * self._dim.Ln
        for j, record in enumerate(self._data):
            line = []
            for i, value in enumerate(record):
                if i in pop_name:
                    pop_data[pop_name.index(i)].append(value)
                else:
                    line.append(value)
            new_data[j] = line

        self._dim = dims(self._dim.Ln, self._dim.Col-len(pos))
        self._data = new_data
        return SeriesSet(dict(zip(pop_name, pop_data)))

    def dropna(self, axis='LINE'):
        '''pop all records that maintains miss value while axis is `LINE` or
        pop all variables that maintains miss value while axis is `COL`
        '''
        pops = []
        if str(axis).upper() in ('0', 'LINE'):
            for i, record in enumerate(self._data):
                if self._nan in record:
                    pops.append(i)

        if str(axis).upper() in ('1', 'COL'):
            for i, sequence in enumerate(zip(*self._data)):
                if self._nan in sequence:
                    pops.append(self._columns[i])

        if len(pops) != 0:
            self.__delitem__(pops)

    def from_file(self, addr, **kwrd):
        '''read dataset from csv or txt file.
        '''
        with open(addr, 'r') as f:
            freader, col_types, nan, prefer = self._check_read_text(f, **kwrd)
            self._data = []
            try:
                for record in freader:
                    line = [self._trans_str2val(
                            i, v, col_types, nan, prefer) \
                            for i, v in enumerate(record)]
                    if len(line) != self._dim.Col:
                        line.extend([self._nan] * (self._dim.Col - len(line)))
                    self._data.append(line)
            except MemoryError:
                self._dim = dims(len(self._data), self._dim.Col)
                warn('since the limitation of memory, DaPy can not read the'+\
                     ' whole file.')

    def reverse(self):
        self._data.reverse()

    def shuffle(self):
        shuffles(self._data)

    def values(self):
        for sequence in zip(*self._data):
            yield list(sequence)
