from collections import namedtuple, OrderedDict, Counter
from copy import deepcopy
from csv import reader
from datetime import date, datetime
from tools import is_seq, is_iter, is_math, is_value, get_sorted_index
from tools import str2value, transfer_funcs
from random import random, shuffle as shuffles
from re import search as re_search

__all__ = ['SeriesSet', 'Frame']

dims = namedtuple('sheet', ['Ln', 'Col'])

class BaseSheet(object):
    '''
    Attributes
    ----------
    _columns : str in list
        the titles in each columns.

    _dim : namedtuple
        the two dimensional span of this data set.

    _miss_symbol : value
        the symbol represented miss value in current seriesset.

    _miss_value : values in list
        the number of miss value in each column.

    _data : OrderedDict or list
        the OrderedDict object contains all the data by columns.
    '''

    def __init__(self, obj=None, columns=None,
                 miss_symbol=None, miss_value=None):
        self._miss_symbol = miss_value
        self._miss_value = []
        miss_symbol = self._check_miss_symbol(miss_symbol)

        if obj is None and columns is None:
            self._columns = []
            self._dim = dims(0, 0)

        elif obj is None and columns is not None:
            if isinstance(columns, str):
                columns = [columns,]
            self._dim, self._columns = dims(0, 0), []
            for name in columns:
                self.append_col([], name)

        elif isinstance(obj, SeriesSet):
            self._init_col(obj)

        elif isinstance(obj, Frame):
            self._init_frame(obj, miss_symbol, columns)

        elif isinstance(obj, (dict, OrderedDict)):
            self._init_dict(obj, miss_symbol, columns)

        elif hasattr(obj, 'items'):
            self._init_dict(dict(obj.items()), miss_symbol, columns)

        elif is_seq(obj) and all(map(is_seq, obj)):
            self._init_like_table(obj, miss_symbol, columns)

        elif is_seq(obj) and all(map(is_value, obj)):
            self._init_like_seq(obj, miss_symbol, columns)

        else:
            raise TypeError("sheet structure does not " +\
                            "support %s type." % type(obj))

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._dim

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, item):
        self._init_col_name(item)

        if isinstance(self._data, list):
            return

        new_data = OrderedDict()
        for i, value in enumerate(self._data.values()):
            new_data[self._columns[i]] = value
        del self._data
        self._data = new_data

    @property
    def miss_symbol(self):
        return self._miss_symbol

    @miss_symbol.setter
    def miss_symbol(self, item):
        if not is_value(item):
            raise TypeError('miss value should be a value object, not %s' % type(item))

        if isinstance(self._data, list):
            for record in self._data:
                for j, value in enumerate(record):
                    if value == self._miss_symbol:
                        record[j] = item
        else:
            for i, mv in enumerate(self._miss_value):
                if mv != 0:
                    seq = self._data[self.columns[i]]
                    for j, value in enumerate(seq):
                        if value == self._miss_symbol:
                            seq[j] = item

        self._miss_symbol = item

    @property
    def miss_value(self):
        return sum(self._miss_value)

    def __getattr__(self, name):
        if name in self._columns:
            return self.__getitem__(name)
        raise AttributeError("DaPy.sheet object has no attribute '%s'" % name)

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
                for variable in self.values():
                    if variable == e:
                        return True

        if is_value(other):
            for record in self:
                for value in record:
                    if value == other:
                        return True
        return False

    def __eq__(self, other):
        if is_seq(other):
            other = Frame(other)
            if other.shape != self.shape:
                return False
            for a, b in zip(self, other):
                if a != b:
                    return False
            return True
        return False

    def __setitem__(self, key, value):
        if isinstance(key, int):
            if key <= self._dim.Ln:
                self.__delitem__(key)
                self.insert(key, value)
            else:
                self.append(value)

        elif isinstance(key, str):
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
                "same time. Try: S['A':'B'][3:10] or S[3:10]['A':'B']"
        for arg in interval:
            if isinstance(arg, str):
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
                obj.append(self.__getitem__(arg))
            else:
                raise TypeError('bad statement as [%s:%s]' % (start, stop))
        return obj

    def __delitem__(self, key):
        if isinstance(key, int):
            self.pop(key)

        elif isinstance(key, str) and key in self._columns:
            self.pop_col(key)

        elif isinstance(key, tuple):
            for every in key:
                self.__delitem__(every)
        else:
            raise KeyError('%s should be int, str or slice.'%key)

    def __getslice__(self, start, stop, step=1):
        if start in self._columns or stop in self._columns:
            return self._getslice_col(*self._check_slice_pos_col(start, stop))
        elif not (not isinstance(start, int) and not isinstance(stop, int)):
            return self._getslice_ln(start, stop, step)
        raise TypeError('bad expression as [%s:%s]' % (start, stop))

    def __getstate__(self):
        instance = self.__dict__.copy()
        instance['_dim'] = tuple(self._dim)
        if isinstance(self._data, OrderedDict):
            instance['_data'] = dict(self._data)
        return instance

    def __setstate__(self, dict):
        '''load this object from a stream file'''
        self._dim = dims(*dict['_dim'])
        self._columns = dict['_columns']
        self._miss_value = dict['_miss_value']
        self._miss_symbol = dict['_miss_symbol']
        if isinstance(dict['_data'], type(dict)):
            self._data = OrderedDict()
            for col in self._columns:
                self._data[col] = dict['_data'][col]
        else:
            self._data = dict['_data']

    def _init_col_name(self, columns):
        if isinstance(columns, str):
            self._columns = [columns + '_%d' % i for i in range(self._dim.Col)]
        elif columns is None or str(columns).strip() == '':
            self._columns = ['C_%d' % i for i in range(self._dim.Col)]
        elif is_seq(columns):
            self._columns = []
            for col in columns[:self._dim.Col]:
                self._columns.append(self._check_col_new_name(col))
            for i in range(self._dim.Col - len(self._columns)):
                self._columns.append(self._check_col_new_name(None))
        else:
            raise TypeError('Column names should be stored in a iterable, not %s' % type(columns))

    def _transform_str_condition(self, cond, axis=0):
        if axis == 0:
            for i in get_sorted_index(self._columns, key=len, reverse=True):
                cond = cond.replace(self._columns[i], 'record[%d]'%i)
            record = self.__getitem__(0) # test record

        elif axis == 1:
            opeartes = {'and': 3, 'or': 2}
            for opearte, bias in opeartes.items():
                counts = cond.count(opearte)
                index = 0
                for i in range(counts):
                    index = cond.index(opearte, index + bias) + bias
                    cond = cond[: index] + ' value' + cond[index: ]
            cond = 'value ' + cond
            value = 30
        else:
            raise AttributeError('axis shoud be 1 or 0.')
        eval(cond)
        try:
            eval(cond)
        except:
            raise SyntaxError('we have some problem in transforming '
                            'your statement, please check syntax again.')
        return cond

    def _check_sequence_type(self, series, miss_symbol):
        if is_value(series):
            return 0, [series] * self._dim.Ln

        if not is_seq(series):
            raise TypeError("can't transfer type `%s`" % type(series) +\
                            "SeriesSet.")

        miss_symbol = self._check_miss_symbol(miss_symbol)
        mv, series = 0, list(series)

        if len(series) < self._dim.Ln:
            series.extend([self._miss_symbol] * (self._dim.Ln - len(series)))

        for i, element in enumerate(series):
            if element in miss_symbol:
                series[i] = self._miss_symbol
                mv += 1
        return mv, series

    def _check_replace_condition(self, col, condition, new_value):
        if col is all:
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

        if not isinstance(condition, str):
            raise TypeError('condition should be written in a python '+\
                            'syntax str object.')

        if not is_value(new_value):
            raise TypeError('SeriesSet does not support %s ' % type(new_value),
                            'as a value type.')
        return col, condition, new_value

    def _check_col_new_name(self, new_name):
        if not new_name:
            return self._check_col_new_name('C_%d' % len(self._columns))
        
        if str(new_name) not in self._columns:
            return str(new_name)

        start_no, titles = 1, ','.join(self._columns) + ','
        while True:
            if not re_search('%s_%d,' % (new_name, start_no), titles):
                return '%s_%d' % (new_name, start_no)
            start_no += 1

    def _check_slice_pos_col(self, i, j):
        if i in self._columns:
            i = self._columns.index(i)
        elif i is None:
            i = 0
        else:
            raise ValueError('can not get the title of %s'%j)

        if j in self._columns:
            j =  self._columns.index(j)
        elif j is None:
            j = self._dim.Col - 1
        else:
            raise ValueError('can not get the title of %s'%j)
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

    def _check_miss_symbol(self, miss_symbol):
        if is_value(miss_symbol):
            return (miss_symbol, self._miss_symbol)
        return tuple(miss_symbol)

    def _check_read_text(self, f, **kwrd):
        first_line = kwrd.get('first_line', 1)
        miss_symbol = kwrd.get('miss_symbol', 'NA')
        miss_symbol = self._check_miss_symbol(miss_symbol)
        self._miss_value = kwrd.get('miss_value', self._miss_value)
        title_line = kwrd.get('title_line', 0)
        sep = kwrd.get('sep', ',')
        prefer_type = kwrd.get('prefer_type', None)

        freader = reader(f, delimiter=sep)
        col = max([len(line) for l_n, line in enumerate(freader)])
        self._miss_value = [0] * col
        self._dim = dims(l_n+1-first_line, col)
        if isinstance(prefer_type, str):
            _col_types = [str] * col
        else:
            _col_types = [None] * col
        if title_line < 0:
            self._init_col_name(None)

        f.seek(0)
        for i, lines in enumerate(freader):
            if i >= first_line:
                for j, item in enumerate(lines):
                    if j in _col_types or item in miss_symbol:
                        continue
                    _col_types[j] = transfer_funcs[type(
                                str2value(item, prefer_type))]
            elif i == title_line:
                if len(lines) < self._dim.Col:
                    lines.extend(['C_%d' % i for i in range(len(self._columns),
                                                col - len(self._columns))])
                self._init_col_name(lines)

            if all(_col_types):
                break
        f.seek(0)
        if first_line != 0:
            for m, record in enumerate(freader):
                if m >= first_line - 1 :
                    break
        return freader, tuple(_col_types), miss_symbol, prefer_type

    def _check_transform_value(self, i, item, _col_types,
                                miss_symbol, prefer_type):
        try:
            if item in miss_symbol:
                self._miss_value[i] += 1
                return self._miss_symbol
            return _col_types[i](item)
        except ValueError:
            return str2value(item, prefer_type)

    def show(self, lines='all'):
        if not isinstance(lines, int) and not isinstance(lines, str):
            raise TypeError('parameter `lines` should be an int() or keyword `all`.')

        if lines == 'all' or 2*lines >= self._dim.Ln:
            lines, omit = -1, 0
            if isinstance(self._data, list):
                temporary_data = self._data
            else:
                temporary_data = zip(*[sequence for sequence in self.values()])
        elif lines <= 0:
            raise ValueError('parameter `lines` should be greater than 0.')
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
                line += ' ' + str(value).center(column_size[i]) + ' ' + '|'
            frame += line[:-1] + '\n'
        return frame[:-1]


class SeriesSet(BaseSheet):
    '''Every variable will be stored by a sequene.
    '''
    def __init__(self, series=None, columns=None,
                 miss_symbol=None, miss_value=None):
        self._data = OrderedDict()
        BaseSheet.__init__(self, series, columns, miss_symbol, miss_value)

    @property
    def info(self):
        from DaPy import describe
        mins, maxs, avgs, stds = [], [], [], []
        for sequence in self._data.values():
            d = describe(sequence)
            mins.append(str(d.Min))
            maxs.append(str(d.Max))
            try:
                avgs.append('%.2f'%d.Mean)
            except TypeError:
                avgs.append(' - ')
            try:
                stds.append('%.2f'%d.S)
            except TypeError:
                stds.append(' - ')

        miss = map(str, self._miss_value)

        blank_size = [max(len(max(self._columns, key=len)), 5) + 2,
                      max(len(max(miss, key=len)), 4) + 2,
                      max(len(max(mins, key=len)), 3) + 2,
                      max(len(max(maxs, key=len)), 3) + 2,
                      max(len(max(avgs, key=len)), 4) + 2,
                      max(len(max(stds, key=len)), 3)]

        # Draw the title line of description
        title_line = 'Title'.center(blank_size[0]) + '|' +\
                     'Miss'.center(blank_size[1]) +'|' +\
                     'Min'.center(blank_size[2]) + '|' +\
                     'Max'.center(blank_size[3]) + '|' +\
                     'Mean'.center(blank_size[4]) + '|' +\
                     'Stdev'.center(blank_size[5]) + '\n'
        for lenth in blank_size:
            title_line += '-'*lenth + '+'

        # Draw the main table of description
        info = str()
        for i, title in enumerate(self._columns):
            info += title.center(blank_size[0]) + '|'
            info += miss[i].center(blank_size[1]) + '|'
            info += mins[i].center(blank_size[2]) + '|'
            info += maxs[i].center(blank_size[3]) + '|'
            info += avgs[i].center(blank_size[4]) + '|'
            info += stds[i].center(blank_size[5]) + '\n'

        print('1.  Structure: DaPy.SeriesSet\n' +\
              '2. Dimensions: Ln=%d | Col=%d\n'%self._dim +\
              '3. Miss Value: %d elements\n'%sum(self._miss_value) +\
              '4.   Describe: \n'+\
              title_line[:-1] + '\n' +\
              info + '='*(6 + sum(blank_size)))

    def _init_col(self, series):
        self._data = deepcopy(series._data)
        self._columns = deepcopy(series._columns)
        self._dim = deepcopy(series._dim)
        self._miss_value = deepcopy(series._miss_value)
        self._miss_symbol = deepcopy(series._miss_symbol)

    def _init_dict(self, series, miss_symbol, columns):
        max_Ln = max(map(len, series.itervalues()))
        self._dim = dims(max_Ln, len(series))
        if columns is None:
            self._init_col_name(series.keys())
        else:
            self._init_col_name(columns)
        for i, sequence in enumerate(series.values()):
            mv, sequence = self._check_sequence_type(sequence, miss_symbol)
            self._miss_value.append(mv)
            self._data[self._columns[i]] = sequence

    def _init_frame(self, series, miss_symbol, columns):
        self._dim = dims(series._dim.Ln, series._dim.Col)
        self._miss_value = deepcopy(series._miss_value)
        if columns is None:
            columns = deepcopy(series._columns)
        self._init_col_name(columns)

        for sequence, title in zip(zip(*series), self._columns):
            self._data[title] = list(sequence)

        if self._miss_symbol != series._miss_symbol:
            for seq in self._data.values():
                for i, value in enumerate(seq):
                    if value in miss_symbol:
                        seq[i] = self._miss_symbol

    def _init_like_seq(self, series, miss_symbol, columns):
        self._dim = dims(len(series), 1)
        self._init_col_name(columns)
        mv, series = self._check_sequence_type(series, miss_symbol)
        self._miss_value = [mv,]
        self._data[self._columns[0]] = series

    def _init_like_table(self, series, miss_symbol, columns):
        lenth_Col = len(max(series, key=len))
        for pos, record in enumerate(series):
            if len(record) < lenth_Col:
                new = list(record)
                new.extend([self._miss_symbol] * (lenth_Col - len(record)))
                series[pos] = new

        self._dim = dims(len(series), lenth_Col)
        self._init_col_name(columns)
        self._miss_value = [0] * self._dim.Col

        for j, sequence in enumerate(zip(*series)):
            mv, series = self._check_sequence_type(sequence, miss_symbol)
            self._miss_value[j] += mv
            self._data[self._columns[j]] = series

    def __repr__(self):
        if self._dim.Ln > 10:
            def write_Ln(i, title, blank):
                item = self._data[title]
                msg = ' '*blank + title + ': <'
                msg += ', '.join([str(value) for value in item[:5]])
                msg += ', ... ,'
                msg += ', '.join([str(value) for value in item[-5:]])
                msg +=  '>\n'
                return msg
        elif self._dim.Ln != 0:
            def write_Ln(i, title, blank):
                item = self._data[title]
                msg = ' '*blank + title + ': <'
                msg += ', '.join([str(value) for value in item])
                msg += '>\n'
                return msg
        else:
            return ''

        msg = str()
        size = len(max(self._columns, key=len))
        for i, title in enumerate(self._columns):
            msg += write_Ln(i, title, size - len(title))
        return msg[:-1]

    def _getslice_col(self, i, j):
        new_data = OrderedDict()
        for title in self._columns[i: j+1]:
            new_data[title] = self._data[title]
        return SeriesSet(new_data, miss_value=self._miss_symbol)

    def _getslice_ln(self, i, j, k):
        return_list = zip(*[self._data[t][i:j:k] for t in self._columns])
        return Frame(return_list, self._columns, None, self._miss_symbol)

    def __getitem__(self, interval):
        if isinstance(interval, int):
            return [self._data[title][interval] for title in self._columns]

        elif isinstance(interval, (tuple, list)):
            return_obj = SeriesSet(miss_value=self._miss_symbol)
            return self._getitem_by_tuple(interval, return_obj)

        elif isinstance(interval, slice):
            return self.__getslice__(interval.start, interval.stop)

        elif isinstance(interval, str):
            return self._data[interval]

        else:
            raise TypeError('SeriesSet index must be int, str and slice, '+\
                            'not %s' % str(type(interval)).split("'")[1])

    def __iter__(self):
        for i in range(self._dim.Ln):
            yield [self._data[title][i] for title in self._columns]

    def __reversed__(self):
        for i in range(self._dim.Ln-1, -1, -1):
            yield self[i]

    def __arrange_by_index(self, self_new_index=None, other_new_index=None):
        if self_new_index:
            for title, sequence in self._data.items():
                self._data[title] = [sequence[j] for j in self_new_index]
        elif other_new_index:
            for title, sequence in self._data.items():
                new_sequence = [None] * self._dim.Ln
                for index, value in zip(other_new_index, sequence):
                    new_sequence[index] = value
                self._data[title] = new_sequence
        else:
            raise RuntimeError('at least one parameter should be filled in.')

    def _replace_typical(self, col, condition, new_value):
        col, condition, new_value = self._check_replace_condition(col, condition, new_value)
        cond = self._transform_str_condition(condition, axis=1)
        seq = self._data[col]
        for i, value in enumerate(seq):
            if eval(cond) is True:
                seq[i] = new_value

    def append(self, item, miss_symbol=None):
        if is_value(item):
            item = [item] * self._dim.Col
        elif not all(map(is_value, item)):
            raise TypeError("insert item is not a support type "+\
                            "with `%s`"%type(item))

        miss_symbol = self._check_miss_symbol(miss_symbol)
        lenth_bias = len(item) - self._dim.Col
        if lenth_bias < 0:
            item.extend([self._miss_symbol] * abs(lenth_bias))
        elif lenth_bias > 0:
            for i in range(self._dim.Col, self._dim.Col + lenth_bias):
                self.append_col([self._miss_symbol] * self._dim.Ln,
                                miss_symbol=self._miss_symbol)

        for i, seq in enumerate(self._data.values()):
            element = item[i]
            if element in miss_symbol:
                self._miss_value[i] += 1
                element = self._miss_symbol
            seq.append(element)

        self._dim = dims(self._dim.Ln+1, self._dim.Col)

    def append_col(self, series, variable_name=None, miss_symbol=None): # OK #
        '''append a series data to the seriesset last
        '''
        variable_name = self._check_col_new_name(variable_name)
        miss_symbol = self._check_miss_symbol(miss_symbol)
        mv, series = self._check_sequence_type(series, miss_symbol)

        # check the lenth of data
        size = len(series)
        if size > self._dim.Ln:
            for i, title in enumerate(self._columns):
                self._miss_value[i] += size - self._dim.Ln
                self._data[title].extend([self._miss_symbol] *(\
                    size - self._dim.Ln))

        self._columns.append(variable_name)
        self._miss_value.append(mv)
        self._dim = dims(size, self._dim.Col+1)
        self._data[variable_name] = series

    def corr(self):
        '''correlation between variables in data -> Frame object
        '''
        from DaPy import corr as f_c
        new_ = Frame([[0] * self._dim.Col for i in range(self._dim.Col)],
                     self._columns)
        for i, sequence in enumerate(self._data.values()):
            for j, next_sequence in enumerate(self._data.values()):
                if i == j:
                    new_[j][i] = 1
                    continue
                r = f_c(sequence, next_sequence)
                new_[i][j], new_[j][i] = r, r
        new_.insert_col(0, self._columns, '_Subjects_')
        return new_

    def count(self, X, point1=None, point2=None):
        '''count X in area (point1, point2)-> Counter object
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
        return dict(counter)

    def count_element(self, col=all):
        if col == all:
            col = self._columns

        elif isinstance(col, int):
            if abs(col) < self._dim.Col:
                col = [self._columns[col]]
            else:
                raise TypeError("unrecognized expression '%s'"%str(col))

        elif col in self._columns:
            col = [col, ]

        elif is_iter(col):
            new = []
            for each in col:
                if each in self._columns:
                    new.append(each)
                elif isinstance(each, int) and abs(each) < self._dim.Col:
                    new.append(self._columns[each])
                else:
                    raise TypeError("`%s` is not a title name"%str(each))
            col = new
        else:
            raise TypeError("unrecognized expression `%s`"%str(col))

        collect_elements = dict()
        for title in col:
            c = Counter(self._data[title])
            collect_elements[title] = c
        return collect_elements

    def extend(self, item):
        '''extend the current SeriesSet with records in set.
        '''
        if isinstance(item, SeriesSet):
            for title, sequence in item.items():
                mv = sequence.count(item.miss_symbol)
                if title not in self._columns:
                    self._columns.append(self._check_col_new_name(title))
                    self._miss_value.append(self._dim.Ln + mv)
                    To = [self._miss_symbol] * self._dim.Ln
                else:
                    self._miss_value[self._columns.index(title)] += mv
                    To = self._data[title]
                To.extend(sequence)
                self._data[title] = To
            self._dim = dims(self._dim.Ln+item._dim.Ln, len(self._columns))

            for i, sequence in enumerate(self._data.values()):
                if len(sequence) != self._dim.Ln:
                    add_miss_size = self._dim.Ln - len(sequence)
                    sequence.extend([self._miss_symbol]*add_miss_size)
                    self._miss_value[i] += add_miss_size

        elif isinstance(item, Frame):
            self.extend(SeriesSet(item))

        elif all(filter(is_iter, item)):
            self.extend(SeriesSet(item, self._columns))

        else:
            raise TypeError('could not extend a single value only.')

    def extend_col(self, other):
        if isinstance(other, SeriesSet):
            lenth = max(self._dim.Ln, other._dim.Ln)
            for title, sequence in other.items():
                title = self._check_col_new_name(title)
                mv, sequence = self._check_sequence_type(sequence, other._miss_symbol)
                self._miss_value.append(mv)
                self._data[title] = sequence
                self._columns.append(title)
            self._dim = dims(lenth, len(self._columns))

        elif isinstance(other, Frame):
            self.extend_col(SeriesSet(other))

        elif all(filter(is_iter, other)):
            new_col = [title + '_1' for title in self._columns]
            self.extend_col(SeriesSet(other, new_col))

        else:
            raise TypeError('could not extend a single value only.')

    def items(self):
        return self._data.items()

    def insert(self, pos, item, miss_symbol=None): # OK #
        '''insert a record to the frame, position in <index>
        '''
        if is_value(item):
            item = [item] * self._dim.Col
        if not (isinstance(pos, int) and is_iter(item)):
            raise TypeError("insert item is not a support"+\
                            "type with %s"%type(item))

        miss_symbol = self._check_miss_symbol(miss_symbol)
        lenth_bias = len(item) - self._dim.Col
        if lenth_bias < 0:
            item.extend([self._miss_symbol] * abs(lenth_bias))
        elif lenth_bias > 0:
            for i in range(self._dim.Col, self._dim.Col + lenth_bias):
                self.append_col([self._miss_symbol] * self._dim.Ln,
                                miss_symbol=self._miss_symbol)
        for i, title in enumerate(self._columns):
            element = item[i]
            if element in miss_symbol:
                self._data[title] = list(self._data[title])
                self._data[title].insert(pos, self._miss_symbol)
                self._miss_value[i] += 1
            else:
                self._data[title].insert(pos, element)
        self._dim = dims(self._dim.Ln + 1, self._dim.Col)

    def insert_col(self, index, series, variable_name=None, miss_symbol=None):
        '''insert a series of data to the frame, position in <index>
        '''
        variable_name = self._check_col_new_name(variable_name)
        miss_symbol = self._check_miss_symbol(miss_symbol)
        mv, series = self._check_sequence_type(series, miss_symbol)

        size_series = len(series)
        if size_series < self._dim.Ln:
            series = list(series)
            mv_2 = self._dim.Ln - size_series
            series.extend([self._miss_symbol] * mv_2)
            mv += mv_2
            size_series = self._dim.Ln
        elif size_series > self._dim.Ln:
            for title in self._columns:
                self._data[title] = list(self._data[title])
                self._data[title].extend([self._miss_symbol] * (
                    size_series - self._dim.Ln))

        self._columns.insert(index, variable_name)
        self._dim = dims(size_series, self._dim.Col + 1)
        self._miss_value.insert(index, mv)

        new_data = OrderedDict()
        if index >= self._dim.Col:
            self._data[variable_name] = series
            return

        for i, title in enumerate(self._data):
            if i == index:
                new_data[variable_name] = series
            new_data[title] = self._data[title]
        self._data = new_data

    def keys(self):
        return self._data.keys()

    def normalized(self, process='NORMAL', col=all, attr=None, get_attr=None):
        if col is all:
            new_col = self._columns
        else:
            new_col = []
            for each in col:
                if each in self._columns:
                    new_col.append(each)
                else:
                    new_col.append(self._columns[each])

        from DaPy import describe
        if process == 'LOG':
            from DaPy import log
        attrs_dic = dict()
        if process == 'NORMAL':
            attrs_structure = namedtuple('Nr_attr', ['Min', 'Range'])
        elif process == 'STANDARD':
            attrs_structure = namedtuple('Sd_attr', ['Mean', 'S'])

        for i, title in enumerate(new_col):
            sequence = self._data[title]

            if attr:
                A, B = attr[title]

            elif process == 'NORMAL':
                statis = describe(sequence)
                A, B = statis.Min, statis.Range

            elif process == 'STANDARD':
                statis = describe(sequence)
                A, B = statis.Mean, statis.Sn

            elif process != 'LOG':
                raise ValueError("unrecognized symbol '%s',"%str(process)+\
                                 "use 'NORMAL', 'STANDARD' and 'LOG'")

            new = [0] * self._dim.Ln
            if process == 'LOG':
                for i, value in enumerate(sequence):
                    try:
                        new[i] = log(value)
                    except ValueError:
                        continue
            else:
                for i, value in enumerate(sequence):
                    try:
                        new[i] = (value - A) / B
                    except ZeroDivisionError:
                        continue
                    except TypeError:
                        continue
                self._data[title] = new

            try:
                attrs_dic[title] = attrs_structure(A, B)
            except UnboundLocalError:
                pass

        if get_attr:
            return attrs_dic
        return

    def merge(self, other, self_key=0, other_key=0, keep=True):
        other = SeriesSet(other)

        if isinstance(self_key, int):
            self_key = self._columns[self_key]
        elif not isinstance(self_key, str):
            raise TypeError('self_key shoule be a string or int type, which '+\
                            'representes the column.')
        if self_key not in self._columns:
            raise IndexError('`%s` is not the '%self_key +\
                             'element in this dataset.')

        if isinstance(other_key, int):
            other_key = other._columns[other_key]
        elif not isinstance(other_key, str):
            raise TypeError('other_key shoule be a string or int type, which '+\
                            'representes the column.')
        if other_key not in other._columns:
            raise IndexError('`%s` is not the '%other_key+\
                             'element in other dataset.')

        if keep not in (True, False, 'other', 'self'):
            raise ValueError('keep should be in (True, False, "other", "self").')

        change_name = []
        for i, col in enumerate(other.columns):
            self._miss_value.append(other._miss_value[i])
            col = self._check_col_new_name(col)
            self._columns.append(col)
            change_name.append(col)

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

        how_many_new_index = new_index - other.shape.Ln
        if how_many_new_index != 0:
            other.extend([[None] * other.shape.Col for i in range(how_many_new_index)])

        hash_temp_index = set(temp_index)
        for i in range(new_index):
            if i not in hash_temp_index:
                temp_index.append(i)
        other.__arrange_by_index(None, temp_index)

        if other.miss_symbol != self._miss_symbol:
            other.symbol = self._miss_symbol
        for title, seq in zip(change_name, other.values()):
            self._data[title] = seq

        self._dim = dims(new_index, len(self._columns))
        for seq in self._data.values():
            seq.extend([None] * (self._dim.Ln - len(seq)))

    def pop_miss_value(self, axis='COL'):
        if str(axis).upper() == 'COL':
            pops = dict()
            for i in range(self._dim.Col-1, -1, -1):
                if self._miss_value[i] != 0:
                    pops[self._columns.pop(i)] = self._data.pop(self._columns[i])
            self._dim = dims(self._dim.Ln, len(self._columns))
            self._miss_value = [0 for i in xrange(self._dim.Col)]
            if pops:
                return SeriesSet(pops)
            return None

        elif str(axis).upper() == 'LINE':
            drop_line_index = []
            for sequence in self._data.values():
                for i, v in enumerate(sequence):
                    if v == self._miss_symbol:
                        drop_line_index.append(i)
            drop = sorted(set(drop_line_index), reverse=True)
            pops = []
            for sequence in self._data.values():
                pops.append([list(sequence).pop(index) for index in drop])
            self._dim = dims(self._dim.Ln - len(drop),
                                       len(self._columns))
            self._miss_value = [0 for i in range(self._dim.Col)]
            if pops:
                return Frame(zip(*pops), self._columns)
            return None
        else:
            raise IndexError("axis should be 'COL' or 'LINE'")

    def select(self, conditions):
        '''pick out records that comply with conditions
        '''
        condition = self._transform_str_condition(conditions, axis=0)
        pick_record_line = []
        for i, record in enumerate(self):
            if eval(condition) is True:
                pick_record_line.append(i)

        return_dict = OrderedDict()
        for title, sequence in self._data.items():
            return_dict[title] = [sequence[i] for i in pick_record_line]

        return SeriesSet(return_dict, self._columns)

    def pop(self, pos=-1):
        '''pop(remove & return) a record from the Frame
        '''
        if not isinstance(pos, int):
            raise TypeError('an integer is required.')

        pop_item = [sequence.pop(pos) for sequence in self._data.values()]

        self._dim = dims(self._dim.Ln - 1, self._dim.Col)
        for i, each in enumerate(pop_item):
            if self._miss_symbol == each:
                self._miss_value[i] -= 1
        return pop_item

    def pop_col(self, *titles):
        pop_name = []
        if not titles:
            titles = [self._columns[-1]]
        for item in titles:
            if isinstance(item, str):
                if item in self._columns:
                    pop_name.append(item)
                else:
                    raise ValueError("`%s` is not in columns title"%item)
            elif isinstance(item, int) and item < self._dim.Col:
                pop_name.append(self._columns[item])
            else:
                raise TypeError('an integer or string, which stands for a '+\
                                'column name is required,'+
                                'not {0} !'.format(type(item)))
        pop_data = dict()
        for title in pop_name:
            pos = self._columns.index(title)
            pop_data[title] = self._data.pop(title)
            self._columns.pop(pos)
            self._miss_value.pop(pos)
        self._dim = dims(self._dim.Ln, self._dim.Col-len(pop_name))
        return SeriesSet(pop_data, pop_name)

    def reverse(self, axis='COL'):
        if axis.upper() == 'COL':
            self._columns.reverse()
            self._miss_value.reverse()
            new_data = OrderedDict()
            for key in reversed(self._data):
                new_data[key] = self._data[key]
            del self._data
            self._data = new_data
        else:
            for sequence in self._data.values():
                sequence.reverse()

    def read_text(self, addr, **kwrd):
        '''read dataset from csv or txt file.

        Attribute
        ---------
        addr : str
            address of source file.

        first_line : int (default=1)
            the first line with data.

        miss_symbol : str (default="NA")
            the symbol of missing value in csv file.

        title_line : int (default=0)
            the line with title, rules design as follow:
            -1 -> there is no title inside;
            >=0 -> the titleline.

        sep : str (default=",")
            the delimiter symbol inside.

        prefer_type : 
        '''
        with open(addr, 'r') as f:
            freader, col_types, miss_symbol, prefer = self._check_read_text(f, **kwrd)
            datas = [[self._miss_symbol] * self._dim.Ln\
                     for i in range(self._dim.Col)]
            try:
                for m, record in enumerate(freader):
                    for i, v in enumerate(record):
                        datas[i][m] = self._check_transform_value(i, v,
                                    col_types, miss_symbol, prefer)
            except MemoryError:
                self._dim = dims(m+1, self._dim.Col)
                warn('since the limitation of memory, DaPy cannot'+\
                     'read the whole file.')
        self._data = OrderedDict(zip(self._columns, datas))

    def values(self):
        return self._data.values()

    def replace(self, *arg):
        if arg == tuple():
            raise KeyError('argument is empty!')

        elif len(arg) == 3 and is_value(arg[-1]) and isinstance(arg[1], str):
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
                "                       ('C2', '> 200', 3)). Use help() "+\
                'for using details.')

    def sort(self, *orders): 
        '''orders as tuple like (column_name, 'DESC')
        '''
        if len(orders) > 1:
            new_ = Frame(self)
            new_.sort(*orders)
            new = SeriesSet(new_)
            self._data = new._data
            return

        if isinstance(orders[0][0], int):
            compare_title = self._columns[orders[0][0]]
        elif orders[0][0] in self._columns:
            compare_title = orders[0][0]
        else:
            raise TypeError("'%s' keyword is "%orders[0][0] +\
                            "not in frame's columns.")

        if orders[0][1] == 'DESC':
            reverse = True
        elif orders[0][1] == 'ASC':
            reverse = False
        else:
            raise TypeError("`%s` is not a recognized symbol." % orders[0][1])

        new_index = get_sorted_index(self._data[compare_title], reverse=reverse)
        self.__arrange_by_index(new_index)

    def shuffle(self):
        new_index = range(self._dim.Ln)
        shuffles(new_index)
        self.__arrange_by_index(new_index)


class Frame(BaseSheet):
    '''Maintains the data as records.
    '''

    def __init__(self, frame=None, columns=None,
                 miss_symbol=None, miss_value=None):
        self._data = []
        BaseSheet.__init__(self, frame, columns,
                            miss_symbol, miss_value)

    @property
    def info(self):
        new_m_v = map(str, self._miss_value)
        max_n = len(max(self._columns, key=len))

        info = ''
        for i in range(self._dim.Col):
            info += ' '*15
            info += self._columns[i].center(max_n) + '| '
            info += ' ' + new_m_v[i] + '\n'

        print('1.  Structure: DaPy.Frame\n' +\
              '2. Dimensions: Ln=%d | Col=%d\n'%self._dim +\
              '3. Miss Value: %d elements\n'%sum(self._miss_value) +\
              '4.    Columns: ' + 'Title'.center(max_n) + '|'+\
                             '  Miss\n'+ info)

    def _init_col(self, obj):
        self._data = [list(record) for record in zip(*obj._data.values())]
        self._miss_value = deepcopy(obj._miss_value)
        self._columns = deepcopy(obj._columns)
        self._dim = dims(obj._dim.Ln, obj._dim.Col)

    def _init_frame(self, frame, miss_symbol, columns):
        self._data = deepcopy(frame._data)
        self._columns = deepcopy(frame._columns)
        self._dim = deepcopy(frame._dim)
        self._miss_value = deepcopy(frame._miss_value)

    def _init_dict(self, frame, miss_symbol, columns):
        self._dim = dims(max(map(len, frame.values())), len(frame))
        self._miss_value = [0] * self._dim.Col
        if columns is None:
            self._init_col_name(frame.keys())
        else:
            self._init_col_name(columns)
        for i, (title, col) in enumerate(frame.items()):
            mv, sequence = self._check_sequence_type(col, miss_symbol)
            frame[title] = sequence
            self._miss_value[i] += mv

        self._data = [list(record) for record in zip(*frame.values())]

    def _init_like_table(self, frame, miss_symbol, columns):
        self._data = map(list, frame)
        dim_Col, dim_Ln = len(max(self._data, key=len)), len(frame)
        self._dim = dims(dim_Ln, dim_Col)
        self._miss_value = [0] * self._dim.Col

        for i, item in enumerate(self._data):
            if len(item) < dim_Col:
                item.extend([self._miss_symbol] * (dim_Col - len(item)))
            for j, value in enumerate(item):
                if value in miss_symbol:
                    self._miss_value[j] = self._miss_value[j] + 1
        self._init_col_name(columns)

    def _init_like_seq(self, frame, miss_symbol, columns):
        self._data = [[value,] for value in frame]
        self._miss_value.append(0)
        self._dim = dims(len(frame), 1)
        self._init_col_name(columns)
        for i, value in enumerate(self._data):
            if value in miss_symbol:
                self._data[i] = self._miss_symbol
                self._miss_value[0] += 1

    def __repr__(self):
        return self.show(30)

    def _getslice_col(self, i, j):
        new_data = [record[i : j+1] for record in self._data]
        return Frame(new_data, self._columns[i : j+1],
                     self._miss_symbol)

    def _getslice_ln(self, i, j, k):
        return Frame(self._data[i:j:k], self._columns, self._miss_symbol)

    def __getitem__(self, interval):
        if isinstance(interval, int):
            return self._data[interval]

        elif isinstance(interval, slice):
            return self.__getslice__(interval.start, interval.stop)

        elif isinstance(interval, str):
            col = self._columns.index(interval)
            return [item[col] for item in self._data]

        elif isinstance(interval, (tuple, list)):
            return_obj = Frame()
            return self._getitem_by_tuple(interval, return_obj)

        else:
            raise TypeError('item should be represented as `slice`, `int`, `str` or `tuple`.')

    def __iter__(self):
        for record in self._data:
            yield record

    def _replace_typical(self, col, condition, new_value):
        col, condition, new_value = self._check_replace_condition(col, condition, new_value)
        cond = self._transform_str_condition(condition, axis=1)
        for record in self._data:
            value = record[col]
            if eval(cond) is True:
                record[col] = new_value

    def append(self, item, miss_symbol=None):
        '''append a new record to the Frame tail
        '''
        if is_value(item):
            item = [item] * self._dim.Col
        elif not all(map(is_value, item)):
            raise TypeError("append item is not a support type "+\
                                "with <'%s'>"%type(item))

        miss_symbol = self._check_miss_symbol(miss_symbol)
        for i, element in enumerate(item):
            if element in miss_symbol:
                self._miss_value[i] += 1

        lenth_bias = len(item) - self._dim.Col
        if lenth_bias < 0:
            item.extend([self._miss_symbol] * abs(lenth_bias))
        elif lenth_bias > 0:
            for record in self._data:
                record.extend([self._miss_symbol] * lenth_bias)

        self._data.append(item)
        self._dim = dims(self._dim.Ln + 1, max(self._dim.Col, len(item)))

    def append_col(self, series, variable_name=None, miss_symbol=None):
        '''append a new variable to the current records tail
        '''
        if miss_symbol is not None:
            miss_symbol = self._miss_symbol

        mv, series = self._check_sequence_type(series, miss_symbol)

        size = len(series) - self._dim.Ln
        if size > 0:
            self._miss_value = [m+size for m in self._miss_value]
            self._data.extend([[self._miss_symbol] * self._dim.Col\
                                for i in range(size)])

        self._miss_value.append(mv)
        for i, element in enumerate(series):
            self._data[i].append(element)
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
                record.extend([self._miss_symbol] * new_title)

            extend_part = [[self._miss_symbol] * len(self._columns)\
                           for i in range(len(other))]
            new_title_index = [self._columns.index(title)
                               for title in other._columns]
            self._dim = dims(len(self) + len(other), len(self._columns))
            self._miss_value.extend([self._dim.Ln] * new_title)

            for i, record in enumerate(other._data):
                for j, value in zip(new_title_index, record):
                    if value == other._miss_symbol:
                        value = self._miss_symbol
                    extend_part[i][j] = value

            self._data.extend(extend_part)

        elif isinstance(other, SeriesSet):
            self.extend(Frame(other))

        elif all(map(is_seq, other)):
            self.extend(Frame(other, self._columns))

        else:
            raise TypeError('can not extend the dataset with this object.')

    def extend_col(self, other):
        if isinstance(other, Frame):
            for title in other._columns:
                self._columns.append(self._check_col_new_name(title))
            self._miss_value.extend(other._miss_value)

            for i, record in enumerate(other._data):
                if i < self._dim.Ln:
                    current_record = self._data[i]
                else:
                    current_record = [self._miss_symbol] * self._dim.Col
                    self._data.append(current_record)
                for value in record:
                    if value == other.miss_symbol:
                        value = self._miss_value
                    current_record.append(value)
            if i < self._dim.Ln:
                for record in self._data[i+1:]:
                    record.extend([self._miss_symbol] * other.shape.Col)
            self._dim = dims(len(self._data), len(self._columns))

        elif isinstance(other, SeriesSet) or all(filter(is_iter, other)):
            self.extend_col(Frame(other))

        else:
            raise TypeError('could not extend a single value only.')

    def insert(self, index, item, miss_symbol=None):
        '''insert a new record to the frame with position `index`
        '''
        if is_value(item):
            item = [item] * self._dim.Col

        elif not isinstance(item, list):
            try:
                index = list(index)
            except:
                raise TypeError("append item should be a iterable, "+\
                                "not <'%s'>"%type(item))

        miss_symbol = self._check_miss_symbol(miss_symbol)

        for i, element in enumerate(item):
            if element == miss_symbol:
                self._miss_value[i] += 1

        lenth_bias = len(item) - self._dim.Col
        if lenth_bias < 0:
            item.extend([self._miss_symbol] * abs(lenth_bias))
        elif lenth_bias > 0:
            for record in self._data:
                record.extend([self._miss_symbol] * lenth_bias)

        self._data.insert(index, item)
        self._dim = dims(self._dim.Ln + 1, max(self._dim.Col, len(item)))

    def insert_col(self, index, series, variable_name=None, miss_symbol=None):
        '''insert a new variable to the current records in position `index`
        '''
        miss_symbol = self._check_miss_symbol(miss_symbol)
        mv, series = self._check_sequence_type(series, miss_symbol)

        size = len(series) - self._dim.Ln
        if size > 0:
            for i in range(self._dim.Col):
                self._miss_value[i] += size
            self._data.extend([[self._miss_symbol] * self._dim.Col\
                                for i in range(size)])

        self._miss_value.insert(index, mv)
        for i, element in enumerate(series):
            self._data[i].insert(index, element)

        self._columns.insert(index, self._check_col_new_name(variable_name))
        self._dim = dims(max(self._dim.Ln, size), self._dim.Col+1)

    def items(self):
        for i, sequence in enumerate(zip(*self._data)):
            yield self._columns[i], list(sequence)

    def keys(self):
        return self._columns

    def select(self, conditions):
        '''pick out records that comply with conditions
        '''
        conditions = self._transform_str_condition(conditions, axis=0)

        return_data = []
        for record in self._data:
            if eval(conditions) is True:
                return_data.append(record)
        if not return_data:
            return None
        return Frame(return_data, self._columns)

    def pop(self, item=-1):
        '''pop(remove & return) a record from the Frame
        '''
        if isinstance(item, int):
            pop_item = self._data.pop(item)
            self._dim = dims(self._dim.Ln - 1, self._dim.Col)
            for i, value in enumerate(pop_item):
                if value == self._miss_symbol:
                    self._miss_value[i] -= 1
            return pop_item

        raise TypeError('an integer is required.')

    def pop_col(self, *titles):
        '''pop(remove & return) a series from the Frame
        '''
        pos = []
        pop_name = []
        if not titles:
            titles = [self._columns[-1]]

        for item in titles:
            if not isinstance(item, (int, str)):
                raise TypeError('an integer or string, which stands for a '+\
                                'column name, is required, '+
                                'not {0} !'.format(type(item)))
            if isinstance(item, str) and item in self._columns:
                pos.append(self._columns.index(item))
                pop_name.append(item)
            elif isinstance(item, int) and abs(item) < self._dim.Col:
                pos.append(item)
                pop_name.append(self._columns[item])
            else:
                raise IndexError('%s is not a column name or index.'%str(item))

        for name in pop_name:
            index = self._columns.index(name)
            self._columns.pop(index)
            self._miss_value.pop(index)

        pop_data = [[] for i in range(len(pop_name))]

        new_data = [0] * self._dim.Ln
        for j, record in enumerate(self._data):
            line = []
            for i, value in enumerate(record):
                if i in pos:
                    pop_data[pos.index(i)].append(value)
                else:
                    line.append(value)
            new_data[j] = line

        self._dim = dims(self._dim.Ln, self._dim.Col-len(pos))
        self._data = new_data
        return SeriesSet(dict(zip(pop_name, pop_data)))

    def pop_miss_value(self, axis='LINE'):
        '''pop all records that maintains miss value while axis is `LINE` or
        pop all variables that maintains miss value while axis is `COL`
        '''
        if axis.upper() == 'LINE':
            pops = []
            for i, record in enumerate(self._data):
                if self._miss_symbol in record:
                    pops.append(i)
            pops = [self._data.pop(i) for i in sorted(pops, reverse=True)]
            self._dim = dims(len(pops), self._dim.Col)
            return Frame(pops, self._columns)

        if axis.upper() == 'COL':
            pop_col = []
            for i, sequence in enumerate(zip(*self._data)):
                if self._miss_symbol in sequence:
                    pop_col.append(i)

            pop_col.reverse()
            new_columns = [self._columns.pop(i) for i in pop_col].reverse()

            new_data = [0] * self._dim.Ln
            for line, record in enumerate(self._data):
                new_data[line] = [record.pop(i) for i in pop_col]
            for record in new_data:
               record.reverse()
            self._dim = dims(self._dim.Ln, len(self._columns))
            return SeriesSet(new_data, new_columns)

        raise ValueError('axis should be (`LINE` or `COL`) symbols only.')

    def read_text(self, addr, **kwrd):
        '''read dataset from csv or txt file.
        '''
        with open(addr, 'r') as f:
            freader, col_types, miss_symbol, prefer = self._check_read_text(f, **kwrd)
            self._data = []
            try:
                for record in freader:
                    line = [self._check_transform_value(
                            i, v, col_types, miss_symbol, prefer) \
                            for i, v in enumerate(record)]
                    if len(line) != self._dim.Col:
                        line.extend([self._miss_symbol] * \
                                (self._dim.Col - len(line)))
                    self._data.append(line)
            except MemoryError:
                self._dim = dims(len(self._data), self._dim.Col)
                warn('since the limitation of memory, DaPy can not read the'+\
                     ' whole file.')

    def reverse(self):
        self._data.reverse()

    def replace(self, *arg):
        if arg == tuple():
            raise KeyError('argument is empty!')

        elif len(arg) == 3 and is_value(arg[-1]) and isinstance(arg[1], str):
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
                "                       ('C2', '> 200', 3)). Use help() "+\
                'for using details.')

    def sort(self, *orders):
        '''S.sort(('A_col', 'DESC'), ('B_col', 'ASC')) --> sort your records.
        '''
        for each in orders:
            if len(each) != 2:
                raise TypeError("Order argument expects some 2-dimensions"+\
                        " tuples like ('A_col', 'DESC')")

        compare_pos = []
        for order in orders:
            if isinstance(order[0], int):
                if abs(order[0]) >= self.shape[1]:
                    raise IndexError("'%d' is out of range" % order[0])
                compare_pos.append(order[0])
            elif order[0] in self._columns:
                compare_pos.append(self._columns.index(order[0]))
            else:
                raise TypeError("'%s' keyword is "%order[0] +\
                                "not in frame's columns.")
        try:
            compare_symbol = [order[1] for order in orders if order[1]\
                              in ('DESC', 'ASC')]
        except:
            raise TypeError("'%s' is not a recognized symbol."%order[1])

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

        self._data = hash_sort(self._data)
        if compare_symbol[0] == 'DESC':
            self._data.reverse()

    def shuffle(self):
        shuffles(self._data)

    def values(self):
        for sequence in zip(*self._data):
            yield list(sequence)
