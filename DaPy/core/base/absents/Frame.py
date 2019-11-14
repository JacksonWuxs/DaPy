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
        for i in xrange(self._dim.Col):
            info += ' ' * 15
            info += self._columns[i].center(max_n) + '| '
            info += ' ' + new_m_v[i] + '\n'

        print('1.  Structure: DaPy.Frame\n' +
              '2. Dimensions: Ln=%d | Col=%d\n' % self._dim +
              '3. Miss Value: %d elements\n' % sum(self._missing) +
              '4.    Columns: ' + 'Title'.center(max_n) + '|' +
              '  Miss\n' + info)

    @property
    def T(self):
        return Frame(self.iter_values(), None, self.nan)

    def _init_col(self, obj, columns):
        if columns is None:
            columns = copy(obj._columns)
        self._data = [list(record) for record in zip(*list(obj.values()))]
        self._missing = copy(obj._missing)
        self._dim = SHEET_DIM(obj._dim.Ln, obj._dim.Col)
        self._init_col_name(columns)

    def _init_frame(self, frame, columns):
        if columns is None:
            columns = copy(obj._columns)
        self._data = deepcopy(frame._data)
        self._dim = copy(frame._dim)
        self._init_col_name(columns)
        self._missing = copy(frame._missing)

    def _init_dict(self, frame, columns):
        if columns is None:
            columns = list(obj.keys())
        frame = copy(frame)
        self._dim = SHEET_DIM(max(map(len, frame.values())), len(frame))
        self._missing = [0] * self._dim.Col
        self._init_col_name(columns)
        for i, (title, col) in enumerate(frame.items()):
            miss, sequence = self._check_sequence(col, self._dim.Ln)
            frame[title] = sequence
            self._missing[i] += miss
        self._data = [list(record) for record in zip(*frame.values())]

    def _init_like_table(self, frame, columns):
        self._data = map(list, frame)
        dim_Col, dim_Ln = len(max(self._data, key=len)), len(frame)
        self._dim = SHEET_DIM(dim_Ln, dim_Col)
        self._missing = [0] * self._dim.Col

        for i, item in enumerate(self._data):
            if len(item) < dim_Col:
                item.extend([self._nan] * (dim_Col - len(item)))
            for j, value in enumerate(item):
                if value == self.nan or value is self.nan:
                    self._missing[j] = self._missing[j] + 1
        self._init_col_name(columns)

    def _init_like_seq(self, frame, columns):
        self._data = [[value, ] for value in frame]
        self._dim = SHEET_DIM(len(frame), 1)
        self._init_col_name(columns)
        self._missing.append(self._check_sequence(frame, len(frame))[0])

    def __repr__(self):
        return self.show(30)

    def _getslice_col(self, i, j):
        new_data = [record[i: j + 1] for record in self._data]
        return Frame(new_data, self._columns[i: j + 1], self._nan)

    def _getslice_ln(self, i, j, k):
        return Frame(self._data[i:j:k], self._columns, self._nan)

    def __getitem__(self, interval):
        if isinstance(interval, int):
            return Row(self, interval)

        elif isinstance(interval, slice):
            return self.__getslice__(interval)

        elif is_str(interval):
            col = self._columns.index(interval)
            return [item[col] for item in self._data]

        elif isinstance(interval, (tuple, list)):
            return_obj = Frame()
            return self._getitem_by_tuple(interval, return_obj)

        else:
            raise TypeError('item must be represented as slice, int, str.')

    def __iter__(self):
        for i in xrange(self._dim.Ln):
            yield Row(self, i)

    def append_row(self, item):
        '''append a new record to the Frame tail
        '''
        item = self._add_row(item)
        self._data.append(item)

    def append_col(self, series, variable_name=None):
        '''append a new variable to the current records tail
        '''
        miss, series = self._check_sequence(series, self._dim.Ln)
        size = len(series) - self._dim.Ln
        if size > 0:
            self._missing = [m + size for m in self._missing]
            self._data.extend(
                [[self._nan] * self._dim.Col for i in xrange(size)])

        self._missing.append(miss)
        for record, element in zip(self._data, series):
            record.append(element)
        self._columns.append(self._check_col_new_name(variable_name))
        self._dim = SHEET_DIM(max(self._dim.Ln, len(series)), self._dim.Col + 1)
        assert len(self._missing) == self._dim.Col == len(self.columns)

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

    def extend(self, other, inplace=False):
        if isinstance(other, Frame):
            if inplace is False:
                self = SeriesSet(Frame)
            new_title = 0
            for title in other._columns:
                if title not in self._columns:
                    self._columns.append(title)
                    new_title += 1

            for record in self._data:
                record.extend([self._nan] * new_title)

            extend_part = [[self._nan] * len(self._columns)
                           for i in xrange(len(other))]
            new_title_index = [self._columns.index(title)
                               for title in other._columns]
            self._dim = SHEET_DIM(len(self) + len(other), len(self._columns))
            self._missing.extend([self._dim.Ln] * new_title)

            for i, record in enumerate(other._data):
                for j, value in zip(new_title_index, record):
                    if value == other._nan:
                        value = self._nan
                    extend_part[i][j] = value
            self._data.extend(extend_part)
            return self

        elif isinstance(other, SeriesSet):
            return self.extend(Frame(other), inplace)

        else:
            return self.extend(Frame(other, self._columns), inplace)

    def join(self, other, inplace=False):
        if isinstance(other, Frame):
            if inplace is False:
                self = Frame(self)
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
                for record in self._data[i + 1:]:
                    record.extend([self._nan] * other.shape.Col)
            self._dim = SHEET_DIM(len(self._data), len(self._columns))
            return self

        else:
            self.join(Frame(other, nan=self.nan), inplace)

    def insert_row(self, index, item):
        '''insert a new record to the frame with position `index`
        '''
        item = self._add_row(item)
        self._data.insert(index, item)

    def insert_col(self, index, series, variable_name=None):
        '''insert a new variable to the current records in position `index`
        '''
        miss, series = self._check_sequence(series)

        size = len(series) - self._dim.Ln
        if size > 0:
            for i in xrange(self._dim.Col):
                self._missing[i] += size
            self._data.extend([[self._nan] * self._dim.Col
                               for i in xrange(size)])

        self._missing.insert(index, miss)
        for i, element in enumerate(series):
            self._data[i].insert(index, element)
        self._columns.insert(index, self._check_col_new_name(variable_name))
        self._dim = SHEET_DIM(max(self._dim.Ln, size), self._dim.Col + 1)

    def items(self):
        for i, sequence in enumerate(zip(*self._data)):
            yield self._columns[i], list(sequence)

    def keys(self):
        return self._columns

    def pop_row(self, pos=-1):
        '''pop(remove & return) a record from the Frame
        '''
        err = 'an int or ints in list is required.'
        assert isinstance(pos, (int, list, tuple)), err
        if isinstance(pos, int):
            pos = [pos, ]
        pos = sorted(pos, reverse=True)
        pop_item = Frame([self._data.pop(pos_)
                          for pos_ in pos], list(self._columns))
        self._dim = SHEET_DIM(self._dim.Ln - len(pos), self._dim.Col)
        self._missing = map(
            lambda x, y: x - y,
            self._missing,
            pop_item._missing)
        return pop_item

    def from_file(self, addr, **kwrd):
        '''read dataset from csv or txt file.
        '''
        raise NotImplementedError('use DaPy.SeriesSet.from_file()')
        
    def reverse(self):
        self._data.reverse()

    def shuffle(self):
        shuffles(self._data)

    def _values(self):
        for sequence in zip(*self._data._data):
            yield list(sequence)

    def values(self):
        for sequence in zip(*self._data):
            yield Series(sequence)


    def pop_col(self, pos=-1):
        '''pop(remove & return) a series from the Frame
        '''
        pop_name = self._check_columns_index(pos)
        for name in pop_name:
            index = self._columns.index(name)
            self._columns.pop(index)
            self._missing.pop(index)

        pop_data = [[] for i in xrange(len(pop_name))]
        new_data = [0] * self._dim.Ln
        for j, record in enumerate(self._data):
            line = []
            for i, value in enumerate(record):
                if i in pop_name:
                    pop_data[pop_name.index(i)].append(value)
                else:
                    line.append(value)
            new_data[j] = line

        self._dim = SHEET_DIM(self._dim.Ln, self._dim.Col - len(pos))
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


