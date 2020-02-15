from unittest import TestCase
from collections import OrderedDict
from DaPy import SeriesSet, nan, Series
from copy import copy

DICT_DATA = OrderedDict(A=[1, 3, 6])
DICT_DATA['B'] = [2, 4, 7]
DICT_DATA['C'] = [3, None, 8]
DICT_DATA['D'] = [4, 6, 9]
SEQ_DATA = [1, 3, None, 2, 4]
TABLE_DATA = [[1, 2, 3, 4],
              [3, 4, None, 6],
              [6, 7, 8, 9]]
TABLE_COL = ['A', 'B', 'C', 'D']
ROW_1 = ['ROW1', 'ROW1', None, 'ROW1']
ROW_2 = ['ROW2', 'ROW2', 'ROW3', 'ROW4', 'ROW5']

class Test0_InitData(TestCase):
    '''Test initialize sheet structures
    '''
    
    def isinit_sheet_success(self, table, data, shape, col, nan, miss):
        for (lcol, table_col), (rcol, data_col) in zip(table.items(), data.items()):
            assert all(table_col == data_col), '%s:%s != %s:%s' % (lcol, table_col, rcol, data_col)
        self.assertEqual(tuple(table.shape), shape)
        self.assertEqual(table.columns, col)
        self.assertEqual(table.nan, nan)
        self.assertEqual(table.missing, miss)
        
    def test_init_table(self):
        # self.isinit_sheet_success(Frame(TABLE_DATA, TABLE_COL), TABLE_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])
        self.isinit_sheet_success(SeriesSet(TABLE_DATA, TABLE_COL, None), DICT_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])

    def test_init_dict(self):
        # self.isinit_sheet_success(Frame(DICT_DATA), TABLE_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])
        self.isinit_sheet_success(SeriesSet(DICT_DATA, nan=None), DICT_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])

    def test_init_seq(self):
        dcol = SeriesSet(SEQ_DATA, 'T1', None)
        #self.isinit_sheet_success(dframe, [[1], [3], [None], [2], [4]], (5, 1), ['T1'], None, [1])
        self.isinit_sheet_success(dcol, OrderedDict(T1=SEQ_DATA), (5, 1), ['T1'], None, [1])

    #def test_init_frame(self):
        #original = Frame(TABLE_DATA, TABLE_COL, None)
        # self.isinit_sheet_success(Frame(original), TABLE_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])
        #self.isinit_sheet_success(SeriesSet(original, nan=None), DICT_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])

    def test_init_col(self):
        original = SeriesSet(TABLE_DATA, TABLE_COL, None)
        # self.isinit_sheet_success(Frame(original), TABLE_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])
        self.isinit_sheet_success(SeriesSet(original, nan=None), DICT_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])
        # self.isinit_sheet_success(Frame(original, 'NAN'), TABLE_DATA, (3, 4), ['NAN_0', 'NAN_1', 'NAN_2', 'NAN_3'], None, [0, 0, 1, 0])
        self.isinit_sheet_success(SeriesSet(original, 'NAN', nan=None), DICT_DATA, (3, 4), ['NAN_0', 'NAN_1', 'NAN_2', 'NAN_3'], None, [0, 0, 1, 0])

    def test_init_empty(self):
        # self.isinit_sheet_success(Frame(), [], (0, 0), [], None, [])
        self.isinit_sheet_success(SeriesSet(nan=None), OrderedDict(), (0, 0), [], None, [])
        # self.isinit_sheet_success(Frame(columns=['A', 'B']), [], (0, 2), ['A', 'B'], None, [0, 0])
        self.isinit_sheet_success(SeriesSet(columns=['A', 'B'], nan=None),
                                  OrderedDict(A=Series([]), B=Series([])),
                                  (0, 2),
                                  ['A', 'B'], None, [0, 0])


class Test2_Transfer(TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        import pandas as pd
        self.pd = pd
        
    def test_numpy_dapy(self):
        arr = self.np.array(TABLE_DATA)
        
        sheet = SeriesSet(arr, nan=None)
        self.assertEqual(tuple(sheet.shape), (3, 4))
        self.assertEqual(sheet.missing, [0, 0, 1, 0])
        self.assertEqual(sheet[0], [1, 2, 3, 4])
        self.assertEqual(sheet.columns, ['C_0', 'C_1', 'C_2', 'C_3'])
        
        arr = self.np.array(sheet)
        self.assertEqual(arr.shape, (3, 4))
        assert (arr[0] == [1, 2, 3, 4]).all()

    def test_pandas_dapy(self):
        df = self.pd.DataFrame(DICT_DATA)

        sheet = SeriesSet(df)
        self.assertEqual(tuple(sheet.shape), (3, 4))
        self.assertEqual(sheet.missing, [0, 0, 1, 0])
        self.assertEqual(sheet.columns, ['A', 'B', 'C', 'D'])

        df = self.pd.DataFrame(sheet.todict())
        self.assertEqual(df.shape, (3, 4))
        assert (df.columns == ['A', 'B', 'C', 'D']).all()


class Test1_CoreOperations(TestCase):
    def setUp(self):
        pass

    def test_getitem(self):
        def _test_getitem_by_int(sheet):
            row = sheet[0]
            from DaPy.core.base.Row import Row
            self.assertEqual(type(row), Row)
            self.assertEqual(row, [1, 2, 3, 4])

        def _test_getitem_by_str(sheet):
            ser = sheet['A']
            self.assertEqual(type(ser), Series)
            self.assertEqual(ser, [1, 3, 6])

        def _test_getitem_by_int_slice(sheet):
            subset = sheet[:2]
            self.assertEqual(tuple(subset.shape), (2, 4))
            self.assertEqual(subset.missing, [0, 0, 1, 0])
            self.assertEqual(subset[0], [1, 2, 3, 4])
            self.assertEqual(subset[1], [3, 4, None, 6])
            self.assertEqual(subset.columns, TABLE_COL)

        def _test_getitem_by_str_slice(sheet):
            subset = sheet['A': 'C']
            self.assertEqual(tuple(subset.shape), (3, 3))
            self.assertEqual(subset.missing, [0, 0, 1])
            self.assertEqual(subset[0], [1, 2, 3])
            self.assertEqual(subset[1], [3, 4, None])
            self.assertEqual(subset.columns, ['A', 'B', 'C'])

        def _test_getitem_by_int_tuple(sheet):
            subset = sheet[0, 0, 1, 1]
            self.assertEqual(tuple(subset.shape), (4, 4))
            self.assertEqual(subset.missing, [0, 0, 2, 0])
            self.assertEqual(subset[0], [1, 2, 3, 4])
            self.assertEqual(subset[1], [1, 2, 3, 4])
            self.assertEqual(subset[2], [3, 4, None, 6])
            self.assertEqual(subset[3], [3, 4, None, 6])
            self.assertEqual(subset.columns, ['A', 'B', 'C', 'D'])

        def _test_getitem_by_str_tuple(sheet):
            subset = sheet['A', 'A', 'C', 'C']
            self.assertEqual(tuple(subset.shape), (3, 4))
            self.assertEqual(subset.missing, [0, 0, 1, 1])
            self.assertEqual(subset[0], [1, 1, 3, 3])
            self.assertEqual(subset[1], [3, 3, None, None])
            self.assertEqual(subset[2], [6, 6, 8, 8])
            self.assertEqual(subset.columns, ['A', 'A_1', 'C', 'C_1'])
        sheet = SeriesSet(TABLE_DATA, TABLE_COL, nan=None)
        _test_getitem_by_int(sheet)
        _test_getitem_by_str(sheet)
        _test_getitem_by_int_slice(sheet)
        _test_getitem_by_str_slice(sheet)
        _test_getitem_by_int_tuple(sheet)
        _test_getitem_by_str_tuple(sheet)

    def test_append(self):
        def _test_append_row(sheet):
            sheet.append_row(ROW_1)
            sheet.append_row(ROW_2)
            sheet.append_row(dict(A=9, B=9, C=9, D=9, E=9))
            self.assertEqual(tuple(sheet.shape), (6, 6))
            self.assertEqual(sheet.missing, [0, 0, 2, 0, 5, 5])
            self.assertEqual(sheet[0], [1, 2, 3, 4, None, None])
            self.assertEqual(sheet[-3], ['ROW1', 'ROW1', None, 'ROW1', None, None])
            self.assertEqual(sheet[-1], [9, 9, 9, 9, None, 9])
            
        def _test_append_col(sheet):
            sheet.append_col(ROW_1)
            sheet.append_col(ROW_2)
            self.assertEqual(tuple(sheet.shape), (5, 6))
            self.assertEqual(sheet.missing, [2, 2, 3, 2, 2, 0])
            self.assertEqual(sheet[0], [1, 2, 3, 4, 'ROW1', 'ROW2'])
            self.assertEqual(sheet[-1], [None, None, None, None, None, 'ROW5'])
        _test_append_row(SeriesSet(TABLE_DATA, TABLE_COL, nan=None))
        _test_append_col(SeriesSet(TABLE_DATA, TABLE_COL, nan=None))

    def test_insert(self):
        def _test_insert_row(sheet):
            sheet.insert_row(0, ROW_1)
            sheet.insert_row(1, ROW_2)
            self.assertEqual(tuple(sheet.shape), (5, 5))
            self.assertEqual(sheet.missing, [0, 0, 2, 0, 4])
            self.assertEqual(sheet.columns, ['A', 'B', 'C', 'D', 'C_4'])
            self.assertEqual(sheet[0], ['ROW1', 'ROW1', None, 'ROW1', None])
            self.assertEqual(sheet[1], ROW_2)
            self.assertEqual(sheet[-1], [6, 7, 8, 9, None])
        def _test_insert_col(sheet):
            sheet.insert_col(0, ROW_1)
            sheet.insert_col(1, ROW_2)
            self.assertEqual(tuple(sheet.shape), (5, 6))
            self.assertEqual(sheet.missing, [2, 0, 2, 2, 3, 2])
            self.assertEqual(sheet.columns, ['C_4', 'C_5', 'A', 'B', 'C', 'D']) 
            self.assertEqual(sheet[0], ['ROW1', 'ROW2', 1, 2, 3, 4])
            self.assertEqual(sheet[2], [None, 'ROW3', 6, 7, 8, 9])
            self.assertEqual(sheet[-1], [None, 'ROW5', None, None, None, None])
        _test_insert_row(SeriesSet(TABLE_DATA, TABLE_COL, nan=None))
        _test_insert_col(SeriesSet(TABLE_DATA, TABLE_COL, nan=None))


    def test_extend(self):
        sheet1 = SeriesSet(TABLE_DATA, TABLE_COL, nan=None)
        sheet2 = SeriesSet(TABLE_DATA, TABLE_COL, nan=None)
        sheet = sheet1.extend(sheet2)
        self.assertEqual(tuple(sheet.shape), (6, 4))
        self.assertEqual(sheet.missing, [0, 0, 2, 0])
        self.assertEqual(sheet.columns, ['A', 'B', 'C', 'D'])
        self.assertEqual(sheet[0], [1, 2, 3, 4])
        self.assertEqual(sheet[3], [1, 2, 3, 4])

    def test_join(self):
        sheet1 = SeriesSet(TABLE_DATA, TABLE_COL, nan=None)
        sheet2 = SeriesSet(TABLE_DATA, TABLE_COL, nan=None)
        sheet2.append_col(['K', 'K'], 'K_col')
        sheet = sheet1.join(sheet2)
        self.assertEqual(tuple(sheet.shape), (3, 9))
        self.assertEqual(sheet.missing, [0, 0, 1, 0, 0, 0, 1, 0, 1])
        self.assertEqual(sheet.columns, ['A', 'B', 'C', 'D', 'A_1', 'B_1', 'C_1', 'D_1', 'K_col'])

    def test_merge(self):
        left = SeriesSet([
                        ['Alan', 35],
                        ['Bob', 27],
                        ['Charlie', 30],
                        ['Daniel', 29]],
                        ['Name', 'Age'],
                         '')
        right = SeriesSet([['Alan', 'M', 35],
                        ['Bob', 'M', 27],
                        ['Charlie', 'F', 30],
                        ['Janny', 'F', 26]],
                        ['Name', 'gender', 'Age'],
                          '')

        new = left.merge(right, 'outer', 'Name', 'Name')
        self.assertEqual(tuple(new.shape), (5, 5))
        self.assertEqual(new.missing, [1, 1, 1, 1, 1])
        self.assertEqual(new.columns, ['Name', 'Age', 'Name_1', 'gender', 'Age_1'])
        self.assertEqual(new[0], ['Alan', 35, 'Alan', 'M', 35])
        self.assertEqual(new[-1], ['', '', 'Janny', 'F', 26])
                                
        new = left.merge(right, 'inner', 'Name', 'Name').sort('Name')
        self.assertEqual(tuple(new.shape), (3, 5))
        self.assertEqual(new.missing, [0, 0, 0, 0, 0])
        self.assertEqual(new.columns, ['Name', 'Age', 'Name_1', 'gender', 'Age_1'])
        self.assertEqual(new[0], ['Alan', 35, 'Alan', 'M', 35])
        self.assertEqual(new[-1], ['Charlie', 30, 'Charlie', 'F', 30])
        
        new = left.merge(right, 'left', 'Name', 'Name')
        self.assertEqual(tuple(new.shape), (4, 5))
        self.assertEqual(new.missing, [0, 0, 1, 1, 1])
        self.assertEqual(new.columns, ['Name', 'Age', 'Name_1', 'gender', 'Age_1'])
        self.assertEqual(new[0], ['Alan', 35, 'Alan', 'M', 35])
        self.assertEqual(new[-1], ['Daniel', 29, '', '', ''])

        new = left.merge(right, 'right', 'Name', 'Name')
        self.assertEqual(tuple(new.shape), (4, 5))
        self.assertEqual(new.missing, [0, 0, 0, 1, 1])
        self.assertEqual(new.columns, ['Name', 'gender', 'Age', 'Name_1', 'Age_1'])
        self.assertEqual(new[0], ['Alan', 'M', 35, 'Alan', 35])
        self.assertEqual(new[-1], ['Janny', 'F', 26, '', ''])

    def test_drop(self):
        data = SeriesSet(TABLE_DATA, TABLE_COL, None)
        data.drop(0, inplace=True)
        data.drop(2, axis=1, inplace=True)
        self.assertEqual(tuple(data.shape), (2, 3))
        self.assertEqual(data.missing, [0, 0, 0])
        self.assertEqual(data.columns, ['A', 'B', 'D'])
        self.assertEqual(data[0], [3, 4, 6])


    def test_reshape(self):
        data = SeriesSet(TABLE_DATA, TABLE_COL, None)
        new = data.reshape((6, 2))
        self.assertEqual(tuple(new.shape), (6, 2))
        self.assertEqual(new.missing, [1, 0])
        self.assertEqual(new.columns, ['C_0', 'C_1'])
        self.assertEqual(new[0], [1, 2])
        self.assertEqual(new[-1], [8, 9])

    def test_get(self):
        data = SeriesSet(TABLE_DATA, TABLE_COL, None)
        self.assertEqual(data.get('TEST'), None)
    
    def test_count(self):
        data = SeriesSet(TABLE_DATA, TABLE_COL, None)
        self.assertEqual(data.count(2), 1)
        self.assertEqual(sorted(data.count([2, 3]).values()), [1, 2])
        self.assertEqual(sorted(data.count([2, 3]).keys()), [2, 3])
        self.assertEqual(data.count(None, (1, 2), (0, 1)), 1)

    def test_count_values(self):
        test = SeriesSet([
                        ['Alan', 35],
                        ['Bob', 27],
                        ['Charlie', 30],
                        ['Daniel', 29],
                        ['Daniel', 29]],
                        ['Name', 'Age'],
                         '')
        self.assertEqual(dict(test.count_values('Name')),
                         {'Alan': 1, 'Bob': 1, 'Charlie': 1, 'Daniel': 2})

    def test_pop(self):
        def pop_row(sheet):
            rows = sheet.pop_row([0, 1])
            self.assertEqual(tuple(rows.shape), (2, 2))
            self.assertEqual(rows.missing, [1, 0])
            self.assertEqual(rows.columns, ['Name', 'Age'])
            self.assertEqual(rows[0], ['Alan', 35])
            self.assertEqual(rows[1], ['', 3])

        def pop_col(sheet):
            rows = sheet.pop_col([0])
            self.assertEqual(tuple(rows.shape), (6, 1))
            self.assertEqual(rows.missing, [1])
            self.assertEqual(rows.columns, ['Name'])
            self.assertEqual(rows[0], ['Alan'])
            self.assertEqual(tuple(sheet.shape), (6, 1))
            self.assertEqual(sheet.columns, ['Age'])

        pop_row(SeriesSet([
                        ['Alan', 35],
                        ['', 3],
                        ['Bob', 27],
                        ['Charlie', 30],
                        ['Daniel', 29],
                        ['Daniel', 29]],
                        ['Name', 'Age'],
                         ''))
        pop_col(SeriesSet([
                        ['Alan', 35],
                        ['', 3],
                        ['Bob', 27],
                        ['Charlie', 30],
                        ['Daniel', 29],
                        ['Daniel', 29]],
                        ['Name', 'Age'],
                         ''))















        
