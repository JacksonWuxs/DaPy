from unittest import TestCase
from collections import OrderedDict
from DaPy import Frame, SeriesSet, nan
from copy import copy

DICT_DATA = OrderedDict(A=[1, 3, 6])
DICT_DATA['B'] = [2, 4, 7]
DICT_DATA['C'] = [3, None, 8]
DICT_DATA['D'] = [4, 6, 9]
SEQ_DATA = [1, 3, None, 2, 4]
TABLE_DATA = [[1, 2, 3, 4], [3, 4, None, 6], [6, 7, 8, 9]]
TABLE_COL = ['A', 'B', 'C', 'D']
ROW_1 = ['ROW1', 'ROW1', None, 'ROW1']
ROW_2 = ['ROW2', 'ROW2', 'ROW3', 'ROW4', 'ROW5']

class Test0_InitData(TestCase):
    '''Test initialize sheet structures
    '''
    
    def isinit_sheet_success(self, table, data, shape, col, nan, miss):
        self.assertEqual(table.data, data)
        self.assertEqual(tuple(table.shape), shape)
        self.assertEqual(table.columns, col)
        self.assertEqual(table.nan, nan)
        self.assertEqual(table.missing, miss)
        
    def test_init_table(self):
        self.isinit_sheet_success(Frame(TABLE_DATA, TABLE_COL), TABLE_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])
        self.isinit_sheet_success(SeriesSet(TABLE_DATA, TABLE_COL), DICT_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])

    def test_init_dict(self):
        self.isinit_sheet_success(Frame(DICT_DATA), TABLE_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])
        self.isinit_sheet_success(SeriesSet(DICT_DATA), DICT_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])

    def test_init_seq(self):
        dframe, dcol = Frame(SEQ_DATA, 'T1'), SeriesSet(SEQ_DATA, 'T1')
        self.isinit_sheet_success(dframe, [[1], [3], [None], [2], [4]], (5, 1), ['T1'], None, [1])
        self.isinit_sheet_success(dcol, OrderedDict(T1=SEQ_DATA), (5, 1), ['T1'], None, [1])

    def test_init_frame(self):
        original = Frame(TABLE_DATA, TABLE_COL)
        self.isinit_sheet_success(Frame(original), TABLE_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])
        self.isinit_sheet_success(SeriesSet(original), dict(DICT_DATA), (3, 4), TABLE_COL, None, [0, 0, 1, 0])

    def test_init_col(self):
        original = SeriesSet(TABLE_DATA, TABLE_COL)
        self.isinit_sheet_success(Frame(original), TABLE_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])
        self.isinit_sheet_success(SeriesSet(original), DICT_DATA, (3, 4), TABLE_COL, None, [0, 0, 1, 0])
        self.isinit_sheet_success(Frame(original, 'NAN'), TABLE_DATA, (3, 4), ['NAN_0', 'NAN_1', 'NAN_2', 'NAN_3'], None, [0, 0, 1, 0])
        self.isinit_sheet_success(SeriesSet(original, 'NAN'), DICT_DATA, (3, 4), ['NAN_0', 'NAN_1', 'NAN_2', 'NAN_3'], None, [0, 0, 1, 0])

    def test_init_empty(self):
        self.isinit_sheet_success(Frame(), [], (0, 0), [], None, [])
        self.isinit_sheet_success(SeriesSet(), OrderedDict(), (0, 0), [], None, [])
        self.isinit_sheet_success(Frame(columns=['A', 'B']), [], (0, 2), ['A', 'B'], None, [0, 0])
        self.isinit_sheet_success(SeriesSet(columns=['A', 'B']), OrderedDict(A=[], B=[]), (0, 2), ['A', 'B'], None, [0, 0])


class Test1_OperaData(TestCase):
    def setUp(self):
        pass

    def test_append(self):
        def _test_append_row(sheet):
            sheet.append_row(ROW_1)
            sheet.append_row(ROW_2)
            self.assertEqual(tuple(sheet.shape), (5, 5))
            self.assertEqual(sheet.missing, [0, 0, 2, 0, 4])
            self.assertEqual(sheet[0], [1, 2, 3, 4, None])
            self.assertEqual(sheet[-2], ['ROW1', 'ROW1', None, 'ROW1', None])

        def _test_append_col(sheet):
            sheet.append_col(ROW_1)
            sheet.append_col(ROW_2)
            self.assertEqual(tuple(sheet.shape), (5, 6))
            self.assertEqual(sheet.missing, [2, 2, 3, 2, 2, 0])
            self.assertEqual(sheet[0], [1, 2, 3, 4, 'ROW1', 'ROW2'])
            self.assertEqual(sheet[-1], [None, None, None, None, None, 'ROW5'])
        _test_append_row(SeriesSet(TABLE_DATA, TABLE_COL))
        _test_append_col(SeriesSet(TABLE_DATA, TABLE_COL))

    def test_insert(self):
        def _test_insert_row(sheet):
            sheet.insert_row(0, ROW_1)
            sheet.insert_row(1, ROW_2)
            self.assertEqual(tuple(sheet.shape), (5, 5))
            self.assertEqual(sheet.missing, [0, 0, 2, 0, 4])
            self.assertEqual(sheet.columns, ['A', 'B', 'C', 'D', 'A_1'])
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
        _test_insert_row(SeriesSet(TABLE_DATA, TABLE_COL))
        _test_insert_col(SeriesSet(TABLE_DATA, TABLE_COL))


    def test_extend(self):
        sheet1 = SeriesSet(TABLE_DATA, TABLE_COL)
        sheet2 = SeriesSet(TABLE_DATA, TABLE_COL)
        sheet = sheet1.extend(sheet2)
        self.assertEqual(tuple(sheet.shape), (6, 4))
        self.assertEqual(sheet.missing, [0, 0, 2, 0])
        self.assertEqual(sheet.columns, ['A', 'B', 'C', 'D'])
        self.assertEqual(sheet[0], [1, 2, 3, 4])
        self.assertEqual(sheet[3], [1, 2, 3, 4])

    def test_join(self):
        sheet1 = SeriesSet(TABLE_DATA, TABLE_COL)
        sheet2 = SeriesSet(TABLE_DATA, TABLE_COL)
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
                        ['Name', 'Age'])
        right = SeriesSet([['Alan', 'M', 35],
                        ['Bob', 'M', 27],
                        ['Charlie', 'F', 30],
                        ['Janny', 'F', 26]],
                        ['Name', 'gender', 'Age'])

        # MERGE with keep_key=True and keep_same=True
        new = left.merge(right, 'Name', 'Name', keep_key=True, keep_same=True)
        self.assertEqual(tuple(new.shape), (5, 5))
        self.assertEqual(new.missing, [1, 1, 1, 1, 1])
        self.assertEqual(new.columns, ['Name', 'Age', 'Name_1', 'gender', 'Age_1'])
        self.assertEqual(new[0], ['Alan', 35, 'Alan', 'M', 35])
        self.assertEqual(new[-1], [None, None, 'Janny', 'F', 26])
                                
        # MERGE with keep_key=True and keep_same=False
        new = left.merge(right, 'Name', 'Name', keep_key=True, keep_same=False)
        self.assertEqual(tuple(new.shape), (5, 4))
        self.assertEqual(new.missing, [1, 1, 1, 1])
        self.assertEqual(new.columns, ['Name', 'Age', 'Name_1', 'gender'])
        self.assertEqual(new[0], ['Alan', 35, 'Alan', 'M'])
        self.assertEqual(new[-1], [None, None, 'Janny', 'F'])
        
        # MERGE with keep_key=False and keep_same=True
        new = left.merge(right, 'Name', 'Name', keep_key=False, keep_same=True)
        self.assertEqual(tuple(new.shape), (5, 3))
        self.assertEqual(new.missing, [1, 1, 1])
        self.assertEqual(new.columns, ['Age', 'gender', 'Age_1'])
        self.assertEqual(new[0], [35, 'M', 35])
        self.assertEqual(new[-1], [None, 'F', 26])

        # MERGE with keep_key=False and keep_same=False
        new = left.merge(right, 'Name', 'Name', keep_key=False, keep_same=False)
        self.assertEqual(tuple(new.shape), (5, 2))
        self.assertEqual(new.missing, [1, 1])
        self.assertEqual(new.columns, ['Age', 'gender'])
        self.assertEqual(new[0], [35, 'M'])
        self.assertEqual(new[-1], [None, 'F'])

        # MERGE with keep_key=other and keep_same=False
        new = left.merge(right, 'Name', 'Name', keep_key='other', keep_same=False)
        self.assertEqual(tuple(new.shape), (5, 3))
        self.assertEqual(new.missing, [1, 1, 1])
        self.assertEqual(new.columns, ['Age', 'Name_1', 'gender'])
        self.assertEqual(new[0], [35, 'Alan', 'M'])
        self.assertEqual(new[-1], [None, 'Janny', 'F'])

        # MERGE with keep_key='self' and keep_same=True
        new = left.merge(right, 'Name', 'Name', keep_key='self', keep_same=True)
        self.assertEqual(tuple(new.shape), (5, 4))
        self.assertEqual(new.missing, [1, 1, 1, 1])
        self.assertEqual(new.columns, ['Name', 'Age', 'gender', 'Age_1'])
        self.assertEqual(new[0], ['Alan', 35, 'M', 35])
        self.assertEqual(new[-1], [None, None, 'F', 26])

    def test_drop(self):
        data = SeriesSet(TABLE_DATA, TABLE_COL)
        data.drop(0)
        data.drop(2, axis=1)
        self.assertEqual(tuple(data.shape), (2, 3))
        self.assertEqual(data.missing, [0, 0, 0])
        self.assertEqual(data.columns, ['A', 'B', 'D'])
        self.assertEqual(data[0], [3, 4, 6])


