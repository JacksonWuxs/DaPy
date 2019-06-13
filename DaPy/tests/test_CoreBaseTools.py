from unittest import TestCase
from collections import OrderedDict
from datetime import datetime
from DaPy.core.base.utils import (
    auto_str2value, argsort, hash_sort, auto_plus_one,
    is_value, is_math, is_iter, is_seq
)

TABLE_DATA = [[1, 2, 3, 4], [3, 4, None, 6], [6, 7, 8, 9], [3, 1, 2, 7]]


class Test_Tools(TestCase):        
    def test_str2value(self):
        self.assertEqual(auto_str2value('3.24'), 3.24)
        self.assertEqual(auto_str2value('3'), 3)
        self.assertEqual(auto_str2value('3.5%'), 0.035)
        self.assertEqual(auto_str2value('20170210'), 20170210)
        self.assertEqual(auto_str2value('20170210', 'datetime'), datetime(2017, 2, 10))
        self.assertEqual(auto_str2value('True'), True)
        self.assertEqual(auto_str2value('no'), False)
        
    def test_GetSortedIndex(self):
        self.assertEqual(argsort([3, 1, 2, 6, 4, 2, 1, 3]),
                         [1, 6, 2, 5, 0, 7, 4, 3])

    def test_hash_sort(self):
        self.assertEqual(
            hash_sort(TABLE_DATA, (0, 'DESC'), (3, 'DESC')),
            [[6, 7, 8, 9],
             [3, 1, 2, 7],
             [3, 4, None, 6],
             [1, 2, 3, 4]])

    def test_autoone(self):
        func = auto_plus_one
        exist = []
        self.assertEqual(func([], 'A'), 'A_1')
        self.assertEqual(func(['A'], 'A'), 'A_1')
        self.assertEqual(func(['A', 'A'], 'A'), 'A_1')
        self.assertEqual(func(['A', 'A_1'], 'A'), 'A_2')

    def test_isvalue(self):
        self.assertEqual(is_value(3), True)
        self.assertEqual(is_value(datetime(2018, 1, 1)), True)
        self.assertEqual(is_value([]), False)

    def test_ismath(self):
        self.assertEqual(is_math(3.5), True)
        self.assertEqual(is_math(datetime(2018, 1, 1)), False)
