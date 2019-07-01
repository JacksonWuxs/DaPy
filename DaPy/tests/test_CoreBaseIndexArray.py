from unittest import TestCase
from collections import OrderedDict
from datetime import datetime
from DaPy.core.base.IndexArray import SortedIndex

TABLE_DATA = [[1, 2, 3, 4], [3, 4, None, 6], [6, 7, 8, 9], [3, 1, 2, 7]]


class Test_Tools(TestCase):
    def setUp(self):
        self.src = [4, 23, 31, 33, 34, 34, 21, 23, 33]
        self.ind = SortedIndex(self.src)
        
    def test_init(self):
        self.assertEqual(self.ind._val, [4, 21, 23, 23, 31, 33, 33, 34, 34])
        self.assertEqual(self.ind._ind, [0, 6, 1, 7, 2, 3, 8, 4, 5])
        self.assertEqual(len(self.ind), 9)
        self.assertEqual(str(self.ind),
                         'SortedIndex([4, 21, 23, 23, 31, 33, 33, 34, 34])')
        
    def test_getitem(self):
        self.assertEqual(self.ind[1], (6, 21))
        self.assertEqual(self.ind[1:3], ([6, 1], [21, 23]))

    def test_between(self):
        self.assertEqual(self.ind.between(23, 33, (True, True), True),
                         [23, 23, 31, 33, 33])
        self.assertEqual(self.ind.between(23, 33, (False, True), True),
                         [31, 33, 33])
        self.assertEqual(self.ind.between(23, 33, (True, False), True),
                         [23, 23, 31])
        self.assertEqual(self.ind.between(23, 33, (False, False), True),
                         [31])

    def test_index(self):
        self.assertEqual(self.ind.index(23), [1, 7])
        self.assertEqual(self.ind.index(4), [0])
        self.assertEqual(self.ind.index(34), [4, 5])
        self.assertEqual(self.ind.index(21), [6])

    def test_equal(self):
        self.assertEqual(self.ind.equal(33), [3, 8])
        self.assertEqual(self.ind.equal(56), [])
        self.assertEqual(self.ind.equal(1), [])

    def test_unequal(self):
        self.assertEqual(self.ind.unequal(33), list(set([0, 1, 2, 4, 5, 6, 7])))

    def test_lower(self):
        self.assertEqual(self.ind.lower(21, True), [0, 6])
        self.assertEqual(self.ind.lower(21, False), [0])

        
