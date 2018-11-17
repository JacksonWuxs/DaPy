import unittest
from DaPy import Frame, SeriesSet

class TestFrameMethods(unittest.TestCase):
    def test_init_records(self):
        frame = Frame([[1, 2, 3, 4],
                        [2, 3, 4, 5],
                        [3, 4, None, 6],
                        [4, float('nan'), 6, 7],
                        [5, 6, 7, 8],
                        [6, 7, 8, 9]],
                        ['D', 'C', 'B', 'A'],
                        miss_symbol=float('nan'))

        self.assertEqual(frame.data,
                         [[1, 2, 3, 4],
                         [2, 3, 4, 5],
                         [3, 4, None, 6],
                         [4, None, 6, 7],
                         [5, 6, 7, 8],
                         [6, 7, 8, 9]])
        self.assertEqual(tuple(frame.shape), (6, 4))
        self.assertEqual(frame.columns, ['D', 'C', 'B', 'A'])
        self.assertEqual(frame.miss_symbol, None)
        self.assertEqual(frame.miss_value, [0, 1, 1, 0])

    def test_merge(self):
        frame1 = Frame([[1, 2, 3, 4],
                        [2, 3, 4, 5],
                        [3, 4, None, 6],
                        [4, float('nan'), 6, 7],
                        [5, 6, 7, 8],
                        [6, 7, 8, 9]],
                        ['D', 'C', 'B', 'A'],
                        miss_symbol=float('nan'))
        frame2 = Frame([[10, 10, 10, 10],
                        [None, 11, 11, 11],
                        [12, 12, 12, None]],
                       ['D', 'C', 'E', 'F'])
        frame1.extend_col(frame2)

        self.assertEqual(frame.data,
                         [[1, 2, 3, 4, ],
                          [2, 3, 4, 5],
                          [3, 4, None, 6],
                          [4, None, 6, 7],
                          [5, 6, 7, 8],
                          [6, 7, 8, 9],
                          [10, 10, None, None, 10, 10],
                          [None, 11, None, None, 11, 11],
                          [])
