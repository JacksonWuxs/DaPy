
class BaseSeries(object):
    def __init__(self):
        pass


class TimeSeries(BaseSeries):
    def __init__(self):
        BaseSeries.__init__(self)


class Series(BaseSeries):
    def __init__(self):
        BaseSeries.__init__(self)

    def __add__(self, right):
        '''[1, 2, 3] + 3 -> [4, 5, 6]
        '''

    def __radd__(self, left):
        '''3 + [1, 2, 3] -> [4, 5, 6]
        '''

    def __sub__(self, right):
        '''[1, 2, 3] - 3 -> [-2, -1 ,0]
        '''
        
    def __isub__(self, left):
        '''3 - [1, 2, 3] -> [2, 1, 0]
        '''        

    def __mul__(self, right):
        '''[1, 2, 3] * 3 -> [3, 6, 9]
        '''

    def __imul__(self, left):
        '''3 * [1, 2, 3] -> [3, 6, 9]
        '''

    def __div__(self, right):
        '''[1, 2, 3] / 2 -> [0.5, 1, 1.5]
        '''

    def __idiv__(self, left):
        '''3 / [1, 2, 3] -> [3, 1.5, 1]
        '''

    def __mod__(self, right):
        '''[1, 2, 3] % 3 -> [0, 0, 1]
        '''

    def __imod__(self, left):
        '''3 % [1, 2, 3] -> [3, 1, 1]
        '''

    def __pow__(self, right):
        '''[1, 2, 3] ** 2 -> [1, 4, 9]
        '''

    def __ipow__(self, left):
        '''2 ** [1, 2, 3] -> [2, 4, 8]
        '''

    def __float__(self):
        '''float([1, 2, 3]) -> [1.0, 2.0, 3.0]
        '''

    def __len__(self):
        '''len([1, 2, 3]) -> 3
        '''

    def __abs__(self):
        '''abs([-1, 2, -3]) -> [1, 2, 3]
        '''

    def __setitem__(self, key, value):
        pass

    def __getitem__(self, key):
        pass

    def all(self):
        pass

    def any(self):
        pass

    def append(self, value):
        pass

    def apply(self, func, args=(), **kwrds):
        pass

    def idxmax(self):
        pass

    def idxmin(self, skipna=False):
        pass

    def sort(self, **kwrds):
        pass

    def between(self, left, right):
        pass

    def corr(self, other, method='pearson'):
        pass

    def count(self, value, range='all'):
        pass

    def count_element(self):
        pass

    def drop(self, label):
        pass

    def drop_duplicates(self, keep=['first', 'last', 'False'], inplace=False):
        pass

    def drop_miss_value(self, inplace=False):
        pass

    def normalize(self):
        pass

    def isna(self):
        pass

    def items(self):
        pass

    def keys(self):
        pass

    def map(self, func):
        pass

    def max(self):
        pass

    def min(self):
        pass

    def mean(self):
        pass

    def quantile(self, q):
        pass
    
    def replace(self):
        pass

    def select(self):
        pass
