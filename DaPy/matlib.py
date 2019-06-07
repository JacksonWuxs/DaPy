from array import array
from .core import Matrix, SeriesSet, Series
from .core import nan, inf
from .core import range, filter, zip, range
from .core import is_math, is_seq, is_iter
from .core.base import STR_TYPE
from collections import namedtuple, deque, Iterable, deque
from itertools import repeat
from warnings import warn
from functools import reduce
import math
    
__all__ = ['dot', 'multiply', 'exp', 'zeros', 'ones', 'C', 'P',
           'cov', 'corr', 'frequency', 'quantiles', '_sum', '_max',
           'distribution','describe', 'mean', 'diag', 'log']

def P(n, k):
    '''"k" is for permutation numbers.
    A permutation is an ordered sequence of elements selected from a given
    finite set, without repetitions, and not necessarily using all elements 
    of the given set.

    Formula
    -------
                  n!
    P(n, k) = ----------
               (n - k)!
    '''
    if k == 0 or n == k:
        return 1
    upper = reduce(multiply, range(1, 1+n))
    down = reduce(multiply, range(1, 1+ n - k))
    return float(upper / down)

def C(n, k):
    '''"C" is for combination numbers.
    A combination number is an un-ordered collection of distinct elements,
    usually of a prescribed size and taken from a given set.

    Formula
    -------
                   n!
    C(n, k) = -----------
               k!(n - k)!
    '''
    if k == 0 or n == k:
        return 1
    upper = reduce(multiply, range(1, 1+n))
    left = reduce(multiply, range(1, 1+k))
    right = reduce(multiply, range(1, 1+ n - k))
    return float(upper / (left * right))

def add(m1, m2):
    if hasattr(m1, '__add__'):
        return m1 + m2

    if is_seq(m1):
        return Matrix(m1) + m2

    raise TypeError('add() expectes elements which can add.')

def _abs(data):
    if hasattr(data, 'shape'):
        new = [0] * data.shape[0]
        if hasattr(data, 'tolist'):
            data = data.tolist()
        for i, line in enumerate(data):
            try:
                new[i] = map(abs, line)
            except TypeError:
                new[i] = abs(line)
        return Matrix(new)
    
    if is_math(data):
        return abs(data)

    if is_iter(data):
        return map(_abs, data)

    raise TypeError('expects an iterable or numeric for exp(), got %s'%type(other))


def multiply(m1, m2):
    if is_math(m1) and is_math(m2):
        return m1 * m2
    if isinstance(m1, Matrix) or isinstance(m2, Matrix):
        return m1 * m2
    return Matrix(m1) * m2
    
def dot(matrix_1, matrix_2):
    if hasattr(matrix_1, 'dot'):
        return matrix_1.dot(matrix_2)

    try:
        col_size_1 = len(matrix_1[0])
        col_size_2 = len(matrix_2[0])
        line_size_1 = len(matrix_1)
        line_size_2 = len(matrix_2)
        columns = None
    except TypeError:
        raise TypeError('unsupported operation dot(), with type'+\
                        ' %s and ,' % type(matrix_1) +\
                        '%s.' % type(matrix_2))
    
    if col_size_1 != line_size_2:
        raise ValueError('shapes (%d, %d) '%(line_size_1, col_size_1)+\
                         'and (%d, %d) not aligned.'%(line_size_2, col_size_2))
    
    new_ = list()
    for i in range(line_size_1):
        new_line = list()
        for pos in range(col_size_2):
            sumup = sum(matrix_1[i][j]*matrix_2[j][pos]
                        for j in range(col_size_1))
            new_line.append(sumup)
        new_.append(new_line)
    return Matrix(new_)

def exp(other):
    if hasattr(other, 'shape'):
        new = [0] * other.shape[0]
        for i, line in enumerate(other):
            new[i] = map(math.exp, line)
        return Matrix(new)
    
    if is_math(other):
        return math.exp(other)

    if is_iter(other):
        new_ = list()
        for item in other:
            new_.append(exp(item))
        return new_

    raise TypeError('expects an iterable or numeric for exp(), got %s'%type(other))

def create_mat(shape, num):
    return Matrix().make(shape[0], shape[1], num)

def cumsum(series):
    series, new = Series(series), Series()
    init = 0
    for value in series:
        init += value
        new.append(init)
    return new

def count(df, value, axis=None):
    assert axis in (None, 0, 1)
    if axis == None:
        if hasattr(container, 'count'):
            return container.count(value)
    if axis == 1:
        if hasattr(container, 'count'):
            return c

def zeros(shape):
    return create_mat(shape, 0)

def ones(shape):
    return create_mat(shape, 1)

def diag(values):
    return Matrix().make_eye(len(values), values)

def diff(seq, lag=1):
    return [seq[i] - seq[i-lag] for i in range(lag, len(seq))]

def log(data, base=2.71828183):
    if is_seq(data):
        if is_seq(data[0]):
            return [map(log, record, [base] * len(record)) for record in data]
        return list(map(log, data, [base] * len(data)))
    return math.log(data, base)

def boxcox(value, lambda_=1, a=0, k=1):
    if lambda_ == 0:
        return log(value)
    return ((value + a) ** lambda_ - 1) / (k + lambda_)

def _sum(data, axis=None):
    '''Sum of sequence elements.

    Parameters
    ----------
    data : array_like
        elements to sum.

    axis : None, 0, 1
        determine how to summary this data.
        None - summary all elements into one value,
        0 - summary the elements in each variable,
        1 - summary the elements in each record.

    Return
    ------
    Sum : float or int
        a number of sum.

    Examples
    --------
    >>> dp.sum([0.5, 1.5])
    2.0
    >>> dp.sum([[0.5, 0.7],
                [0.2, 1.5]])
    2.9
    >>> dp.sum([[0, 1],
                [0, 5]], axis=1) # sum of each record
    [1, 5]
    >>> dp.sum([[0, 1],
                [0, 5]], axis=0) # sum of each variable
    [0, 6]
    '''
    if hasattr(data, 'sum'):
        return data.sum(axis)
        
    if axis is None:
        if all(map(is_math, data)):
            return sum(data)
        return reduce(sum, data)
    
    if axis == 1:
        return map(sum, data)

    if axis == 0:
        return _sum([line for line in zip(*data)], axis=1)

def _max(data, axis=None):
    data = SeriesSet(data)
    if axis is None:
        return max(map(max, data))
    if axis == 0:
        return map(max, data.values())
    if axis == 1:
        return map(max, data)

def mean(data, axis=None):
    '''average of sequence elements.

    Parameters
    ----------
    data : array_like
        elements to average.

    axis : None, 0, 1
        determine how to summary this data.
        None - average value of all elements into one value,
        0 - average value of the elements in each variable,
        1 - average value of the elements in each record.

    Return
    ------
    number : number or numbers in list
        the mean of data

    Examples
    --------
    >>> a = [[1, 2], [3, 4]]
    >>> dp.mean(a)
    2.5
    >>> dp.mean([[0.5, 0.7],
                 [0.2, 1.5]])
    0.725
    >>> dp.mean([[0, 1],
                 [0, 5]], axis=1) # mean of each record
    [0.5, 2.5]
    >>> dp.mean([[0, 1],
                 [0, 5]], axis=0) # mean of each variable
    [0.0, 3.0]
    '''
    if axis is None:
        if hasattr(data, 'shape'):
            return float(_sum(data, axis)) / _sum(data.shape)
        if is_seq(data) or isinstance(data, Series):
            if is_seq(data[0]):
                return float(_sum(data, axis)) / (len(data[0]) + len(data))
            return float(_sum(data, axis)) / len(data)

    if hasattr(data, 'shape'):
        size = float(data.shape[axis])
    elif axis == 1:
        size = float(len(data))
    else:
        size = float(len(data[0]))
        
    result = Matrix([value / size for value in _sum(data, axis)]).T
    if result.shape.Ln == result.shape.Col == 1:
        return result[0][0]
    return result

def std(data, axis=None):
    Ex = mean(data)
    Ex2 = sum((i ** 2 for i in data)) / len(data)
    return (Ex2 - Ex**2) ** 0.5

def cov(x, y=None, **kwrds):
    '''
    formula:  cov(x,y) = E(xy) - E(x)E(y) 
    '''
    # high level data structure
    if hasattr(x, 'shape') or y is None:
        if hasattr(x, 'tolist'):
            x = x.tolist()
        size = len(x)
        covX = [[0] * size for t in range(size)]
        for i, x_1 in enumerate(x):
            for j, x_2 in enumerate(x):
                cov_num = cov(x_1, x_2)
                covX[i][j] = cov_num
                covX[j][i] = cov_num
        return Matrix(covX)

    # sequence level data structure
    try:
        X, Y = array('f', x), array('f', y)
    except TypeError:
        X, Y = array('f'), array('f')
        for x, y in zip(x, y):
            if is_math(x) and is_math(y):
                X.append(x)
                Y.append(y)

    assert len(X) == len(Y), 'two variables have different lenth.'
    size = float(len(X))
    if size == 0:
        warn('x and y has no efficient numbers.')
        return 0
    
    Ex, Ey = kwrds.get('Ex', None), kwrds.get('Ey', None)
    if not Ex:
        Ex = sum(X) / size
    if not Ey:
        Ey = sum(Y) / size
    return sum(((x-Ex) * (y-Ey) for x, y in zip(X, Y))) / (size - 1)

def corr(x, y, method='pearson'):
    '''calculate the correlation between X and Y

    Parameters
    ----------
    x, y : array-like
        sequence of values to calculate the correlation

    method : str (default="pearson")
        the method used to calculate correlation.
        ("pearson" and "spearman" are supported).
    '''
    assert isinstance(method, STR_TYPE), 'method should be a str or unicode'
    assert method in ('pearson', 'spearman'), 'method should be "pearson" or "spearman"'
    if method.lower() == 'pearson':
        return _corr_pearson(x, y)

    if method.lower() == 'spearman':
        return _corr_spearman(x, y)  

def _corr_spearman(x, y):
    '''calculate the spearman rank correlation between X and Y

    Formula
    -------
                   6
    Rs = 1 - -------------SIGMA(di^2)
               n(n^2 - 1)

    Reference
    ---------
    He X & Liu W. Applied Regression Analysis. China People's University
    Publication House. 2015.
    '''
    def rank(X):
        lst, last_value, last_rank = [1], X[0], 1.0
        for rank, value in enumerate(X[1:], 2):
            if value == last_value:
                lst.append(float(last_rank))
                continue
            lst.append(float(rank))
            last_value, last_rank = value, rank
        return lst
        
    data = SeriesSet({'X': x, 'Y': y})
    n = data.shape.Ln
    data = data.sort(('X', 'DESC'))
    data.append_col(rank(data.X), 'xRank')
    data = data.sort(('Y', 'DESC'))
    data.append_col(rank(data.Y), 'yRank')
    return _corr_pearson(data.xRank, data.yRank)

def _corr_pearson(x, y):
    '''calculate the pearson correlation between X and Y

    formula
    -------
                    cov(x,y)
    corr(x,y) = -----------------
                 std(x) * std(y)
    '''
    return cov(x, y) / (cov(x, x) * cov(y, y)) ** 0.5
        
def frequency(data, cut=0.5):
    statistic = namedtuple('Frequency', ['Lower', 'Equal', 'Upper'])
    Group1, Group2, Group3 = 0, 0, 0
    size = float(len(data))
    for each in data:
        if each < cut:
            Group1 += 1
        elif each > cut:
            Group3 += 1
        else:
            Group2 += 1
    return statistic(Group1/size, Group2/size, Group3/size)

def quantiles(data, shapes=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]):
    data = sorted(data)
    groups = list()
    lenth = len(data) + 1
    for point in shapes:
        try:
            groups.append(data[int(lenth*point)])
        except:
            pass
    return groups

def distribution(data, breaks=10, x_label=False):
    assert isinstance(breaks, int)
    data = Series(data)
    groups = [0] * breaks
    maxn, minn = max(data), min(data)
    ranges = maxn - minn
    size = len(data)
    breaks = [minn+i*ranges/breaks for i in range(1, breaks+1)]
    for record in data:
        for i,cut_point in enumerate(breaks):
            if cut_point >= record:
                groups[i] += 1
                break
    if x_label:
        return ([minn+i*ranges/(breaks*2) for i in range(1, breaks+1)],
                [float(i)/size for i in groups])
    return [float(i)/size for i in groups]


def describe(data):
    '''
    Help you compute some basic describe statistic indexes.
    It only supports 1 dimention data.

    Parameter
    ---------
    data : array - Like
        The sequence store your data inside.

    Return
    ------
    NamedTuple(Mean, S, Sn, CV, Range, Min, Max, Skew, Kurt)

    Mean : float
        mean of data.

    S : float
        adjusted variance of data.

    Sn : float
        sample variance of data.

    CV : float
        coeffient variance of the data.

    Min : value
        the minimun value of the data.

    Max : value
        the maximun value of the data.

    Range : value
        the range of the data.

    Formulas
    --------
    <1> E(x) = sum(x)/n                             # Average of samples
    <2> D(x) = E(x^2) - E(x)^2                      # Sample Variance
    <3> D(x)' = n/(n-1) * D(x)                      # Modified Sample Variance
    <4> CV = D(x) / E(x)                            # Coefficient of Variation
                E(x^3) - 3*E(x)*D(x) - E(x)^3
    <5> S(x) = ------------------------------------ # Skewness of samples
                            D(x)^1.5
    <6> K(x) = E(x)^4 / D(x)^2 - 3                  # Excess Kurtosis of samples
                
    
    '''
    statistic = namedtuple('STAT',
                           ['Mean', 'S', 'Sn', 'CV', 'Range',
                            'Min', 'Max', 'Skew', 'Kurt'])
    try:
        data = array('f', x)
    except:
        data = array('f', filter(lambda x: is_math(x) and x != nan, data))
    size = float(len(data))
    if size == 0:
        return statistic(*[None] * 9)

    Ex = sum(data) / size
    Ex2 = sum((i ** 2 for i in data)) / size
    Ex3 = sum((i ** 3 for i in data)) / size
    Ex4 = sum((i ** 4 for i in data)) / size

    std = (Ex2 - Ex**2) ** 0.5
    std_n = size / (size - 1.0) * std
    if std == 0:
        S,K = '-','-'
    else:
        S = (Ex3 - 3 * Ex * std ** 2 - Ex ** 3) / std ** 1.5
        K = Ex4 / std ** 4 - 3
    min_, max_ = min(data), max(data)

    if is_math(min_) and is_math(max_):
        rang = max_ - min_
    else:
        rang = None
    
    if Ex == 0:
        return statistic(Ex, std, std_n, None, rang, min_, max_, S, K)
    return statistic(Ex, std, std_n, std/Ex, rang, min_, max_, S, K)
