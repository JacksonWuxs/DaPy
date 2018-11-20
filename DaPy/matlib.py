from array import array
from core import Matrix, is_math, is_seq, is_iter
from collections import namedtuple, deque, Iterable, deque
from datetime import datetime
from time import struct_time
from warnings import warn
import math

__all__ = ['dot', 'multiply', 'exp', 'zeros', 'ones', 'C', 'P',
           'cov', 'corr', 'frequency', 'quantiles', '_sum',
           'distribution','describe', 'mean']

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
            sumup = sum(matrix_1[i][j]*matrix_2[j][pos]\
                        for j in range(col_size_1))
            new_line.append(sumup)
        new_.append(new_line)
    return Matrix(new_, check=False)

def exp(other):
    if hasattr(other, 'shape'):
        new = [0] * other.shape[0]
        for i, line in enumerate(other):
            new[i] = map(math.exp, line)
        return Matrix(new, check=False)
    
    if is_math(other):
        return math.exp(other)

    if is_iter(other):
        new_ = list()
        for item in other:
            new_.append(exp(item))
        return new_

    raise TypeError('expects an iterable or numeric for exp(), got %s'%type(other))

def create_mat(shape, num):
    matrix = mat()
    matrix.make(shape[0], shape[1], num)
    return matrix

def zeros(shape):
    return create_mat(shape, 0)

def ones(shape):
    return create_mat(shape, 1)

def log(data):
    if is_seq(data):
        if is_seq(data[0]):
            return [map(math.log, record) for record in data]
        return map(math.log, data)
    return math.log(data)


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
    if axis is None:
        if all(map(is_math, data)):
            return sum(data)
        return sum(sum(data, list()))
    
    if axis == 1:
        return map(sum, data)

    if axis == 0:
        return _sum([line for line in zip(*data)], axis=1)

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
    >>> dp.sum([[0, 1],
                [0, 5]], axis=1) # mean of each record
    [0.5, 2.5]
    >>> dp.sum([[0, 1],
                [0, 5]], axis=0) # mean of each variable
    [0.0, 3.0]
    '''
    if axis is None:
        if hasattr(data, 'shape'):
            return float(_sum(data, axis)) / _sum(data.shape)
        if is_seq(data):
            if is_seq(data[0]):
                return float(_sum(data, axis)) / (len(data[0]) + len(data))
            return float(_sum(data, axis)) / len(data)

    if hasattr(data, 'shape'):
        size = float(data.shape[axis])
    elif axis == 1:
        size = float(len(data))
    else:
        size = float(len(data[0]))
    return [value / size for value in _sum(data, axis)]

def cov(x, y):
    '''
    formula:  cov(x,y) = E(xy) - E(x)E(y) 
    '''
    if len(x) != len(y):
        raise ValueError('two variables have different lenth.')
    
    X, Y = array('f'), array('f')
    for x, y in zip(x, y):
        if is_math(x) and is_math(y):
            X.append(x)
            Y.append(y)
            
    size = float(len(X))
    if size == 0:
        warn('x and y has no efficient numbers.')
        return 0
    
    Exy = sum([x*y for x, y in zip(X, Y)])/size
    Ex = sum(X)/size
    Ey = sum(Y)/size
    return Exy - Ex * Ey
     
def corr(data_1, data_2):
    '''
    formula: cor(x,y) = cov(x,y) / (std(x)*std(y))
    '''
    static_1 = describe(data_1)
    static_2 = describe(data_2)
    covariance = cov(data_1, data_2)
    try:
        return covariance / (static_1.Sn * static_2.Sn)
    except TypeError:
        return None
    except ZeroDivisionError:
        return float('+inf')
        
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
    groups = [0 for i in range(breaks)]
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
    NamedTuple(Mean, S, Sn, CV, Min, Max, Range)

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
    data = filter(is_math, data)
    size = len(data)

    try:
        Ex = sum(data) / float(size)
        Ex2 = sum(map(pow, data, [2]*size)) / float(size)
        Ex3 = sum(map(pow, data, [3]*size)) / float(size)
        Ex4 = sum(map(pow, data, [4]*size)) / float(size)
    except ZeroDivisionError:
        return statistic(None, None, None, None, None, None, None, None, None)
    
    std = (Ex2 - Ex**2)**0.5
    if size > 1:
        std_n = size / (size - 1.0) * std
    else:
        std_n = std

    S = (Ex3 - 3*Ex*std**2 - Ex**3) / std ** 1.5
    K = Ex4 / std ** 4 - 3
    min_, max_ = min(data), max(data)
    
    if Ex == 0:
        return statistic(Ex, std, std_n, None, max_-min_, min_, max_, S, K)
    return statistic(Ex, std, std_n, std/Ex, max_-min_, min_, max_, S, K)
