from collections import namedtuple, deque, Iterable, deque
from datetime import datetime
from time import struct_time
from array import array
from tools import is_math, is_iter, is_seq
import math

__all__ = ['cov', 'corr', 'frequency', 'quantiles', '_sum',
           'distribution','describe', 'mean']


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

def cov(data_1, data_2):
    '''
    formula:  cov(x,y) = E(xy) - E(x)E(y) 
    '''
    try:
        if len(data_1) != len(data_2):
            raise ValueError('two variables have different lenth.')
        size = float(len(data_1))
        xy = [x*y for x, y in zip(data_1, data_2) if is_math(x) and is_math(y)]
        Exy = sum(xy)/size
        Ex = sum(filter(is_math, x))/size
        Ey = sum(filter(is_math, x))/size
    except ZeroDivisionError:
        return None
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
                           ['Mean', 'S', 'Sn', 'CV', 
                            'Min', 'Max', 'Skew', 'Kurt'])
    data = filter(is_math, data)
    size = len(data)

    try:
        Ex = sum(data) / float(size)
        Ex2 = sum(map(pow, data, [2]*size)) / float(size)
        Ex3 = sum(map(pow, data, [3]*size)) / float(size)
        Ex4 = sum(map(pow, data, [4]*size)) / float(size)
    except ZeroDivisionError:
        return statistic(None, None, None, None, None, None, None)
    
    std = (Ex2 - Ex**2)**0.5
    if size > 1:
        std_n = n / (n - 1.0) * std
    else:
        std_n = std

    S = (Ex3 - 3*Ex*std**2 - Ex**3) / std ** 1.5
    K = Ex4 / std ** 4 - 3
    
    if Ex == 0:
        return statistic(Ex, std, std_n, None, min(data), max(data), S, K)
    return statistic(Ex, std, std_n, std/Ex, min(data), max(data), S, K)
