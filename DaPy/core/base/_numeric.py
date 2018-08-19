from collections import namedtuple, deque, Iterable, deque
from datetime import datetime
from time import struct_time
from array import array
from _function import is_math, is_iter, is_seq
import math

__all__ = ['cov', 'corr', 'frequency', 'quantiles', '_sum',
           'distribution','describe', 'mean']

def log(data):
    if is_seq(data):
        return map(math.log, data)
    return math.log(data)


def _sum(data, axis=None):
    '''Sum of sequence elements.

    Parameters
    ----------
    data : array_like
        Elements to sum.

    Return
    ------
    Sum : float or int
        a number of sum.

    Examples
    --------
    >>> dp.sum([0.5, 1.5])
    2.0
    >>> dp.sum([[0.5, 0.7], [0.2, 1.5]])
    2.9
    >>> dp.sum([[0, 1], [0, 5]], axis=1)
    [1, 5]
    >>> dp.sum([[0, 1], [0, 5]], axis=0)
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

def mean(data):
    summary = _sum(data)
    if hasattr(data, 'dim'):
        size = data.dim.Ln * data.dim.Col
    elif is_iter(data[0]):
            size = len(data) * len(data[0])
    else:
        size = len(data)
    return summary/float(size)

def cov(data_1, data_2):
    '''
    formula:  cov(x,y) = E(xy) - E(x)E(y) 
    '''
    try:
        size = min(float(len(data_1)), float(len(data_2)))
        xy = [data_1[i] * data_2[i] for i in range(int(size))]
        Exy = sum(xy)/size
        Ex = sum(data_1)/size
        Ey = sum(data_2)/size
    except TypeError:
        data_1 = [v for v in data_1 if is_math(v)]
        data_2 = [v for v in data_2 if is_math(v)]
        return cov(data_1, data_2)
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
    <1> E(x) = sum(x)/n          # Average of samples
    <2> D(x) = E(x^2) - E(x)^2   # Sample Variance
    <3> D(x)' = n/(n-1) * D(x)   # Modified Sample Variance
    '''
    statistic = namedtuple('STAT',
                           ['Mean', 'S', 'Sn',
                            'CV', 'Min', 'Max', 'Range'])
    size = len(data)

    try:
        try:
            Ex = sum(data)/float(size)
            Ex2 = sum([i**2 for i in data])/float(size)
        except TypeError:
            data = filter(is_math, data)
            return describe(data)
    except ZeroDivisionError:
        return statistic(None, None, None, None, None, None, None)
    
    std_n = (Ex2 - Ex**2)**0.5
    if size > 1:
        std = (size/(size-1.0)*(Ex2 - Ex**2))**0.5
    else:
        std = std_n
    
    if Ex == 0:
        return statistic(Ex, std, std_n, None, min(data),
                         max(data), max(data)-min(data))
    return statistic(Ex, std, std_n, std/Ex, min(data),
                     max(data), max(data)-min(data))
