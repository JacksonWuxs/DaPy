from collections import namedtuple, deque
from structure import DataSet, Matrix, SeriesSet, Table, Frame
import math

__all__ = ['cov', 'corr', 'CountFrequency', 'CountQuantiles', 'sum_',
           'CountDistribution','Statistic', 'mean', 'exp', 'dot']

def sum_(data):
    if isinstance(data, (Matrix, Frame, Table)):
        return sum([sum(record) for record in data])

    if isinstance(data, SeriesSet):
        return sum([sum(data[title]) for title in data._columns])
  
    if isinstance(data[0], (int, float, long)):
        return sum(data)

    return sum([sum(line) for line in data])

def mean(data):
    sums = sum_(data)
    try:
        size = data.dim.Ln * data.dim.Col
    except AttributeError:
        if isinstance(data[0], (tuple, list)):
            size = len(data) * len(data[0])
        else:
            size = len(data)
    return sums/size

def exp(other):
    try:
        new = list()
        for i in xrange(other.dim.Ln):
            new.append([math.exp(other[i][j]) for j in xrange(other.dim.Col)])
        return Matrix(new, check=False)
    
    except AttributeError:
        for i in xrange(len(other)):
            other[i] = math.exp(other[i])
        return Matrix(other)

def dot(matrix_1, matrix_2):
    try:
        col_size_1 = matrix_1.dim.Col
        col_size_2 = matrix_2.dim.Col
        line_size_1 = matrix_1.dim.Ln
        line_size_2 = matrix_2.dim.Ln
        columns = matrix_2.columns
    except AttributeError:
        col_size_1 = len(matrix_1[0])
        col_size_2 = len(matrix_2[0])
        line_size_1 = len(matrix_1)
        line_size_2 = len(matrix_2)
        columns = None
    except TypeError:
        raise TypeError('unsupported operation dot, with type'+\
                        '< %s > and '%str(type(matrix_1)) +\
                        '< %s > and '%str(type(matrix_2)))
    
    if col_size_1 != line_size_2:
        raise ValueError('shapes (%d, %d) '%(line_size_1, col_size_1)+\
                         'and (%d, %d) not aligned.'%(line_size_2, col_size_2))
    
    new_ = list()
    for i in xrange(line_size_1):
        new_line = list()
        for pos in xrange(col_size_2):
            sumup = sum(matrix_1[i][j]*matrix_2[j][pos]\
                        for j in xrange(col_size_1))
            new_line.append(sumup)
        new_.append(new_line)

    return Matrix(new_, columns, check=False)

def cov(data_1, data_2):
    '''
    formula:  cov(x,y) = E(xy) - E(x)E(y) 
    '''

    # Some intermediate variables
    size = min(float(len(data_1)), float(len(data_2)))
    xy = [data_1[i]*data_2[i] for i in range(int(size))]
    Exy = sum(xy)/size
    Ex = sum(data_1)/size
    Ey = sum(data_2)/size

    return Exy - Ex*Ey
     
def corr(data_1, data_2):
    '''
    formula: cor(x,y) = cov(x,y)/(std(x)*std(y))
    '''
    static_1 = Statistic(data_1)
    static_2 = Statistic(data_2)
    covariance = cov(data_1, data_2)

    return covariance/(static_1.Sn * static_2.Sn)
 
def CountFrequency(data, cut=0.5):
    statistic = namedtuple('Frequency',
                           ['Lower', 'Equal', 'Upper'])
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

def CountQuantiles(data, shapes=[0.05,0.1,0.25,0.5,0.75,0.9,0.95]):
    data = sorted(data)
    groups = list()
    lenth = len(data) + 1
    for point in shapes:
        try:
            groups.append(data[int(lenth*point)])
        except:
            pass
    return groups

def CountDistribution(data, breaks=10, x_label=False):
    groups = [0 for i in range(breaks)]
    maxn, minn = max(data), min(data)
    ranges = maxn - minn
    size = len(data)
    breaks = [minn+i*ranges/breaks for i in range(1,breaks+1)]
    for record in data:
        for i,cut_point in enumerate(breaks):
            if cut_point >= record:
                groups[i] += 1
                break
    if x_label:
        return (breaks, [float(i)/size for i in groups])
    return [float(i)/size for i in groups]

def Statistic(data):
    '''
    Help you calculate some basic describe statistic indexes.'
    formulas:
        <1> E(x) = sum(x)/n          # Average of samples
        <2> D(x) = E(x^2) - E(x)^2   # Sample Variance
        <3> D(x)' = n/(n-1) * D(x)   # Modified Sample Variance
    '''
    statistic = namedtuple('STAT',
                           ['Mean', 'S', 'Sn',
                            'CV', 'Min', 'Max', 'Range'])
    size = len(data)

    Ex = sum(data)/float(size)
    Ex2 = sum([i**2 for i in data])/float(size)
    
    std_n = (Ex2 - Ex**2)**0.5
    std = (size/(size-1.0)*(Ex2 - Ex**2))**0.5
    
    if Ex == 0:
        return statistic(Ex, std, std_n, None, min(data),
                         max(data), max(data)-min(data))
    return statistic(Ex, std, std_n, std/Ex, min(data),
                     max(data), max(data)-min(data))
