from bisect import bisect_right, bisect_left
from copy import copy
from collections import Counter
from operator import itemgetter
from itertools import repeat

from .core import DataSet, Frame, SeriesSet, Matrix as mat, Series
from .core import is_seq, is_math, is_value, range, filter, zip, xrange

def merge(sheets=(), keys=(), how='inner'):
    '''laterally merge multiple datasets into a new dataset.
    More info with help(dp.DataSet.merge)

    Parameters
    ----------
    sheets : 2-D data sheet(s)
        
    keys : int, str and list
        the key column in each dataset.
        `int` -> the number of key column index;
        `str` -> the name of key column;
        `list` -> the number or names for each key column in each dataset

    how : 'inner', 'outer', 'left', 'right' (default='inner')
        how to handle rows which not match the columns
        `left` -> Keep only all rows in the current sheet;
        `right` -> Keep only all rows in the other sheet;
        `inner` -> Keep only rows from the common parts of two tables;
        `outer` -> Keep all rows from both sheets;

    Return
    ------
    sheet : merged dataset

    Example
    -------
    >>> import DaPy as dp
    >>> data1 = dp.SeriesSet([['A', 39, 'F'],
                            ['B', 40, 'F'], ['C', 38, 'M']],
                          ['Name', 'Age', 'Gender'])
    >>> data2 = dp.SeriesSet([['A', 'F', True], ['B', 'F', False],
                            ['C', 'M', True]],
                          ['Name', 'Gender', 'Married'])
    >>> data3 = dp.SeriesSet([['A', 'China'], ['B', 'US'],
                        ['C', 'Japan'], ['D', 'England']],
                        ['Name', 'Country'])
    >>> data = [data1, data2, data3]
    >>> dp.merge(data, 0, 'inner')['Name', 'Age', 'Gender', 'Married', 'Country'].show()
     Name | Age | Gender | Married | Country
    ------+-----+--------+---------+---------
      A   |  39 |   F    |   True  |  China  
      B   |  40 |   F    |  False  |    US   
      C   |  38 |   M    |   True  |  Japan
    >>> dp.merge(data, 0, 'outer')['Name', 'Age', 'Gender', 'Married', 'Country'].show()
     Name | Age | Gender | Married | Country
    ------+-----+--------+---------+---------
      A   |  39 |   F    |   True  |  China  
      B   |  40 |   F    |  False  |    US   
      C   |  38 |   M    |   True  |  Japan  
     nan  | nan |  nan   |   nan   | England 
    '''
    if not is_seq(keys):
        keys = [keys] * len(sheets)
    assert len(keys) == len(sheets), 'keys should have same lenth as datas.'

    if len(sheets) == 1:
        if isinstance(sheets[0], DataSet):
            return merge(sheets[0].data, keys, how)
        raise RuntimeError('only one sheets, can not merge.')
    
    merged, left = SeriesSet(sheets[0]), keys[0]
    for right, data in zip(keys[1:], sheets[1:]):
        merged = merged.merge(data, how, left, right)
        left = right
    return merged

def delete(data, index, axis=0):
    if isinstance(index, int):
        index = [index, ]

    if isinstance(data, (SeriesSet, Frame)):
        if isinstance(data, Frame):
            data = Frame(data)
        else:
            data = SeriesSet(data)
        if axis == 1:
            index = tuple([data.columns[i] for i in index])
        else:
            index = tuple(sorted(index, reverse=True))
        del data[index]
        return data
        
    if not isinstance(data, mat):
        return delete(mat(copy(data)), index, axis)
    
    if axis == 1:
        new = []
        for line in data:
            for i in index:
                del line[i]
            new.append(line)
        return mat(new)
    
    if axis == 0:
        index = sorted(index, reverse=True)
        new = [line for i, line in enumerate(data) if i not in index]
        return mat(new)

def concatenate(tup, axis=0):
    '''Stack 1-D data as columns or rows into a 2-D SeriesSet
        concatenate([A, B], axis=0) -> Horizentally combine A & B
        concatenate([A, B], axis=1) -> Vertically join A & B
        Detail see: DaPy.column_stack or DaPy.row_stack
    '''
    if axis == 1:
        return column_stack(tup)
    return row_stack(tup)

def column_stack(tup):
    '''Stack 1-D data as columns into a 2-D dataset.

    Parameters
    ----------
    tup : sequence of 1-D or 2-D data.
        Arrays to stack. All of them must have the same first dimension.

    Retures
    -------
    stack

    Examples
    --------
    >>> one = [1, 1, 1]
    >>> two = [2, 2, 2]
    >>> else_ = [[3, 3, 3], [4, 4, 4] [5, 5, 5]]
    >>> dp.columns_stack([one, two, else])
    matrix(|1.0  2   3   3   3 |
           |1.0  2   4   4   4 |
           |1.0  2   5   5   5 |)
    '''
    if isinstance(tup, tuple):
        tup = list(tup)
        
    if isinstance(tup[0], (Frame, SeriesSet)):
        if isinstance(tup[0], Frame):
            data = Frame(tup[0])
        else:
            data = SeriesSet(tup[0])

        for other in tup[1:]:
            data.join(other)
        return data

    if isinstance(tup[0], mat):
        new = tup[0]
        for other in tup[1:]:
            assert is_value(other) is False, 'can not stack a number into a column.'
            other = mat(other)
            assert other.shape.Ln == new.shape.Ln
            for current_row, other_row in zip(new,  other):
                current_row.extend(other_row)
        new._dim = mat.dims(new.shape.Ln, new.shape.Col + other.shape.Col)
        return new
    
    if not isinstance(tup[0], mat):
        tup[0] = mat(copy(tup[0]))
        return column_stack(tup)

def _repeat(val, times):
    '''create a series which contains `val` for `times`'''
    return Series(repeat(val, times))

def row_stack(tup):
    if isinstance(tup, tuple):
        tup = list(tup)

    if isinstance(tup[0], Series):
        tup[0] = SeriesSet(tup[0])

    if isinstance(tup[0], (Frame, SeriesSet)):
        if isinstance(tup[0], Frame):
            data = Frame(tup[0])
        else:
            data = SeriesSet(tup[0])

        for other in tup[1:]:
            data.extend(other)
        return data
    
    if not isinstance(tup[0], mat):
        if hasattr(tup[0], 'tolist'):
            tup[0] = tup[0].tolist()
        tup[0] = mat(copy(tup[0]))
        return row_stack(tup)
    
    new = copy(tup[0]).src
    for data in tup[1:]:
        if is_value(data):
            raise TypeError('can not stack a number into a row.')

        if hasattr(data, 'tolist'):
            data = data.tolist()
        if all(map(is_seq, data)):
            for row in data:
                new.append(row)
        else:
            new.append(data)
    return mat(new)

def get_ranks(series, duplicate='mean'):
    '''return the rank of each value in the series

    In this function you can choose how to rank the data
    which has multiple same value. The cumsumption of
    time is O(N*logN + 3N).

    Parameters
    ----------
    series : array-like
        the data you expect to sort
    
    duplicate : str (default='mean')
        how to calculate the rank for same data
        `mean` -> use average rank when appear same values
        `first` -> use the first rank of same values
        `last` -> use the last rank of same values

    Return
    ------
    series : the ranks of each values in the series

    Examples
    --------
    >>> from DaPy import get_rank
    >>> get_rank([3, 3, 2, 5, 7, 1, 4], duplicate='mean')
    [3.5, 3.5, 2, 6, 7, 1, 5]
    '''
    assert duplicate in ('mean', 'first', 'last')
    sort_series = sorted(series) # O(N*logN)
    ranks = Counter(sort_series) # O(N)
    rank = 1
    
    while rank <= len(sort_series): # O(N)
        value = sort_series[rank - 1]
        num = ranks[value]
        if num == 1 or duplicate == 'first':
            ranks[value] = rank
        elif duplicate == 'mean':
            ranks[value] = (2 * rank + (num - 1)) / 2.0
        elif duplicate == 'last':
            ranks[value] = rank + num - 1
        rank += num
    return [ranks.__getitem__(v) for v in series] # O(N)

def get_dummies(data, value=1, dtype='mat'):
    '''Convert categorical variable into dummy variables

    Parameters
    ----------
    data : array-like
        the data you expect to convert, it should be a 1D sequence data

    value : value-type (default=1)
        the value which will be used as a mark in the return object

    dtype : str, data structure (default='mat')
        the type of return object

    Examples
    --------
    >>> from DaPy import get_dummies
    >>> get_dummies([1, 1, 3, 4, 2, 3, 4, 1])
    matrix(┏       ┓
           ┃1 0 0 0┃
           ┃1 0 0 0┃
           ┃0 0 1 0┃
           ┃0 0 0 1┃
           ┃0 1 0 0┃
           ┃0 0 1 0┃
           ┃0 0 0 1┃
           ┃1 0 0 0┃
           ┗       ┛)
    >>> get_dummies(data=list('abdddcadacc'), value='f', dtype='frame')
     a | c | b | d
    ---+---+---+---
     f | 0 | 0 | 0 
     0 | 0 | f | 0 
     0 | 0 | 0 | f 
     0 | 0 | 0 | f 
     0 | 0 | 0 | f 
     0 | f | 0 | 0 
     f | 0 | 0 | 0 
     0 | 0 | 0 | f 
     f | 0 | 0 | 0 
     0 | f | 0 | 0 
     0 | f | 0 | 0 
    '''
    assert is_value(value), 'parameter should be a value, not %s' % type(value)
    assert is_seq(data), 'converted object should be a sequence'
    assert str(dtype).lower() in ('frame', 'set', 'mat', 'matrix', 'seriesset')
    set_data = sorted(set(data))
    settle = dict(zip(set_data, xrange(len(set_data))))
    dummies = [[0] * len(settle) for i in range(len(data))]

    for record, original in zip(dummies, data):
        record[settle[original]] = value

    if callable(dtype):
        return dtype(record)
    
    if dtype.lower() == 'frame':
        return Frame(dummies, settle)
    
    if dtype.lower() in ('set', 'seriesset'):
        return SeriesSet(dummies, settle)
    
    if dtype.lower() in ('mat', 'matrix'):
        return mat(dummies)

def get_categories(array, cut_points, group_name, boundary=(False, True)):
    '''values in `array` are divided into groups according to `cut_points`

    This function uses bisect library to impletment a lookup operation.
    The comsumption of time is O(N*logK), where K is the number of cut points.

    Parameters
    ----------
    array : array-like
        the data you expect to be grouped

    cut_points : values in list
        the boundaries of each subgroup

    group_names : str in list
        the name of each group

    boundary : bools in tuple (default=(False, True))
        how to divide the values which exactely match the boundary

    Return
    ------
    group_list : group_names in the list

    Example
    -------
    >>> from DaPy import get_group
    >>> scores = [57, 89, 90, 100]
    >>> cuts = [60, 70, 80, 90]
    >>> grades = ['F', 'D', 'C', 'B', 'A']
    >>> get_group(scores, cuts, grades, boundary=(False, True))
    ['F', 'B', 'B', 'A']
    >>> get_group(scores, cuts, grades, boundary=(True, False))
    ['F', 'B', 'A', 'A']
    '''
    assert is_seq(array), '`array` must be a sequence'
    assert is_seq(cut_points), '`cut points` must be held with a sequence'
    assert is_seq(group_name), '`group name` must be held with a sequence'
    assert isinstance(boundary, tuple) and len(boundary) == 2, '`boundary` must be a 2 dimention tuple'
    assert boundary.count(True) == 1, '`boundary` must have only single True'
    assert sorted(cut_points) == cut_points, '`cut_points` must be arranged by asceding'
    assert len(cut_points) == len(group_name) - 1
    
    if boundary[0] is True:
        return [group_name[bisect_right(cut_points, x)] for x in array]
    return [group_name[bisect_left(cut_points, x)] for x in array]
    






        
