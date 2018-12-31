from copy import copy
from .core import DataSet, Frame, SeriesSet, Matrix as mat
from .core import is_seq, is_math, is_value, range, filter, zip

def merge(*datas, **kwrds):
    '''laterally merge multiple datasets into a new dataset.
    More info with help(dp.DataSet.merge)

    Parameters
    ----------
    keys : int, str and list
        the key column in each dataset.
        `int` -> the number of key column index;
        `str` -> the name of key column;
        `list` -> the number or names for each key column in each dataset

    Return
    ------
    sheet : merged dataset

    Example
    -------
    >>> import DaPy as dp
    >>> data1 = dp.Frame([['A', 39, 'F'], ['B', 40, 'F'], ['C', 38, 'M']],
                          ['Name', 'Age', 'Gender'])
    >>> data2 = dp.Frame([['A', 'F', True], ['B', 'F', False], ['C', 'M', True]],
                          ['Name', 'Gender', 'Married'])
    >>> data3 = [['A', 'China'], ['B', 'US'], ['C', 'Japan'], ['D', 'England']]
    >>> dp.merge(data1, data2, data3, keys=0, keep_key='self', keep_same=False).show()
     Name | Age  | Gender | Married |   C_1  
    ------+------+--------+---------+---------
      A   |  39  |   F    |    F    |  China  
      B   |  40  |   F    |    F    |    US   
      C   |  38  |   M    |    M    |  Japan  
     None | None |  None  |   None  | England 
    '''
    keys = kwrds.get('keys', 0)
    keep_key = kwrds.get('keep_key', True)
    keep_same = kwrds.get('keep_same', True)
    if not is_seq(keys):
        keys = [keys] * len(datas)
    assert len(keys) == len(datas), 'keys should have same lenth as datas.'

    if len(datas) == 1:
        if isinstance(datas[0], DataSet):
            return merge(*datas[0].data, **kwrds)
        raise RuntimeError('only one sheets, can not merge.')
    
    first_data, last_key = SeriesSet(datas[0]), keys[0]
    for key, data in zip(keys[1:], datas[1:]):
        first_data.merge(data, last_key, key, keep_key, keep_same)
        last_key = key
    return first_data    

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
        return mat(new, False)
    
    if axis == 0:
        index = sorted(index, reverse=True)
        new = [line for i, line in enumerate(data) if i not in index]
        return mat(new, False)

def column_stack(tup):
    '''Stack 1-D data as columns into a 2-D data.

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
    >>> else = [[3, 3, 3], [4, 4, 4] [5, 5, 5]]
    >>> np.columns_stack([one, two, else])
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
            data.extend_col(other)
        return data
    
    if not isinstance(tup[0], mat):
        if hasattr(tup[0], 'tolist'):
            tup[0] = tup[0].tolist()
        tup[0] = mat(copy(tup[0]), True)
        return column_stack(tup)
    
    new = tup[0]
    for data in tup[1:]:
        if is_value(data):
            raise TypeError('can not stack a number into a column.')

        if hasattr(data, 'tolist'):
            data = data.tolist()
        if all(map(is_seq, data)):
            for current, append in zip(new, data):
                current.extend(append)
        else:
            for current, append in zip(new, data):
                current.append(append)
    return mat(new)

def row_stack(tup):
    if isinstance(tup, tuple):
        tup = list(tup)

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
        tup[0] = mat(copy(tup[0]), True)
        return row_stack(tup)
    
    new = copy(tup[0]).data
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
    >>> get_dummies(list('abdddcadacc'), 'f', 'frame')
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

    settle = tuple(set(data))
    dummies = [[0] * len(settle) for i in range(len(data))]
    for record, original in zip(dummies, data):
        record[settle.index(original)] = value
        
    if isinstance(dtype, str) is False:
        return dtype(record)
    if dtype.lower() == 'frame':
        return Frame(dummies, settle)
    if dtype.lower() in ('set', 'seriesset'):
        return SeriesSet(dummies, settle)
    if dtype.lower() == 'mat':
        return mat(dummies)

    
        
























        
