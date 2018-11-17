from copy import deepcopy
from core import Frame, SeriesSet, Matrix as mat
from core import is_seq, is_math, is_value

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
        return delete(mat(deepcopy(data)), index, axis)
    
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
        if isinstance(data, Frame):
            data = Frame(data)
        else:
            data = SeriesSet(data)

        for other in tup[1:]:
            data.extend(other)
        return data
    
    if not isinstance(tup[0], mat):
        tup[0] = mat(deepcopy(tup[0]), True)
        return column_stack(tup)
    
    new = tup[0]
    for data in tup[1:]:
        if is_value(data):
            raise TypeError('can not stack a number a column.')

        if all(map(is_seq, data)):
            for current, append in zip(new, data):
                current.extend(append)
        else:
            for current, append in zip(new, data):
                current.append(append)
    return mat(new)
