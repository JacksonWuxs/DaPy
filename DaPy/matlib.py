from .core import Matrix, is_math, is_iter
import math

def dot(matrix_1, matrix_2):
    try:
        return matrix_1.dot(matrix_2)
    except AttributeError:
        col_size_1 = len(matrix_1[0])
        col_size_2 = len(matrix_2[0])
        line_size_1 = len(matrix_1)
        line_size_2 = len(matrix_2)
        columns = None
    except TypeError:
        raise TypeError('unsupported operation dot, with type'+\
                        ' %s and ,'%str(type(matrix_1)) +\
                        '%s and '%str(type(matrix_2)))
    
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

