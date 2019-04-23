import numpy as np
import DaPy as dp
from timeit import Timer

X = [[17, 2, 9, 2],
     [21, 8, 1, 46],
     [4, 3, 2, 13],
     [23, 1, 31, 3]]

X_dp = dp.mat(X)
X_np = np.mat(X)

def numpy_multi():
    X_np * X_np
    X_np * 32

def dapy_multi():
    X_dp * X_dp
    X_dp * 32

def numpy_dot():
    X_np.T.dot(X_np)

def dapy_dot():
    X_dp.T.dot(X_dp)

def numpy_attribute():
    X_np.T
    X_np.I
    np.linalg.det(X_np)

def dapy_attribute():
    X_dp.T
    X_dp.I
    X_dp.D

if __name__ == '__main__':
    t1 = Timer('numpy_multi()', 'from __main__ import numpy_multi, X_np').timeit(20000)
    t2 = Timer('dapy_multi()', 'from __main__ import dapy_multi, X_dp').timeit(20000)
    t3 = Timer('numpy_dot()', 'from __main__ import numpy_dot, X_dp').timeit(20000)
    t4 = Timer('dapy_dot()', 'from __main__ import dapy_dot, X_dp').timeit(20000)
    t5 = Timer('numpy_attribute()', 'from __main__ import numpy_attribute, X_dp').timeit(200)
    t6 = Timer('dapy_attribute()', 'from __main__ import dapy_attribute, X_dp').timeit(200)
    print 'Numpy is %s time faster than DaPy in matrix multiple' % (t2 / t1)
    print 'Numpy is %s time faster than DaPy in matrix dot' % (t4 / t3)
    print 'Numpy is %s time faster than DaPy in matrix attributes(T, D, I)' % (t6 / t5)
