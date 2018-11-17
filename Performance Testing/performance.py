# user/bin/python2
#########################################
# Author         : Xuansheng Wu           
# Email          : wuxsmail@163.com 
# created        : 2018-01-01 00:00 
# Last modified  : 2018-11-17 11:09
# Filename       : performance.py
# Description    : testify the efficiency
#                  among DaPy, numpy and
#                  pandas.
#########################################
import DaPy as dp
import pandas as pd
import numpy as np
from time import clock

def test_Pandas(files):
    # Testing of Pandas
    t1 = clock()
    data_pandas = pd.DataFrame(pd.read_csv(files))
    t2 = clock()
    for index in data_pandas.itertuples():
        this_line = index
    t3 = clock()
    data_pandas.sort_values(by='Price')
    t4 = clock()
    data_pandas.to_csv('test_Pandas.csv', index=0)
    t5 = clock()

    
    print '                Pandas (%s)' % pd.__version__
    print '    Load Time: %.2f s '%(t2-t1)
    print 'Traverse Time: %.2f s '%(t3-t2)
    print '    Sort Time: %.2f s '%(t4-t3)
    print '    Save Time: %.2f s '%(t5-t4)
    print '   Total Time: %.2f s '%(t5-t1)

def test_Numpy(files):
    # Testing of numpy
    t1 = clock()
    data_numpy = np.loadtxt(files, skiprows=1, delimiter=',', dtype={
        'names': ('Date', 'Time', 'Price', 'Volume', 'Token', 'LastToken', 'LastMaxVolume'),
        'formats': ('i4', 'S1', 'i4', 'i4', 'i4', 'i4', 'i4')},)
    t2 = clock()
    for index in data_numpy:
        this_line = index
    t3 = clock()
    data_numpy = np.sort(data_numpy, order='Price')
    t4 = clock()
    data_numpy.tofile('test_Numpy.csv', sep=',')
    t5 = clock()

    
    print '                Numpy (%s)' % np.__version__
    print '    Load Time: %.2f s '%(t2-t1)
    print 'Traverse Time: %.2f s '%(t3-t2)
    print '    Sort Time: %.2f s '%(t4-t3)
    print '    Save Time: %.2f s '%(t5-t4)
    print '   Total Time: %.2f s '%(t5-t1)


def test_DaPy(files):
    # Testing of DaPy
    t1 = clock()
    data_dapy = dp.read(files)
    t2 = clock()
    data_dapy.toframe()
    data_dapy = data_dapy.data
    t2_ = clock()
    for line in data_dapy:
        this_line = line 
    t3 = clock()
    data_dapy.sort(('Price', 'ASC'))
    t4 = clock()
    dp.save('test_DaPy.csv', data_dapy)
    t5 = clock()
    
    print '                Dapy (%s)' % dp.__version__
    print '    Load Time: %.2f s '%(t2-t1)
    print 'Traverse Time: %.2f s '%(t3-t2_)
    print '    Sort Time: %.2f s '%(t4-t3)
    print '    Save Time: %.2f s '%(t5-t4)
    print '   Total Time: %.2f s '%(t5-t1)
    
def main(files):
    test_Pandas(files)
    test_Numpy(files)
    test_DaPy(files)
    
if __name__ == '__main__':
    main('read_csv.csv')
    raw_input('Continue...')
