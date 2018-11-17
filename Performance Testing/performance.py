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
import time

def test_Pandas(files):
    # Testing of Pandas
    t1 = time.clock()
    data_pandas = pd.DataFrame(pd.read_csv(files))
    t2 = time.clock()
    for index in data_pandas.itertuples():
        this_line = index
    t3 = time.clock()
    data_pandas.sort_values(by='Price')
    t4 = time.clock()

    
    print '                Pandas'
    print '    Load Time: %.2f s '%(t2-t1)
    print 'Traverse Time: %.2f s '%(t3-t2)
    print '    Sort Time: %.2f s '%(t4-t3)
    print '   Total Time: %.2f s '%(t4-t1)

def test_Numpy(files):
    # Testing of numpy
    t1 = time.clock()
    data_numpy = np.loadtxt(files, skiprows=1, delimiter=',', dtype={
        'names': ('Date', 'Time', 'Price', 'Volume', 'Token', 'LastToken', 'LastMaxVolume'),
        'formats': ('i4', 'S1', 'i4', 'i4', 'i4', 'i4', 'i4')},)
    t2 = time.clock()
    for index in data_numpy:
        this_line = index
    t3 = time.clock()
    data_numpy[np.lexsort(data_numpy.T[2, None])]
    t4 = time.clock()

    
    print '                Numpy'
    print '    Load Time: %.2f s '%(t2-t1)
    print 'Traverse Time: %.2f s '%(t3-t2)
    print '    Sort Time: %.2f s '%(t4-t3)
    print '   Total Time: %.2f s '%(t4-t1)


def test_DaPy(files):
    # Testing of DaPy
    t1 = time.clock()
    data_dapy = dp.read(files, 'frame')
    t2 = time.clock()
    for line in data_dapy:
        this_line = line 
    t3 = time.clock()
    data_dapy._data.sort(('Price', 'ASC'))
    t4 = time.clock()


    print '                Dapy'
    print '    Load Time: %.2f s '%(t2-t1)
    print 'Traverse Time: %.2f s '%(t3-t2)
    print '    Sort Time: %.2f s '%(t4-t3)
    print '   Total Time: %.2f s '%(t4-t1)
    
def main(files):
    #test_Pandas(files)
    test_Numpy(files)
    #test_DaPy(files)
    
if __name__ == '__main__':
    main('read_csv.csv')
    raw_input('Continue...')
