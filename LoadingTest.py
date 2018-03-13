import DaPy as dp
import pandas as pd
import numpy as np
import time
import sys
import gc
import array
import binascii

def test_load(files):
    gc.disable()

    # Testing of Pandas
    t1 = time.time()
    
    data_pandas = pd.DataFrame(pd.read_csv(files))
    
    t2 = time.time()
    
    for index in data_pandas.index:
        this_line = data_pandas.loc[index].values[:]
        
    t3 = time.time()

    # Tesing of Datapy
    data_DaPy = dp.DataSet(files)
    data_DaPy.readframe()
    
    t4 = time.time()

    for item in data_DaPy:
        this_line = item

    t5 = time.time()

    # Tesing of Numpy
    data_numpy = np.genfromtxt(files, dtype=None, delimiter=',', names=True)

    t6 = time.time()

    for line in data_numpy:
        this_line = line

    t7 = time.time()

    # File Information
    with open(files,'r') as f:
        data_file = f.read()

    # Calculating the memory of each data set
    size_pandas = sys.getsizeof(data_pandas)
    size_DaPy = sys.getsizeof(data_DaPy.data)
    size_numpy = sys.getsizeof(data_numpy)
    size_file = sys.getsizeof(data_file)

    gc.enable()
    gc.collect()

    print '\n'
    print '               Datapy | Pandas | Numpy | File'
    print '    Load Time:{0:^4.2f}\t| {1:^4.2f}\t| {2: ^4.2f}\t|  -'.format(t4-t3, t2-t1, t6-t5)
    print 'Traverse Time:{0:^4.1f}\t| {1:^4.1f}\t| {2: ^4.1f}\t|  -'.format(t5-t4, t3-t2, t7-t6)
    print '  Total Spent:{0:^4.1f}\t| {1:^4.1f}\t| {2: ^4.1f}\t|  -'.format(t5-t3, t3-t1, t7-t5)
    print '  Memory Size: %dMB\t| %dMB\t| %dMB\t| %dMB'%(size_DaPy//1048575,size_pandas//1048575,size_numpy//1048575,size_file//1048575)
    print '\n' 
    
test_load('1718.csv')
