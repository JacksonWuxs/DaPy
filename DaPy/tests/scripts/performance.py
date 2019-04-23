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
    return t2-t1, t3-t2, t4-t3, t5-t4, t5-t1

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
    return t2-t1, t3-t2, t4-t3, t5-t4, t5-t1


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
    return t2-t1, t3-t2_, t4-t3, t5-t4, t5-t1
    
def main(files):
    dp_ = dp.Frame(None, ['Load', 'Traverse', 'Sort', 'Save', 'Total'])
    np_ = dp.Frame(None, ['Load', 'Traverse', 'Sort', 'Save', 'Total'])
    pd_ = dp.Frame(None, ['Load', 'Traverse', 'Sort', 'Save', 'Total'])
    for i in range(100):
        dp_.append(test_DaPy(files))
        np_.append(test_Numpy(files))
        pd_.append(test_Pandas(files))

    summary = dp.Frame(None,
            ['engine', 'Load', 'Traverse', 'Sort', 'Save', 'Total', 'Version'])
    summary.append(['DaPy', dp.mean(dp_['Load']), dp.mean(dp_['Traverse']),
                    dp.mean(dp_['Sort']), dp.mean(dp_['Save']),
                    dp.mean(dp_['Total']), dp.__version__])
    summary.append(['Numpy', dp.mean(np_['Load']), dp.mean(np_['Traverse']),
                    dp.mean(np_['Sort']), dp.mean(np_['Save']),
                    dp.mean(np_['Total']), np.__version__])
    summary.append(['Pandas', dp.mean(pd_['Load']), dp.mean(pd_['Traverse']),
                    dp.mean(pd_['Sort']), dp.mean(pd_['Save']),
                    dp.mean(pd_['Total']), pd.__version__])

    file_ = dp.DataSet()
    file_.add(summary, 'Summary Table')
    file_.add(dp_, 'DaPy')
    file_.add(np_, 'Numpy')
    file_.add(pd_, 'Pandas')
    file_.save('Performance_result.xls')
    
if __name__ == '__main__':
    t = clock()
    main('read_csv.csv')
    print clock() - t
