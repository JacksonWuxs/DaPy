from os import path

__all__ = ['read', 'save', 'encode']


def read(addr, dtype='col', **kward):
    '''dp.read('file.xlsx') -> return DataSet object
    '''
    from core import DataSet
    data = DataSet()
    data.read(addr, dtype, **kward)
    return data

def save(addr, data, encode='utf-8'):
    '''dp.save('file.xlsx', [1,2,3,4]) -> save dataset into file
    '''
    from core import DataSet
    data = DataSet(data)
    data.save(addr, encode)

def encode(code='cp936'):
    import sys
    stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
    reload(sys)
    sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde
    sys.setdefaultencoding(code)
    return


