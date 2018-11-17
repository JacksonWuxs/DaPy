from os import path

__all__ = ['read', 'save', 'encode']


def read(addr, dtype='col', **kward):
    '''dp.read('file.xlsx') -> return DataSet object
        more info on help(dp.DataSet.read)
    '''
    from core import DataSet
    data = DataSet()
    data.read(addr, dtype, **kward)
    return data

def save(addr, data, **kward):
    '''dp.save('file.xlsx', [1,2,3,4]) -> save dataset into file
        more info on help(dp.DataSet.save)
    '''
    from core import DataSet
    if not isinstance(data, DataSet):
        data = DataSet(data, sheet=kward.get('sheet', 'sheet0'))
    data.save(addr, **kward)

def encode(code='cp936'):
    '''change the python environment encode
    '''
    import sys
    stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
    reload(sys)
    sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde
    sys.setdefaultencoding(code)
    return


