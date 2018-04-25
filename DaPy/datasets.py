from structure import DataSet
from os.path import dirname, join

__all__ = ['wine', 'iris', 'example']
module_path = dirname(__file__)

def wine():
    file_path = join(module_path, 'data/wine')
    
    data = DataSet(join(file_path, 'data.csv'))
    data.readcol()
    with open(join(file_path, 'info.txt')) as f:
        DESC = f.read()

    return data, DESC

def iris():
    file_path = join(module_path, 'data/iris')
    
    data = DataSet(join(file_path, 'data.csv'))
    data.readcol()
    with open(join(file_path, 'info.txt')) as f:
        DESC = f.read()

    return data, DESC

def example():
    file_path = join(module_path, 'data/sample.csv')
    data = DataSet(file_path)
    data.readcol()
    return data
