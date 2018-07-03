from DaPy import read
from os.path import dirname, join

__all__ = ['wine', 'iris', 'example']
module_path = dirname(__file__)

def wine():
    file_path = join(module_path, 'wine')
    data = read(join(file_path, 'data.csv'))
    with open(join(file_path, 'info.txt')) as f:
        DESC = f.read()
    return data, DESC

def iris():
    file_path = join(module_path, 'iris')
    data = read(join(file_path, 'data.csv'))
    with open(join(file_path, 'info.txt')) as f:
        DESC = f.read()
    return data, DESC

def example():
    file_path = join(module_path, 'sample.csv')
    data = read(file_path)
    return data
