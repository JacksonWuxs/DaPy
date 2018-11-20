from tools import str2value as dp_str2value, is_math
from cProfile import run
from csv import reader
import re

pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-=]?\.?[0-9]\d*$')
with open('read_csv.csv', 'r') as f:
    freader = reader(f)
    data = [record for record in f]
numbers = map(str, range(10000000))

def is_number_re(value):
    if pattern.match(value):
        return True
    return False

def is_number_if(value):
    try:
        float(value)
    except ValueError:
        return False
    return True

def test_script(func):
    for i, line in enumerate(data):
        for value in line:
            func(value)
        if i == 10000:
            break

def test_script2(func):
    for each in numbers:
        func(each)

if __name__ == '__main__':
##    run("test_script(dp_str2value)")
##    run("test_script(zyq_str2value)")
    run("test_script2(is_math)")
    run("test_script2(is_number)")
    run('test_script2(is_numbers)')
