from re import compile as re_compile
from re import findall
from datetime import datetime
from .base import SeriesSet, Series
from .base.utils import fast_str2value, auto_str2value
from .base.Sheet import subset_quickly_append_col

legal_types = {'integer': int, 'int': int, 'smallint': int, 'tinyint': int,
               'decimal': float, 'numeric': float,
               'char': str, 'varchar': str, 'date': datetime}

CREATE_PATTERN = re_compile(u'^create table [`]{0,1}([_A-Za-z0-9\u4e00-\u9fa5]{1,})[`]{0,1}[ ]{0,1}\(([\s\S]*)\)[ |;]', flags=2)
COLUMN_PATTERN = re_compile(u'[`]{0,1}([_A-Za-z0-9\u4e00-\u9fa5]{1,})[`]{0,1} (%s)' % '|'.join(legal_types.keys()))
def parse_create_statement(string):
    patterns = CREATE_PATTERN.findall(string.strip())
    assert len(patterns) >= 1, '`%s` is not a SQL statement' % string
    assert len(patterns) == 1, 'please set the statement one by one'
    table_name, contents = patterns[0]
    names, types = [], []
    for var, dtype in COLUMN_PATTERN.findall(contents):
        names.append(var)
        types.append(legal_types[dtype])
    return table_name, names, types

INSERT_PATTERN = re_compile(u"insert into [`]{0,1}([_A-Za-z0-9\u4e00-\u9fa5]{1,})[`]{0,1}([ ]{0,1}\([\s\S]*\))? values[ ]{0,1}(\([\s\S]*\))", flags=2)
STR_CLEAN_PATTERN = re_compile(u'''(^["|""|'|'']|["|""|'|'']$)''')
# RECORD_PATTERN = re_compile(u'(\()([\s\S]*?)(\))(,|;)')
RECORD_PATTERN = re_compile(u',(?![^\(]*\))')
SPLIT_PATTERN = re_compile(u',(?=(?:[^"]*"[^"]*")*[^"]*$)')
def parse_insert_statement(string, dtypes=None, nan=None):
    patterns = INSERT_PATTERN.findall(string.strip())
    assert len(patterns) >= 1, '`%s` is not a SQL statement' % string
    assert len(patterns) == 1, 'please set the statements one by one'
    del string

    table, columns, records = patterns[0][0], patterns[0][1], patterns[0][2]
    records = [tuple(STR_CLEAN_PATTERN.sub('', _.strip()) for _ in SPLIT_PATTERN.split(row[1:-1])) for row in RECORD_PATTERN.split(records)]

    if columns:
        columns = columns.strip()[1:-1].split(',')
        len_col = len(columns)
        for record in records:
            assert len(record) == len_col, "row values (%s) doesn't match columns (%s)" % (record, columns)
    else:
        columns = [None] * len(max(records, key=len))

    if dtypes is None:
        dtypes = []
        for row in records:
            if 'NULL' not in row:
                for i, val in enumerate(row):
                    val = auto_str2value(val, None)
                    str_type = str(val.__class__).split()[1][1:-2].split('.')[0]
                    dtypes.append(fast_str2value[str_type])
                break
    assert len(dtypes) == len(columns), 'lenth of data types is not match column size: %s != %s' % (dtypes, columns)

    data = tuple(Series() for _ in range(len(columns)))
    miss = [0] * len(columns)
    for row in records:
        for i, (seq, tran, val) in enumerate(zip(data, dtypes, row)):
            if val == "NULL":
                seq.append(nan)
                miss[i] += 1
            else:
                seq.append(tran(val))

    sheet = SeriesSet(nan=nan)
    for col, seq, miss in zip(columns, data, miss):
        subset_quickly_append_col(sheet, col, seq, miss)
    return sheet
    
    

if __name__ == '__main__':
    print(parse_create_statement("CREATE TABLE Persons(Id_P int,LastName varchar(255),FirstName varchar(255),Address varchar(255),City varchar(255))"))
    print(parse_insert_statement("INSERT INTO Persons VALUES ('Gates', 'Bill', 'Xuanwumen 10', 'Beijing'), ('System', 'CPU', 'Memory', 35);").show())
    print(parse_insert_statement("INSERT INTO Persons (age, name) VALUES (10, 'Jack')").show())
