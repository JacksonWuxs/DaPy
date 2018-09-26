from base import SeriesSet, Frame, Matrix
from os.path import split

def create_sheet(dtype, data, titles, miss_symbol, miss_value):
    if dtype.upper() == 'COL' or dtype.upper() == 'SERIESSET':
        return SeriesSet(data, titles, miss_symbol, miss_value)

    elif dtype.upper() == 'FRAME':
        return Frame(data, titles, miss_symbol, miss_value)

    elif dtype.upper() == 'MATRIX':
        return Matrix(data, titles)

    else:
        raise RuntimeError('unrecognized symbol of data type')

def parse_addr(addr):
    file_path, file_name = split(addr)
    if file_name.count('.') > 1:
        file_base = '.'.join(file_name.split('.')[:-1])
        file_type = file_name.split('.')[-1]
    else:
        file_base, file_type = file_name.split('.')
    return file_path, file_name, file_base, file_type

def parse_sql(addr, fname, dtype, miss_symbol, miss_value):
    try:
        import sqlite3 as sql3
    except ImportError:
        raise ImportError('DaPy uses `sqlite3` to access a database.')
    
    with sql3.connect(addr) as conn:
        cur = conn.cursor()
        cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
        table_list = cur.fetchall()
        for table in table_list:
            table = str(table[0])
            cur.execute('SELECT * FROM %s' % table)
            data = cur.fetchall()

            cur.execute('PRAGMA table_info(%s)' % table)
            titles = [title[1] for title in cur.fetchall()]

            try:
                yield (create_sheet(dtype, data, titles, miss_symbol, miss_value),
                       table)
            except ValueError:
                warn('%s is empty.' % titles)

def parse_sav(addr, fname, dtype, miss_symbol, miss_value, fbase):
    try:
        import savReaderWriter
    except ImportError:
        raise ImportError('DaPy uses `savReaderWriter` to open a .sav file')

    with savReaderWriter.SavReader(addr) as reader:
        titles = reader.getSavFileInfo()[2]
        data = list(reader)
        return (create_sheet(dtype, data, titles, miss_symbol, miss_value), fbase)

def parse_excel(dtype, addr, first_line, title_line, miss_symbol, miss_value):
    try:
        import xlrd
    except ImportError:
        raise ImportError('DaPy uses `xlrd` to parse a %s file' % ftype)

    book = xlrd.open_workbook(addr)
    for sheet in book.sheets():
        data = [0] * (sheet.nrows - first_line)
        for index, i in enumerate(range(first_line, sheet.nrows)):
            data[index] = [cell.value for cell in sheet.row(i)]

        if title_line >= 0:
            try:
                titles = [cell.value for cell in sheet.row(title_line)]
            except IndexError:
                titles = None
        else:
            titles = None
            
        try:
            yield (create_sheet(dtype, data, titles, miss_symbol, miss_value), sheet.name)
        except ValueError:
            warn('%s is empty.'%sheet)
            
