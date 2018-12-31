from .base import SeriesSet, Frame, Matrix
from .base import is_seq, is_iter, is_value, is_math
from .base.tools import str2value
from os.path import split
from warnings import warn

def create_sheet(dtype, data, titles, miss_symbol, miss_value):
    if dtype.upper() in ('COL', 'SERIESSET'):
        return SeriesSet(data, titles, miss_symbol, miss_value)

    elif dtype.upper() == 'FRAME':
        return Frame(data, titles, miss_symbol, miss_value)

    elif dtype.upper() == 'MATRIX':
        return Matrix(data, titles)

    else:
        raise RuntimeError('unrecognized symbol of data type')

def parse_addr(addr):
    if addr.startswith('http'):
        fname = addr.split(':')[1].split('.')
        if fname[0].startswith('name'):
            return None, None, fname[1], 'web'
        return None, None, fname[0], 'web'
    
    file_path, file_name = split(addr)
    if file_name.count('.') > 1:
        file_base = '.'.join(file_name.split('.')[:-1])
        file_type = file_name.split('.')[-1]
    else:
        try:
            file_base, file_type = file_name.split('.')
        except ValueError:
            raise ValueError('your address can not be parsed, a legal address '+\
                             'seems like "test.xls"')
    return file_path, file_name, file_base, file_type

def parse_sql(addr, dtype, miss_symbol, miss_value):
    try:
        import sqlite3 as sql3
    except ImportError:
        raise ImportError('DaPy uses "sqlite3" to access a database.')
    
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
            except UnicodeEncodeError:
                warn("'ascii' can not encode characters, use dp.io.encode to fix.")

def parse_sav(addr, dtype, miss_symbol, miss_value):
    try:
        import savReaderWriter
    except ImportError:
        raise ImportError('DaPy uses "savReaderWriter" to open a .sav file, '+\
                          'please try command: pip install savReaderWriter.')

    with savReaderWriter.SavReader(addr) as reader:
        titles = reader.getSavFileInfo()[2]
        data = list(reader)
        return create_sheet(dtype, data, titles, miss_symbol, miss_value)

def parse_excel(dtype, addr, first_line, title_line, miss_symbol, miss_value):
    try:
        import xlrd
    except ImportError:
        raise ImportError('DaPy uses "xlrd" to parse a excel file, '+\
                          'please try command: pip install xlrd.')

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
        except UnicodeEncodeError:
            warn('"ascii" can not encode characters, use dp.io.encode() to fix.')

def parse_html(text, dtype, miss_symbol, miss_value, sheetname):
    try:
        from bs4 import BeautifulSoup as bs
    except ImportError:
        raise ImportError('DaPy uses "bs4" to parse a .html file, '+\
                          'please try command: pip install bs4.')

    soup = bs(text, 'html.parser')
    for table in soup.findAll('table'):
        sheet = table.attrs.get('class', [sheetname])[0]
        title = table.find('thead')
        if title:
            title = [col.string for col in title.findAll(['td', 'th'])]
            
        records = []
        try:
            for record in table.find('tbody').findAll(['tr', 'div']):
                current_record = []
                for value in record.findAll(['td', 'th']):
                    if value.find(['a', 'div']) is not None:
                        current_record.append(value.find(['a', 'div']).text)
                    else:
                        current_record.append(value.text)
                records.append(map(str2value, current_record))

            try:
                yield (create_sheet(dtype, records, title, miss_symbol, miss_value), sheet)
            except UnicodeEncodeError:
                warn('"ascii" can not encode characters, use dp.io.encode() to fix.')
        except AttributeError:
            warn('Table "%s" can not be auto parsed.' % sheet)
            
def write_txt(f, data, newline, delimiter, encode, decode):
    def writeline(f, record):
        msg = delimiter.join(map(str, record)) + newline
        f.write(msg.decode(decode).encode(encode))

    if hasattr(data, 'columns'):
        writeline(f, data.columns)

    if isinstance(data, (Frame, Matrix, SeriesSet)) or \
       (is_seq(data) and is_seq(data[0])):
        for line in filter(is_seq, data):
            writeline(f, line)

    elif hasattr(data, 'items'):
        if all(map(is_seq, data.values())):
            writeline(f, data.keys())
            temp = SeriesSet(data)
            for line in temp:
                writeline(f, line)
                
        elif all(map(is_value, data.values())):
            for key, value in data.items():
                writeline(f, [key, value])
                
        else:
            raise ValueError('DaPy can save the object with same value styles only.')

    elif is_seq(data):
        for record in data:
            if is_seq(record):
                writeline(f, record)
            else:
                writeline(f, [record])
    else:
        raise ValueError('DaPy can save a sequence object, dict-like object ' +\
                     'and sheet-like object only.')

def write_xls(worksheet, data, decode, encode):
    def writeline(f, i, record):
        for j, value in enumerate(record):
            if isinstance(value, (str, unicode)):
                value = value.decode(decode).encode(encode)
            f.write(i, j, value)
            
    if hasattr(data, 'columns'):
        start = 1
        writeline(worksheet, 0, data.columns)
    else:
        start = 0

    try:
        if isinstance(data, (Frame, Matrix, SeriesSet)) or \
           (is_seq(data) and is_seq(data[0])):
            for i, row in enumerate(data, start):
                writeline(worksheet, i, row)
                    
        elif hasattr(data, 'items'):
            if all(map(is_seq, data.values())):
                writeline(0, 0, data.keys())
                temp = SeriesSet(data)
                for i, line in enumerate(temp, 1):
                    writeline(worksheet, i, line)
                    
            elif all(map(is_value, data.values())):
                for i, (key, value) in enumerate(data.items()):
                    writeline(worksheet, i, (key, value))
                    
            else:
                raise ValueError('DaPy can save the object with same value styles only.')

        elif is_seq(data):
            for i, record in enumerate(data, start):
                if is_seq(record):
                    writeline(worksheet, i, record)
                else:
                    writeline(worksheet, i, [record])

        else:
            raise ValueError('DaPy can save a sequence object, dict-like object ' +\
                             'and sheet-like object only.')
    except ValueError:
        warn('.xls format only allows 65536 lines per sheet.')

def write_html(f, data, encode, decode):
    def writeline(f, record):
        msg = '<tr><td>' + '</td><td>'.join(map(str, record)) + '</td></tr>'
        f.write(msg.decode(decode).encode(encode))

    if hasattr(data, 'columns'):
        f.write('<thead>')
        writeline(f, data.columns)
        f.write('</thead>')

    f.write('<tbody>')
    if isinstance(data, (Frame, Matrix, SeriesSet)) or \
       (is_seq(data) and is_seq(data[0])):
        for line in filter(is_seq, data):
            writeline(f, line)

    elif hasattr(data, 'items'):
        if all(map(is_seq, data.values())):
            writeline(f, data.keys())
            temp = SeriesSet(data)
            for line in temp:
                writeline(f, line)
                
        elif all(map(is_value, data.values())):
            for key, value in data.items():
                writeline(f, [key, value])
                
        else:
            raise ValueError('DaPy can save the object with same value styles only.')

    elif is_seq(data):
        for record in data:
            if is_seq(record):
                writeline(f, record)
            else:
                writeline(f, [record])
    else:
        raise ValueError('DaPy can save a sequence object, dict-like object ' +\
                     'and sheet-like object only.')
    f.write('</tbody>')

def write_db(conn, sheet, data, if_exists):
    cur = conn.cursor()
    if not isinstance(data, (Frame, SeriesSet)):
        data = SeriesSet(data)
        
    tables = cur.execute(\
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    tables = [table[0] for table in tables]
    if sheet in tables and if_exists == 'replace':
        cur.execute(u'DROP TABLE IF EXISTS %s' % unicode(sheet))
        tables.remove(sheet)
    elif sheet in tables and if_exists == 'fail':
        raise ValueError('table "%s" already exists, ' % sheet +\
                         'change keyword ``if_exists``.')
    elif sheet in tables and if_exists == 'append':
        cols = cur.execute(u'PRAGMA table_info(%s)' % unicode(sheet)).fetchall()
        if [col[1] for col in cols] != data.columns:
            raise ValueError('The columns in exist table are not match those '+\
                             'in saving table `%s`. ' % sheet)
    
    if sheet not in tables:
        cols = []
        for column, records in data.items():
            if not all(map(is_math, records)):
                cols.append('%s STRING' % column)
            elif all(map(isinstance, records, [float] * data.shape.Ln)):
                cols.append('%s INTEGER' % column)
            else:
                cols.append('%s REAL' % column)
        cur.execute(u'CREATE TABLE %s(%s)' % (unicode(sheet), ','.join(cols)))

    INSERT = 'INSERT INTO %s VALUES (%s)' % (\
                sheet, ','.join(['?'] * data.shape.Col))
    for record in data:
        cur.execute(INSERT, record)
    conn.commit()
