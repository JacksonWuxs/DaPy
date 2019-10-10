from os.path import split
from warnings import warn
from datetime import datetime
from itertools import repeat

from .base import Frame, Matrix, SeriesSet, Series, is_iter, is_math, is_seq, is_value
from .base import auto_str2value, fast_str2value, STR_TYPE, zip
from .sqlparser import parse_insert_statement, parse_create_statement


def create_sheet(dtype, data, titles, nan):
    if dtype.upper() in ('COL', 'SERIESSET'):
        return SeriesSet(data, titles, nan)

    elif dtype.upper() == 'FRAME':
        return Frame(data, titles, nan)

    elif dtype.upper() == 'MATRIX':
        return Matrix(data, titles)

    else:
        raise RuntimeError('unrecognized symbol of data type')

def parse_addr(addr):
    if addr.lower().startswith('http'):
        fname = addr.split(':')[1].split('.')
        if fname[0].startswith('name'):
            return None, None, fname[1], 'web'
        return None, None, fname[0], 'web'

    if addr.lower().startswith('mysql'):
        assert addr.count(':') == 3 and addr.count('@') == 1
        spliter = addr[6]
        addr, file_name = addr[8:].split(spliter, 1)
        file_name = file_name.split(spliter)
        file_type = 'mysql'
        file_path, file_base = addr.split('@')
        return file_path, file_name, file_base, file_type

    maybe_error = 'you may connect a mysql database, try to write `addr` like: "mysql://[username]:[password]@[server_ip]:[server_port]/[database_name]"'
    if addr.count('@') == 1 and addr.count(':') >= 2 and addr.count('.') == 0:
        maybe_error
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

def parse_mysql_server(cur, fname):
    if len(fname) == 1:
            cur.execute('SHOW TABLES;')
            for table in cur.fetchall():
                fname.append(table[0])
        
    for table in fname[1:]:
        cur.execute('SELECT column_name FROM information_schema.columns WHERE table_name="%s";' % table)
        columns = [_[0] for _ in cur.fetchall()]
        cur.execute('SELECT * FROM %s;' % table)
        yield SeriesSet(cur.fetchall(), columns), '%s_%s' % (fname[0], table)

def parse_db(cur, dtype, nan):
    cur.execute('SELECT name FROM sqlite_master WHERE type="table"')
    table_list = cur.fetchall()
    for table in table_list:
        table = str(table[0])
        cur.execute('PRAGMA table_info(%s)' % table)
        titles = [title[1] for title in cur.fetchall()]
        cur.execute('SELECT * FROM %s' % table)

        try:
            yield create_sheet(dtype,  cur.fetchall(), titles, nan),  table
        except UnicodeEncodeError:
            warn("'ascii' can not encode characters, use dp.io.encode to fix.")

def parse_sav(doc, dtype, nan):
    titles = doc.getSavFileInfo()[2]
    data = list(readocder)
    return create_sheet(dtype, data, titles, nan)

def parse_excel(dtype, addr, fline, tline, nan):
    try:
        import xlrd
    except ImportError:
        raise ImportError('DaPy uses "xlrd" to parse a excel file, '+\
                          'please try command: pip install xlrd.')

    book = xlrd.open_workbook(addr)
    for sheet, name in zip(book.sheets(), book.sheet_names()):
        try: 
            series_set = SeriesSet(None, None, nan)
            for cols in range(sheet.ncols):
                column = Series(sheet.col_values(cols))[fline:]
                title = column.get(tline)
                if tline >= 0:
                    column.pop(tline)
                series_set.append_col(series=column,
                                      variable_name=title)
            yield series_set, name
        except UnicodeEncodeError:
            warn('can not decode characters, use `DaPy.io.encode()` to fix.')

def parse_html(text, dtype, miss_symbol, nan, sheetname):
    try:
        from bs4 import BeautifulSoup as bs
    except ImportError:
        raise ImportError('DaPy uses "bs4" to parse a .html file, '+\
                          'please try command: pip install bs4.')
    if not is_iter(miss_symbol):
        miss_symbol = [miss_symbol]
        
    soup = bs(text.replace('\n', ''), 'html.parser')

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
                    while value.find(['a', 'div']) is not None:
                        value = value.find(['a', 'div'])
                    if value.text in miss_symbol:
                        current_record.append(nan)
                    else:
                        current_record.append(auto_str2value(value.text))
                if current_record != []:
                    records.append(current_record)

            try:
                yield (create_sheet(dtype, records, title, nan), sheet)
            except UnicodeEncodeError:
                warn('"ascii" can not encode characters, use dp.io.encode() to fix.')
        except RuntimeError:
            warn('Table "%s" can not be auto parsed.' % sheet)

def parse_sql(doc, nan):
    command = ''
    for row in doc:
        if row[:6].lower() in ('create', 'insert'):
            command = row.replace('\n', '')
            
        elif command:
            command += row.replace('\n', '')

        if command.endswith(';'):
            if command.lower().startswith('create table'):
                table_name, columns, dtypes = parse_create_statement(command)

            if command.lower().startswith('insert into'):
                sheet = parse_insert_statement(command, dtypes, nan)
                sheet.columns = columns
                yield sheet, table_name
            command = ''


type2str = {int: 'int', float:'float', str:'varchar', datetime:'datetime'}
def write_sql(doc, sheet, sheet_name):
    doc.write('DROP TABLE IF EXISTS `%s`;\n' % sheet_name)
    doc.write('SET character_set_client = utf8mb4;\n')
    doc.write('CREATE TABLE `%s` (\n' % sheet_name)
    for key, column in sheet.items():
        for val in column:
            if sheet._isnan(val) is False:
                doc.write('`%s` %s,\n' % (key, type2str[type(val)]))
                break
    doc.write('  PRIMARY KEY (`%s`)\n' % sheet.columns[0])
    doc.write(') ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;\n')
    
    doc.write('LOCK TABLES `%s` WRITE;\n' % sheet_name)
    doc.write('INSERT INTO `%s` VALUES ' % sheet_name)
    string_nan = str(sheet.nan)
    for i, row in enumerate(sheet.iter_rows()):
        if i != 0:
            doc.write(',')
        doc.write(str(row).replace(string_nan, 'NULL'))
    else:
        doc.write(';\n')
    doc.write('UNLOCK TABLES;\n\n')

def write_txt(f, data, newline, delimiter):
    def writeline(f, record):
        f.write(delimiter.join(map(str, record)) + newline)

    if hasattr(data, 'columns'):
        writeline(f, data.columns)

    if isinstance(data, (Frame, SeriesSet)):
        for line in data.iter_rows():
            writeline(f, line)

    elif hasattr(data, 'items'):
        if all(map(is_iter, data.values())):
            writeline(f, data.keys())
            temp = SeriesSet(data)
            for line in temp:
                writeline(f, line)
                
        elif all(map(is_value, data.values())):
            for key, value in data.items():
                writeline(f, [key, value])

        else:
            raise ValueError('DaPy can save the object with same value styles only.')

    elif is_iter(data):
        for record in data:
            if is_iter(record):
                writeline(f, record)
            else:
                writeline(f, [record])
    else:
        raise ValueError('DaPy can save a sequence object, dict-like object ' +\
                     'and sheet-like object only.')

def write_xls(worksheet, data):
    def writeline(f, i, record):
        for j, value in enumerate(record):
            f.write(i, j, value)
            
    if hasattr(data, 'columns'):
        start = 1
        writeline(worksheet, 0, data.columns)
    else:
        start = 0

    try:
        if isinstance(data, (Frame, Matrix, SeriesSet)):
            for i, row in enumerate(data, start):
                writeline(worksheet, i, row)
                    
        elif hasattr(data, 'items'):
            if all(map(is_iter, data.values())):
                writeline(worksheet, 0, data.keys())
                temp = SeriesSet(data)
                for i, line in enumerate(temp, 1):
                    writeline(worksheet, i, line)
                    
            elif all(map(is_value, data.values())):
                for i, (key, value) in enumerate(data.items()):
                    writeline(worksheet, i, (key, value))
                    
            else:
                raise ValueError('DaPy can save the object with same value styles only.')

        elif is_iter(data) and is_iter(data[0]):
            for i, record in enumerate(data, start):
                if is_iter(record):
                    writeline(worksheet, i, record)
                else:
                    writeline(worksheet, i, [record])

        else:
            raise ValueError('DaPy can save a sequence object, dict-like object ' +\
                             'and sheet-like object only.')
    except ValueError:
        warn('.xls format only allows 65536 lines per sheet.')

def write_html(f, data):
    def writeline(f, record):
        f.write('<tr><td>' + '</td><td>'.join(map(str, record)) + '</td></tr>')

    if hasattr(data, 'columns'):
        f.write('<thead>')
        writeline(f, data.columns)
        f.write('</thead>')

    f.write('<tbody>')
    if isinstance(data, (Frame, Matrix, SeriesSet)) or \
       (is_iter(data) and is_iter(data[0])):
        for line in filter(is_iter, data):
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

    elif is_iter(data):
        for record in data:
            if is_iter(record):
                writeline(f, record)
            else:
                writeline(f, [record])
    else:
        raise ValueError('DaPy can save a sequence object, dict-like object ' +\
                     'and sheet-like object only.')
    f.write('</tbody>')

def write_db(cur, sheet, data, if_exists, mode):
    
    if not isinstance(data, (Frame, SeriesSet)):
        data = SeriesSet(data)
    
    SELECT_STATEMENT = {
        'mysql': [u"SHOW TABLES;", u'SELECT column_name FROM information_schema.columns WHERE table_name="%s"', u'%s'], 
        'sqlite3': [u"SELECT name FROM sqlite_master WHERE type='table'", u'PRAGMA table_info(%s)', u'?']
    }
    cur.execute(SELECT_STATEMENT[mode][0])
    tables = cur.fetchall()
    tables = [table[0] for table in tables]
    if sheet in tables and if_exists == 'replace':
        cur.execute(u'DROP TABLE IF EXISTS %s' % sheet)
        tables.remove(sheet)
    elif sheet in tables and if_exists == 'fail':
        raise ValueError('table "%s" already exists, ' % sheet +\
                         'change keyword ``if_exists``.')
    elif sheet in tables and if_exists == 'append':
        cols = cur.execute(SELECT_STATEMENT[mode][1] % sheet)
        cols = cur.fetchall()
        if [col[1] for col in cols] != data.columns:
            raise ValueError('The columns in exist table are not match those '+\
                             'in saving table `%s`. ' % sheet)
    
    if sheet not in tables:
        cols = []
        for column, records in data.items():
            column = column.replace(' ', '').replace('-', '').replace(':', '')
            if all(map(isinstance, records, repeat(float, data.shape.Ln))):
                cols.append('`%s` float' % column)
            elif all(map(is_math, records)):
                cols.append('`%s` int' % column)
            elif all(map(isinstance, records, repeat(datetime, data.shape.Ln))):
                cols.append('`%s` date' % column)
            else:
                cols.append('`%s` varchar(30)' % column)
        cur.execute(u'CREATE TABLE `%s` (%s);' % (sheet, ','.join(cols)))

    INSERT = 'INSERT INTO %s VALUES (%s);' % (sheet, ','.join(repeat(SELECT_STATEMENT[mode][2], data.shape.Col)))
    for record in data.iter_rows():
        cur.execute(INSERT, record)
