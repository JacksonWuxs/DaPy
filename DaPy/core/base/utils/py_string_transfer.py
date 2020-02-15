from distutils.util import strtobool

str2int = int
str2float = float
str2pct = lambda val: float(val.replace('%', '')) / 100.0

def str2bool(val):
    try:
        if val == u'\u662f' or strtobool(val) == 1:
            return True
    except ValueError:
        pass
    return False

def str2datetime(val):
    if ' ' in value:
        day, time = value.split(' ')
    elif ':' in value:
        time = value
    elif '-' in value:
        day = value
    day, time = tuple(map(int, day.split('-'))), tuple(map(int, time.split(':')))
    return datetime(day[0], day[1], day[2], time[0], time[1], time[2])
