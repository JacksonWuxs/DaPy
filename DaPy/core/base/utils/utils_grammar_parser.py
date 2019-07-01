from DaPy.core.base.constant import PYTHON2, PYTHON3

from distutils.util import strtobool
try:
    from dateutil.parser import parse as strtodate
except ImportError:
    from datetime import datetime
    def strtodate(value, day='1900-1-1', time='0:0:0'):
        if ' ' in value:
            day, time = value.split(' ')
        elif ':' in value:
            time = value
        elif '-' in value:
            day = value
        day, time = map(int, day.split('-')), map(int, time.split(':'))
        return datetime(day[0], day[1], day[2], time[0], time[1], time[2])

def _str2date(value):
    try:
        return strtodate(value)
    except ValueError:
        return value

def _str2bool(value):
    try:
        if value == u'\u662f' or strtobool(value) == 1:
            return True
    except ValueError:
        pass
    return False

def _str2percent(value):
    return float(value.replace('%', '')) / 100.0


