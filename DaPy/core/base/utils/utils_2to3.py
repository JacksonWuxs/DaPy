from DaPy.core.base.constant import PYTHON3, PYTHON2

if PYTHON2 is True:
    from itertools import izip, imap, ifilter, izip_longest as zip_longest
    from string import split, strip
    range, xrange, map, zip, filter = range, xrange, imap, izip, ifilter
        
if PYTHON3 is True:
    from itertools import zip_longest
    xrange = range
    range = lambda x: list(xrange(x))
    split, map, filter, zip, strip = str.split, map, filter, zip, str.strip

try:
    import cPickle as pickle
except ImportError:
    import pickle
