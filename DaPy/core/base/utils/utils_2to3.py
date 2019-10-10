from DaPy.core.base.constant import PYTHON3, PYTHON2

if PYTHON2 is True:
    from itertools import izip, imap, ifilter, izip_longest as zip_longest
    from string import split, strip
    range, xrange, map, zip, filter = range, xrange, imap, izip, ifilter
        
if PYTHON3 is True:
    from itertools import zip_longest
    xrange, split, map, filter, zip, strip = range, str.split, map, filter, zip, str.strip
    def range(x, y=None, z=1):
        if y is None:
            x, y = 0, x
        return list(xrange(x, y, z))
    
try:
    import cPickle as pickle
except ImportError:
    import pickle
