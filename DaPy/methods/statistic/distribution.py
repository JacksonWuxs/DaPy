from DaPy.core.base import LogWarn

def unsupportedTest(*args, **kwrds):
    return '-'

try:
    from scipy.stats import f, t, binom, norm
    Fcdf, Tcdf, Bcdf, Ncdf = f.cdf, t.cdf, binom.cdf, norm.cdf
except ImportError:
    Fcdf, Tcdf, Bcdf, Ncdf = unsupportedTest, unsupportedTest, unsupportedTest, unsupportedTest
    LogWarn('DaPy uses `scipy` to compute p-value, try: pip install scipy.')
