from collections import namedtuple
from DaPy.core import is_math, SeriesSet, DataSet, Series

try:
    from scipy.stats import f
except ImportError:
    def unsupportedTest(*args, **kwrds):
        return '-'
    Fcdf = UnsupportTest
    warn('DaPy uses scipy to compute p-value, try: pip install scipy.')
else:
    Fcdf = f.cdf

__all__ = ['ANOVA']

def ANOVA(*classes):
    assert len(classes) > 1, 'ANOVA() expects more than 1 comparing class.'

    new_classes = list()
    for sequence in classes:
        sequence = Series(filter(is_math, sequence))
        assert len(sequence) > 1, 'ANOVA() expects more than 1 samples in each class.'
        new_classes.append(sequence)

    n = sum(map(len, new_classes))
    ni = list(map(len, new_classes))
    r = len(new_classes)
    Ti = map(sum, new_classes)
    G = sum([sum(map(lambda x: x ** 2, sequence)) for sequence in new_classes])
    
    totals = sum([Ti[i]**2/float(ni[i]) for i in range(r)])
    Sa = totals - sum(Ti)**2/float(sum(ni))
    Se = G - totals
    MSa = Sa / float(r - 1)
    MSe = Se / float(sum(ni) - r)
    F = MSa / MSe
    return  DataSet(SeriesSet([
        ['Regression', round(Sa, 4), r - 1, round(MSa, 4), round(F, 4), round(1 - Fcdf(F, r-1, n-r), 4)],
        ['Residual', round(Se, 4), n - r, round(MSe), '', ''],
        ['Total', round(Sa + Se, 4), n - 1, '', '', '']],
        ['Model', 'Sum Squr', 'DF', 'Mean Squr', 'F', 'p-value']), 'ANOVA')
        
