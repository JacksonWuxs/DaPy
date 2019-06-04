from collections import namedtuple, Counter
from DaPy.core import is_math, is_seq, SeriesSet, DataSet, Series
from DaPy.matlib import mean
from .distribution import Fcdf, Tcdf, Bcdf, Ncdf
from math import sqrt

__all__ = ['SignTest']

RunTestResult = namedtuple('RunTestResult', ['RejectInterval', 'R'])
CoxStautResult = namedtuple('CoxStautTestResult', ['H0', 'n', 'pvalue'])
SignTestResult = namedtuple('SignTestResult', ['K', 'n', 'center', 'pvalue'])
WicoxonTestResult = namedtuple('WicoxonTestResult', ['Z', 'n', 'center', 'pvalue'])

def SignTest(series, center, side='both', alpha=0.05):
    '''Sign Test is one of the most oldest method for Non-parametric Statistics

    Parameters
    ----------
    series : array-like
        a series of data you expect to inferen

    compare : float or int
        a value you expect to compare with (always the mode of series)

    side : str (default='both')

    alpha : float (default=0.5)
        the level of significant

    Return
    ------
    SeriesSet : the result of test

    References
    ----------
    Xin Wang & T.J Chu, Non-parametric Statistics (Second Edition),
    Tsinghua Publish, 2014.
    '''
    assert side in ('both', 'upper', 'lower')
    assert is_seq(series), 'Sign test expects sequence object stored data'
    assert all(map(is_math, series)), 'Sign test expects numerical data'
    assert is_math(center) is True, 'the value to compare must be a number'
    greater = [_ for _ in series if _ > center]
    smaller = [_ for _ in series if _ < center]
    n = len(greater) + len(smaller)
    if side == 'both':
        k, side = min(len(greater), len(smaller)), 2
        pvalue = min(Bcdf(k=k, n=n, p=0.5) * 2, 1)
    else:
        if side == 'upper':
            k = len(smaller)
        else:
            k = len(greater)
        pvalue = 1 - Bcdf(k=k, n=n, p=0.5)
    return SignTestResult(k, n, center, round(pvalue, 4))

def CoxStautTest(series, H0='increase'):
    assert H0 in ('increase', 'decrease', 'no-trend')
    assert is_seq(series), 'Sign test expects sequence object stored data'
    assert all(map(is_math, series)), 'Sign test expects numerical data'
    series = [right - left for left, right in zip(series, series[len(series)//2:])]
    if H0 == 'increase':
        r = SignTest(series, 0, 'lower')
    elif H0 == 'decrease':
        r =  SignTest(series, 0, 'upper')
    else:
        r =  SignTest(series, 0)
    return CoxStautResult(H0, r.n, r.pvalue)

def WicoxonTest(series, center, side='both', alpha=0.05):
    assert is_seq(series), 'Sign test expects sequence object stored data'
    assert all(map(is_math, series)), 'Sign test expects numerical data'
    assert is_math(center) is True, 'the value to compare must be a number'
    assert side == 'both', "don't supprot single side test in thie version"
    symbol = [1 if _ > center else 0 for _ in series]
    data = SeriesSet({'X': series, 'SYMBOL': symbol})

    # distance to the compare center
    data['ABS'] = data.X.apply(lambda x:abs(x - center))

    # rank records by distance
    data = data.sort(('ABS', 'ASC'))
    counts = Counter(data.ABS)
    Rank, rank_ = [], 1
    while rank_ <= data.shape.Ln:
        num = counts[data.ABS[rank_ - 1]]
        if num > 1:
            Rank.extend([(2 * rank_ + (num - 1)) / 2.0] * num)
        else:
            Rank.append(rank_)
        rank_ += num
    data['RANK'] = Rank
    # calculate the sum of ranking
    W_pos = sum(data.select(lambda row: row['SYMBOL'] == 1).RANK)
    W_neg = sum(data.select(lambda row: row['SYMBOL'] == 0).RANK)
    W = min(W_pos, W_neg)

    # calculate the Statistic
    n, C = float(data.shape.Ln), 0.5
    if W < n * (n + 1) / 4:
        C = -0.5
    up = W - n * (n + 1) / 4 + C
    down = sqrt(n * (n + 1) * (2 * n + 1) / 24)
    Z = up / down
    pvalue = Ncdf(Z, 0, 1)
    return WicoxonTestResult(round(Z, 4), n, center, round(pvalue, 4))

def RunTest(series, side='both', alpha=0.05):
    assert all(map(lambda x: x in (0, 1), series)), 'data must be 1 or 0 in Run Test'
    assert side in ('both', 'upper', 'lower')
    if alpha == 'both':
        alpha = side / 2.0
    n, n1 = len(series), sum(series)
    n0 = n - n1
    R, last = 1, series[0]
    for value in series[1:]:
        if value != last:
            last = value
            R += 1
            
    r1 = round((2.0 * n1 * n0) / n * (1 + 1.96 / sqrt(n)), 2)
    r0 = round((2.0 * n1 * n0) / n * (1 - 1.96 / sqrt(n)), 2)
    return RunTestResult([r0, r1], R)
