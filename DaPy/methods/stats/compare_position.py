from collections import namedtuple, Counter
from DaPy.core import is_math, is_seq, SeriesSet, DataSet, Series
from DaPy.matlib import mean, median, C
from DaPy.operation import get_ranks
from .distribution import Fcdf, Tcdf, Bcdf, Ncdf
from math import sqrt

BrownMoodResult = namedtuple('BrownMoodTestResult', ['Statistic', 'pvalue', 'Decision'])
WilcoxonMannWhitneyResult = namedtuple('WilcoxonMannWhitneyTestResult', ['Statistic', 'pvalue', 'Decision'])
RunTestResult = namedtuple('RunTestResult', ['RejectInterval', 'R'])
CoxStautResult = namedtuple('CoxStautTestResult', ['H0', 'n', 'pvalue'])
SignTestResult = namedtuple('SignTestResult', ['K', 'n', 'center', 'pvalue'])
WicoxonTestResult = namedtuple('WicoxonTestResult', ['Z', 'n', 'center', 'pvalue'])
IndTTest = namedtuple('IndependentTTest', ['T', 'n', 'center', 'pvalue'])

def IndependentTTest(x, center=0, alpha=.05):
    n, mu = len(x), mean(x)
    sigma = sum(map(lambda x: (x - mu) ** 2, x)) / (n - 1)
    t = (mu - center) / sqrt(sigma / (n - 1))
    pvalue = Tcdf(t, n-1)
    return IndTTest(t, n, center, round(pvalue, 4))

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
    
    series = [r - l for l, r in zip(series, series[len(series)//2:])]

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
    data['ABS'] = data.X.apply(lambda x: abs(x - center))

    # rank records by distance
    data['RANK'] = get_ranks(data.ABS)
    
    # calculate the sum of ranking
    W_pos = sum(data.select(lambda row: row['SYMBOL'] == 1).RANK)
    W_neg = sum(data.select(lambda row: row['SYMBOL'] == 0).RANK)
    W = min(W_pos, W_neg)

    # calculate the Statistic
    n, C = float(data.shape.Ln), 0.5
    if W < n * (n + 1) / 4:
        C = -0.5
    Z = (W - n * (n + 1) / 4 + C) / (sqrt(n * (n + 1) * (2 * n + 1) / 24))
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

def BrownMoodTest(X, Y, side='equal', alpha=0.05):
    '''Brown-Mood Test compares the medians of two populations

    Parameters
    ----------
    X : array-like
        a series of data you expect to inferen

    Y : array-like
        a series of data you expect to inferen

    side : str (default='both')
        `both` -> H1: Xmed != Ymed
        `upper` -> H1: Xm > Ym
        `lower` -> H1: Xm < Ym

    alpha : float (default=0.5)
        the level of significant

    Return
    ------
    TestResult : namedtuple(Statistic, p-value, Decision)

    Example
    -------
    >>> from DaPy.methods.stats import median_test
    >>> X = [10, 8, 12, 16, 5, 9, 7, 11, 6]
    >>> Y = [12, 15, 20, 18, 13, 14, 9, 16]
    >>> median_test.BrownMood(X, Y, side='lower')
    BrownMoodTestResult(Statistic=-2.0748, pvalue=0.0190, Decision='H1: Mx < My')

    References
    ----------
    Xin Wang & T.J Chu, Non-parametric Statistics (Second Edition),
    Tsinghua Publish, 2014.
    '''
    assert side in ('equal', 'upper', 'lower')
    assert is_seq(X) and is_seq(Y), 'Brown-Mood test expects sequence object stored data'
    assert all(map(is_math, X)) and all(map(is_math, Y)), 'Brown-Mood test expects numerical data'

    Mxy = median(list(X) + list(Y))
    large_X = len([i for i in X if i > Mxy])
    large_Y = len([i for i in Y if i > Mxy])
    less_X = len([i for i in X if i < Mxy])
    less_Y = len([i for i in Y if i < Mxy])
    m, n = large_X + less_X, large_Y + less_Y
    t = large_X + large_Y
    k = min(m, t)

    upper = large_X - m * t / (m + n)
    lower = sqrt(float(m * n * t * (m + n - t)) / (m + n) ** 3)
    Z = upper / lower
    pvalue = Ncdf(Z, 0, 1)
    pvalue = min(pvalue, 1 - pvalue)
    if side == 'equal':
        pvalue *= 2
    if side == 'lower' and Z < 0 and pvalue <= alpha:
        return BrownMoodResult(Z, pvalue, 'H1: Mx < My')
    elif side == 'upper' and Z > 0 and pvalue <= alpha:
        return BrownMoodResult(Z, pvalue, 'H1: Mx > My')
    elif side == 'equal' and pvalue <= alpha:
        return BrownMoodResult(Z, pvalue, 'H1: Mx != My')
    else:
        return BrownMoodResult(Z, pvalue, 'H0: Mx == My')


def WilcoxonMannWhitneyTest(X, Y, side='equal', alpha=0.05):
    '''Wilcoxon-Mann-Whitney Test compares the ranks of two populations

    Parameters
    ----------
    X : array-like
        a series of data you expect to inferen

    Y : array-like
        a series of data you expect to inferen

    side : str (default='both')
        `both` -> H1: Xmiu != Ymiu
        `larger` -> H1: Xmiu > Ymiu
        `smaller` -> H1: Xmiu < Ymiu

    alpha : float (default=0.5)
        the level of significant

    Return
    ------
    TestResult : namedtuple(Statistic, p-value, Decision)

    Example
    -------
    >>> from DaPy.methods.stats import median_test
    >>> X = [10, 8, 12, 16, 5, 9, 7, 11, 6]
    >>> Y = [12, 15, 20, 18, 13, 14, 9, 16]
    >>> median_test.BrownMood(X, Y, side='lower')
    BrownMoodTestResult(Statistic=-2.0748, pvalue=0.0190, Decision='H1: Mx < My')

    References
    ----------
    Xin Wang & T.J Chu, Non-parametric Statistics (Second Edition),
    Tsinghua Publish, 2014.
    '''
    assert side in ('equal', 'larger', 'smaller')
    assert is_seq(X) and is_seq(Y), 'W-M-W test expects sequence object stored data'
    assert all(map(is_math, X)) and all(map(is_math, Y)), 'W-M-W  test expects numerical data'

    # clean data
    combine_col = list(X) + list(Y)
    node_col = [i for i in Counter(combine_col).values() if i != 1]
    rank_pair_data = dict(zip(combine_col, get_ranks(combine_col)))
    rank_Y = [rank_pair_data[y] for y in Y]
    rank_X = [rank_pair_data[x] for x in X]

    # choose which hypothesis
    n, m = len(Y), len(X)
    Wx, Wy = sum(rank_X), sum(rank_Y)
    if side == 'equal':
        Wy = min(Wx, Wy)

    # do some statistic
    Wxy = Wy - n * (n + 1) / 2.0
    mn, m_n_1 = float(m * n), m + n + 1.0
    upper = Wxy - mn / 2.0
    down_left = m_n_1 / 12.0
    down_right = (sum([i ** 3 for i in node_col]) - sum(node_col)) / (12.0 * (m + n) * m_n_1)
    Z = upper / sqrt(mn * (down_left - down_right))
    pvalue = Ncdf(Z, 0, 1)
    pvalue = min(pvalue, 1 - pvalue)
    if side == 'equal':
        pvalue *= 2
    if side == 'smaller' and Z > 0 and pvalue <= alpha:
        return WilcoxonMannWhitneyResult(Z, pvalue, 'H1: Mx < My')
    elif side == 'larger' and Z < 0 and pvalue <= alpha:
        return WilcoxonMannWhitneyResult(Z, pvalue, 'H1: Mx > My')
    elif side == 'equal' and pvalue <= alpha:
        return WilcoxonMannWhitneyResult(Z, pvalue, 'H1: Mx != My')
    else:
        return WilcoxonMannWhitneyResult(Z, pvalue, 'H0: Mx == My')
