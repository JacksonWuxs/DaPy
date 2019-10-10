from collections import namedtuple
from DaPy.core import is_math, is_seq, is_str
from DaPy.core import SeriesSet
from DaPy.operation import get_ranks
from .distribution import Fcdf, Ncdf
from math import sqrt

__all__ = ['ANOVA']

ANOVA_result = namedtuple('one_way_ANOVA_Result', ['F', 'pvalue'])
MoodTestResult = namedtuple('MoodTestResult', ['Statistic', 'pvalue', 'Decision'])

def ANOVA(data, cluster):
    if not isinstance(data, SeriesSet):
        data = SeriesSet(data)
    assert data.shape.Col > 1, 'ANOVA() expects more than 1 comparing group.'
    assert data.shape.Ln > 2, 'at least 2 records in the data'
    assert is_str(cluster), '`cluster` must be a string object to represent the categorical variable in the data'
    assert is_str(control) or control == None, '`control` must be False or a string object'
    assert report in (True, False)
    assert cluster in data.columns
    cluster = [cluster]

    value_column = tuple(set(data.columns) - set(cluster))[0]
    SST = data[value_column].std()
    
    total_mean = data[value_column].mean()
    SSA, SSE, r, n = 0.0, 0.0, 0.0, data.shape.Ln
    for label, subset in data.iter_groupby(cluster):
        seq = subset[value_column]
        r += 1
        SSA += len(seq) * (seq.mean() - total_mean) ** 2
        SSE += len(seq) * seq.std() ** 2
    MSA = SSA / (r - 1.0)
    MSE = SSE / (n - r) if SSE != 0 else 0.00001
    F = MSA / MSE 
    return ANOVA_result(F, 1 - Fcdf(F, r-1, n-r))

def MoodTest(X, Y, side='equal', alpha=0.05):
    assert side in ('equal', 'upper', 'lower')
    assert is_seq(X) and is_seq(Y), 'Mood test expects sequence object stored data'
    assert all(map(is_math, X)) and all(map(is_math, Y)), 'Mood test expects numerical data'

    # clean data
    m, n = len(X), len(Y)
    combine_col = list(X) + list(Y)
    rank_pair_data = dict(zip(combine_col, get_ranks(combine_col)))
    
    # statistic something
    hypothesis_mean = (m + n + 1) / 2.0
    M = sum([(rank_pair_data[x] - hypothesis_mean) ** 2 for x in X])
    EM = m * (m + n + 1) * (m + n - 1) / 12.0
    VM = m * n * (m + n + 1) * (m + n + 2) * (m + n - 2) / 180.0

    # calculate the statistic value
    Z = (M - EM) / sqrt(VM)
    if m + n <= 30:
        Z += 1 / (2.0 * sqrt(VM))

    pvalue = Ncdf(Z, 0, 1)
    pvalue = min(pvalue, 1 - pvalue)
    if side == 'equal':
        pvalue *= 2
    if side == 'smaller' and Z < 0 and pvalue <= alpha:
        return MoodTestResult(Z, pvalue, 'H1: var(X) < var(Y)')
    elif side == 'larger' and Z > 0 and pvalue <= alpha:
        return MoodTestResult(Z, pvalue, 'H1: var(X) > var(Y)')
    elif side == 'equal' and pvalue <= alpha:
        return MoodTestResult(Z, pvalue, 'H1: var(X) != var(Y)')
    else:
        return MoodTestResult(Z, pvalue, 'H0: var(X) == var(Y)')

    
    
