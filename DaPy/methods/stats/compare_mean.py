from collections import namedtuple
from math import sqrt

from DaPy.core import Series
from DaPy.matlib import mean, std

from .distribution import Tcdf

IndTTest = namedtuple('IndependentTTest', ['T', 'n', 'center', 'pvalue'])


def IndependentTTest(x, center=0, alpha=.05):
    n = len(x)
    mu = mean(x)
    sigma = sum(map(lambda x: (x - mu) ** 2, x)) / (n - 1)
    t = (mu - center) / sqrt(sigma / (n - 1))
    pvalue = Tcdf(t, n-1)
    return IndTTest(t, n, center, round(pvalue, 4))
