from DaPy.matlib import mean, std, sqrt
from DaPy.core import Series
from scipy.stats import t as t_dist

def ttest(x, mu=0, sigma=1):
      x = Series(x)
      sample_mu = mean(x)
      sample_sigma = std(x)
      n = len(x)
      t = (sample_mu - mu) / (sample_sigma / sqrt(n))
      pvalue = t_dist.isf(alpha / 2.0, df=n-1)
      return t, p
