from collections import namedtuple
from math import sqrt

from DaPy.core import DataSet, Frame
from DaPy.core import Matrix as mat
from DaPy.core import is_math, is_seq
from DaPy.matlib import _abs as abs
from DaPy.matlib import _sum as sum
from DaPy.matlib import corr, log, mean
from DaPy.methods.activation import UnsupportTest
from DaPy.methods.tools import _engine2str, _str2engine

__all__ = ['LinearRegression']

try:
    from scipy.stats import f, t  
except ImportError:
    Fcdf, Tcdf = UnsupportTest, UnsupportTest
    warn('DaPy uses scipy to compute p-value, try: pip install scipy.')
else:
    Fcdf, Tcdf = f.cdf, t.cdf

class LinearRegression:
    '''Linear Regression Model

    This class will help you develop a ``Linear Regression Model`` calculated
    by Least Squares rule, which is one of the most common solution in prediction.

    Return
    ------
    LR_Model : use LR_Model to do prediction.

    Examples
    --------
    >>> from DaPy.methods import LinearRegression
    >>> x = [1,2,3,4,5,5,6,6,7,8,9]
    >>> y = [13, 14, 14, 15, 15, 16,16, 16, 17,17, 18]
    >>> lr = LinearRegression()
    >>> lr.fit(x, y)
    >>> lr.predict(x)
    matrix([[13.09701493, 13.69552239, 14.29402985, 14.89253731, 15.49104478,
             15.49104478, 16.08955224, 16.08955224, 16.6880597 , 17.28656716,
             17.88507463]])
    >>> lr.report()
    sheet:Model Summary
    ===================
     Constant |  Beta  | Sigma^2 | T-value
    ----------+--------+---------+---------
     12.4985  | 0.5985 |  0.101  | 14.6981 

    sheet:F-test
    ============
        Args    | Sum Squr | DF | Mean Squr | F-value 
    ------------+----------+----+-----------+----------
     Regression | 21.8183  | 1  |  21.8183  | 216.0336 
      Residual  |  0.909   | 9  |    0.0    |          
       Total    | 22.7273  | 10 |           |          
    >>> lr([3])
    matrix([[14.29402985]])
    >>>

    Reference
    ---------
    Xu X, Wang R. Probability Theory & Mathematical Statistics.
        Shanghai: Shanghai Jiaotong University Press. 2013.
    He X, Liu W. Applied Regression Analysis.
        Beijing: China People University Press. 2015.
    
    '''
    def __init__(self, engine='Numpy', beta=None, weight=None):
        self._engine = _str2engine(engine)
        self._beta = self._engine.mat(beta)
        self._W = self._engine.mat(weight)
        self._report = DataSet()

    @property
    def engine(self):
        '''Return the calculating tool that you are using
        '''
        return _engine2str(self._engine)

    @engine.setter
    def engine(self, value):
        '''Reset the calculating library (DaPy or Numpy)
        '''
        self._engine = _str2engine(value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, new_beta):
        self._beta = self._engine.mat(new_beta)

    @property
    def SSE(self):
        if hasattr(self, '_SSE'):
            return self._SSE
        return None

    @property
    def SSR(self):
        if hasattr(self, '_SSR'):
            return self._SSR
        return None

    @property
    def report(self):
        return self._report

    def __call__(self, x):
        return self.predict(x)

    def _get_weight(self, W, cols, X):
        if W is None:
            return self._engine.diag([1] * X.shape[0])

        if hasattr(W, 'shape'):
            assert W.shape[0] == W.shape[1], 'weight matrix should be a square.'
            assert W.shape[0] == X.shape[0], 'weight should have same lenth with data.'
            return self._engine.mat(W)

        if isinstance(W, dict):
            assert len(W) == 1, 'weighted variable should be only one.'
            variable, power = reduce(list.__add__, W.items())
            if isinstance(variable, (str, unicode)):
                variable = cols.index(variable)
            assert abs(variable) < X.shape[1], 'settle variable index out of range.'
            return self._engine.diag((X[:, variable] ** (-power)).tolist())

        if is_seq(W) is True:
            assert all(map(is_math, W)), 'sequence of weight should be number inside.'
            assert len(W) == X.shape[0], 'weight should have same lengh with data.'
            return self._engine.diag(W)
        
    def fit(self, X, Y, W=None, **kwrds):
        '''
        Parameter
        ---------
        X : matrix-like
            A sequence stores some variables of the records.

        Y : matrix-like
            A sequence stores some targets of the records.

        W : dict, array-like & bool (default=None)
            when this parameter is not None, this model will be fit with
            Weighted Least Squares algorithm.
            
            dict -> {variable name: power value}. when you use a dict as your
            parameter, it will auto-calculate the weight matrix for each
            observation.

            array-like -> [W1, W2, W3, ..., Wn]. The model will build the matrix
            in light of the weight array.
            

        Cp: dict, 2D-tuple (default=None)
            (m, SSEm) -> when this parameter fills with 2D-tuple, the first
            position in this tuple is the dimension size of total model and the
            other position is the SSE of Full-Regression-model. Get SSE from 
            other model with statement as ">>> lr.SSE"
            {'SSE': value, 'm': value} -> fill with dict.

        Formulas
        --------

        beta = (X'X)^(-1)X'Y

        Reference
        ---------
        He X & Liu W. Applied Regression Analysis. China People's University
        Publication House. 2015.
        '''
        if hasattr(X, 'columns'):
            kwrds['variables'] = X.columns
        X, Y = mat(X), mat(Y)
        X_, Y_ = self._engine.mat(X), self._engine.mat(Y)
        n, p = X.shape
        cols = kwrds.get('variables', ['X%d' % i for i in range(1, p+1)])
        if kwrds.get('C', True) is True:
            cols.insert(0, 'Constant')
            X_ = self._engine.column_stack([[1] * X_.shape[0], X_])
        
        W = self._get_weight(W, cols, X)
        self._fit_mle(X_, Y_, W)
        
        y_hat = self._beta.T.dot(X_.T) # Engine.mat
        self._res = Y - mat(y_hat.tolist()).T # DaPy.mat
        y_bar = self._engine.mean(Y_)
        self._SSR = sum(mat((y_hat-y_bar).tolist()) ** 2)
        self._SSE = sum(self._res ** 2)
        R2 = round(self._SSR / (self._SSE+self._SSR), 4)
                    
        self._report.add(self.GetSummary(R2, n, p), 'Model Summary')
        self._report.add(self.GetANOVA(n, p), 'ANOVA')
        self._report.add(self.GetCoeff(X_, n, p, cols), 'Coefficients')
        self._report.add(self.GetCorr(X, n ,p, cols), 'Residual Correlation')
        self._report.add(self.GetPerf(R2, n, p, kwrds.get('Cp', self._SSE)),
                         'Method Performance')

    def GetSummary(self, R2, n, p):
        rho_up = [v1[0] * v2[0] for v1, v2 in zip(self._res, self._res[1:])]
        rho_low = map(lambda x: x[0]**2, self._res[1:])
        table = Frame(None, ['R', u'R\u00B2', u'Adj-R\u00B2', 'DW'])
        table.append([round(sqrt(R2), 4), R2,
                      round(1-((1-R2)*(n-1))/(n-p-1.0), 4),
                      round(2 - 2*sum(rho_up) / sum(rho_low), 4)])
        return table

    def GetANOVA(self, n, p):
        M, N = p, n - p - 1.0
        F = (self._SSR/M) / (self._SSE/N)
        sig = 1 - Fcdf(F, M, N)
        table = Frame(None,
                ['Source', 'df', 'Sum Square', 'Mean Square', 'F', 'Sig.'],
                miss_value='')
        table.append(['Regression', int(M), self._SSR, self._SSR/M, F, '%.4f' % sig])
        table.append(['Residual', int(N), self._SSE, round(self._SSE/N, 4)])
        table.append(['Total', int(N+M), self._SSE+self._SSR])
        return table

    def GetCoeff(self, X, n, p, cols):
        c = X.T.dot(X).I.tolist()
        if cols[0] == 'Constant':
            c = [sqrt(c[i][i]) for i in range(int(p)+1)]
        else:
            c = [sqrt(c[i][i]) for i in range(int(p))]
        sigma_hat = sqrt(self._SSE / (n - p -1.0))
        betas = self._beta.T.tolist()[0]
        t = [round(beta_ / (c_ * sigma_hat), 4) for c_, beta_ in zip(c, betas)]
        sigs = map(lambda x: round(2 * Tcdf(min(x, -x), n-p), 4), t)
        table = Frame(None, ['Method', 'Beta', 't', 'Sig.'])
        for col, beta_, t_, sig_ in zip(cols, betas, t, sigs):
            table.append([col, beta_, t_, sig_])
        return table

    def GetCorr(self, X, n, p, cols):
        abs_res = abs(self._res.T).T.tolist()
        table = Frame(None, ['Variable', 'Spearman', 't', 'Sig.'])
        for i, col_ in enumerate(cols[len(cols) // X.shape[1]:]):
            seq = X[:, i].T.tolist()[0]
            rs = round(corr(seq, abs_res, 'spearman'), 4)
            t = round(sqrt(n - 2) * rs / sqrt(1 - rs**2), 4)
            sig = 2 * round(Tcdf(min(t, -t), n-2), 4)
            table.append([col_, rs, t, sig])
        return table

    def GetPerf(self, R2, n ,p, Cp):
        table = Frame(None, [u'R\u00B2\u2090', 'AIC', u'C\u209A'])

        assert isinstance(Cp, (dict, list, tuple, float))
        if isinstance(Cp, dict):
            m, SSEm = Cp['m'], Cp['SSE']
        if isinstance(Cp, (list, tuple)):
            m, SSEm = Cp
        if isinstance(Cp, (int, float)):
            m, SSEm = (p, self._SSE)
        Cp = round((n - m - 1.0) * self._SSE / SSEm - n + 2 * p, 3)
        table.append([round(1 - (n-1) / (n-p-1.0) * (1 - R2), 4),
                      round(n * log(self._SSE) + 2 * p, 2), Cp])
        return table
                     
    def _fit_mle(self, X, Y, W):
        self._beta = X.T.dot(W).dot(X).I.dot(X.T).dot(W).dot(Y)

    def predict(self, X):
        if not isinstance(X, mat):
            X = mat(X)
        X = self._engine.column_stack([[1] * X.shape[0], X])
        return self._beta.T.dot(mat(X).T)
