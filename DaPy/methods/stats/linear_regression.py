from collections import namedtuple
from DaPy import Frame, DataSet, mean, mat
from DaPy.methods.tools import _str2engine, _engine2str

__all__ = ['LinearRegression']

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
    Xiaoling Xu, Ronghua Wang. (2013). Probability Theory & Mathematical Statistics.
        Shanghai: Shanghai Jiaotong University Press.
    '''
    def __init__(self, engine='Numpy', beta_=None, alpha_=None):
        self._engine = _str2engine(engine)
        self._beta = self._engine.mat(beta_)
        self._alpha = self._engine.mat(alpha_)
        self._mode = None
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
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):
        self._alpha = self._engine.mat(new_alpha)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, new_beta):
        self._beta = self._engine.mat(new_beta)

    @property
    def report(self):
        return self._report

    def __call__(self, x):
        return self.predict(x)
        
    def fit(self, x, y, solve='MLE', F=True, R=True, T=True):
        '''
        Parameter
        ---------
        X : matrix-like
            A sequence stores some variables of the records.

        Y : matrix-like
            A sequence stores some targets of the records.

        solve : str (default="MLE")
            how to get the beta matrix.

        Formulas
        --------

        beta = (X'X)^(-1)X'Y

        Reference
        ---------
        He X & Liu W. Applied Regression Analysis. China People's University
        Publication House. 2015.
        '''
        assert solve.lower() == 'mle', 'solve should be "MLE" in this version'
        X, Y = self._engine.mat(mat(x)), self._engine.mat(mat(y))
        X = self._engine.column_stack([[1] * X.shape[0], X])
        self._fit_mle(X, Y)

        if True in (F, R, T):
            Y_bar, Y_hat = self._engine.mean(Y), self._beta.T.dot(X.T)
            SSR = round(self._engine.sum(mat((Y_hat-Y_bar).tolist()) ** 2), 4)
            SSE = round(self._engine.sum(mat((Y - Y_hat.T).tolist()) ** 2), 4)
            SST = round(SSE + SSR, 4)
            R_sqrt = SSR / SST
            n, p = float(X.shape[0]), X.shape[1]-1.0
            M, N = p, n - p - 1.0
            
        if R is True:
            table = Frame(None, ['R', u'R\u00B2', u'Adj-R\u00B2'])
            table.append([round(R_sqrt**0.5, 4), round(R_sqrt, 4), round(1-((1-R_sqrt)*(n-1))/N, 4)])
            self._report.add(table, 'Model Summary')
            
        if F is True:
            F = round((SSR/p) / (SSE/(n-p-1)), 4)
            try:
                from scipy.stats import f
                sig = f.pdf(F, M, N)
            except ImportError:
                sig = '-'
                warn('F-test bases on the scipy lib, try: pip install scipy.')
            table = Frame(None,
                    ['Source', 'df', 'Sum Square', 'Mean Square', 'F', 'Sig.'],
                    miss_value='')
            table.append(['Regression', int(M), SSR, SSR/p, F, '%.4f' % sig])
            table.append(['Residual', int(N), SSE, round(SSE/(n-p-1), 4)])
            table.append(['Total', int(n-1), SST])
            self._report.add(table, 'ANOVA')

        if T is True:
            if hasattr(x, 'columns'):
                cols = x.columns
            else:
                cols = ['X%d' % i for i in range(1, X.shape[1]+1)]
            c = X.T.dot(X).I.tolist()
            c = [c[i][i] ** 0.5 for i in range(int(p+1))]
            sigma_hat = (SSE / N) ** 0.5
            betas = self._beta.T.tolist()[0]
            t = [round(beta_ / (c_ * sigma_hat), 4) for c_, beta_ in zip(c, betas)]
            try:
                from scipy.stats import t as t_test
                sigs = map(lambda x: round(t_test.pdf(x, n-p), 4), t)
            except ImportError:
                sigs = ['-'] * len(t)
                warn('t-test bases on the scipy lib, try: pip install scipy.')
            
            table = Frame(None, ['Method', 'Beta', 't', 'Sig.'])
            table.append(['Constant', '%.4f' % betas[0], t[0], sigs[0]])
            for i, (beta_, t_, sig_) in enumerate(zip(betas[1:], t[1:], sigs[1:])):
                table.append([cols[i], round(beta_, 6), t_, sig_])
            self._report.add(table, 'Coefficients')
                
    def _fit_mle(self, X, Y):
        self._beta = X.T.dot(X).I.dot(X.T).dot(Y)

    def predict(self, X):
        if not isinstance(X, mat):
            X = mat(X)
        X = self._engine.column_stack([[1] * X.shape[0], X])
        return self._beta.T.dot(mat(X).T)
