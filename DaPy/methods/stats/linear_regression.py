from collections import namedtuple
from math import sqrt

from DaPy.core import DataSet, Frame, SeriesSet, Matrix as mat
from DaPy.core import is_math, is_seq
from DaPy.matlib import _abs as abs
from DaPy.matlib import _sum as sum
from DaPy.matlib import corr, log, mean
from DaPy.methods.tools import engine2str, str2engine, _create_plot_reg
from DaPy.operation import column_stack

from warnings import warn

__all__ = ['LinearRegression']

try:
    from scipy.stats import f, t  
except ImportError:
    def unsupportedTest(*args, **kwrds):
        return '-'
    Fcdf, Tcdf = unsupportedTest, unsupportedTest
    warn('DaPy uses `scipy` to compute p-value, try: pip install scipy.')
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
    >>> lr.report.show()
    sheet:Model Summary
    ===================
       R    |  R²  | Adj-R² |   DW  
    --------+------+--------+--------
     0.9798 | 0.96 | 0.9556 | 3.4255 
    sheet:ANOVA
    ===========
       Source   | df |   Sum Square   |  Mean Square  |       F       |  Sig 
    ------------+----+----------------+---------------+---------------+--------
     Regression | 1  | 21.8183175034  | 21.8183175034 | 216.033587103 | 0.0000 
      Residual  | 9  | 0.908955223881 |     0.101     |               |        
       Total    | 10 | 22.7272727273  |               |               |        
    sheet:Coefficients
    ==================
      Method  |      Beta      |    t    | Sig
    ----------+----------------+---------+------
     Constant | 12.4985074627  | 54.7277 | 0.0  
        X1    | 0.598507462687 | 14.6981 | 0.0  
    sheet:Residual Correlation
    ==========================
     Variable | Spearman | t | Sig
    ----------+----------+---+------
    sheet:Method Performance
    ========================
      R²ₐ   | AIC  |  Cₚ
    --------+------+-----
     0.9556 | 0.95 | 0.0 
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
    def __init__(self, engine='Numpy', beta=None, weight=None, constant=True):
        self._engine = _str2engine(engine)
        self._beta = self._engine.mat(beta)
        self._W = self._engine.mat(weight)
        self._C = constant
        self._report = DataSet(log=False)
        self._plot = None

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
    def constant(self):
        if self._C == 1:
            return True
        return False

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

    @property
    def plot(self):
        return self._plot

    def __call__(self, x):
        return self.predict(x)

    def _mul(self, a, b):
        return self._engine.multiply(a, b)

    def _mean(self, x, axis=None):
        return self._engine.mean(x, None)

    def _pow(self, x, power=2):
        if isinstance(power, int):
            return reduce(self._mul, [x] * power)
        x = mat(x.tolist())
        return x ** 0.5
        
    def _sum(self, x, axis=None):
        return self._engine.sum(x, axis)
    
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

    def _stepwise(self, X, Y, W, enter, drop, verbal=True):
        print(' - Step 0 Enter Variable: %s' % X.columns[0])
        use, useless = X.columns[:1], X.columns[1:]
        for step, new in enumerate(useless, 1):
            testify = use + [new]
            self._fit(X[testify], Y, W, report='basic')
            if self._report.Coefficients.data.Sig[-1] <= enter:
                print(' - Step %d Enter Variable: %s' % (step, new))
                keep_use = set(self._backward(testify, X, Y, W, drop, False))
                drop_use = list(set(use) - keep_use)
                use = list(keep_use)
                useless.extend(drop_use)
                if drop_use != []:
                    print(' - Step %d Delete Variable: %s' % (step, '|'.join(drop_use)))
        else:
            self.fit(X[use], Y, W)
            return use
                                  

    def _backward(self, variables, X, Y, W, enter, verbal=True):
        step = 0
        while len(variables) != 0:
            self._fit(X[variables], Y, W, report='basic')
            report = self._report.Coefficients.data
            coef_max = max(report.Sig[self._C:])
            if coef_max <= enter:
                return variables
            dropout = variables.pop(report.Sig[self._C:].index(coef_max))
            if verbal is True:
                print(' - Step %d Delete Variable: %s' % (step, dropout))
        else:
            warn('there is no significant variable inside the (X).')

    def _forward(self, variables, X, Y, W, enter, verbal=True):
        use = []
        for step, variable in enumerate(variables):
            self._fit(X[use + variable], Y, W, report='basic')
            if self._report.Coefficients.data.Sig[-1] <= enter:
                use.append(variable)
                if verbal is True:
                    print(' - Step %d Enter Variable: %s' % (step, variable))
        else:
            return use     
        
    def _fit(self, X, Y, W=None, **kwrds):
        if hasattr(X, 'columns'):
            kwrds['variables'] = list(X.columns)
        X, Y = self._engine.mat(mat(X)), self._engine.mat(mat(Y))
        y_, c = self._fit_ls(X, Y, W, kwrds['variables'])
        self._create_report(y_, c, X, Y, **kwrds)
        self._plot = _create_plot_reg(y_, Y, self._res)

    def _fit_ls(self, x, y, w, cols):
        '''train model with (weight) least square method

        Parameter
        ---------
        x : matrix-like
        y : matrix-like
        w : dict, array-like & bool
        cols : str in list

        Returns
        -------
        y_hat : matrix-like
        c : array-like
        '''
        if self.constant is True:
            cols.insert(0, 'Constant')
            x = self._engine.column_stack([[1] * x.shape[0], x])
        w = self._get_weight(w, cols, x)
        self._beta = x.T.dot(w).dot(x).I.dot(x.T).dot(w).dot(y)
        c = x.T.dot(x).I.tolist()
        y_hat = self._beta.T.dot(x.T)
        return y_hat, [sqrt(abs(c[i][i])) for i in range(x.shape[1])]

    def _create_report(self, y_hat, c, x, y, **kwrds):
        self._res = y - mat(y_hat.tolist()).T
        self._SSR = sum(self._pow(y_hat - self._mean(y)))
        self._SSE = sum(self._pow(self._res))

        R2 = round(self._SSR / (self._SSE+self._SSR), 4)
        Cp = kwrds.get('Cp', self._SSE)
        n, p = x.shape
        cols = kwrds.get('variables', ['X%d' % i for i in range(1, p+1)])

        self._report = DataSet(log=False)
        if kwrds.get('report', 'entire').lower() in ('basic', 'entire'):
            self._report.add(self._Summary(R2, n, p), 'Model Summary')
            self._report.add(self._ANOVA(n, p), 'ANOVA')
            self._report.add(self._Coeff(c, n, p, cols), 'Coefficients')
        if kwrds.get('report', 'entire').lower() in ('advance', 'entire'):
            self._report.add(self._Corr(x, n ,p, cols), 'Residual Correlation')
            self._report.add(self._Perf(R2, n, p, Cp), 'Performance')
            self._report.add(self._Coll(x, cols), 'Collinearity')
            self._report.add(self._Outliers(x, y_hat, n, p), 'Outliers')
        self.report.log = True
        
    def _Summary(self, R2, n, p):
        rho_up = [v1[0] * v2[0] for v1, v2 in zip(self._res, self._res[1:])]
        rho_low = map(lambda x: x[0]**2, self._res[1:])
        table = Frame(None, ['R', u'R\u00B2', u'Adj-R\u00B2', 'DW'])
        table.append([round(sqrt(R2), 4), R2,
                      round(1-((1-R2)*(n-1))/(n-p-1.0), 4),
                      round(2 - 2*sum(rho_up) / sum(rho_low), 4)])
        return table

    def _ANOVA(self, n, p):
        M, N = p, n - p - 1.0
        F = round((self._SSR/M) / (self._SSE/N), 4)
        sig = 1 - Fcdf(F, M, N)
        table = Frame(None,
                ['Source', 'df', 'Sum Square', 'Mean Square', 'F', 'Sig'],
                miss_value='')
        table.append(['Regression', int(M), self._SSR, self._SSR/M, F, '%.4f' % sig])
        table.append(['Residual', int(N), self._SSE, round(self._SSE/N, 4)])
        table.append(['Total', int(N+M), self._SSE+self._SSR])
        return table

    def _Coeff(self, c, n, p, cols):
        sigma_hat = sqrt(self._SSE / (n - p -1.0))
        betas = self._beta.T.tolist()[0]
        t = [round(beta_ / (c_ * sigma_hat), 4) for c_, beta_ in zip(c, betas)]
        sigs = map(lambda x: round(2 * Tcdf(min(x, -x), n-p), 4), t)
        table = Frame(None, ['Method', 'Beta', 't', 'Sig'])
        for col, beta_, t_, sig_ in zip(cols, betas, t, sigs):
            table.append([col, beta_, t_, sig_])
        return table

    def _Corr(self, X, n, p, cols):
        abs_res = abs(self._res).tolist()
        table = Frame(None, ['Variable', 'Spearman', 't', 'Sig'])
        for i, col_ in enumerate(cols[self._C:]):
            seq = X[:, i].T.tolist()[0]
            rs = round(corr(seq, abs_res, 'spearman'), 4)
            t = round(sqrt(n - 2) * rs / sqrt(1 - rs**2), 4)
            sig = 2 * round(Tcdf(min(t, -t), n-2), 4)
            table.append([col_, rs, t, sig])
        return table

    def _Perf(self, R2, n ,p, Cp):
        assert isinstance(Cp, (dict, list, tuple, float))
        if isinstance(Cp, dict):
            m, SSEm = Cp['m'], Cp['SSE']
        if isinstance(Cp, (list, tuple)):
            m, SSEm = Cp
        if isinstance(Cp, (int, float)):
            m, SSEm = (p, self._SSE)
        Cp = round((n - m - 1.0) * self._SSE / SSEm - float(n) + 2.0 * p, 3)
        RMSE = round(sqrt(self._mean(self._pow(self._res))), 6)

        table = Frame(None, [u'R\u00B2\u2090', 'AIC', u'C\u209A', 'RMSE'])
        table.append([round(1 - (n-1) / (n-p-1.0) * (1 - R2), 4),
                      round(n * log(self._SSE) + 2 * p, 2), Cp, RMSE])
        return table

    def _Coll(self, X, cols):
        X -= self._mean(X, axis=1)
        X /= self._engine.std(X, axis=0)
        C = X.T.dot(X).I.tolist()
        Cj = [C[i][i] for i in range(len(C[0]))]

        table = Frame(None, ['Variable', 'VIF', 'Tolerance'])
        for col, Cj_ in zip(cols, Cj):
            table.append([col, round(Cj_, 1), 1 / Cj_])
        return table
    
    def _Outliers(self, X, Y_h, n, p):
        H = X.dot(X.T.dot(X).I).dot(X.T).tolist()
        hs = mat([H[i][i] for i in range(n)])
        sigma_hat = sqrt(self._SSE / (n - p))
        SRE = self._res / (sigma_hat * (self._pow(1 - hs, 0.5) ))
        up, down = n - p - 2, n - p - 1.0
        SRE_del = self._mul(SRE, self._pow(up / (down - self._pow(SRE)), 0.5))
        CookDis = self._pow(self._res) / ((p + 1) * sigma_hat ** 2) * hs / self._pow(1 - hs)
        mean_h = 3 * (p+1.0) / n

        table = Frame(None, ['Index', 'y_', 'error', 'hi', 'SRE(i)', 'CookDis', 'Influential', 'Outlier'])
        for i, (y, e, h, sre, d) in enumerate(zip(Y_h.tolist()[0], self._res.T.tolist()[0],
                                    hs.T.tolist()[0], SRE_del.T.tolist()[0],
                                   CookDis.T.tolist()[0])):

            record = [i, round(y, 4), round(e, 4), round(h, 3), round(sre, 3), round(d, 3), '', '']
            if h > mean_h:
                record[6] = 'YES'
            if sre > 3 or d > 1:
                record[7] = 'YES'
            table.append(record)
        return table

    def fit(self, X, Y, W=None, method='enter', **kwrds):
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

        method : str (default='Enter')
            'enter' -> enter all variables to train model
            'backward' -> use backward method to select suitable variables
            'foreward' -> use foreward method to select suitable variables
            'stepwise' -> use stepwise method to select suitable variables

        constant : bool (default=True)
            add a constant variable into model or not

        Cp: dict, 2D-tuple (default=None)
            (m, SSEm) -> when this parameter fills with 2D-tuple, the first
            position in this tuple is the dimension size of total model and the
            other position is the SSE of Full-Regression-model. Get SSE from 
            other model with statement as ">>> lr.SSE"
            {'SSE': value, 'm': value} -> fill with dict.

        Report : str (default='entire')
            'basic' -> testify the model with following tables:
                        "Model Summary", "ANOVA" & "Coefficients"
            'advanced' -> testify the model with following tables:
                        "Residual Correlation", "Performance" & "Outliers"
            'entire' -> build and testify the model with a full report including
                        all above tables

        Formulas
        --------

        beta = (X'X)^(-1)X'Y

        Reference
        ---------
        He X & Liu W. Applied Regression Analysis. China People's University
        Publication House. 2015.
        '''
        assert isinstance(method, (str, unicode)), 'method parameter should be a str'
        assert method.lower() in ('enter', 'backward', 'foreward', 'stepwise')
        self._C = kwrds.get('constant', 1)
        if hasattr(X, 'columns') is False:
            kwrds['variables'] = ['C_%d' % i for i in range(mat(X).shape.Col)]
        if method.lower() == 'enter':
            self._fit(X, Y, W, **kwrds)
            return

        assert hasattr(X, 'columns'), 'X should have variable names.'
        if method.lower() == 'backward':
            return self._backward(list(X.columns), X, Y, W, kwrds.get('enter', 0.05))

        if method.lower() == 'stepwise':
            enter, drop = kwrds.get('enter', 0.1), kwrds.get('drop', 0.15)
            return self._stepwise(X, Y, W, enter, drop)

        if method.lower() == 'foreward':
            return self._foreward(list(X.columns), X, Y, W, kwrds.get('enter', 0.05), **kwrds)

    def drop_outliers(self, X, Y, **kwrds):
        X, Y = Frame(X), Frame(Y, 'Y')
        dropout = SeriesSet(None, list(X.columns))
        while "YES" in self.report['Outliers']['Outlier']:
            delete_index = self.report['Outliers'].select('Outlier == "YES"')['Index']
            pop_x, pop_y = X.pop(delete_index), Y.pop(delete_index)
            dropout.extend(column_stack([pop_x, pop_y]))
            self._fit(X, Y, report='advance', **kwrds)
            print(" - Total Delete Records: %d" % dropout.shape.Ln)

        self._fit(X, Y, report='entire', **kwrds)
        ds = DataSet(column_stack([X, Y]), 'Normal', log=False)
        ds.add(dropout, 'Abnormal')
        return ds

    def predict(self, X):
        if not isinstance(X, mat):
            X = mat(X)
        if self.constant is True:
            X = self._engine.column_stack([[1] * X.shape[0], X])
        return self._beta.T.dot(mat(X).T)

    def performance(self, X, Y):
        result, target = mat(self.predict(X).tolist()), mat(Y)
        return ' - Regression RMSE: %.4f' % sqrt(self._engine.mean((target - result) ** 2))


