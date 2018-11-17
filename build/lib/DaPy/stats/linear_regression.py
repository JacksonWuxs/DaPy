from collections import namedtuple
from DaPy.core import Frame, DataSet, mean, Matrix

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
    >>> from DaPy.linear_regression import LinearRegression
    >>> x = [1,2,3,4,5,5,6,6,7,8,9]
    >>> y = [13, 14, 14, 15, 15, 16,16, 16, 17,17, 18]
    >>> lr = LinearRegression()
    >>> lr.fit(x, y)
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
    >>> lr(3)
    14.294029850746268
    >>>

    Reference
    ---------
    Xiaoling Xu, Ronghua Wang. (2013). Probability Theory & Mathematical Statistics.
        Shanghai: Shanghai Jiaotong University Press.
    '''
    def __init__(self, beta_=None, alpha_=None, solve='MSE'):
        self._solve = solve
        self._beta = beta_
        self._alpha = alpha_
        self._mode = None
        self._f_test = None

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):
        self._alpha = float(new_alpha)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def alpha(self, new_beta):
        self._beta = float(new_beta)

    def __call__(self, x):
        return self._beta * x + self._alpha

    def report(self):
        return DataSet([self._mode, self._f_test],
                       ['Model Summary', 'F-test'])
        
    def fit(self, X, Y):
        '''
        Parameter
        ---------
        X : numbers in sequence
            A sequence stores some variables of the records.

        Y : numbers in sequence
            A sequence stores some targets of the records.
        '''
        len_x, len_y = float(len(X)), float(len(Y))
        if len_x != len_y:
            raise ValueError('size of two sequences are not match.')
        lxx = sum(map(lambda x: pow(x, 2), X)) - pow(sum(X), 2)/len_x
        lyy = sum(map(lambda x: pow(x, 2), Y)) - pow(sum(Y), 2)/len_y
        lxy = sum([x * y for x, y in zip(X, Y)]) - sum(X)*sum(Y)/len_y

        self._beta = lxy / lxx
        self._alpha = mean(Y) - self._beta * mean(X)
        
        # T-test and Variance analysis
        sigma_2 = sum([(Y[i] - self(X[i])) ** 2 for i in range(len_x)]) / (len_x - 2)
        T = self._beta * (lxx ** 0.5) / sigma_2 ** 0.5
        U = pow(self._beta, 2) * lxx
        Se = lyy - U
        MS2 = Se / float(len_x - 2)
        F = U / MS2

        self._mode = Frame([
            [round(self._alpha, 4), round(self._beta, 4), round(sigma_2, 4), round(T, 4)]],
            ['Constant', 'Beta', 'Sigma^2', 'T-value'])
            
        self._f_test = Frame([
            ['Regression', round(U, 4), 1, round(U, 4), round(F, 4)],
            ['Residual', round(Se, 4), len_x - 2, round(MS2), ''],
            ['Total', round(lyy, 4), len_x - 1, '', '']],
            ['Args', 'Sum Squr', 'DF', 'Mean Squr', 'F-value'])
