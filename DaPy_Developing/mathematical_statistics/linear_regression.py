from collections import namedtuple
from DaPy.core import Table, DataSet, mean

__all__ = ['LinearRegression']

def LinearRegression(X, Y):
    '''Linear Regression Model

    This function will help you develop a ``Linear Regression Model`` calculated
    by Least Squares, which is the most common used in data analysis .

    Parameter
    ---------
    X : numbers in sequence
        A sequence stores some variables of the records.

    Y : numbers in sequence
        A sequence stores some targets of the records.

    Return
    ------
    LR_Model : Namedtuple(Alpha, Beta, Sigma^2, T-test, MS1, MS2, F-test)

    Reference
    ---------
    Xiaoling Xu, Ronghua Wang. (2013). Probability Theory & Mathematical Statistics.
        Shanghai: Shanghai Jiaotong University Press
    '''
    # Calculate Model
    len_x, len_y = len(X), len(Y)
    if len_x != len_y:
        raise ValueError('size of two sequences are not match.')

    lxx = sum(map(lambda x: pow(x, 2), X)) - pow(sum(X), 2)/float(len_x)
    lyy = sum(map(lambda x: pow(x, 2), Y)) - pow(sum(Y), 2)/float(len_y)
    lxy = sum([X[i] * Y[i] for i in xrange(len_x)]) -\
          sum(X)*sum(Y)/float(len_y)

    beta = lxy / lxx
    alpha = mean(Y) - beta * mean(X)
    def y_(x):
        return beta * x + alpha

    # T-test and Variance analysis
    sigma_2 = sum([(Y[i] - y_(X[i])) ** 2 for i in range(len_x)]) / (len_x - 2)
    T = beta * (lxx ** 0.5) / sigma_2 ** 0.5
    U = pow(beta, 2) * lxx
    Se = lyy - U
    MS2 = Se / float(len_x - 2)
    F = U / MS2

    return_model = Table([
        [round(alpha, 4), round(beta, 4), round(sigma_2, 4), round(T, 4)]],
        ['Constant', 'Beta', 'Sigma^2', 'T-value'])
    return_F_test = Table([
        ['Regression', round(U, 4), 1, round(U, 4), round(F, 4)],
        ['Residual', round(Se, 4), len_x - 2, round(MS2)],
        ['Total', round(lyy, 4), len_x - 1]],
        ['', 'Sum Squr', 'DF', 'Mean Squr', 'F-value'])
    
    return DataSet([return_model, return_F_test], ['Model Summary', 'ANOVA'])
