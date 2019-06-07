from DaPy.core import Matrix, SeriesSet, Series
from DaPy.core import LogInfo, LogWarn, LogErr
from DaPy.matlib import zeros, mean
from math import sqrt
from .utils import clf_label

def ConfuMat(Y, y_, labels=None):
    '''calculate confution Matrix'''
    if y_.shape[1] != 1:
        y_ = clf_label(y_, labels)
    if Y.shape[1] != 1:
        Y = clf_label(Y, labels)
    Y = [value[0] if isinstance(value, list) else value for value in Y]
    y_ = [value[0] if isinstance(value, list) else value for value in y_]
    labels = list(set(Y) | set(y_))
    confu = zeros((len(labels) + 1, len(labels) + 1))
    temp = SeriesSet({'Y': Y, 'y': y_})
    
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            confu[i][j] = len(temp.select(lambda row: row[0] == l1 and row[1] == l2))
        confu[i][-1] = sum(confu[i])
        
    for j in range(len(labels) + 1):
        confu[-1][j] = sum(confu[:, j].tolist())
    return confu

def Accuracy(confumat):
    upper = sum([confumat[i][i] for i in range(confumat.shape[1] - 1)])
    return round(upper / float(confumat[-1][-1]) * 100, 4)

def Kappa(confumat):
    as_ = confumat[:, -1].tolist()[:-1]
    bs_ = confumat[-1][:-1]
    Po = Accuracy(confumat) /100
    upper = sum([a * b for a, b in zip(as_, bs_)])
    Pe = float(upper) / confumat[-1][-1] ** 2
    return (Po - Pe) / (1 - Pe)

def Performance(predictor, data, target, mode='reg'):
    assert mode in ('clf', 'reg'), "`mode` must be `clf` or `reg` only."
    assert len(data) == len(target),"the number of target data is not equal to variable data"
    target = Series(target)
    
    if mode == 'clf':
        if hasattr(predictor, 'predict_proba'):
            result = Matrix(predictor.predict_proba(data))

        predict = Matrix(predictor.predict(data).tolist())
        confuMat = ConfuMat(result, target)
        LogInfo('Classification Accuracy: %.4f' % Accuracy(confuMat) + '%')
        LogInfo('Classification Kappa: %.4f' % Kappa(confuMat))
        return confuMat
    
    elif mode == 'reg':
        predict = Series(predictor.predict(data).tolist())
        mean_abs_err = score.MAE(target, predict)
        mean_sqrt_err = score.MSE(target, predict)
        R2 = score.R2_score(target, predict)
        mean_abs_percent_erro = score.MAPE(target, predict)
        LogInfo('Regression MAE: %.4f' % mean_abs_err)
        LogInfo('Regression MSE: %.4f' % mean_sqrt_err)
        LogInfo('Regression MAPE: %.4f' % mean_abs_percent_erro)
        LogInfo(u'Regression R²: %.4f' % R2)
        

class score:
    
    '''performace score to evalulate a regressor'''

    @staticmethod
    def MAE(target, predict):
        return mean(abs(target - predict))

    @staticmethod
    def MSE(target, predict):
        return mean((target - predict) ** 2)

    @staticmethod
    def R2_score(target, predict):
        SSE = sum((predict - target) ** 2)
        SST = sum((target - mean(target)) ** 2)
        return 1 - SSE / SST

    @staticmethod
    def MAPE(target, predict):
        return mean(abs((target - predict) / target))
