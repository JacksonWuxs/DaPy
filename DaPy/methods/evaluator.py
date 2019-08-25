from DaPy.core import Matrix, SeriesSet, Series
from DaPy.core import LogInfo, LogWarn, LogErr, is_seq
from DaPy.matlib import zeros, mean
from math import sqrt


def ConfuMat(Y, y_, labels=None):
    '''calculate confution Matrix'''
    labels = sorted(set(Y) | set(y_))
    confu = zeros((len(labels) + 1, len(labels) + 1))
    temp = SeriesSet({'Y': Y, 'y': y_})
    for i, l1 in enumerate(labels):
        subtemp = temp.select(lambda row: row[0] == l1)
        for j, l2 in enumerate(labels):
            confu[i, j] = len(subtemp.select(lambda row: row[1] == l2))
        confu[i, -1] = sum(confu[i])
        
    for j in range(len(labels) + 1): 
        confu[-1, j] = sum(confu[:, j].tolist()[0])
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

def Auc(target, predict, n_bins=100):
    pos_len = sum(target)
    neg_len = len(target) - pos_len
    total = pos_len * neg_len
    pos_histogram = [0] * n_bins
    neg_histogram = [0] * n_bins
    bin_width = 1.0 / n_bins
    for tar, pre in zip(target, predict):
        nth_bin = int(pre / bin_width)
        if tar == 1:
            pos_histogram[nth_bin] += 1
        else:
            neg_histogram[nth_bin] += 1

    accumulate_neg, satisfied_pair = 0, 0
    for pos_his, neg_his in zip(pos_histogram, neg_histogram):
        satisfied_pair += (pos_his * accumulate_neg + pos_his * neg_his*0.5)
        accumulate_neg += neg_his
    return satisfied_pair / float(total)

def Performance(predictor, data, target, mode='reg'):
    assert mode in ('clf', 'reg'), "`mode` must be `clf` or `reg` only."
    assert len(data) == len(target),"the number of target data is not equal to variable data"

    if mode == 'clf':
        result = predictor.predict(data)
        if hasattr(result, 'shape') is False:
            result = SeriesSet(result)
        if hasattr(target, 'shape') is False:
            target = SeriesSet(target)
            assert target.shape[1] == 1, 'testify target must be a sequence'
            target = target[target.columns[0]]
            
        confuMat = ConfuMat(target, result, predictor.labels)
        LogInfo('Classification Accuracy: %.4f' % Accuracy(confuMat) + '%')
        LogInfo('Classification Kappa: %.4f' % Kappa(confuMat))
        if confuMat.shape[1] == 3:
            proba = predictor.predict_proba(data)
            if proba.shape[1] == 2:
                proba = proba[:, 0]
            target = Series(1 if _ == predictor.labels[0] else 0 for _ in target)
            LogInfo('Classification AUC: %.4f' % Auc(target, proba))
        return confuMat
    
    elif mode == 'reg':
        target = Series(target)
        predict = Series(predictor.predict(data).T.tolist()[0])
        mean_abs_err = Score.MAE(target, predict)
        mean_sqrt_err = Score.MSE(target, predict)
        R2 = Score.R2_score(target, predict)
        mean_abs_percent_erro = Score.MAPE(target, predict)
        LogInfo('Regression MAE: %.4f' % mean_abs_err)
        LogInfo('Regression MSE: %.4f' % mean_sqrt_err)
        LogInfo('Regression MAPE: %.4f' % mean_abs_percent_erro)
        LogInfo(u'Regression R²: %.4f' % R2)
        

class Score(object):
    
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
