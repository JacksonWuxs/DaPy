from DaPy.core import Matrix, SeriesSet
from DaPy.core import LogInfo, LogWarn, LogErr
from DaPy.matlib import zeros, mean
from math import sqrt
from .tools import clf_label

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

def Performance(clf, data, target, mode='clf'):
    if len(data) != len(target):
        raise IndexError("the number of target data is not equal to variable data")

    result, target = Matrix(clf.predict_proba(data)), Matrix(target)
    if mode == 'clf':
        confuMat = ConfuMat(result, target)
        LogInfo('Classification Accuracy: %.4f' % Accuracy(confuMat) + '%')
        LogInfo('Classification Kappa: %.4f' % Kappa(confuMat))
        return confuMat
    
    elif mode == 'reg':
        LogInfo('Regression MSE: %.4f' % sqrt(mean((target - result) ** 2)))
    
    else:
        raise ValueError("`mode` supports `clf` or `reg` only.")
