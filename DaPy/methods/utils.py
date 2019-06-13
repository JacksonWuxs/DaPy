from DaPy.core import Frame, is_iter
from DaPy.matlib import describe
from collections import namedtuple
from operator import itemgetter

__all__ = ['_label', 'score_binary_clf']

_binary_perf_result = namedtuple('binary_clf', ['TP', 'FN', 'FP', 'TN'])

def check_input_data(data, engine):
    assert hasattr(engine, 'mat')
    data = engine.mat(data)
    if data.shape[0] < data.shape[1]:
        data = data.T
    if hasattr(data, 'astype'):
        data = data.astype('float32')
    return data    

def plot_reg(y_hat, y, res):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        warn('DaPy uses `matplotlib` to draw pictures, try: pip install matplotlib.')
        return None

    plt.subplot(311)
    plt.title('Prediction of Model')
    plt.xlabel('Samples')
    plt.ylabel('Prediction')
    plt.plot(y.T.tolist()[0], color='blue', alpha=0.65, label='Actual')
    plt.plot(y_hat.tolist()[0], color='red', alpha=0.7, label='Predict')
    plt.legend()
    
    plt.subplot(312)
    plt.title('Distribution of Residual')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.hist(res, max(10, len(y_hat) // 5), color='blue', alpha=0.6)
    
    plt.subplot(313)
    plt.title('Residual')
    plt.xlabel('Samples')
    plt.ylabel('Residual')
    sigma = [describe(res.T.tolist()[0]).Sn] * y_hat.shape[1]
    plt.plot(res, color='blue', alpha=0.6)
    plt.plot([0] * y_hat.shape[1], color='black', linestyle='--', alpha=0.5)
    plt.plot(sigma, color='black', alpha=0.25, linestyle='--')
    plt.plot(map(lambda x: -x, sigma), color='black', alpha=0.25, linestyle='--')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95,
            wspace=0.2, hspace=0.8)
    return plt

def tolist(lst):
    if hasattr(lst, 'tolist'):
        return lst.tolist()
    elif hasattr(lst, 'list'):
        return lst.list()
    elif is_iter(lst):
        return list(lst)
    raise TypeError('unsupported type, try [%s] instead' % lst)

def engine2str(obj):
    return str(obj).split()[1][1: -1]

def str2engine(obj):
    if obj.lower() == 'numpy':
        import numpy as engine
        engine.seterr(divide='ignore', invalid='ignore')
    elif obj.lower() == 'dapy':
        import DaPy as engine
    else:
        raise ValueError('do not support engine as %s, "DaPy" or "numpy" only.'%obj)
    return engine

def clf_label(sequence, groupby=None):
    sequence = tolist(sequence)
    if not groupby:
        groupby = dict()
    elif isinstance(groupby, list):
        groupby = dict(zip(range(len(sequence)), map(str, groupby)))
    elif not isinstance(groupby, dict):
        raise TypeError('groupby shoud input a list of label names or dict object.')

    labelled = list()
    for record in sequence:
        label = max(enumerate(record), key=itemgetter(1))[0]
        labelled.append(groupby.get(label, label))
    return labelled
    
def score_binary_clf(clf, X, Y, cutpoint=0.5, confu_mat=False):
    TP, FN, FP, TN = 0.0, 0.0, 0.0, 0.0
    if hasattr(clf, 'predict_prob'):
        mypredict = _tolist(clf.predict_prob(X))
    else:
        mypredict = _tolist(clf.predict(X))
        
    for predict, true in zip(mypredict, _tolist(Y)):
        if true >= cutpoint:
            if predict >= cutpoint:
                TP += 1
            else:
                FN += 1
        else:
            if predict >= cutpoint:
                FP += 1
            else:
                TN += 1

    if not confu_mat:
        return _binary_perf_result(TP, FN, FP, TN)
    return Frame([
        ['Actual Positive', TP, FN, TP + FN],
        ['Actual Negative', FP, TN, FP + TN],
        ['Total', TP + FP, FN + TN, total]],
        ['', 'Predict Positive', 'Predict Negative', 'Total'])
