from DaPy.core import Frame, is_iter
from collections import namedtuple

__all__ = ['_label', 'score_binary_clf']

_binary_perf_result = namedtuple('binary_clf', ['TP', 'FN', 'FP', 'TN'])

def _tolist(lst):
    if hasattr(lst, 'tolist'):
        return lst.tolist()
    elif hasattr(lst, 'list'):
        return lst.list()
    elif is_iter(lst):
        return list(lst)
    raise TypeError('unsupported type, try [%s] instead' % lst)

def _engine2str(obj):
    return str(obj).split()[1][1: -1]

def _str2engine(obj):
    if obj.lower() == 'numpy':
        import numpy as engine
        engine.seterr(divide='ignore', invalid='ignore')
    elif obj.lower() == 'dapy':
        import DaPy as engine
    else:
        raise ValueError('do not support engine as %s, "DaPy" or "numpy" only.'%obj)
    return engine

def _label(Y_, groupby=None):
    if not groupby:
        groupby = dict()
    elif isinstance(groupby, list):
        groupby = dict(zip(range(len(Y_)), map(str, groupby)))
    elif not isinstance(groupby, dict):
        raise TypeError('groupby shoud input a list of label names or dict object.')
        
    labelled = list()
    for record in _tolist(Y_):
        label = record.index(max(record))
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
    
    
