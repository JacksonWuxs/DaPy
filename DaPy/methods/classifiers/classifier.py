from DaPy.core import Series, SeriesSet
from DaPy.core import is_seq
from copy import copy


def proba2label(seq, labels):
    if hasattr(seq, 'shape') is False:
        seq = SeriesSet(seq)
    if seq.shape[1] > 1:
        return clf_multilabel(seq, labels)
    return clf_binlabel(seq, labels)

def clf_multilabel(seq, groupby=None):
    if is_seq(groupby):
        groupby = dict(enumerate(map(str, groupby)))
    if not groupby:
        groupby = dict()
    assert isinstance(groupby, dict), '`labels` must be a list of str or dict object.'
    max_ind = seq.argmax(axis=1).T.tolist()[0]
    return Series(groupby.get(int(_), _) for _ in max_ind)
    
def clf_binlabel(seq, labels, cutpoint=0.5):
    return Series(labels[0] if _ >= cutpoint else labels[1] for _ in seq)


class BaseClassifier(object):
    def __init__(self):
        self._labels = []

    @property
    def labels(self):
        return copy(self._labels)
    
    def _calculate_accuracy(self, predict, target):
        pred_labels = predict.argmax(axis=1).T.tolist()[0]
        targ_labels = target.argmax(axis=1).T.tolist()[0]
        return sum(1.0 for p, t in zip(pred_labels, targ_labels) if p == t) / len(predict)
    
    def predict_proba(self, X):
        '''
        Predict your own data with fitted model

        Paremeter
        ---------
        data : matrix
            The new data that you expect to predict.

        Return
        ------
        Matrix: the predict result of your data.
        '''
        X = self._engine.mat(X)
        return self._forecast(X)

    def predict(self, X):
        '''
        Predict your data with a fitted model and return the label

        Parameter
        ---------
        data : matrix
            the data that you expect to predict

        Return
        ------
        Series : the labels of each record
        '''
        return proba2label(self.predict_proba(X), self._labels)
