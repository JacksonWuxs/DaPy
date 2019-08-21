from DaPy.methods.core import BaseLinearModel
from .classifier import BaseClassifier
from DaPy.core.base import Series

class LogistClassifier(BaseLinearModel, BaseClassifier):
    def __init__(self, engine='numpy', learn_rate=0.005, l1_penalty=0, l2_penalty=0, fit_intercept=True):
        BaseLinearModel.__init__(self, engine, learn_rate, l1_penalty, l2_penalty, fit_intercept)
        BaseClassifier.__init__(self)
        self._sigmoid = self._activator('sigm')

    def _check_target_labels(self, target):
        self._labels = sorted(set(target))
        assert len(self._labels) == 2, 'the number of labels in Y must be 2.'
        return Series(1 if _ == self._labels[0] else 0 for _ in target)
    
    def _forecast(self, X):
        return self._sigmoid(X.dot(self._weight) + self._bias)
    
    def fit(self, X, Y, epoch=500, early_stop=True, verbose=False):
        Y = self._check_target_labels(Y)
        self._fit(X, Y, epoch, early_stop, verbose)
        return self


