from itertools import repeat
from math import sqrt
from random import randint, random
from time import clock, localtime

from DaPy.core import DataSet, LogInfo
from DaPy.core import Matrix as mat
from DaPy.core import Series

from .base import (Activators, BaseEngineModel, check_activations, eng2str,
                   str2eng)
from .bp_model import BaseBPModel


class BaseLinearModel(BaseBPModel):
    def __init__(self, engine, learn_rate, l1_penalty, l2_penalty, fit_intercept):
        BaseBPModel.__init__(self, engine, learn_rate, l1_penalty, l2_penalty)
        self.intercept = fit_intercept
        self._weight = None
    
    @property
    def intercept(self):
        return self._intercept
    
    @intercept.setter
    def intercept(self, set_intercept):
        assert set_intercept in (True, False)
        self._intercept = set_intercept
    
    def _backward(self, X, gradient):
        delta = X.T.dot(gradient)                                # gradient error
        pen1 = self.l1_penalty * self._engine.sign(self._weight) # Level 1 penlty
        pen2 = self.l2_penalty * self._weight                    # Level 2 penlty
        self._weight -= self.learn_rate * (delta + pen1 + pen2)
        self._bias -= self.learn_rate * self._sum(gradient)

    def _calculate_accuracy(self, predict, target):
        return self._mean(self._pow(predict - target, 2)) / len(predict)

    def _calculate_backward_error(self, predict, target):
        return 2 * (predict - target)

    def _create(self, X, Y):
        self._weight = self._engine.mat([random() for _ in range(X.shape[1])]).T
        self._bias = random()
    
    def _fit(self, X, Y, epoch=500, early_stop=True, verbose=False):
        '''create a new model and train it.

        This model will help you create a model which suitable for this
        dataset and then train it. It will use the function self.create()
        at the first place, and call self.train() function following.
        '''
        X, Y = self._engine.mat(X), self._engine.mat(Y).T
        assert Y.shape[1] == 1, 'Y must be 1 dimentions'
        assert X.shape[0] == Y.shape[0], "number of records in X doesn't match Y"
        self._create(X, Y)
        self._train(X, Y, epoch, verbose, early_stop)
        return self

