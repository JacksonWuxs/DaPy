from copy import copy

from DaPy.core import Matrix, Series, SeriesSet
from DaPy.methods.core import BaseMLP
from DaPy.operation import get_dummies
from DaPy.methods import evaluator
from .classifier import BaseClassifier


class MLPClassifier(BaseMLP, BaseClassifier):
    
    def __init__(self, engine='numpy', learn_rate=0.05, l1_penalty=0, l2_penalty=0, upfactor=1.05, downfactor=0.95):
        BaseMLP.__init__(self, engine, learn_rate, l1_penalty, l2_penalty, upfactor, downfactor)
        BaseClassifier.__init__(self)

    def _check_target_labels(self, target):
        target = SeriesSet(target)
        if target.shape.Col == 1:
            target = get_dummies(target[target.columns[0]], dtype='SeriesSet')
        self._labels = target.columns
        self._final_func = 'softmax'
        return self._engine.mat(list(target.iter_rows()))

    def fit(self, X, Y, n_epoch=500, n_layers=None,
            activators='sigm', early_stop=False, verbose=True):
        '''Fit your model

        Parameters
        ----------
        X : matrix/2-darray
            The feature matrix in your training dataset.

        Y : matrix/2-darray
            The target matrix in your training dataset.

        n_epoch : int (default=200)
            The number of loops that train the model.

        n_layers : int, int in list (default=None)
            the number of cells in each hidden layer, if model has more than
            one hidden layers, input numbers of cells in each layers as list.

        activators : str, str in list (default='relu')
            the active function used to calculate in each layers
            options: `relu`, `sigm`, `tanh`, `line`, `radb`, `softmax`

        early_stop : float or None (default=False)

        verbose : Bool (default=True)
            - True: print the information while training model.
            - False: do not print any information.

        Return
        ------
        None
        '''
        target = self._check_target_labels(Y)
        self._fit(X, target, n_epoch, n_layers, activators, early_stop, verbose)
        return self

    
