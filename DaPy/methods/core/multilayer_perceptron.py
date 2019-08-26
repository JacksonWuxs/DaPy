from itertools import repeat
from math import sqrt
from random import randint
from time import clock, localtime

from DaPy.core import DataSet, LogInfo
from DaPy.core import Matrix as mat
from DaPy.core import Series

from .base import Dense, Input, Output, check_activations
from .bp_model import BaseBPModel

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl


CELL = u'\u25CF'.encode('utf-8')
ONE_CELL = u'\u2460'.encode('utf-8')


class BaseMLP(BaseBPModel):
    '''
    Base MultiLayers Perceptron Model (MLP)

    Parameters
    ----------
    engine : data process library
        The opearting model which is used to calculate the active function.

    alpha : float (default=1.0)
        Initialized learning ratio

    beta : float (default=0.1)
        Initialized the adjusting ratio

    upfactor : float (default=1.05)

    downfactor : float (default=0.95)

    layers : list
        The list contains hidden layers.
        
    Examples
    --------
    >>> mlp = dp.MLP() # Initialize the multilayers perceptron model
    >>> mlp.create(3, 1, [3, 3, 3]) # Create a new model with 3 hidden layers
     - Create structure: 3 - 3 - 3 - 3 - 1
    >>> mlp.train(data, target) # Fit your data
     - Start: 2018-7-1 14:47:16 Remain: 9.31 s	Error: 44.02%
        Completed: 10.00 	Remain Time: 8.57 s	Error: 7.64%
        Completed: 20.00 	Remain Time: 7.35 s	Error: 3.82%
        Completed: 29.99 	Remain Time: 6.35 s	Error: 2.65%
        Completed: 39.99 	Remain Time: 5.40 s	Error: 2.06%
        Completed: 49.99 	Remain Time: 4.48 s	Error: 1.70%
        Completed: 59.99 	Remain Time: 3.58 s	Error: 1.46%
        Completed: 69.99 	Remain Time: 2.68 s	Error: 1.28%
        Completed: 79.98 	Remain Time: 1.79 s	Error: 1.15%
        Completed: 89.98 	Remain Time: 0.90 s	Error: 1.04%
        Completed: 99.98 	Remain Time: 0.00 s	Error: 0.96%
     - Total Spent: 9.0 s	Error: 0.9578 %
    >>> mlp.performance(new_data) # Test the performance of model
     - Classification Correct: 98.2243%
    >>> mlp.predict_proba(your_data) # Predict your real task.
    '''
        
    def __init__(self, engine, learn_rate, l1_penalty, l2_penalty, upfactor, downfactor):
        '''initialize a multilayers perceptron model

        Parameter
        ---------
        engine : str (default='numpy')
            The calculating engine for this model. Since the 20 times
            efficiency in mathematical calculating than DaPy, we highly
            recommend that you setup numpy as your engine.

        alpha : float (default=0.025)
            The learning rate baseline for automatic adjustment.

        beta : float (default=0.025)
            The fixed learning rate.

        upfactor : float (default=1.05)
            The automatic adjustment rate to speed up the convergence while it
            is in positive movement.

        downfactor : float (default=0.95)
            The automatic adjustment rate to slow down the diverge while it is
            in nagetive movement.

        '''
        BaseBPModel.__init__(self, engine, learn_rate, l1_penalty, l2_penalty)
        assert 1 < upfactor, '`upfactor` must be greater than 1'
        assert downfactor < 1, '`downfactor` must be less than 1'
        self._upfactor = upfactor               # Upper Rate
        self._downfactor = downfactor           # Down Rate
        self._layers = []                       # restore each Dense layers
        self._size = 0

    @property
    def weight(self):
        '''Return a diction stored two weight matrix.'''
        return dict((str(layer), layer._weight) for layer in self._layers)

    @property
    def layers(self):
        return self._layers

    @property
    def loss_func(self):
        return self._loss_func

    @loss_func.setter
    def loss_func(self, new):
        assert new in ('MSE', 'binary_classifier', 'multiple_classifer')
        self._loss_func = new

    def __repr__(self):
        max_size_y = max([layer.shape[1] for layer in self._layers]) * 2
        size_x = [layer.__repr__() for layer in self._layers]
        print_col = list()
        for layer in self._layers:
            blank = max_size_y / layer.shape[1]
            normal = [blank * i for i in range(1, 1 + layer.shape[1])]
            bias = - (max_size_y - max(normal) + min(normal)) / 2 
            print_col.append([bias + pos for pos in normal])

        output = '  '.join(size_x) + '\n'
        output += '-' * len(output) + '\n'
        output += '   '.join([ONE_CELL.center(len(name)) for name in size_x[:-1]]) + '\n'
        for i in range(1, 1 + max_size_y):
            line = ''
            for j, layer in enumerate(self._layers):
                if i in print_col[j]:
                    line += CELL.center(len(size_x[j])) + '   '
                else:
                    line += ' '.center(len(size_x[j])) + '  '
            output += line + '\n'
        output += '---------------------------\n'
        output += 'Tips:' + CELL + ' represents the normal cell in layer; \n'
        output += '     ' + ONE_CELL + ' represents the automatically added offset cells.'
        return output

    def __setstate__(self, pkl):
        BaseBPModel.__setstate__(self, pkl)
        self._layers = pkl['_layers']
        self._upfactor = pkl['_upfactor']
        self._downfactor = pkl['_downfactor']
        for layer in self._layers:
            if layer.strfunc is not None:
                layer.activation = self._activator(layer.strfunc)

    def add_layer(self, layer):
        self._layers.append(layer)
        self._size += 1
        
    def _create(self, n_in, n_out, layers=None, activators=None):
        ''' Create a new MLP model with multiable hiddent layers.

        Parameters
        ----------
        input_cell : int
            The dimension of your features.

        output_cell : int
            The dimension of your target.

        layers : int or list (default:None)
            The cells in hidden layer, defualt will create a 1 hidden layer MLP
            with a experience formula. If you want to build more than 1 hidden
            layer, you should input a numbers in list.

            activators : str, str in list (default: None)
            The active function of each hidden layer.
        
        reg_act : str (default='softmax')
            the activation of output layer

        Return
        ------
        info : the built up structure of this MLP.

        Example
        -------
        >>> mlp = MLP()
        >>> mlp.create(9, 4, [2, 2, 3], ['sigm', 'sigm', 'sigm', 'line'])
        ' - Create structure: 9 - 2 - 2 - 3 - 4'
        '''
        layers = self._check_cells(n_in, layers, n_out)
        funcs = self._check_funcs(activators, len(layers) - 1)
        assert len(funcs) == len(layers) - 2, 'the number of activations does not match layers.'
        
        self.add_layer(Input(self._engine, n_in))
        for in_, out_, strfunc in zip(layers[:-2], layers[1:-1], funcs):
            actfunc = self._activator(strfunc)
            self.add_layer(Dense(self._engine, in_, out_, actfunc, strfunc))
        self.add_layer(Output(self._engine, out_, n_out, self._activator(self._final_func), self._final_func))
        LogInfo('Structure | ' + ' - '.join(['%s:%d' % (layer, layer.shape[1]) for layer in self.layers]))

    def _check_cells(self, input_, hidden, output):
        if hidden is None:
            return [input_, int(sqrt(input_ + output)) + randint(1,11), output]
        if isinstance(hidden, int) and hidden > 0:
            return [input_, hidden, output]
        if all(map(lambda x: isinstance(x, int), hidden)):
            return [input_] + list(hidden) + [output]
        raise ValueError('hidden layer must be integer or list of integers')

    def _check_funcs(self, funcs, lenth_layer):
        if funcs is None:
            return ['sigm'] * (lenth_layer - 1)
        return check_activations(funcs)

    def _calculate_backward_error(self, predict, target):
        return 2 * (target - predict)

    def _forecast(self, output):
        for i, layer in enumerate(self._layers):
            output = layer.propagation(output)
        return output

    def _fit(self, X, Y, epoch=500, layers=None, activators=None, threashold=0.05, verbose=False):
        '''create a new model and train it.

        This model will help you create a model which suitable for this
        dataset and then train it. It will use the function self.create()
        at the first place, and call self.train() function following.
        '''
        X, Y = self._engine.mat(X), self._engine.mat(Y)
        assert X.shape[0] == Y.shape[0], "number of records in X doesn't match Y"
        self._create(X.shape[1], Y.shape[1], layers, activators)
        self._train(X, Y, epoch, verbose, threashold)

    def _backward(self, x, gradient):
        self._learn_rate = self._auto_adjust_learn_rate()
        for i, layer in enumerate(self._layers[::-1]):
            gradient = layer.backward(gradient, self.learn_rate)

    def _auto_adjust_learn_rate(self):
        return self._learn_rate

    def save(self, addr):
        '''Save your model to a .pkl file
        '''
        file_ = self._check_addr(addr, 'wb')
        try:
            pkl.dump(self, file_)
        finally:
            file_.close()
        return self

    def load(self, fp):
        file_ = self._check_addr(fp, 'rb')
        try:
            self.__setstate__(pkl.load(fp).__getstate__())
        finally:
            file_.close()
