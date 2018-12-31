from math import sqrt
from random import gauss, randint, uniform

from DaPy.core import Matrix
from DaPy.methods.functions import activation as acfs
from DaPy.methods.tools import _engine2str, _str2engine


class Layer:
    def __init__(self, engine, index, function):
        self._engine = engine
        self._func = acfs[function.lower()]
        self._index = index

    @property
    def engine(self):
        return str(self._engine).split("'")[1].lower()

    @engine.setter
    def engine(self, other):
        self._engine = other

    @property
    def function(self):
        return str(self._func).split(' ')[1].lower()

    @function.setter
    def function(self, other):
        self._func = acfs[ohter]

    @property
    def index(self):
        return self._index

    def __getstate__(self):
        obj = self.__dict__.copy()
        obj['_engine'] = _engine2str
        return obj

    def __setstaet__(self, dict):
        self._func = dict['_func']
        self._index = dict['_index']
        self._engine = _str2engine(dict['_engine'])
        if '_weight' in dict:
            self._weight = dict['_weight']
            

    def propagation(self, x):
        pass

    def backward(self, x, error, y, alpha, beta):
        pass


class Dense_Layer(Layer):
    '''A type of common layer for multilayer perceptron

    This kind of structure can help you quickly develop a new
    machine learning model.
    '''
    
    def __init__(self, engine, in_cells, out_cells, layer_index,
                 activation='sigm', weight_mode='Xavier'):
        Layer.__init__(self, engine, layer_index, activation)
        self.__init__weight(in_cells, out_cells, weight_mode)

    @property
    def shape(self):
        return self._weight.shape

    def __init__weight(self, in_cells, out_cells, mode='MSRA'):
        '''inintialized the weight matrix in this layer.

        Paramters
        ---------
        in_cells : int
            the number of input variables.

        out_cells : int
            the number of output variables.

        mode : str (default: 'MSRA')
            the method of initialization the weight of this layer, use
            MSRA, Xavier or Gauss.

        Return
        ------
        None 
        '''
        if mode in ('MSRA', 'He'):
            t1, t2 = 0, sqrt(2.0 / in_cells)
            f = gauss

        elif mode == 'Xavier':
            low = in_cells + out_cells
            t1, t2 = -sqrt(6.0 / low), sqrt(6.0 / low)
            f = uniform

        elif mode == 'Gauss':
            t1, t2 = 0, 1
            f = gauss

        else:
            raise TypeError('the mode for initiall weight only supports MSRA, '+\
                            'Xavier, and Gauss.')
        
        weight = [1] * in_cells
        for i in range(in_cells):
            weight[i] = [f(t1, t2) for j in range(out_cells)]
        self._weight = Matrix(weight)

    def __repr__(self):
        if not isinstance(self._index, int):
            return self._index
        return 'Dense_%s' % self._index

    def propagation(self, x):
        return self._func(self._engine.dot(x, self._weight) + 1, self._engine)

    def backward(self, x, error, y, alpha, beta):
        delta = error * self._func(y, self._engine, True)
        error = self._engine.dot(delta, self._weight.T)
        self._weight += self._engine.dot(x.T, delta) * (alpha + beta)
        return error
