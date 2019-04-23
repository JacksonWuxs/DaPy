from math import sqrt
from random import gauss, randint, uniform

from DaPy.core.base import Matrix, STR_TYPE
from DaPy.methods.tools import engine2str, str2engine


class Layer(object):
    def __init__(self, engine, function):
        self._engine = engine
        self._func = function

    def __repr__(self):
        return self.__name__

    @property
    def engine(self):
        if self._engine is None:
            return ''
        return engine2str(self._engine)

    @engine.setter
    def engine(self, other):
        if isinstance(other, STR_TYPE):
            if other != '':
                self._engine = str2engine(other)
            else:
                self._engine = None
        else:
            self._engine = other

    @property
    def activation(self):
        return self._func.__name__

    @activation.setter
    def activation(self, other):
        assert callable(other), 'activation should bu callable object'
        self._func = other

    def __getstate__(self):
        obj = self.__dict__.copy()
        obj['_engine'] = self.engine
        obj['_func'] = None
        return obj

    def __setstaet__(self, dict):
        self._func = dict['_func']
        self.engine = dict['_engine']
        if '_weight' in dict:
            self._weight = dict['_weight']

    def propagation(self):
        pass

    def backward(self):
        pass

class Input(Layer):
    '''Input layer in the model
    '''
    
    __name__ = 'Input'
    
    def __init__(self, in_cells, *args, **kwrds):
        assert isinstance(in_cells, int)
        Layer.__init__(self, None, None)
        self._in_cells = in_cells
        self._weight = 0

    @property
    def shape(self):
        return (self._in_cells, self._in_cells)

    def propagation(self, x):
        assert x.shape[1] == self._in_cells
        return x

    def backward(self, x, error, y, alpha, beta):
        pass

class Dense(Layer):
    '''A type of common layer for multilayer perceptron

    This kind of structure can help you quickly develop a new
    machine learning model.
    '''

    __name__ = 'Dense'
    
    def __init__(self, engine, in_cells, out_cells, activation, init_weight='Xavier'):
        Layer.__init__(self, engine, activation)
        self.__init__weight(in_cells, out_cells, init_weight)

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

        activation : function
        '''
        if mode in ('MSRA', 'He'):
            t1, t2, f = 0, sqrt(2.0 / in_cells), gauss

        elif mode == 'Xavier':
            low = in_cells + out_cells
            t1, t2, f = -sqrt(6.0 / low), sqrt(6.0 / low), uniform

        elif mode == 'Gauss':
            t1, t2, f = 0, 1, gauss

        else:
            raise TypeError('the mode for initiall weight only supports MSRA, '+\
                            'Xavier, and Gauss.')
        
        weight = [[f(t1, t2) for j in range(out_cells)] for i in range(in_cells)]
        self._weight = Matrix(weight)

    def propagation(self, x):
        return self._func(self._engine.dot(x, self._weight))

    def backward(self, x, error, y, alpha, beta):
        delta = error * self._func(y, True)
        self._weight += self._engine.dot(x.T, delta) * (alpha + beta)
        return self._engine.dot(delta, self._weight.T)
