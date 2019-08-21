from math import sqrt
from random import gauss, randint, uniform

from DaPy.core.base import Matrix

from .models import BaseEngineModel


class Layer(BaseEngineModel):
    def __init__(self, engine, function):
        BaseEngineModel.__init__(self, engine)
        self.activation = function

    def __repr__(self):
        return self.__name__

    @property
    def activation(self):
        return self._func.__name__

    @activation.setter
    def activation(self, other):
        assert callable(other) or other is None, 'activation should bu callable object'
        self._func = other

    def __getstate__(self):
        obj = self.__dict__.copy()
        obj['_engine'] = self.engine
        return obj

    def __setstaet__(self, dict):
        self._func = dict['_func']
        self.engine = dict['_engine']
        if '_weight' in dict:
            self._weight = dict['_weight']

    def propagation(self, *args, **kwrds):
        pass

    def backward(self, *args, **kwrds):
        pass

class Input(Layer):
    '''Input layer in the model'''
    
    __name__ = 'Input'
    
    def __init__(self, engine, in_cells, *args, **kwrds):
        Layer.__init__(self, engine, None)
        self._in_cells = in_cells
        self._weight = self._engine.zeros((in_cells, in_cells))

    @property
    def shape(self):
        return (self._in_cells, self._in_cells)

    def propagation(self, x):
        assert x.shape[1] == self._in_cells
        self._input = self._output = x
        return x
    

class Dense(Layer):
    '''A type of common layer for multilayer perceptron

    This kind of structure can help you quickly develop a new
    machine learning model.
    '''

    __name__ = 'Dense'
    
    def __init__(self, engine, n_in, n_out, activation, init_weight='Xavier'):
        Layer.__init__(self, engine, activation)
        self._init_parameters(n_in, n_out, init_weight)

    @property
    def shape(self):
        return self._weight.shape

    def _init_parameters(self, in_cells, out_cells, mode='MSRA'):
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
            t1, t2, f = - sqrt(6.0 / low), sqrt(6.0 / low), uniform

        elif mode == 'Gauss':
            t1, t2, f = 0, 1, gauss

        else:
            raise TypeError('the mode for initiall weight only supports MSRA, '+\
                            'Xavier, and Gauss.')
        
        weight = [[f(t1, t2) for j in range(out_cells)] for i in range(in_cells)]
        self._weight = self._engine.mat(weight)

    def propagation(self, input_):
        self._input = input_
        self._output = self._func(input_.dot(self._weight))
        return self._output

    def backward(self, gradient, alpha):
        gradient = self._mul(gradient, self._func(self._output, True))
        self._weight += self._input.T.dot(gradient) * alpha
        return self._dot(gradient, self._weight.T)


class Output(Dense):
    '''Output layer in the model'''
    
    __name__ = 'Output'

    def __init__(self, engine, n_in, n_out, activation, init_weight='Xavier'):
        Dense.__init__(self, engine, n_in, n_out, activation, init_weight)


