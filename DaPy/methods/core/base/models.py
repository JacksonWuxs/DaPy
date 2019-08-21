
from itertools import repeat
from DaPy.core.base import is_str
from .utils import str2eng, eng2str

class BaseEngineModel(object):

    def __init__(self, engine='numpy'):
        self.engine = engine
       
    @property
    def engine(self):
        '''Return the calculating tool that you are using'''
        return eng2str(self._engine)

    @engine.setter
    def engine(self, new_engine):
        '''Reset the calculating library (DaPy or Numpy)'''
        if is_str(new_engine):
            new_engine = str2eng(new_engine)

        for func in ('abs', 'dot', 'multiply', 'mean', 'log', 'sum', 'exp'):   
            assert hasattr(new_engine, func), "Your engine does't have attribute %s" % func

        self._engine = new_engine
        if hasattr(self, '_activator'):
            self._activator.engine = self._engine

    def __call__(self, x):
        return self.predict(x)

    def __getstate__(self):
        pickle_self = self.__dict__.copy()
        pickle_self['_engine'] = eng2str(self._engine)
        return pickle_self

    def __setstate__(self, pkl):
        self.engine = pkl['_engine']

    def _abs(self, x):
        return self._engine.abs(x)
    
    def _dot(self, left, right):
        return self._engine.dot(left, right)
    
    def _exp(self, x):
        return self._engine.exp(x)

    def _mul(self, a, b):
        return self._engine.multiply(a, b)

    def _mean(self, x, axis=None):
        return self._engine.mean(x, None)

    def _pow(self, x, power=2):
        if isinstance(power, int):
            return reduce(self._mul, repeat(x, power))

        single_power_x = x
        for terms in range(power):
            x = self._mul(x, single_power_x)
        return x
    
    def _log(self, x, grad=2):
        return self._engine.log(x, grad)
        
    def _sum(self, x, axis=None):
        return self._engine.sum(x, axis)

    def _check_addr(self, addr, mode):
        if isinstance(addr, STR_TYPE):
            return open(addr, mode)
        return addr

    def _check_input_X_matrix(self, mat):
        assert hasattr(mat, 'shape'), 'input X must have attribute `mat.T`'
        assert hasattr(mat, 'T'), 'input X must have attribute `mat.T`'
        assert mat.shape[0] >= mat.shape[1], 'number of rows in X matrix is less than the number of columns'
        if hasattr(mat, 'astype'):
            mat = mat.astype('float32')
        return mat


