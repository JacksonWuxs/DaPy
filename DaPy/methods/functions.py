from warnings import filterwarnings
from DaPy.core.base import STR_TYPE
from DaPy.methods.utils import engine2str, str2engine
filterwarnings('ignore')

ACTIVATIONS = ('sigm', 'tanh', 'line', 'radb', 'relu', 'softmax')
def check_activations(func):
    if isinstance(func, (list, tuple)) is False:
        func = [func]
    for every in func:
        assert isinstance(every, STR_TYPE)
        assert every.lower() in ACTIVATIONS, 'invalid activation symbol, tanh, sigm, relu, softmax, radb and line.'
    return list(func)

class activators(object):
    def __init__(self, engine):
        self._engine = engine

    def __getstate__(self):
        obj = self.__dict__.copy()
        obj['_engine'] = engine2str(self._engine)
        return obj

    def __setstate__(self, dict):
        self._engine = str2engine(dict['_engine'])

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, _engine):
        self._engine = _engine

    def get_actfunc(self, func_name):
        assert isinstance(func_name, STR_TYPE)
        func_name = func_name.lower()
        if func_name == 'sigm':
            return self.sigmoid
        if func_name == 'tanh':
            return self.tanh
        if func_name == 'line':
            return self.linear
        if func_name == 'radb':
            return self.radb
        if func_name == 'relu':
            return self.relu
        if func_name == 'softmax':
            return self.softmax
        raise KeyError('undefined function as %s' % func_name)

    def sigmoid(self, x, diff=False):
        if diff:
            return (1 - x) * x
        return 1.0 / (1.0 + self.engine.exp(-x))

    def tanh(self, x, diff=False):
        if diff:
            return 1.0 - x * x
        poss, negt = self.engine.exp(x), self.engine.exp(-x)
        return (poss - negt) / (poss + negt)

    def linear(self, x, diff=False):
        if diff:
            return self.engine.ones(x.shape)
        return x

    def radb(self, x, diff=False):
        if diff:
            return -2.0 * x * self.engine.exp( -x * x)
        return self.engine.exp(-x * x)

    def relu(self, x, diff=False):
        if diff:
            return (abs(x) + x) / x
        return (abs(x) + x) / 2

    def softmax(self, x, diff=False):
        if diff:
            return x - x * x
        exp_x = self.engine.exp(x)
        sum_x = self.engine.sum(exp_x, axis=1)
        return exp_x / sum_x.reshape((len(sum_x), 1))
        output = [1] * len(new_x)
        for i, record in enumerate(new_x):
            div = sum_x[i]
            output[i] = [value / div for value in record]
        return self.engine.mat(output)

