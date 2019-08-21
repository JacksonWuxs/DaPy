from warnings import filterwarnings
from .models import BaseEngineModel
from .utils import eng2str, str2eng
filterwarnings('ignore')

ACTIVATIONS = set(('sigm', 'tanh', 'line', 'radb', 'relu', 'softmax'))

def check_activations(func):
    if isinstance(func, (list, tuple)) is False:
        func = [func]
    for every in func:
        assert str(every).lower() in ACTIVATIONS, 'invalid activation symbol, tanh, sigm, relu, softmax, radb and line.'
    return list(func)

class Activators(BaseEngineModel):
    def __init__(self, engine):
        BaseEngineModel.__init__(self, engine)

    def __call__(self, func_name):
        return self.get_actfunc_by_str(func_name)

    def get_actfunc_by_str(self, func_name):
        func_name = str(func_name).lower()
        assert func_name in ACTIVATIONS, 'invalid activation symbol, tanh, sigm, relu, softmax, radb and line.'
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
        # if func_name == 'softmax':
        return self.softmax

    def sigmoid(self, x, diff=False):
        if diff:
            return self._mul(x, 1 - x)
        return 1.0 / (1.0 + self._exp(-x))

    def tanh(self, x, diff=False):
        if diff:
            return 1.0 - self._mul(x, x)
        poss, negt = self._exp(x), self._exp(-x)
        return (poss - negt) / (poss + negt)

    def linear(self, x, diff=False):
        if diff:
            return self.engine.ones(x.shape)
        return x

    def radb(self, x, diff=False):
        temp = self.engine._mul(-x, x)
        if diff:
            return self._multiply(-2.0 * x, self._exp(temp))
        return self._exp(temp)

    def relu(self, x, diff=False):
        if diff:
            return (abs(x) + x) / (2 * x)
        return (abs(x) + x) / 2

    def softmax(self, x, diff=False):
        if diff:
            return x - self._mul(x, x)
        exp_x = self._exp(x)
        sum_x = self._sum(exp_x, axis=1)
        return exp_x / sum_x



