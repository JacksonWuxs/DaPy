from DaPy.methods.core import BaseLinearModel

class LinearRegressor(BaseLinearModel):
    def __init__(self, engine='numpy', learn_rate=0.05, l1_penalty=0, l2_penalty=0, fit_intercept=True):
        BaseLinearModel.__init__(self, engine, learn_rate, l1_penalty, l2_penalty, fit_intercept)

    def _forecast(self, X):
        return X.dot(self._weight) + self._bias
    
    def fit(self, X, Y, epoch=500, early_stop=True, verbose=False):
        self._fit(X, Y, epoch, early_stop, verbose)
        return self

    def predict(self, X):
        X = self._engine.mat(X)
        return self._forecast(X)
