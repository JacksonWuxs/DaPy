from time import clock
from DaPy import LogInfo, Series
from .base import BaseEngineModel, Activators

class PageRank(BaseEngineModel):
    def __init__(self, engine, weight, random_walk_rate=0.1):
        BaseEngineModel.__init__(self, engine)
        self._weight = self._mat(weight)
        self.random_walk_rate = random_walk_rate

    @property
    def random_walk_rate(self):
        return self._alpha

    @random_walk_rate.setter
    def random_walk_rate(self, rate):
        assert isinstance(rate, float)
        assert 0 <= rate <= 1
        self._alpha = rate

    def __call__(self, X_mat):
        return self.transform(X_mat)

    def transform(self, X_mat, min_error=0.000001, max_iter=1000):
        X_mat = self._mat(X_mat).T
        assert X_mat.shape[1] == 1, 'X should be 1-D sequence'
        assert X_mat.shape[0] == self._weight.shape[1], 'items in the X not fit the shape of weight matrix'
        
        for round_ in range(10000):
            X_next = self._alpha * self._dot(self._weight, X_mat) + (1.0 - self._alpha) / X_mat.shape[0]
            error = self._sum(self._abs(X_next - X_mat))
            X_mat = X_next
            if error < min_error:
                LogInfo('   Early stopped iteration')
                break
        return X_mat
            
        

if __name__ == '__main__':
    weight = [
        [0, 0.9, 0, 0],
        [0.333, 0, 0, 0.5],
        [0.333, 0, 1, 0.5],
        [0.333, 0.5, 0, 0]
        ]
    initial = [0.25, 1, 0.25, 0.25]
    pageranker = PageRank("numpy", weight)
    print pageranker(initial)
