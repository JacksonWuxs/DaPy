from time import clock
from DaPy import LogInfo, Series
from .base import BaseEngineModel

class PageRank(BaseEngineModel):
    def __init__(self, engine, random_walk_rate=0.85):
        BaseEngineModel.__init__(self, engine)
        self.random_walk_rate = random_walk_rate

    @property
    def random_walk_rate(self):
        return self._alpha

    @random_walk_rate.setter
    def random_walk_rate(self, rate):
        assert isinstance(rate, float)
        assert 0 <= rate <= 1
        self._alpha = rate

    def __setstate__(self, args):
        BaseEngineModel.__setstate__(self, args)
        self._alpha = args['_alpha']

    def __call__(self, X_mat, stochastic_matrix=None, min_error=0.0001, max_iter=1000):
        return self.transform(X_mat, stochastic_matrix, min_error, max_iter)

    def transform(self, X_mat, stochastic_matrix=None, min_error=0.0001, max_iter=1000):
        X_mat = self._mat(X_mat).T
        if stochastic_matrix is False:
            weight = self._weight
        self._weight = weight = self._mat(stochastic_matrix)
        assert isinstance(max_iter, int) and max_iter >= 1
        assert X_mat.shape[1] == 1, 'X should be 1-D sequence'
        assert X_mat.shape[0] == weight.shape[1], 'items in the X not fit the shape of weight matrix'
        
        for round_ in range(max_iter):
            X_next = self._alpha * self._dot(weight, X_mat) + (1.0 - self._alpha) / X_mat.shape[0]
            error = self._sum(self._abs(X_next - X_mat))
            X_mat = X_next
            if error < min_error:
                LogInfo('   Early stopped iteration')
                break
        return Series(X_mat.T.tolist()[0])
            
        

if __name__ == '__main__':
    weight = [
        [0, 0.9, 0, 0],
        [0.333, 0, 0, 0.5],
        [0.333, 0, 1, 0.5],
        [0.333, 0.5, 0, 0]
        ]
    initial = [0.25, 1, 0.25, 0.25]
    pageranker = PageRank("numpy")
    print(pageranker(initial, weight))
