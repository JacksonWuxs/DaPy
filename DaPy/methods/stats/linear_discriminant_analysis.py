from DaPy import Frame, DataSet, mat, cov, mean
from DaPy.methods.tools import _str2engine, _engine2str

__all__ = ['LinearDiscriminantAnalysis']

class LinearDiscriminantAnalysis:
    def __init__(self, engine='numpy', solve='Linear'):
        self._engine = _str2engine(engine)
        self._solve = solve
        if solve.upper() == 'FISHER' and self.engine != 'numpy':
            raise AttributeError('numpy supports Fisher solution only.')

    @property
    def I(self):
        return self._I

    @property
    def C(self):
        return self._C

    @property
    def engine(self):
        '''Return the calculating tool that you are using
        '''
        return _engine2str(self._engine)

    @engine.setter
    def engine(self, value):
        '''Reset the calculating library (DaPy or Numpy)
        '''
        self._engine = _str2engine(value)

    def _calculate_xbar(self, X):
        return [self._engine.mean(x, axis=0) for x in X]

    def _calculate_Sp(self, X):
        df = sum([len(x) for x in X]) - len(X)
        S = [self._engine.cov(x.T) * (x.shape[0]-1) for x in X]
        Sp = (self._engine.mat(reduce(self._engine.add, S)) / df)
        return Sp

    def _solve_linear(self, X):
        X_bar = self._calculate_xbar(X)
        Sp = self._calculate_Sp(X).I
        I = [Sp.dot(x.T) for x in X_bar]
        C = [(-0.5) * x.dot(Sp).dot(x.T) for x in X_bar]
        return I, C

    def _solve_fisher(self, X):
        size, col = map(len, X), len(X[0].tolist()[0])
        X_bar = self._calculate_xbar(X)
        X_T = reduce(self._engine.add,
                     [n*x for n, x in zip(size, X_bar)]) / sum(size)
        H = 0
        for n, x in zip(size, X_bar):
            out_diff = (x-X_T).T
            H += out_diff.dot(out_diff.T) * n
            
        E = 0
        for x, x_bar in zip(X, X_bar):
            in_diff = (x - x_bar).T
            E += in_diff.dot(in_diff.T)

        delta = E.I.dot(H)
        L = filter(lambda x: x > 0, self._engine.linalg.eig(delta)[0])
        T = []
        for lamda in L:
            print '--' * 20
            lamda_eye = self._engine.diag([lamda] * col)
            t = self._engine.linalg.solve(delta, lamda_eye)
            
        
    def _predict_linear(self, X):
        results = [i.T.dot(X.T) + c for i, c in zip(self._I, self._C)]
        return self._engine.column_stack(results)

    def fit(self, *X):
        '''
        Parameters
        ----------
        X : a sequence of sample variables which seperated by class already.
        '''
        X = map(self._engine.mat, X)
        shape = max([x.shape[1] for x in X])
        if not all([shape == x.shape[1] for x in X]):
            raise ValueError('variables between classes should be the same.')
        
        if self._solve.upper() == 'LINEAR':
            self._I, self._C = self._solve_linear(X)

        if self._solve.upper() in ['FISHER', 'TYPICAL']:
            self._solve_fisher(X)

    def predict(self, X):
        X = self._engine.mat(mat(X))
        if self._solve.upper() == 'LINEAR':
            return self._predict_linear(X)
                       
    def predict_proba(self, X):
        result = self.predict(X).tolist()
        proba_result = []
        for line in result:
            sum_ = sum(map(abs, line))
            proba_result.append([abs(v) / sum_ for v in line])
        return proba_result

