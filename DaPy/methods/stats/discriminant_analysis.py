from DaPy.core import DataSet, Matrix as mat, SeriesSet
from DaPy.matlib import cov, mean
from DaPy.operation import column_stack, row_stack
from DaPy.methods.tools import str2engine, engine2str
from DaPy.methods.evaluator import Accuracy, Kappa, ConfuMat

__all__ = ['LinearDiscriminantAnalysis']

class DiscriminantAnalysis(object):
    def __init__(self, engine='numpy', solve='FISHER'):
        self._engine = str2engine(engine)
        self._solve = solve
        self._confumat = None
        self._report = DataSet()
        if solve.upper() == 'FISHER' and self.engine != 'numpy':
            raise AttributeError('numpy supports Fisher solution only.')

    @property
    def I(self):
        return self._I

    @property
    def C(self):
        return self._C

    @property
    def report(self):
        return self._report

    @property
    def confumat(self):
        return self._confumat

    @property
    def engine(self):
        '''Return the calculating tool that you are using
        '''
        return engine2str(self._engine)

    @engine.setter
    def engine(self, value):
        '''Reset the calculating library (DaPy or Numpy)
        '''
        self._engine = str2engine(value)

    def _create_report(self, **kwrds):
        self._report = DataSet()
        if self._solve.upper() == 'FISHER':
            self._report.add(self._Info(kwrds['shape']), 'Model Summary')
            self._report.add(self._Summary(), 'Model Information')
        self._report.add(self._Perf(kwrds['X']), 'Performance')

    def _Summary(self):
        table = SeriesSet(None, ['Function', 'Eigenvalue', 'Rate (%)', 'Cumulative (%)'])
        acf = 0
        for i, (val, valrate) in enumerate(zip(self._value, self._valrate), 1):
            acf += valrate
            table.append_row(['Func%d'%i, round(val, 4), round(valrate * 100, 4), round(acf * 100, 4)])
        return table
    
    def _Info(self, shape):
        table = SeriesSet()
        table.append_col(['X%d' % i for i in range(1, shape+1)], 'Variables')
        for i, vec in enumerate(self._vector, 1):
            table.append_col(vec.tolist()[0], 'Func%d' % i)
        return table

    def _Perf(self, X):
        if self._confumat is None:
            self._confumat = self._calculate_confumat(X)
        table = SeriesSet(None, ['Method', 'Accuracy (%)', 'Kappa'], nan='-')
        table.append_row([self._solve.upper(), Accuracy(self._confumat), Kappa(self._confumat)])
        return table
        
    def _calculate_confumat(self, X):
        Y, y_ = [], []
        for i, x in enumerate(X):
            base = [0] * x.shape[1]
            base[i] = 1
            Y.extend([base for k in range(x.shape[0])])
            y_.extend(self.predict(x).tolist())
        return ConfuMat(mat(Y), mat(y_))        

    def _calculate_xbar(self, X):
        return [self._engine.mean(x, axis=0) for x in X]

    def _calculate_Sp(self, X):
        df = sum([len(x) for x in X]) - len(X)
        S = [self._engine.cov(x.T) * (x.shape[0]-1) for x in X]
        Sp = (self._engine.mat(reduce(self._engine.add, S)) / df)
        return Sp

    def _fit_linear(self, X):
        X_bar = self._calculate_xbar(X)
        Sp = self._calculate_Sp(X).I
        I = [Sp.dot(x.T) for x in X_bar]
        C = [(-0.5) * x.dot(Sp).dot(x.T) for x in X_bar]
        return I, C

    def _fit_fisher(self, X, acr=1.):
        size, col = map(len, X), len(X[0].tolist()[0])
        X_bar = self._calculate_xbar(X)
        X_T = reduce(self._engine.add,
                     [n*x for n, x in zip(size, X_bar)]) / sum(size)
        Sp = self._calculate_Sp(X).I
        
        H = 0
        for n, x in zip(size, X_bar):
            out_diff = (x-X_T).T
            H += out_diff.dot(out_diff.T) * n
            
        E = 0
        for x, x_bar in zip(X, X_bar):
            in_diff = (x - x_bar).T
            E += in_diff.dot(in_diff.T)
        Sp_ = E / (size[0] - col)
        delta = E.I.dot(H.T)
        values, vectors = self._engine.linalg.eig(delta)
        vectors = [vec for val, vec in zip(values, vectors) if val > 0]
        values = [val for val in values if val >=0]
        values_rate = [float(val) / sum(values) for val in values]
        acr_ = 0
        for r, valrate in enumerate(values_rate, 1):
            acr_ += valrate
            if acr_ >= acr:
                break
        return (vectors[:r], values[:r], values_rate[:r], X_bar)
        
    def _predict_linear(self, X):
        results = [X.dot(i) + c for i, c in zip(self._I, self._C)]
        return self._engine.column_stack(results)

    def _predict_fisher(self, X):
        results = []
        for center in self._center:
            stand_X = X - center
            results.append(row_stack([vec.dot(stand_X.T) for vec in self._vector]))
        results = [1.0 / self._engine.sum(vec ** 2, 0) for vec in results]
        return self._engine.column_stack(results)

    def fit(self, *X, **kwrds):
        '''
        Parameters
        ----------
        X : matrix-like
            a sequence of sample variables which seperated by class already.

        ACR : float (default=0.8)
            Expected accumulated contribution rate
        '''
        X = map(self._engine.mat, X)
        shape = max([x.shape[1] for x in X])
        assert all([shape == x.shape[1] for x in X]), 'variables between classes should be the same.'
        
        if self._solve.upper() == 'LINEAR':
            self._I, self._C = self._fit_linear(X)

        if self._solve.upper() in ['FISHER', 'TYPICAL']:
            ACR = kwrds.get('ACR', 0.8)
            (self._vector, self._value, self._valrate, self._center) = self._fit_fisher(X, ACR)
        self._create_report(shape=shape, X=X)

    def predict_proba(self, X):
        X = self._engine.mat(mat(X))
        if self._solve.upper() == 'LINEAR':
            score = self._engine.abs(self._predict_linear(X))
        if self._solve.upper() == 'FISHER':
            score = self._engine.abs(self._predict_fisher(X))
        return score / self._engine.sum(score, 1).reshape((X.shape[0], 1))
                       
    def predict(self, X):
        X_proba = self.predict_proba(X)
        return X_proba / self._engine.max(X_proba, 1).reshape(X_proba.shape[0], 1)
