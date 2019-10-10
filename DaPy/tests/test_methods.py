from unittest import TestCase

from DaPy import datasets, io, Series, exp
from DaPy.methods.classifiers import MLPClassifier
from DaPy.methods.evaluator import Performance, Kappa
io.encode()

class Test_Tools(TestCase):
    def setUp(self):
        iris, info = datasets.iris()
        iris = iris.data
        iris.normalized()
        iris.shuffle(inplace=True)
        self.X, self.Y = iris[:'petal width'], iris['class']
        
    def test_mlpclf(self):
        mlp = MLPClassifier('numpy', 0.001)
        mlp.fit(self.X[:120], self.Y[:120], 500, verbose=False)
        confu = Performance(mlp, self.X[120:], self.Y[120:], 'clf')
        self.assertEqual(Kappa(confu) > 0.75,  True)
##
##from DaPy.methods.regressors import LinearRegressor
##from DaPy.methods.classifiers import LogistClassifier
##from random import random
##
##X = [[2, 3, 7], [1, 4, 2], [3, 9, 1], [5, 1, 4], [-2, -7, 5], [-5, -1, 2]]
##Y = list(map(lambda x1, x2, x3: x1 * 5 - x2 * 3 + 1 + random(), *zip(*X)))
##lr = LinearRegressor('numpy', 0.005, 0, 0, True)
##lr.fit(X, Y, epoch=200, early_stop=False)
##Performance(lr, X, Y)
##
##X = [[1, 2], [1, 3], [0, 3], [1, 2], [0, 1], [1, 2], [0, 0], [0, 1]]
##Y = ['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F']
##lr = LogistClassifier()
##lr.fit(X, Y, epoch=200, early_stop=False)
##print(Performance(lr, X, Y, 'clf'))

##class1 = iris.select('Iris-setosa == 1')[:' petal width']
##class2 = iris.select('Iris-versicolor == 1')[:' petal width']
##class3 = iris.select('Iris-virginica == 1')[:' petal width']
##
##lda = LDA()
##lda.fit(class1, class2, class3)
##lda.report.show()
##lda.predict(X[120:])
##
##lda = LDA(solve='fisher')
##lda.fit(class1, class2, class3)
##lda.report.show()
##lda.predict(X[120:])
