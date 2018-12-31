from DaPy import datasets
from DaPy.methods import MLP, LDA

iris, info = datasets.iris()
iris.normalized()
iris.shuffle()
X, Y = iris[:' petal width'], iris['Iris-setosa':]

mlp = MLP()
mlp.fit(X[:120], Y[:120], 5000)
mlp.performance(X[120:], Y[120:])
mlp.report.show()
mlp.plot_error()


class1 = iris.select('Iris-setosa == 1')[:' petal width']
class2 = iris.select('Iris-versicolor == 1')[:' petal width']
class3 = iris.select('Iris-virginica == 1')[:' petal width']

lda = LDA()
lda.fit(class1, class2, class3)
lda.report.show()
lda.predict(X[120:])

lda = LDA(solve='fisher')
lda.fit(class1, class2, class3)
lda.report.show()
lda.predict(X[120:])
