from DaPy import datasets
from DaPy import methods

data, info = datasets.iris()

data.get_dummies('class')
data.drop('class', axis=1)
data.shuffle()
data.normalized()
X, Y = data[:'petal width'], data['class=setosa':]

my_clf = methods.MLP()
my_clf.fit(X[:100], Y[:100])
my_clf.plot_error()
methods.Performance(clf, X[100:], Y[100:], mode'clf')
my_clf.save('my_clf.pkl')

