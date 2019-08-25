from DaPy import datasets
from DaPy.methods.classifiers import MLPClassifier
from DaPy.methods.evaluator import Performance

data, info = datasets.iris()
data.info
data = data.shuffle().normalized()
X, Y = data[:'petal width'], data['class']

my_clf = MLPClassifier().fit(X[:100], Y[:100])
my_clf.plot_error()
Performance(my_clf, X[100:], Y[100:], mode='clf')
my_clf.save('my_clf.pkl')

from cPickle import load
mlp = load(open('my_clf.pkl', 'r'))
