from collections import OrderedDict, Counter, defaultdict
from math import log
from DaPy import mat, concatenate, SeriesSet
from copy import copy, deepcopy

class DecisionTree(object):
    def __init__(self, max_depth=None):
        self._feature_name = []
        self._class_name = []
        self._shannon = {}
        self._root = {}

    def __getitem__(self, key):
        return self._root[key]
    
    @property
    def n_features(self):
        return len(self._feature_name)

    @property
    def n_outputs(self):
        return len(self._class_name)

    @property
    def root(self):
        return self._root

    def items(self):
        for key,value in self._root.items():
            yield key, value

    def keys(self):
        for key in self._root.keys():
            yield key

    def _cal_shannon(self, data):
        size = float(len(data))
        labels = Counter(data)
        shannon = 0.0
        for key,value in labels.items():
            prob = value / size
            shannon -= prob * log(prob, 2)
        return shannon
    
    def _get_best_feature(self, X, Y, minInfo=0.0):
        size = float(X.shape[0])
        baseEntropy = self._cal_shannon(Y)
        bestInfoGain, bestFeature = minInfo, X.shape[1] - 1
        for index in range(X.shape[1]):
            feature = X[X.columns[index]]
            values = set(feature)
            thisEntropy = 0.0
            for value in values:
                subY = [Y[i] for i,row in zip(Y.index, X) if row[index] == value]
                prob = len(subY) / size
                thisEntropy += prob * self._cal_shannon(subY)
            infoGain = baseEntropy - thisEntropy
            if infoGain > bestInfoGain:
                bestInfoGain, bestFeature = infoGain, index
        return bestFeature, infoGain

    def _create_tree(self, X, Y, feature_name):
        if len(set(Y)) == 0:
            return self._class_name[Y[0]]
        if X.shape[1] == 0:
            value = Counter(Y).most_common()[0][0]
            if isinstance(value, int) or str(value).isdigit():
                value = self._class_name[int(value)]
            return value
        
        bestFeature, bestShannon = self._get_best_feature(X, Y)
        bestLabel = feature_name.pop(bestFeature)
        shannonHistory = self._shannon.setdefault(bestLabel, [0.0, 0])
        shannonHistory[0] += bestShannon
        shannonHistory[1] += 1
        subColumn = X.columns
        subColumn.pop(bestFeature)
        subX = X[subColumn]
        subtree = {}

        for value in set(X[X.columns[bestFeature]]):
            equal_value_index = [i for i,_value in enumerate(X[X.columns[bestFeature]]) if _value == value]
            x, y = subX[equal_value_index], Y[[Y.index[i] for i in equal_value_index]]
            subtree[value] = self._create_tree(x, y, deepcopy(feature_name))
        return {bestLabel: subtree}


    def fit(self, X, Y, class_name=None, feature_name=None):
        if feature_name is None:
            feature_name = ['Feature(%d)' % i for i in range(X.shape[1])]
        if class_name is None:
            class_name = ['Class(%d)' % i for i in range(len(set(Y)))]
        X = SeriesSet(X, feature_name)
        self._feature_name = copy(feature_name)
        self._class_name = copy(class_name)
        self._root = self._create_tree(X, Y, feature_name)

    def predict_once(self, row):
        node = self._root
        for i in range(self.n_features):
            feature = list(node.keys())[0]
            compare_value = row[self._feature_name.index(feature)]
            for value,subnode in node[feature].items():
                if compare_value == value:
                    if isinstance(subnode, dict) is False:
                        return subnode
                    node = subnode
        else:
            raise ValueError('cannot get the result')

    def predict(self, X):
        assert X.shape[1] != self.n_features
        return [self.predict_once(row) for row in X]

    def export_graphviz(self, outfile=None):
        global nodeNum, doc
        doc = ''
        doc += 'digraph Tree{\n'
        doc += 'node [shape=box, style="rounded", color="black", fontname=helvetica] ;\n'
        doc += 'edge [fontname=helvetica] ;\n'
        nodeNum = 0
        
        def plotNode(string, num):
            global doc
            doc += '%d [label="%s"];\n' % (num, string)

        def plotRelation(father, current, string):
            global doc
            if father != '':
                doc += '%s -> %d [headlabel="%s"];\n' % (father, current, string)
            
        def plotTree(tree, father, relation):
            global nodeNum
            firstStr = list(tree.keys())[0]
            plotNode(firstStr, nodeNum)
            plotRelation(father, nodeNum, relation)
            subTree = tree[firstStr]
            fatherNode = nodeNum
            for key,value in subTree.items():
                nodeNum += 1
                if isinstance(value, dict):
                    plotTree(value, fatherNode, key)
                else:
                    plotNode(value, nodeNum)
                    plotRelation(fatherNode, nodeNum, key)
        
        plotTree(self._root, '', 'ROOT')
        doc += '}'
        if hasattr(outfile, 'write') is False:
            return doc
        outfile.write(doc)

    @property
    def most_important_feature(self, top='all'):
        assert isinstance(top, int) or str(top).lower() == 'all'
        if str(top).lower() == 'all':
            top = self.n_features
        shannons = [(key, total/times) for key,(total,times) in self._shannon.items()]
        return sorted(shannons, key=lambda x: x[1], reverse=True)[:top]
        

if __name__ == '__main__':
    with open('lenses.txt') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
        lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate', 'Target']
        lensesLabelsTem = SeriesSet(lenses, lensesLabels)
    X, Y = lensesLabelsTem[:'tearRate'], lensesLabelsTem.Target
    mytree = DecisionTree()
    mytree.fit(X, Y, ['Yes', 'No', 'Unsure'], lensesLabels
    graph = pydotplus.graph_from_dot_data(mytree.export_graphviz())
    graph.write_pdf(r'C:\Users\JacksonWoo\Desktop\boston.pdf')














        
        
