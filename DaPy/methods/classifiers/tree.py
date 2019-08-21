from collections import Counter
from math import log
from DaPy import SeriesSet, Series
from copy import copy, deepcopy
from pprint import pformat

class DecisionTreeClassifier(object):
    '''Implement of decision tree with C4.5 algorithm'''
    def __init__(self, max_depth=None):
        self._feature_name = []
        self._class_name = []
        self._shannon = {}
        self._root = {}

    def __getitem__(self, key):
        return self._root[key]

    def __repr__(self):
        return pformat(self._root)
    
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
        for key, value in labels.items():
            prob = value / size
            shannon -= prob * log(prob, 2)
        return shannon
    
    def _get_best_feature(self, X, Y, tol_gain_ratio=0.0001):
        size = float(X.shape[0])
        base_entropy = self._cal_shannon(Y)
        best_gain_ratio, best_feature = tol_gain_ratio, None
        
        for feature in X.columns:
            if feature == '__target__':
                continue
            
            current_entropy, current_iv = 0.0, 0.00001
            for feature_value, subset_feature in X.iter_groupby(feature):
                prob = subset_feature.shape[0] / size
                subset_shannon = self._cal_shannon(subset_feature['__target__'])
                current_entropy -= prob * subset_shannon
                current_iv -= prob * log(prob, 2)
                
            gain_ratio = (base_entropy - current_entropy) / abs(current_iv)
            if gain_ratio > best_gain_ratio:
                best_gain_ratio, best_feature = gain_ratio, feature
                
        return best_feature, best_gain_ratio

    def _create_tree(self, X, Y, feature_name):
        if len(set(Y)) == 1:
            return Y[0]

        most_common_Y = Counter(Y).most_common()[0][0]
        
        if X.shape[1] == 1 and X.columns[0] == '__target__':
            return most_common_Y
        
        best_feature, best_info_gain = self._get_best_feature(X, Y)
        if best_feature is None:
            return most_common_Y

        feature_name.remove(best_feature)
        self._shannon.setdefault(best_feature,
                                 [best_info_gain, 1])
        subColumn = X.columns
        subColumn.remove(best_feature)
        subX = X[subColumn]
        subtree = {'???': most_common_Y}
        feature_column = X[best_feature]

        for value in set(feature_column):
            equal_value_index = [i for i, _ in enumerate(feature_column) if _ == value]
            x = subX[equal_value_index]
            y = subX[equal_value_index]['__target__']
            subtree[value] = self._create_tree(x, y, deepcopy(feature_name))
        return {best_feature: subtree}

    def fit(self, X, Y):
        X, Y = SeriesSet(X), Y
        feature_name = X.columns
        # we combine the X and Y in a dataset
        X.append_col(Y, '__target__')

        self._feature_name = copy(feature_name)
        self._class_name = set(['Class=%s' % value for value in Y])
        self._root = self._create_tree(X, Y, feature_name)
        return self

    def predict_once(self, row):
        node = self._root
        for i in range(self.n_features):
            feature = list(node.keys())[0]
            compare_value = row[self._feature_name.index(feature)]
            for value, subnode in node[feature].items():
                if compare_value == value:
                    if isinstance(subnode, dict) is False:
                        return subnode
                    node = subnode
                    break
            else:
                return node[feature]['???']
        return self._root['???']

    def predict(self, X):
        assert X.shape[1] == self.n_features
        return Series(self.predict_once(row) for row in X)

    def export_graphviz(self, outfile=None):
        global nodeNum, doc
        doc = ''
        doc += 'digraph Tree{\n'
        doc += 'node [shape=box, style="rounded", fontname=helvetica] ;\n'
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
            if len(tree) > 0:
                firstStr = list(tree.keys())[0]
                plotNode(firstStr, nodeNum)
                plotRelation(father, nodeNum, relation)
                subTree = tree[firstStr]
                fatherNode = nodeNum
                for key, value in subTree.items():
                    nodeNum += 1
                    if isinstance(value, dict):
                        plotTree(value, fatherNode, key)
                    elif key != '???':
                        plotNode(value, nodeNum)
                        plotRelation(fatherNode, nodeNum, key)
        
        plotTree(self._root, '', 'ROOT')
        doc += '}'
        if hasattr(outfile, 'write') is False:
            return doc
        outfile.write(doc)

    def most_important_feature(self, top='all'):
        assert isinstance(top, int) or str(top).lower() == 'all'
        if str(top).lower() == 'all':
            top = self.n_features
        shannons = [(key, total/times) for key,(total,times) in self._shannon.items()]
        return sorted(shannons, key=lambda x: x[1], reverse=True)[:top]
        

if __name__ == '__main__':
    test_data = SeriesSet({
        'color': ['green', 'dark', 'dark', 'green', 'white',
                  'green', 'dark', 'dark', 'dark', 'green',
                  'white', 'white', 'green', 'white', 'dark', 'white', 'green'],
        'root': ['fully rolled', 'fully rolled', 'fully rolled', 'fully rolled', 'fully rolled', 'slightly rolled ','slightly rolled ', 'slightly rolled ',
                 'slightly rolled ', 'straight', 'slightly rolled ', 'fully rolled', 'slightly rolled ', 'slightly rolled ','slightly rolled ', 'fully rolled', 'fully rolled'],
                 
        'response': ['boom', 'low', 'boom', 'low', 'boom', 'boom', 'boom', 'boom',
                 'low', 'clear', 'clear', 'boom', 'boom', 'low', 'boom', 'boom', 'low'],
        'texture': ['clear'] * 6 + ['slightly paste', 'clear', 'slightly paste', 'clear', 'paste', 'paste',
                 'slightly paste', 'slightly paste', 'clear', 'paste', 'slightly paste'],
        'navel': ['dent'] * 5 + ['slightly dent'] * 4 + ['flat'] * 3 + ['dent'] * 2 + \
                ['slightly dent', 'flat', 'slightly dent'],
        'touch': ['hard slip'] * 5 + ['soft sticky ', 'soft sticky ', 'hard slip', 'hard slip', 'soft sticky ', 'hard slip',
                 'soft sticky ', 'hard slip', 'hard slip', 'soft sticky ', 'hard slip', 'hard slip'],
        'good': ['good'] * 8 + ['bad'] * 9})
    test_data = test_data[['color', 'root', 'response',  'texture',  'navel', 'touch', 'good']]
    print(test_data.show())
    X, Y = test_data[:'touch'], test_data['good']
    mytree = DecisionTree()
    mytree.fit(X, Y)
    import pydotplus
    graph = pydotplus.graph_from_dot_data(mytree.export_graphviz())
    graph.write_pdf(r'C:\Users\JacksonWoo\Desktop\boston.pdf')
    print(mytree)
    print(mytree.predict(X))
    print(mytree.predict(SeriesSet([['red', 'red', 'clear', 'None', 'None', 'soft sticky']])))












        
        
