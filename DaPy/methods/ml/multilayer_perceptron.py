from math import sqrt
from random import randint
from time import clock, localtime

from DaPy.core import DataSet
from DaPy.core import Matrix as mat
from DaPy.core import is_math
from DaPy.core import LogInfo, LogWarn, LogErr
from DaPy.core.base import STR_TYPE
from DaPy.methods.functions import activators
from DaPy.methods.utils import engine2str, str2engine
from DaPy.methods.evaluator  import ConfuMat, Accuracy, Kappa

from .Layers import Dense, Input

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

CELL = u'\u25CF'.encode('utf-8')
ONE_CELL = u'\u2460'.encode('utf-8')


class MLP(object):
    '''
    MultiLayers Perceptron (MLP)

    Parameters
    ----------
    engine : data process library
        The opearting model which is used to calculate the active function.

    alpha : float (default=1.0)
        Initialized learning ratio

    beta : float (default=0.1)
        Initialized the adjusting ratio

    upfactor : float (default=1.05)

    downfactor : float (default=0.95)

    layer_list : list
        The list contains hidden layers.
        

    Examples
    --------
    >>> mlp = dp.MLP() # Initialize the multilayers perceptron model
    >>> mlp.create(3,1, [3, 3, 3]) # Create a new model with 3 hidden layers
     - Create structure: 3 - 3 - 3 - 3 - 1
    >>> mlp.train(data, target) # Fit your data
     - Start: 2018-7-1 14:47:16 Remain: 9.31 s	Error: 44.02%
        Completed: 10.00 	Remain Time: 8.57 s	Error: 7.64%
        Completed: 20.00 	Remain Time: 7.35 s	Error: 3.82%
        Completed: 29.99 	Remain Time: 6.35 s	Error: 2.65%
        Completed: 39.99 	Remain Time: 5.40 s	Error: 2.06%
        Completed: 49.99 	Remain Time: 4.48 s	Error: 1.70%
        Completed: 59.99 	Remain Time: 3.58 s	Error: 1.46%
        Completed: 69.99 	Remain Time: 2.68 s	Error: 1.28%
        Completed: 79.98 	Remain Time: 1.79 s	Error: 1.15%
        Completed: 89.98 	Remain Time: 0.90 s	Error: 1.04%
        Completed: 99.98 	Remain Time: 0.00 s	Error: 0.96%
     - Total Spent: 9.0 s	Error: 0.9578 %
    >>> mlp.performance(new_data) # Test the performance of model
     - Classification Correct: 98.2243%
    >>> mlp.predict_proba(your_data) # Predict your real task.
    '''
        
    def __init__(self, engine='numpy',  alpha=0.025,
                 beta=0.025, upfactor=1.05, downfactor=0.95, ):
        '''initialize a multilayers perceptron model

        Parameter
        ---------
        engine : str (default='numpy')
            The calculating engine for this model. Since the 20 times
            efficiency in mathematical calculating than DaPy, we highly
            recommend that you setup numpy as your engine.

        alpha : float (default=0.025)
            The learning rate baseline for automatic adjustment.

        beta : float (default=0.025)
            The fixed learning rate.

        upfactor : float (default=1.05)
            The automatic adjustment rate to speed up the convergence while it
            is in positive movement.

        downfactor : float (default=0.95)
            The automatic adjustment rate to slow down the diverge while it is
            in nagetive movement.

        '''
        self._upfactor = upfactor               # Upper Rate
        self._downfactor = downfactor           # Down Rate
        self._alpha = alpha                     # Learning Rate
        self._beta = beta                       # transforming Rate
        self._layers = []
        self._engine = str2engine(engine)      # which library for camputing
        self._activator = activators(self._engine)
        self._size = 0
        self._report = DataSet()

    @property
    def weight(self):
        '''Return a diction stored two weight matrix.
        '''
        weights = {}
        for layer in self._layers:
            weights[layer.__repr__()] = layer._weight
        return weights

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
        self._activator.engine = self._engine

    @property
    def report(self):
        return self._report

    @property
    def layers(self):
        return self._layers

    def __repr__(self):
        max_size_y = max([layer.shape[1] for layer in self._layers]) * 2
        size_x = [layer.__repr__() for layer in self._layers]
        print_col = list()
        for layer in self._layers:
            blank = max_size_y / layer.shape[1]
            normal = [blank * i for i in range(1, 1 + layer.shape[1])]
            bias = - (max_size_y - max(normal) + min(normal)) / 2 
            print_col.append([bias + pos for pos in normal])

        output = '  '.join(size_x) + '\n'
        output += '-' * len(output) + '\n'
        output += '   '.join([ONE_CELL.center(len(name)) for name in size_x[:-1]]) + '\n'
        for i in range(1, 1 + max_size_y):
            line = ''
            for j, layer in enumerate(self._layers):
                if i in print_col[j]:
                    line += CELL.center(len(size_x[j])) + '   '
                else:
                    line += ' '.center(len(size_x[j])) + '  '
            output += line + '\n'
        output += '---------------------------\n'
        output += 'Tips:' + CELL + ' represents the normal cell in layer; \n'
        output += '     ' + ONE_CELL + ' represents the automatically added offset cells.'
        return output

    def __getstate__(self):
        obj = self.__dict__.copy()
        del obj['_Error'], obj['_activator'], obj['_report']
        obj['_engine'] = engine2str(self._engine)
        return obj

    def __setstate__(self, dict):
        self._engine = str2engine(dict['_engine'])
        self._layers = dict['_layers']
        self._alpha = dict['_alpha']
        self._beta = dict['_beta']
        self._upfactor = dict['_upfactor']
        self._downfactor = dict['_downfactor']
        self._Error = []
        self._activator = activators(self._engine)
        self._report = DataSet()

    def add_layer(self, layer):
        self._layers.append(layer)
        self._size += 1
        
    def create(self, input_cell, output_cell, hidden_cell=None, func=None):
        ''' Create a new MLP model with multiable hiddent layers.

        Parameters
        ----------
        input_cell : int
            The dimension of your features.

        output_cell : int
            The dimension of your target.

        hidden_cell : int or list (default:None)
            The cells in hidden layer, defualt will create a 1 hidden layer MLP
            with a experience formula. If you want to build more than 1 hidden
            layer, you should input a numbers in list.

        func : str, str in list (default: None)
            The active function of each layer.

        Return
        ------
        info : the built up structure of this MLP.

        Example
        -------
        >>> mlp = MLP()
        >>> mlp.create(9, 4, [2, 2, 3], ['sigm', 'sigm', 'sigm', 'line'])
        ' - Create structure: 9 - 2 - 2 - 3 - 4'
        '''
        cells = self._check_cells(input_cell, hidden_cell, output_cell)
        funcs = self._check_funcs(func, len(cells) - 1)
        assert len(funcs) == len(cells) - 1, 'the number of activations and layers do not match.'
        
        self.add_layer(Input(input_cell))
        for i, (in_, out_, func_) in enumerate(zip(cells[:-1], cells[1:], funcs)):
            actfun = self._activator.get_actfunc(func_)
            self.add_layer(Dense(self._engine, in_, out_, actfun))
            
        LogInfo('Create structure: ' + ' - '.join(['%s:%d' % (layer, layer.shape[1]) for layer in self.layers]))

    def _check_cells(self, input_, hidden, output):
        if hidden is None:
            return [input_, int((input_ + output)**0.5) + randint(1,11), output]
        if isinstance(hidden, int) and hidden > 0:
            return [input_, hidden, output]
        if all(map(isinstance, hidden, [int] * len(hidden))):
            return [input_] + list(hidden) + [output]
        raise ValueError('hidden layer must be integer or list of integers')

    def _check_funcs(self, funcs, lenth_layer):
        if funcs is None:
            return ['tanh'] * (lenth_layer - 1) + ['sigm']
        return check_activations(funcs)
    
    def train(self, X, Y, train_time=500, verbose=True, mini_error=0.05):
        '''Fit your model

        Parameters
        ----------
        data : matrix/2-darray
            The feature matrix in your training dataset.

        target : matrix/2-darray
            The target matrix in your training dataset.

        train_time : int (default:5000)
            The number of epoch that trains the model.

        info : Bool (default:True)
            - True: print the information while training model.
            - False: do not print any information.

        plot : Bool (default:True)
            - True: draw the result of loss function.
            - False: do not draw any thing

        Return
        ------
        None
        '''
        X = mat(X)
        Y = mat(Y)                       # Target Matrix
        self._Error = [1,]                      # Mistakes Recorder
        mini_error = mini_error * 100
        n, p = X.shape
        start = clock()                          # Training Start

        for term in range(1, train_time + 1): # Make a Loop
            # Foreward Propagating
            results = self._foreward(X)
            # Back Propagation
            self._backward(results, Y)

            if self._Error[-1] < mini_error:
                LogInfo('Early stop with the target error rates.')
                break

            # show out information
            if verbose and term % 20 == 0:
                spent = clock() - start
                finish_rate = (term/(train_time+1.0))*100
                last = spent/(finish_rate/100) - spent
                LogInfo('Finished: %.1f\t'%finish_rate + '%' +\
                      'Rest: %.2fs\t'%last +\
                      'ME: %.2f'%self._Error[-1] + '%')
                      
            elif term == 5:
                spent = clock() - start
                finish_rate = (term/(train_time+1.0))*100
                last = spent/(finish_rate/100) - spent
                LogInfo('Remain: %.2fs\tInit Error: %.2f' % (last,self._Error[1])) 
                
        LogInfo('Total Spent: %.1f s\tFinal Error: %.4f'%(clock()-start,
                                                self._Error[-1]) + '%')

    def _foreward(self, data):
        output = [data, ]
        for i, layer in enumerate(self._layers):
            output.append(layer.propagation(output[-1]))
        return output

    def _backward(self, outputs, y_true):
        error = y_true - outputs[-1]
        r = self._engine.mean(self._engine.abs(error))
        self._Error.append(r * 100)
        self._alpha = self._autoadjustlr()
        for i, layer in enumerate(self._layers[::-1]):
            error = layer.backward(outputs[self._size - i - 1],
                                   error, outputs[self._size - i],
                                   self._alpha, self._beta)

    def _autoadjustlr(self):
        if self._Error[-1] > self._Error[-2]:
            return self._alpha * self._upfactor
        return self._alpha * self._downfactor

    def fit(self, X, Y, train_time=500, hidden_cells=None,
            func=None, mini_error=0.05, verbose=False):
        '''create a new model and train it.

        This model will help you create a model which suitable for this
        dataset and then train it. It will use the function self.create()
        at the first place, and call self.train() function following.
        '''
        X, Y = mat(X), mat(Y)
        self.create(X.shape.Col, Y.shape.Col, hidden_cells, func)
        self.train(X, Y, train_time, verbose, mini_error)
                                        
    def predict_proba(self, data):
        '''
        Predict your own data with fitting model

        Paremeter
        ---------
        data rix
            The new data that you expect to predict.

        Return
        ------
        Matrix: the predict result of your data.
        '''
        return mat(self._foreward(mat(data))[-1])

    def save(self, addr):
        '''Save your model to a .pkl file
        '''
        if isinstance(fp, STR_TYPE):
            fp = open(fp, 'wb')
        try:
            pkl.dump(self, open(addr, 'wb'))
        finally:
            fp.close()

    def load(self, fp):
        if isinstance(fp, STR_TYPE):
            fp = open(fp, 'rb')
        try:
            self.__setstate__(pkl.load(fp).__getstate__())
        finally:
            fp.close()

    def plot_error(self):
        '''use matplotlib library to draw the error curve during the training.
        '''
        try:
            import matplotlib.pyplot as plt
            plt.title('%d Layers MLP Model Training Result'%(self._size - 1))
            plt.plot(self._Error[1:])
            plt.ylabel('Error %')
            plt.xlabel('Epoch')
            plt.show()
        except ImportError:
            raise ImportError('DaPy uses `matplotlib` to draw picture, try: pip install matplotlib.')

