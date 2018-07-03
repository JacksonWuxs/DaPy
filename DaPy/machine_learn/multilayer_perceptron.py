from random import randint
from DaPy.core import Matrix, is_math
from time import clock, localtime 
from warnings import warn
from .Layers import Fully_Connected_Layer
from .contents import CELL, ONE_CELL
from .active_functions import functions

try:
    import cPickle as pkl
except ImportError:
    import Pickle as pkl
    

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
    >>> mlp.predict(your_data) # Predict your real task.
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
        
        self._upfactor = upfactor     # Upper Rate
        self._downfactor = downfactor # Down Rate
        self._alpha = alpha           # Learning Rate
        self._beta = beta             # transforming Rate

        if engine.lower() == 'numpy':
            import numpy as f
        elif engine.lower() == 'dapy':
            import DaPy as f
        else:
            raise ValueError('do not support engine as %s'%engine)
        self._engine = f              # which library we are using

    @property
    def weight(self):
        '''Return a diction stored two weight matrix.
        '''
        weights = {}
        for layer in self._layer_list:
            weights[layer.__repr__()] = layer._weight
        return weights

    @property
    def engine(self):
        '''Return the calculating tool that you are using
        '''
        return str(self._engine).split("'")[1]

    @engine.setter
    def engine(self, value):
        '''Reset the calculating library (DaPy or Numpy)
        '''
        if value.lower() == 'numpy':
            import numpy as engine
            engine.seterr(divide='ignore', invalid='ignore')
            
        elif value.lower() == 'dapy':
            import DaPy as engine
            
        else:
            raise ValueError('do not support engine as %s, "DaPy" or "numpy" only.'%value)

        self._engine = engine
        for layer in self._layer_list:
            layer.engine = engine

    def __repr__(self):
        max_size_y = max([layer.shape[1] for layer in self._layer_list]) * 2
        size_x = [layer.__repr__() for layer in self._layer_list]
        print_col = list()
        for layer in self._layer_list:
            blank = max_size_y / layer.shape[1]
            normal = [blank * i for i in range(1, 1 + layer.shape[1])]
            bias = - (max_size_y - max(normal) + min(normal)) / 2 
            print_col.append([bias + pos for pos in normal])

        output = '  '.join(size_x) + '\n'
        output += '-' * len(output) + '\n'
        output += '   '.join([ONE_CELL.center(len(name)) for name in size_x[:-1]]) + '\n'
        for i in range(1, 1 + max_size_y):
            line = ''
            for j, layer in enumerate(self._layer_list):
                if i in print_col[j]:
                    line += CELL.center(len(size_x[j])) + '   '
                else:
                    line += ' '.center(len(size_x[j])) + '  '
            output += line + '\n'
        output += '---------------------------\n'
        output += 'Tips:' + CELL + ' represents the normal cell in layer; \n'
        output += '     ' + ONE_CELL + ' represents the automatically added offset cells.'
        return output
        
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

        func : str (default: None)
            The active function of each layer.

        Return
        ------
        info : the built up structure of this MLP.

        Example
        -------
        >>> mlp = MLP()
        >>> mlp = mlp.create(9, 4, [2, 2, 3], ['sigm', 'sigm', 'sigm', 'line'])
        ' - Create structure: 9 - 2 - 2 - 3 - 4'
        '''
        # the layer cells in the list
        cells = [input_cell]
        if not hidden_cell:
            # Formula comes from experience
            cells.append(int((input_cell + output_cell)**0.5) + randint(1,11))
        elif isinstance(hidden_cell, int) and hidden_cell > 0:
            cells.append(hidden_cell)
        elif all(map(isinstance, hidden_cell, [int] * len(hidden_cell))):
            cells.extend(list(hidden_cell))
        else:
            raise AttributeError('unrecognized symbol `%s`'%hidden_cell)
        cells.append(output_cell)

        funcs = ['line']
        if not func:
            funcs += ['sigm'] * (len(cells) - 1)
            
        elif isinstance(func, list):
            for every in func:
                if every not in functions:
                    raise ValueError('invalid functions symbol as `%s`, '%every +\
                                     'tanh, sigm, relu, softmax, radb and line.' )
                funcs.append(every)
        if len(funcs) != len(cells):
            raise ValueError('the number of functions and layers do not match.')
        
        self._layer_list = [Fully_Connected_Layer(self._engine, cells[0], cells[0], 'Input_Layer', 'line')]
        for i, cell in enumerate(cells[1:-1]):
            new_layer = Fully_Connected_Layer(self._engine, cells[i], cell, i + 1, funcs[i + 1])
            self._layer_list.append(new_layer)
        self._layer_list.append(Fully_Connected_Layer(self._engine, cells[-2], cells[-1],
                                      'Output_Layer', funcs[-1]))
        self._size = len(self._layer_list)
        return ' - Create structure: ' + ' - '.join(map(str, cells))
            
    def train(self, data, target, train_time=500, info=True):
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
        X = Matrix(data)
        Y = Matrix(target)                       # Target Matrix
        self._Error = [1, ]                      # Mistakes Recorder

        start = clock()  # Training Start
        for term in range(1, train_time + 1): # Make a Loop
            # Foreward Propagating
            results = self._foreward(X)
            # Back Propagation
            self._backward(results, Y)

            # show out information
            if info and term%((train_time+1)//10)==0:
                spent = clock() - start
                finish_rate = (term/(train_time+1.0))*100
                last = spent/(finish_rate/100) - spent
                print('    Completed: %.2f \t'%finish_rate +\
                      'Remain Time: %.2f s\t'%last +\
                      'Error: %.2f'%self._Error[-1] + '%')
                      
            elif term == 10:
                spent = clock() - start
                finish_rate = (term/(train_time+1.0))*100
                last = spent/(finish_rate/100) - spent
                print(' - Start: ' +\
                      '-'.join(map(str, localtime()[:3])) + ' ' +\
                      ':'.join(map(str, localtime()[3:6])) + ' ' +\
                      'Remain: %.2f s\t'%last +\
                      'Error: %.2f'%self._Error[-1] + '%') 
                
        print(' - Total Spent: %.1f s\tError: %.4f'%(clock()-start,
                                                self._Error[-1]) + '%')

    def _foreward(self, data):
        output = [data, ]
        for i, layer in enumerate(self._layer_list):
            output.append(layer.propagation(output[-1]))
        return output

    def _backward(self, outputs, y_true):
        error = y_true - outputs[-1]
        r = self._engine.mean(abs(error))
        self._Error.append(r * 100)
        self._alpha = self.__autoadjustlr()
        for i, layer in enumerate(self._layer_list[::-1]):
            error = layer.backward(outputs[self._size - i - 1],
                                   error, outputs[self._size - i],
                                   self._alpha, self._beta)

    def __autoadjustlr(self):
        if self._Error[-1] > self._Error[-2]:
            return self._alpha * self._upfactor
        return self._alpha * self._downfactor
                                        
    def predict(self, data):
        '''
        Predict your own data with fitting model

        Paremeter
        ---------
        data : matrix
            The new data that you expect to predict.

        Return
        ------
        Matrix: the predict result of your data.
        '''
        return Matrix(self._foreward(Matrix(data))[-1])

    def topkl(self, addr):
        '''Save your model to a .pkl file
        '''
        with open(addr, 'w') as f:
            pkl.dump(dict(weight=[layer._weight for layer in self._layer_list],
                          func=[layer.function for layer in self._layer_list]), f)

    def show_error(self):
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
            warnings.warn('DaPy uses `matplotlib` library to draw picture.')
            
    def readpkl(self, addr):
        '''Load your model from a .pkl file
        '''
        with open(addr) as f:
            data = pkl.load(f)
        for i, weight in enumerate(data['weight']):
            layer = Layer(self._engine, 0, 0, i, data['func'][i])
            layer._weight = weight
            self._layer_list.append(layer)

    def performance(self, data, target, mode='clf'):
        if len(data) != len(target):
            raise IndexError("the number of target data is not equal to variable data")

        result, target = self.predict(data), Matrix(target)
        if mode == 'clf':
            error = 0
            if all(map(is_math, target)):
                for i in range(len(data)):
                    if target[i][0] > 0.5 and result[i][0] < 0.5:
                        error += 1
                    if target[i][0] < 0.5 and result[i][0] > 0.5:
                        error += 1
            else:
                for t, r in zip(target, result):
                    if t.index(max(t)) != r.index(max(r)):
                        error += 1
            return ' - Classification Correct: %.4f'%(\
                (1 - float(error)/len(target))*100) + '%'
        
        elif mode == 'reg':
            return ' - Regression MSE: %.4f'%self._f.mean(\
                Matrix(target - result) ** 2)
        
        else:
            raise ValueError("`mode` supports `clf` or `reg` only.")
