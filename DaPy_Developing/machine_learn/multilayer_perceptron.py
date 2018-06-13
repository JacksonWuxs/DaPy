from random import randint
from DaPy.core import Matrix, Table
from time import clock
from warnings import warn
from copy import deepcopy
 
try:
    import cPickle as pkl
except ImportError:
    import Pickle as pkl
    

class MLP(object):
    '''
    MultiLayers Perceptron (MLP)

    Parameters
    ----------
    data : DaPy.Matrix or numpy.array (default=None)
        The variable part of the training dataset.

    target : DaPy.Matrix or numpy.array (default=None)
        The Target part of the training dataset.

    unumpy : Bool (default=True)
        Whether use numpy.

    alpha : float (default=1.0)
        Initialized learning ratio

    beta : float (default=0.1)
        Initialized the adjusting ratio

    momentun : float (default=0.1)
        Momentum adjustment factor

    upfactor : float (default=1.05)
        

    Examples
    --------
    >>> mlp = dp.MLP() # Initialize the multilayers perceptron model
    >>> mlp.create(3,1) # Create a new model
    >>> mlp.train(data, target) # Fit your data
    >>> mlp.test(new_data) # Test the performance of model
    >>> mlp.predict(your_data) # Predict your real task.
    '''
        
    def __init__(self, unumpy=True,  alpha=1,
                 beta=0.1, move=0.1, upfactor=1.05, downfactor=0.95, ):
        
        self._upfactor = upfactor     # Upper Rate
        self._downfactor = downfactor # Down Rate
        self._alpha = alpha           # Learning Rate
        self._beta = beta             # transforming Rate
        self._move = move             # Movement Element
        self._numpy = unumpy          # which library we are using
        if unumpy:
            try:
                import numpy as f
            except ImportError:
                import DaPy as f
                warn('could not find Numpy in your computer. Using DaPy'+\
                             'functions now.')
        else:
            import DaPy as f
        self._f = f

    @property
    def weight(self):
        '''Return a diction stored two weight matrix.
        '''
        return {'L1': self._weight1,
                'L2': self._weight2}

    @property
    def unumpy(self):
        '''Return the calculating tool that you are using
        '''
        return self._numpy

    @unumpy.setter
    def unumpy(self, value):
        '''Reset the calculating library (DaPy or Numpy)

        Parameter
        ---------
        Value : Bool
            - True: using numpy.
            - False: using DaPy.

        Return
        ------
        None

        Examples
        --------
        >>> mlp.unumpy = True
        >>> mlp.unumpy = False
        '''
        if value:
            self._numpy = True
            try:
                import numpy as f
            except ImportError:
                import DaPy as f
                warn('could not find Numpy in your computer. Using DaPy'+\
                             'functions now.')
        else:
            import DaPy as f
            self._numpy = False
        self._f = f
        
    def create(self, input_cell, output_cell, hidden_cell='RANDOM'):
        ''' Create a new MLP model with 1 hiddent layer.

        Parameters
        ----------
        input_cell : int
            The dimension of your features.

        output_cell : int
            The dimension of your target.

        hidden_cell : int (default:'RANDOM')
            The cells in hidden layer, defualt will random create with
            a experience formula.

        Return
        ------
        None
        '''
        self._weight1 = Matrix()    # Weight Matrix in Input Layer
        self._weight2 = Matrix()    # Weight Matrix in Hiddent Layer

        n, l = input_cell, output_cell

        if str(hidden_cell).upper() == 'RANDOM':
            # Formula comes from experience
            m = int((n+l)**0.5) + randint(1,11) 
        elif str(hidden_cell).isdigit():
            m = hidden_cell
        else:
            raise AttributeError('unrecognized symbol <%s>'%hidden_cell)
            
            
        self._weight1.make_random(n + 1, m)
        self._weight1 = self._weight1*2 - 1
        
        self._weight2.make_random(m, l)
        self._weight2 = self._weight2*2 - 1     

        print ' - Create structure: %d - %d - %d'%(n + 1, m, l)
    
    def train(self, data, target, train_time=5000, info=True, plot=True):
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
        self._data = self.__add_bias(data)
        self._result = target           # Target Matrix
        self._Error = [1, ]           # Mistakes Recorder
        self.__transform()


        start = clock()  # Training Start
        for term in range(1, train_time + 1): # Make a Loop
            # Foreword Propagating
            out1 = self.__sigmoid(self._data.dot(self._weight1))
            out2 = self.__sigmoid(out1.dot(self._weight2))
            # Back Propagation
            self.__updata_weight(out2, out1)
            
            # show out information
            if info and term%((train_time+1)//10)==0:
                spent = clock() - start
                finish_rate = (term/(train_time+1.0))*100
                last = spent/(finish_rate/100) - spent
                print '    Completed: %.2f \t'%finish_rate +\
                      'Remain Time: %.2f s\t'%last +\
                      'Error: %.2f'%self._Error[-1] + '%'
                      
            elif info and term == 1:
                print ' - Start Training...'
                print ' - Initial Error: %.2f'%self._Error[-1],'%'
                
        self._data, self.result = None, None
                
        print ' - Total Spent: %.1f s\tError: %.4f'%(clock()-start,
                                                self._Error[-1]), '%'
        if plot:
            try:
                import matplotlib.pyplot as plt
                plt.title('Multi-Layer Perceptron Training')
                plt.plot(self._Error[1:])
                plt.ylabel('Error %')
                plt.xlabel('Train Time')
                plt.show()
            except ImportError:
                warnings.warn('could not find matplotlib in your computer.')
    
    def __updata_weight(self, out2, out1):
        # calculate errors
        L2_error = self._result - out2 
        L2_delta = L2_error * self.__sigmoid(out2,True)
        L1_error = L2_delta.dot(self._weight2.T)
        L1_delta = L1_error * self.__sigmoid(out1,True)
        change_L2 = out1.T.dot(L2_delta)
        change_L1 = self._data.T.dot(L1_delta)
        
        # record the error
        r = self._f.mean(abs(L2_error))/self._f.mean(abs(self._result))
        self._Error.append(100*r)

        # adjust the weight matrix 
        self._alpha = self.__autoadjustlr()
        self._weight1 += change_L1*self._alpha + change_L1*self._move
        self._weight2 += change_L2*self._alpha + change_L2*self._move

    def __autoadjustlr(self):
        if self._Error[-1] > self._Error[-2]:
            return self._alpha * self._upfactor
        else:
            return self._alpha * self._downfactor

    def __sigmoid(self, x, diff=False):
        if diff:
            return x*(x*(-1)+1)
        return 1.0/(self._f.exp(x*(-self._beta))+1)

    def __transform(self):
        if self._numpy:
            self._data = self._f.array(self._data)
            self._result = self._f.array(self._result)
            self._weight1 = self._f.array(self._weight1)
            self._weight2 = self._f.array(self._weight2)
        else:
            self._data = Matrix(self._data)
            self._result = Matrix(self._result)
            self._weight1 = Matrix(self._weight1)
            self._weight2 = Matrix(self._weight2)

    def __add_bias(self, data):
        if self._numpy:
            append_list = self._f.array([[1 for i in range(len(data))]])
            return self._f.c_[data, append_list.T]
        else:
            _data = Table(deepcopy(data), None)
            _data.append_col('New_', [1 for i in range(len(_data))])
            return Matrix(_data)
                                        
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
        data = self.__add_bias(data)
        out1 = self.__sigmoid(self._f.dot(data, self._weight1))
        out2 = self.__sigmoid(self._f.dot(out1, self._weight2))
        return Matrix(out2)

    def topkl(self, addr):
        '''Save your model to a .pkl file
        '''
        with open(addr, 'w') as f:
            if isinstance(self._weight1, Matrix):
                pkl.dump([self._weight1._matrix,
                          self._weight2._matrix], f)
            else:
                pkl.dump([self._weight1, self._weight2], f)

    def readpkl(self, addr):
        '''Load your model from a .pkl file
        '''
        with open(addr) as f:
            data = pkl.load(f)
            self._weight1 = data[0]
            self._weight2 = data[1]

        if self._numpy:
            self._weight1 = self._f.array(self._weight1)
            self._weight2 = self._f.array(self._weight2)
        else:
            self._weight1 = Matrix(self._weight1)
            self._weight2 = Matrix(self._weight2)

    def test(self, data, target, mode='CLASSIFICATION'):
        if len(data) != len(target):
            raise IndexError("target data is not equal to variable data")
        
        result = self.predict(data)
        if mode == 'CLASSIFICATION':
            error = 0
            if len(target[0]) == 1:
                for i in xrange(len(data)):
                    if target[i][0] > 0.5 and result[i][0] < 0.5:
                        error += 1
                    if target[i][0] < 0.5 and result[i][0] > 0.5:
                        error += 1
            else:
                for i in xrange(len(data)):
                    if target[i].index(max(target[i])) != result[i].index(max(result[i])):
                        error += 1
            return 'Classification Correct: %.4f'%((1 - float(error)/len(target))*100) + '%'
        
        elif mode == 'REGRESSION':
            return self._f.mean(abs(L2_error))/self._f.mean(abs(self._result))
        
        else:
            raise ValueError("unrecognized mode: %s"%str(mode))
