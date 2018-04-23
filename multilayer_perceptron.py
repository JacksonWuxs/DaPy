import random
import structure as s
import time
import warnings
try:
    import cPickle as pkl
except ImportError:
    import Pickle as pkl

class MLP(object):
    def __init__(self, data=None, target=None, unumpy=True, name='MyMLP'):
        self._data = data             # Input Matrix
        self._result = target         # Target Matrix
        self._name = name             # Name of MLP
        self._numpy = unumpy          # which library we are using
        if unumpy:
            try:
                import numpy as f
            except ImportError:
                import formulas as f
                warning.warn('could not find Numpy in your computer. Using DaPy'+\
                             'functions now.')
        else:
            import formulas as f
        self._f = f
        
    def create(self, cells='RANDOM'):
        ''' We will create a new MLP model with 1 hiddent layer.
        '''
        self._weight1 = s.Matrix()    # Weight Matrix in Input Layer
        self._weight2 = s.Matrix()    # Weight Matrix in Hiddent Layer

        if self._data and self._result:
            n, l = len(self._data[0]), len(self._result[0])
        else:
            raise ValueError("couldn't create a new MLP without any train data")
        
        if str(cells).upper() != 'RANDOM':
            m = self.cell 
        else:
            m = int((n+l)**0.5) + random.randint(1,11) # Random cells in HL
            
        self._weight1.make_random(n, m)
        self._weight1 = self._weight1*2 - 1
        
        self._weight2.make_random(m, l)
        self._weight2 = self._weight2*2 - 1

        if self._numpy:
            self._data = self._f.array(self._data)
            self._result = self._f.array(self._result)
            self._weight1 = self._f.array(self._weight1)
            self._weight2 = self._f.array(self._weight2)      

        return "Create with %d hidden Layer's cells"%m
    
    def train(self, train_time=50000, error=10, alpha=1, beta=0.01,
              info=True, move=0.1, upfactor=1.05, downfactor=0.95, plot=True):
        
        self._upfactor = upfactor     # Upper Rate
        self._downfactor = downfactor # Down Rate
        self._alpha = alpha           # Learning Rate
        self._beta = beta             # transforming Rate
        self._error = error           # Max Mistake Time
        self._Error = [1, ]           # Mistakes Recorder
        self._move = move             # Movement Element
        
        start = time.clock()  # Training Start
        for term in xrange(1, train_time+1): # Make a Loop
            # Foreword Propagating
            out1 = self.__sigmoid(self._f.dot(self._data, self._weight1))
            #print out1
            out2 = self.__sigmoid(self._f.dot(out1, self._weight2))
            #print out2
            
            # Back Propagation
            self.__updata_weight(out2, out1)
            
            # show out information
            if info and term%((train_time+1)//10)==0:
                spent = time.clock() - start
                finish_rate = (term/(train_time+1.0))*100
                last = spent/(finish_rate/100) - spent
                print 'Completed: %.2f \tRemaining Time: %.2f s'%(finish_rate,
                                                              last)
            elif info and term == 1:
                print '%s Start Training...'%self._name
                print 'Initial Error: %.2f'%self._Error[-1],'%'
                
        print 'Total Spent: %.1f s\tErrors: %f'%(time.clock()-start,
                                                self._Error[-1]), '%'
        if plot:
            try:
                import matplotlib.pyplot as plt
                plt.title('Multi-Layer Perceptron Training')
                plt.plot(self._Error[1:])
                plt.ylabel('Error %')
                plt.show()
            except:
                warnings.warn('could not find matplotlib in your computer.')
    
    def __updata_weight(self, out2, out1):
        # calculate errors
        L2_error = self._result - out2 
        L2_delta = L2_error * self.__sigmoid(out2,True)
        L1_error = self._f.dot(L2_delta, self._weight2.T) 
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
        return (self._f.exp(x*(-self._beta))+1)**(-1)
        
    def predict(self, data):
        out1 = self.__sigmoid(self._f.dot(data, self._weight1))
        out2 = self.__sigmoid(self._f.dot(out1, self._weight2))
        return s.Matrix(out2)

    def topkl(self, addr):
        with open(addr, 'w') as f:
            pkl.dump([self._weight1, self._weight2], f)

    def readpkl(self, addr):
        with open(addr) as f:
            data = pkl.load(f)
            self._weight1 = data[0]
            self._weight2 = data[1]

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
