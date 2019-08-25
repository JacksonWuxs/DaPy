from time import clock

from DaPy import LogInfo, Series
from .base import BaseEngineModel, Activators

class BaseBPModel(BaseEngineModel):
    def __init__(self, engine, learn_rate, l1_penalty, l2_penalty):
        BaseEngineModel.__init__(self, engine)
        self.learn_rate = learn_rate
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self._activator = Activators(self.engine)
        self._accuracy = None
        self._cost_history = Series()             # Mistake Recorder

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def cost_history(self):
        return self._cost_history

    @property
    def learn_rate(self):
        return self._learn_rate

    @learn_rate.setter
    def learn_rate(self, new_rate):
        assert isinstance(new_rate, (int, float))
        assert 0 < new_rate < 1, '`learnning rate` must be between 0 to 1'
        self._learn_rate = new_rate

    @property
    def l1_penalty(self):
        return self._l1
    
    @l1_penalty.setter
    def l1_penalty(self, new_l1):
        assert isinstance(new_l1, (int, float))
        assert new_l1 >= 0, '`l1 penalty` must be greater than 0'
        self._l1 = new_l1
    
    @property
    def l2_penalty(self):
        return self._l2
    
    @l2_penalty.setter
    def l2_penalty(self, new_l2):
        assert isinstance(new_l2, (int, float))
        assert new_l2 >= 0, '`l2_penalty` must be greater than 0'
        self._l2 = new_l2

    def __getstate__(self):
        pkl = BaseEngineModel.__getstate__(self)
        del pkl['_activator']
        return pkl

    def __setstate__(self, pkl):
        BaseEngineModel.__setstate__(self, pkl)
        self._activator = Activators(self.engine)
        self.learn_rate = pkl['_learn_rate']
        self.l1_penalty = pkl['_l1']
        self.l2_penalty = pkl['_l2']
        self._cost_history = pkl['_cost_history']
        self._accuracy = pkl['_accuracy']

    def _train(self, X, Y, epoch=500, verbose=True, early_stop=False):
        assert early_stop in (True, False)
        show_log, log_level = 1, 1

        start = clock()
        for term in range(1, epoch + 1):
            # foreward propagation
            predict = self._forecast(X)
            # record the errors
            self._accuracy = self._calculate_accuracy(predict, Y)
            diff = self._calculate_backward_error(predict, Y)
            self._cost_history.append(self._sum(self._abs(diff)) / len(X))
            # back propagation
            self._backward(X, diff)
            
            # check whather to early stop the iteration
            if early_stop and len(self._cost_history) > 10:
                upper_term = 0
                for i in range(1, 11):
                    if self._cost_history[-i] >= self._cost_history[-i-1]:
                        upper_term += 1
                if upper_term >= 10:
                    LogInfo('Early stoped')
                    break

            # print training information
            if verbose and term % show_log == 0:
                spent = clock() - start
                finish_rate = (term / (epoch+1.0))*100
                last = spent / (finish_rate/100) - spent
                LogInfo('Finished: %.1f' % finish_rate + '%\t' +\
                        'Epoch: %d\t' % term +\
                        'Rest Time: %.2fs\t' % last +\
                        'Accuracy: %.2f' % self._accuracy + '%')
                if term > 10 ** (1 + log_level):
                    log_level += 1
                    show_log = 10 ** log_level

        time = clock() - start
        LogInfo('Finish Train | Time:%.1fs\tEpoch:%d\tAccuracy:%.2f'%(time, term, self._accuracy * 100) + '%')

    def plot_error(self):
        '''use matplotlib library to draw the error curve during the training.
        '''
        try:
            import matplotlib.pyplot as plt
            plt.title('Model Training Result')
            plt.plot(self._cost_history[1:])
            plt.ylabel('Error %')
            plt.xlabel('Epoch')
            plt.show()
        except ImportError:
            raise ImportError('DaPy uses `matplotlib` to draw picture, try: pip install matplotlib.')
