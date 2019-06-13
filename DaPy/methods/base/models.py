
from DaPy.core import DataSet, Frame, SeriesSet, Matrix as mat
from DaPy.core import is_math, is_seq

class BaseModel(object):

    def __init__(self, engine='numpy'):
        self._engine = str2engine(engine)      # which library for camputing
        
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
