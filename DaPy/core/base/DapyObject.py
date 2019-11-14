from threading import Lock
from functools import wraps


class Object(object):
    '''The base object for DaPy to gurantee threading safety'''
    def __init__(self):
        self._thread_lock = None

    @property
    def thread_safety(self):
        if self._thread_lock is None:
            return False
        return True

    @thread_safety.setter
    def thread_safety(self, mode):
        assert mode in (True, False), 'setting `thread_safety` with True or False'
        if mode is True:
            if self._thread_lock is None:
                self._thread_lock = Lock()
        else:
            self._thread_lock = None

    @property
    def THREAD_LOCK(self):
        return self._thread_lock


def check_thread_locked(func):
    def locked_func(self, *args, **kwrds):
        lock = self.THREAD_LOCK
        if not lock:
            return func(self, *args, **kwrds)
        try:
            lock.acquire()
            return func(self, *args, **kwrds)
        finally:
            lock.release()
    return locked_func
