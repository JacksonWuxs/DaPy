from DaPy import LogWarn

def eng2str(obj):
    return str(obj).split()[1][1: -1]

def str2eng(obj):
    if str(obj).lower() not in ('numpy', 'dapy', 'none'):
        LogWarn('Your model is not in Numpy or DaPy!')
    
    if str(obj).lower() == 'numpy':
        import numpy as engine
        engine.seterr(divide='ignore', invalid='ignore')
        return engine
    
    if str(obj).lower() == 'dapy':
        import DaPy as engine
        return engine
    
    return None
