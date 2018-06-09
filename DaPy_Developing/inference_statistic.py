from collections import namedtuple
from .core import is_math

def ANOVA(*classes):
    if len(classes) <= 1:
        raise ValueError('ANOVA expects more than 1 comparing types.')

    new_classes = list()
    for sequence in classes:
        sequence = filter(is_math, sequence)
        if len(sequence) <= 1:
            raise ValueError('ANOVA expects more than 1 samples in each type.')
        new_classes.append(sequence)

    ni = [len(s) for s in new_classes]
    r = len(new_classes)
    Ti = [sum(s) for s in new_classes]
    G = 0
    for sequence in new_classes:
        for value in sequence:
            G += value ** 2

    totals = sum([Ti[i]**2/float(ni[i]) for i in range(r)])
    Sa = totals - sum(Ti)**2/float(sum(ni))
    Se = G - totals
    MSa = Sa / float(r - 1)
    MSe = Se / float(sum(ni) - r)
    F = MSa / MSe

    ANOVA_R = namedtuple('ANOVA_RESULT', ['Sa', 'Se', 'MSa', 'MSe', 'F'])
    return ANOVA_R(Sa, Se, MSa, MSe, F)
        
        
    
