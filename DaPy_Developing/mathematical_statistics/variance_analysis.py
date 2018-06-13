from collections import namedtuple
from DaPy.core import is_math, Table, DataSet

__all__ = ['ANOVA']

def ANOVA(*classes):
    if len(classes) <= 1:
        raise ValueError('ANOVA() expects more than 1 comparing types.')

    new_classes = list()
    for sequence in classes:
        sequence = filter(is_math, sequence)
        if len(sequence) <= 1:
            raise ValueError('ANOVA() expects more than 1 samples in each type.')
        new_classes.append(sequence)

    ni = map(len, new_classes)
    r = len(new_classes)
    Ti = map(sum, new_classes)
    G = sum(map(lambda x: x ** 2, sequence) for sequence in new_classes)

    totals = sum([Ti[i]**2/float(ni[i]) for i in range(r)])
    Sa = totals - sum(Ti)**2/float(sum(ni))
    Se = G - totals
    MSa = Sa / float(r - 1)
    MSe = Se / float(sum(ni) - r)
    F = MSa / MSe

    return  DataSet(Table([
        ['Regression', round(Sa, 4), r - 1, round(MSa, 4), round(F, 4)],
        ['Residual', round(Se, 4), n - r, round(MSe)],
        ['Total', round(Sa + Se, 4), n - 1]],
        ['Model', 'Sum Squr', 'DF', 'Mean Squr', 'F - Value']), 'ANOVA')
        
