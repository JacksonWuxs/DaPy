from DaPy.core import Matrix

def Accuracy(confumat):
    upper = sum([confumat[i][i] for i in range(confumat.shape[1] - 1)])
    return round(upper / float(confumat[-1][-1]) * 100, 4)

def Kappa(confumat):
    as_ = confumat[:, -1].tolist()[:-1]
    bs_ = confumat[-1][:-1]
    Po = Accuracy(confumat) /100
    upper = sum([a * b for a, b in zip(as_, bs_)])
    Pe = float(upper) / confumat[-1][-1] ** 2
    return (Po - Pe) / (1 - Pe)
    
def sigmoid(x, engine, diff=False):
    if diff:
        return (1 - x) * x
    return 1.0 / (1.0 + engine.exp(-x))

def tanh(x, engine, diff=False):
    if diff:
        return 1.0 - x * x
    poss, negt = engine.exp(x), engine.exp(-x)
    return (poss - negt) / (poss + negt)

def linear(x, engine, diff=False):
    if diff:
        return engine.ones(x.shape)
    return x

def radb(x, engine, diff=False):
    if diff:
        return -2.0 * x * engine.exp( -x * x)
    return engine.exp(-x * x)

def relu(x, engine, diff=False):
    if diff:
        return (abs(x) + x) / x
    return (abs(x) + x) / 2

def softmax(x, engine, diff=False):
    if diff:
        return x - x * x
    new_x = engine.exp(x)
    sum_x = engine.sum(new_x, axis=1)
    output = [1] * len(new_x)
    for i, record in enumerate(new_x):
        div = sum_x[i]
        output[i] = [value / div for value in record]
    return Matrix(output, False)

activation = {'sigm': sigmoid,
             'tanh': tanh,
             'line': linear,
             'radb': radb,
             'relu': relu,
             'softmax': softmax}
