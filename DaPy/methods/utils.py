from DaPy.core import SeriesSet, is_iter, Series
from DaPy.matlib import describe
from collections import namedtuple
from operator import itemgetter

__all__ = ['_label', 'score_binary_clf']

_binary_perf_result = namedtuple('binary_clf', ['TP', 'FN', 'FP', 'TN'])


def plot_reg(y_hat, y, res):
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        warn('DaPy uses `matplotlib` to draw pictures, try: pip install matplotlib.')
        return None

    plt.subplot(311)
    plt.title('Prediction of Model')
    plt.xlabel('Samples')
    plt.ylabel('Prediction')
    plt.plot(y.T.tolist()[0], color='blue', alpha=0.65, label='Actual')
    plt.plot(y_hat.tolist()[0], color='red', alpha=0.7, label='Predict')
    plt.legend()
    
    plt.subplot(312)
    plt.title('Distribution of Residual')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.hist(res, max(10, len(y_hat) // 5), color='blue', alpha=0.6)
    
    plt.subplot(313)
    plt.title('Residual')
    plt.xlabel('Samples')
    plt.ylabel('Residual')
    sigma = [describe(res.T.tolist()[0]).Sn] * y_hat.shape[1]
    plt.plot(res, color='blue', alpha=0.6)
    plt.plot([0] * y_hat.shape[1], color='black', linestyle='--', alpha=0.5)
    plt.plot(sigma, color='black', alpha=0.25, linestyle='--')
    plt.plot(map(lambda x: -x, sigma), color='black', alpha=0.25, linestyle='--')

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.95,
            wspace=0.2, hspace=0.8)
    return plt


