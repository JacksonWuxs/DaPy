from collections import Counter, defaultdict
from heapq import nlargest
from itertools import chain
from math import log10
from operator import itemgetter

from .base import BaseEngineModel

HEAD = '<HEAD>'
END = '<END>'
ITEMGETTER1 = itemgetter(1)
ITEMGETTER0 = itemgetter(0)

def count_iter(iterable):
    count = 0.0
    while True:
        try:
            next(iterable)
        except StopIteration:
            return count
        count += 1.0
        
class TfidfCounter(BaseEngineModel):
    def __init__(self, ngram=1, engine='numpy'):
        BaseEngineModel.__init__(self, engine)
        self.ngram = ngram
        self.tfidf = {}

    @property
    def ngram(self):
        return self._ngram

    @ngram.setter
    def ngram(self, num_words):
        assert isinstance(num_words, int), 'n_gram parameter must be int'
        assert num_words >= 1, 'n_gram parameter must greater than 1.'
        self._ngram = num_words
        self._h = (HEAD,) * (self.ngram - 1)
        self._e = (END,) * (self.ngram - 1)

    @property
    def tfidf(self):
        return self._tfidf

    @tfidf.setter
    def tfidf(self, values):
        assert isinstance(values, dict)
        self._tfidf = values
        
    def __setstate__(self, args):
        BaseEngineModel.__setstate__(self, args)
        self.ngram = args['_ngram']

    def _pad(self, string):
        return self._h + tuple(string) + self._e

    def _get_ngram(self, string):
        if self._ngram == 1:
            for token in string:
                yield token
        else:
            string = self._pad(string)
            for i in range(len(string) - self._ngram + 1):
                yield string[i:i+self._ngram]

    def _nlargest(self, n):
        return nlargest(n, self.tfidf.items(), key=ITEMGETTER1)

    def nlargest(self, n):
        return dict(self._nlargest(n))
        
    def fit(self, documents, labels, min_freq=1.0, threashold=0.01):
        assert len(documents) == len(labels), 'number of documents must equal to number of labels'
        
        num_words = 0.0
        count_ngram = defaultdict(Counter)
        for row, label in zip(documents, labels):
            for pair in self._get_ngram(row):
                count_ngram[pair][label] += 1
                num_words += 1.0

        labels = Counter(labels)
        for label, freq in labels.items():
            labels[label] = freq * threashold
        
        selector = lambda value: value[1] >= labels[value[0]]
        self.tfidf, D = {}, len(labels) + 1.0
        for pair, counts in count_ngram.items():
            num_word = sum(counts.values())
            if num_word >= min_freq:
                tf = num_word / num_words
                df = count_iter(filter(selector, counts.items())) + 1.0
                self.tfidf[pair] = tf * log10(D / df)
        return self.tfidf

    def transform(self, documents, max_num_tokens=500):
        tokens = map(ITEMGETTER0, self._nlargest(max_num_tokens))
        tokens = dict((token, index) for index, token in enumerate(tokens))
        shape = len(tokens)
        
        embeddings = []
        for row in documents:
            embedding = [0.0] * shape
            for pair in self._get_ngram(row):
                if pair in tokens:
                    embedding[tokens[pair]] += 1
            embeddings.append(embedding)
        return self._engine.vstack(embeddings)

    def fit_transform(self, documents, labels, min_freq=1.0, threashold=0.01, max_num_tokens=500):
        self.fit(documents, labels, min_freq, threashold)
        return self.transform(documents, max_num_tokens)
            

if __name__ == '__main__':
    documents = [
        ['This', 'is', 'a', 'lucky', 'day'],
        ['This', 'is', 'not', 'a', 'lucky', 'day'],
        ]
    counter = TfidfCounter(2)
    print(counter.fit_transform(documents, [1, 0]))
    print(counter.tfidf)
