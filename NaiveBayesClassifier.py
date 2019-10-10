import numpy as np


class NaiveBayesClassifier:

    def __init__(self, labels):
        self.X = None
        self.y = None
        self.delta = None
        self.labels = labels
        self.prior = None

    def fit(self, X, y):
        self.X = X.astype(np.float64)
        self.y = y
        self.delta = self.compute_delta()
        self.prior = self.compute_prior()

    def compute_prior(self):
        prior = np.zeros((len(self.labels)))
        for i in range(len(self.labels)):
            prior[i] = np.log((np.count_nonzero(self.y == self.labels[i]) + 1) / (self.y.shape[0] + 3))
        return prior

    def get_prior(self, label):
        index = self.labels.index(label)
        return self.prior[index]

    def compute_delta(self):
        delta = np.zeros((3, 27))
        for i in range(len(self.labels)):
            label = self.labels[i]
            numer = np.reshape(self.y == label, (-1, 1))*self.X
            numer = np.sum(numer, axis=0).astype(np.float64)
            denum = np.sum(np.reshape(self.y == label, (-1, 1))*self.X, keepdims=True).astype(np.float64)
            # delta[i] = numer/denum
            delta[i] = np.log(numer + 1) - np.log(denum + 27)
        return delta
        # return delta

    def get_likelihood(self, x):
        likelihood = np.zeros((len(self.labels), 1))
        x = np.reshape(x, (-1))
        for i in range(len(self.labels)):
            likelihood[i] = np.sum(self.delta[i]*np.log(x))
        return likelihood

    def get_delta_by_label(self, label):
        index = self.labels.index(label)
        return self.delta[index]

    def predict(self, x):
        likelihood = self.get_likelihood(x)
        index = np.argmax(self.prior + likelihood)
        return self.labels[index]

