import numpy as np
import pandas as pd
from collections import Counter
import string

from NaiveBayesClassifier import NaiveBayesClassifier

def getFileContent(fname):
    f = open(fname, 'r')
    contents = f.read()
    contents = contents.replace('\n', '')
    return contents


def getAttributes(fname):
    dict = {}
    for ch in features:
        dict[ch] = 0

    counts = Counter(getFileContent(fname))
    newCounts = {k: [counts[k]] for k in features}

    df = pd.DataFrame.from_dict(newCounts)
    df = df.reindex(columns=features)
    return df

def q1_3():
    print(np.exp(classifier.get_prior('e')))
    print(np.exp(classifier.get_prior('j')))
    print(np.exp(classifier.get_prior('s')))

    float_formatter = lambda x: "%.5f" % x
    # np.set_printoptions(formatter={'float_kind': float_formatter})
    latex_formatter = lambda x: "%f, " % x
    np.set_printoptions(formatter={'float_kind': latex_formatter})

    delta_e = np.exp(classifier.get_delta_by_label('e'))
    print('theta_e : ', delta_e, 'sum: ', np.sum(delta_e))

    delta_j = np.exp(classifier.get_delta_by_label('j'))
    print('theta_j : ', delta_j, np.sum(delta_j))

    delta_s = np.exp(classifier.get_delta_by_label('s'))
    print('theta_s : ', delta_s, np.sum(delta_s))
    print('Done')

def q4():
    fname = 'languageID/e10.txt'
    dfRow = getAttributes(fname)
    bag_of_words = np.reshape(dfRow.to_numpy(), (-1))
    for i in range(bag_of_words.shape[0]):
        print('x_{%c}=%d, '%(features[i], bag_of_words[i]))
    likelihood = classifier.get_likelihood(bag_of_words)
    for i in range(len(labels)):
        print('p(x|y = %c) = %E'%(labels[i], np.exp(likelihood[i])))
    for i in range(len(labels)):
        prior = classifier.get_prior(labels[i])
        posterior = likelihood[i] + prior
        print('p(y=%c|x) = %E'%(labels[i], np.exp(posterior)))
    print('predicted class = %c'%(classifier.predict(bag_of_words)))
    print('Done')

labels = ['e', 'j', 's']
features = list(string.ascii_lowercase) + [' ']
df = pd.DataFrame(columns=features + ['label'])
for label in labels:
    for i in range(0, 10):
        fname = 'languageID/%c%d.txt' % (label, i)
        dfRow = getAttributes(fname)
        dfRow['label'] = label
        df = df.append(dfRow, ignore_index=True)
dfX = df.loc[:, features]
dfY = df.loc[:, 'label']
X = dfX.to_numpy()
y = dfY.to_numpy()

classifier = NaiveBayesClassifier(labels)
classifier.fit(X, y)
q1_3()
q4()
print('Done')
