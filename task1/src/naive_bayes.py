from collections import Counter
import task1.src.ClassifierBase as classifier
from task1.src.data_parser import Parser
import numpy as np

HANDLES_DICT = {0: "@realDonaldTrump",
                1: "@joebiden",
                2: "@ConanOBrien",
                3: "@TheEllenShow",
                4: "@KimKardashian",
                5: "@KingJames",
                6: "@ladygaga",
                7: "@Cristiano",
                8: "@jimmykimmel",
                9: "@Schwartzenegger",
                }


class naive_bayes(classifier.ClassifierBase):



    def fit(self, X, y):
        word_counts = Parser.get_word_counts(X, y)
        words = Parser.get_unique_words()
        self.probs = compute_probabilities(words, word_counts)


    def classify(self, X):
        labels = []
        for sent in X:
            results = np.array([])
            for person in range(10):
                result = 1
                for word in X:
                    result *= self.probs[(word, person)]
                results[person] = result
            labels.append(np.argmax(results))
        return np.array(labels)


    def score(self, X, y):
        labels = self.classify(X)
        diff = np.sum(y == labels)
        return diff


def compute_probabilities(words, word_counts):

    probabilities = Counter()

    for person in range(10):
        num_words = word_counts[person].sum()
        for word in words:
            appearances = word_counts[person][word]
            probabilities[(word, person)] = (appearances + 1) / (num_words + 1)  # add-1 smoothing

    return probabilities

