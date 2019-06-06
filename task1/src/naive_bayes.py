from collections import Counter
import task1.src.ClassifierBase as classifier
from task1.src.data_parser import Parser


class naive_bayes(classifier):
    def __init__(self) -> None:
        self.probs = None
        self.vocabulary = None


    def fit(self, X, y):
        word_counts = Parser.get_word_counts(X, y)
        self.vocabulary = Parser.get_unique_words()
        self.probs = compute_probabilities(word_counts)


    def classify(self, X):
        labels = []
        for sent in X:
            results = np.array([])
            for person in range(10):
                result = 1
                for word in X:
                    if word in self.vocabulary:
                        result *= self.probs[(word, person)]
                results[person] = result
            labels.append(np.argmax(results))
        return np.array(labels)


    def score(self, X, y):
        labels = self.classify(X)
        diff = np.sum(y == labels)
        return diff


    def compute_probabilities(self, word_counts):

        probabilities = Counter()

        for person in range(10):
            num_words = word_counts[person].sum()
            for word in self.vocabulary:
                appearances = word_counts[person][word]
                probabilities[(word, person)] = (appearances + 1) / (num_words + 1)  # add-1 smoothing

        return probabilities

