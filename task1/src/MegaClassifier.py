import json

from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import LinearSVC

from task1.src.ClassifierBase import ClassifierBase
from task1.src.CNNClassifier import CNNClassifier
from task1.src.genericsklclassifier import GenericSKLClassifier
import numpy as np

SKLEARNERS = (
    (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
    (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
    (PassiveAggressiveClassifier(max_iter=50, tol=1e-3), "Passive-Aggressive"),
    (LinearSVC(penalty='l1', dual=False, tol=1e-3), "linear1"),
    [LinearSVC(penalty='l2', dual=False, tol=1e-3), "linear2"],
    (SGDClassifier(alpha=.0001, max_iter=50, penalty='l2'), "sgd with l2 loss"),
    (SGDClassifier(alpha=.0001, max_iter=50, penalty='elasticnet'), "elasticnet"),
    (MultinomialNB(alpha=.01), "multiNB"),
    (BernoulliNB(alpha=.01), "bernouliNB"),
    (ComplementNB(alpha=.1), "complementNB"))

NN_LEARNERS = (CNNClassifier(),)

LEARNERS = (*SKLEARNERS, *NN_LEARNERS)


class MegaClassifier(ClassifierBase):

    def __init__(self) -> None:
        super().__init__()
        self.learners = []
        self.best_learner = None

    def fit(self, X, y):
        for clf in SKLEARNERS:
            classifier = GenericSKLClassifier(clf)
            classifier.fit(X, y)
            self.learners.append(classifier)
        for clf in NN_LEARNERS:
            # clf.fit(X, y)
            self.learners.append(clf)

    def classify(self, X):
        self.best_learner = LEARNERS[BEST_LEARNER_MAPPING[0]]
        self.weights_path = BEST_LEARNER_MAPPING[1]
        return self.best_learner.classify(X)

    def score(self, X, y):
        results = np.zeros(len(self.learners))
        names =
        for i, clf in enumerate(self.learners):
            results[i] = clf.score(X, y)
        best_learner_idx = np.argmin(results)[0]
        self.best_learner = self.learners[best_learner_idx]
        print("Best learner: %s, best score: %s" % (self.best_learner.__name__, results[best_learner_idx]))

        indices = np.arange(len(results))

        # results = [[x[i] for x in results] for i in range(2)]
        clf_names = [clf.__name__ for clf in self.learners]
        # clf_names, score, training_time, test_time = results
        # training_time = np.array(training_time) / np.max(training_time)
        # test_time = np.array(test_time) / np.max(test_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, scores, .2, label="score", color='navy')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)
        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)
        plt.tight_layout()
        plt.show()