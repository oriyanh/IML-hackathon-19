from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.svm import LinearSVC

from task1.src.ClassifierBase import ClassifierBase
from task1.src.CNNClassifier import CNNClassifier
from task1.src.genericsklclassifier import GenericSKLClassifier, BEST_LEARNER_MAPPING
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

NN_LEARNER = (CNNClassifier(), "CNN Learner")

LEARNERS = (*SKLEARNERS, NN_LEARNER)


class MegaClassifier(ClassifierBase):

    def __init__(self) -> None:
        super().__init__()
        self.learners = []
        self.best_learner = None

    def fit(self, X, y):
        for clf in SKLEARNERS:
            classifier = GenericSKLClassifier(*clf)
            classifier.fit(X, y)
            self.learners.append(classifier)
        self.learners.append(NN_LEARNER[0])

    def classify(self, X):
        best_learner_index = BEST_LEARNER_MAPPING[0]
        if best_learner_index < 10:
            self.best_learner = GenericSKLClassifier(LEARNERS[best_learner_index][0], LEARNERS[best_learner_index][1])
        else:
            self.best_learner = LEARNERS[best_learner_index][0]
        self.weights_path = BEST_LEARNER_MAPPING[1]
        return self.best_learner.classify(X)

    def score(self, X, y):
        scores = np.zeros(len(self.learners))
        for i, clf in enumerate(self.learners):
            scores[i] = clf.score(X, y.astype(np.int))
        best_learner_idx = np.argmax(scores)
        self.best_learner = self.learners[best_learner_idx]
        print(f"Best learner: {LEARNERS[best_learner_idx][1]}, best score: {scores[best_learner_idx]}")
