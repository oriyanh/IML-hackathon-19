import os

from sklearn import metrics
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from task1.src.ClassifierBase import ClassifierBase
from task1.src.Commons import *

WEIGHTS_PATH = os.path.join(OUT_DIR_PATH, "final_classifier_weights.joblib")



class GenericSKLClassifier(ClassifierBase):

    def __init__(self, clf, vocab=None):
        super(ClassifierBase).__init__()
        self.clf = clf
        self.vectorizer = TfidfVectorizer(stop_words='english',
                                          vocabulary=vocab)



    def fit(self, X, y):
        X_train = self.vectorizer.fit_transform(X)

        self.clf.fit(X_train, y)
        self.save_weights()



    def classify(self, X):
        clf = load(WEIGHTS_PATH)
        X_test = self.vectorizer.transform(X)
        return clf.predict(X_test)


    def score(self, X, y):
        pred = self.classify(X)
        return metrics.accuracy_score(y, pred)


    def save_weights(self):
        dump(self.clf, WEIGHTS_PATH)