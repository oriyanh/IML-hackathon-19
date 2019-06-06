from sklearn import metrics
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer


class final_classifier:

    def __init__(self, clf):
        self.clf = clf
        self.vectorizer = CountVectorizer(stop_words='english')



    def fit(self, X, y):
        X_train = self.vectorizer.fit_transform(X)

        # todo: make sure x y correct
        self.clf.fit(X_train, y)




    def classify(self, X):
        clf = load('weights.joblib')
        X_test = self.vectorizer.transform(X)
        return clf.predict(X_test)


    def score(self, X, y):
        pred = self.classify(X)
        return metrics.accuracy_score(y, pred)


    def save_weights(self):
        dump(self.clf, 'weights.joblib')