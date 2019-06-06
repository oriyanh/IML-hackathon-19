import pickle

from sklearn import metrics
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer

from task1.src.ClassifierBase import ClassifierBase
from task1.src.Commons import *





class GenericSKLClassifier(ClassifierBase):

    def __init__(self, clf, name=None):
        super(ClassifierBase).__init__()
        self.clf = clf
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.name = name
        self.weights_path = os.path.join(WEIGHTS_DIR_PATH, self.name + "_weights.joblib")
        self.vectorizer_path = os.path.join(WEIGHTS_DIR_PATH, self.name + "_vectorizer.pickle")
        pass

    def fit(self, X, y):
        X_train = self.vectorizer.fit_transform(X)

        with open(self.vectorizer_path, 'wb') as handle:
            pickle.dump(self.vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.clf.fit(X_train, y.astype(np.int))
        self.save_weights()



    def classify(self, X):
        clf = load(self.weights_path)
        with open(self.vectorizer_path, 'rb') as handle:
            self.vectorizer = pickle.load(handle)
        X_test = self.vectorizer.transform(X)
        return clf.predict(X_test)


    def score(self, X, y):
        pred = self.classify(X)
        return metrics.accuracy_score(y, pred)


    def save_weights(self):
        dump(self.clf, self.weights_path)