import os
import nltk
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC

from task1.src.final_classifier import final_classifier
from task1.src.Commons import OUT_DIR_PATH, split_training_validation_sets
from task1.src.data_parser import Parser
from task1.src.naive_bayes import naive_bayes
from sklearn.linear_model import SGDClassifier

def main():
    nltk.download("wordnet")
    training_set_path = os.path.join(OUT_DIR_PATH, 'training_set.csv')
    test_set_path = os.path.join(OUT_DIR_PATH, 'test_set.csv')
    X, y = Parser.load_csv_to_array(training_set_path)
    # data = Parser.preprocess_data(X)
    # generators = Parser.tokenize_tweets(X)
    # for gen in generators:
    # 	values = list(gen)
    # 	pass
    S, V = split_training_validation_sets(X, y, 0.8)
    vocab = Parser.get_vocabulary(X)
    X_train, X_test = S[0], V[0]
    y_train, y_test = np.array(S[1], dtype="float"), np.array(V[1], dtype="float")

    X_test, y_test = Parser.load_csv_to_array(test_set_path)
    # X_test = Parser.preprocess_data(X_test)
    y_test = np.array(y_test, dtype="float")

    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
            (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
            (NearestCentroid(), "Perceptron"),
            (PassiveAggressiveClassifier(max_iter=50, tol=1e-3), "Passive-Aggressive"),
            (LinearSVC(penalty='l1', dual=False, tol=1e-3), "linear1"),
            (LinearSVC(penalty='l2', dual=False, tol=1e-3), "linear2"),
            (SGDClassifier(alpha=.0001, max_iter=50, penalty='l1'), "sgd1"),
            (SGDClassifier(alpha=.0001, max_iter=50, penalty='l2'), "sgd2"),
            (SGDClassifier(alpha=.0001, max_iter=50, penalty='l2'), "elasticnet"),
            (MultinomialNB(alpha=.01), "multiNB"),
            (BernoulliNB(alpha=.01), "bernouliNB"),
            (ComplementNB(alpha=.1), "complementNB")):
        print(name)
        clf = final_classifier(clf)
        clf.fit(X_train, y_train)
        print(clf.score(X_test, y_test))

pass

if __name__ == "__main__":
    main()
