import os
import nltk
from sklearn.linear_model import RidgeClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from task1.src.genericsklclassifier import GenericSKLClassifier
from task1.src.Commons import OUT_DIR_PATH, split_training_validation_sets
from task1.src.data_parser import Parser
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
    results = []
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
            (Perceptron(max_iter=50, tol=1e-3), "Perceptron"),
            (NearestCentroid(), "Nearest Centroid"),
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
        clf = GenericSKLClassifier(clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(score)
        clf_descr = str(clf).split('(')[0]
        results.append((name, score))


    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(2)]
    clf_names, score = results
    # clf_names, score, training_time, test_time = results
    # training_time = np.array(training_time) / np.max(training_time)
    # test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)
    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)
    plt.tight_layout()
    plt.show()
pass

if __name__ == "__main__":
    main()
