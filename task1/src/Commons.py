import os
import numpy as np
from sklearn.linear_model import RidgeClassifier, Perceptron, \
    PassiveAggressiveClassifier, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC

# from task1.src.CNNClassifier import CNNClassifier

TWEETS_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tweets_data"))
OUT_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "out"))
WEIGHTS_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "weights"))

NAMES_FILE = r"names.txt"
if os.path.exists(TWEETS_DATA_PATH):
	CSV_FILE_NAMES = list(filter(lambda f: f.endswith(".csv") and
										   not f.endswith("tweets_test_demo.csv"), os.listdir(TWEETS_DATA_PATH)))
	CSV_FILE_PATHS = [os.path.join(TWEETS_DATA_PATH, f) for f in CSV_FILE_NAMES]

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


def split_training_validation_sets(X, y, ratio):
	num_samples = X.shape[0]
	num_training_samples = int(num_samples * ratio)
	training_set = X[:num_training_samples], y[:num_training_samples]
	validation_set = X[num_training_samples:], y[num_training_samples:]
	return training_set, validation_set

BEST_LEARNER_MAPPING = (4, os.path.join(WEIGHTS_DIR_PATH, "linear2_weights.joblist"))