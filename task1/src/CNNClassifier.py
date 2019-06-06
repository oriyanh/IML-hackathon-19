from keras.layers import Input, Conv2D, Activation, Add, Dense, \
    Flatten, GlobalMaxPooling2D
from keras.models import Model, Sequential
from keras.backend import batch_flatten
from keras.optimizers import Adam
import gensim
import nltk
from task1.src.ClassifierBase import ClassifierBase
# from task1.src.data_parser import get_corpus
from task1.src.Commons import HANDLES_DICT
import numpy as np


class CNNClassifier(ClassifierBase):

    corpus = None
    models = []

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        label, handle = y
        self.corpus = get_corpus(X)
        model = gensim.models.Word2Vec(self.corpus, size=100, window=5,
                                       min_count=1, workers=4)
        keras_model = Sequential()
        keras_model.add(model.wv.get_keras_embedding())
        keras_model.layers[0].trainable = False




    def classify(self, X):
        pass

    def score(self, X, y):
        pass

if __name__ == "__main__":
    cnn = CNNClassifier()
