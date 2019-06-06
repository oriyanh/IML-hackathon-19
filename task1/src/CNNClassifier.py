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
import tensorflow as tf


class CNNClassifier(ClassifierBase):

    corpus = None
    models = []

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        self.corpus = get_corpus(X)
        for stem in self.corpus:
            word_model = gensim.models.Word2Vec(stem, size=100, window=5,
                                           min_count=1, workers=4)
            inner_layers = word_model.wv.get_keras_embedding()
            inner_layers.trainable = False
            a = Input((None, None, inner_layers.output_dim))
            b = inner_layers()(a)
            gram_1 = Conv2D(Conv2D(3, (1, 1)))(b)
            gram_2 = Conv2D(Conv2D(3, (2, 2)))(b)
            gram_3 = Conv2D(Conv2D(3, (3, 3)))(b)
            all = Add()([gram_1, gram_2, gram_3])
            flatten = GlobalMaxPooling2D()(all) # psuedo flatten
            dense = Dense(300, activation=tf.nn.softmax)(flatten)
            dense = Dense(300, activation=tf.nn.softmax)(dense)
            dense = Dense(300, activation=tf.nn.softmax)(dense)
            out = Dense(10, activation=tf.nn.softmax)(dense)

            model = Model(inputs=a, outputs=out)
            model.summary()
            model.fit()
            self.models.append(model)

    def classify(self, X):
        pass

    def score(self, X, y):
        pass

if __name__ == "__main__":
    cnn = CNNClassifier()
