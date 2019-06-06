from keras.layers import Input, Conv2D, Activation, Add, Dense, \
    Flatten, GlobalMaxPooling2D, Concatenate
from keras.models import Model, Sequential
from keras.backend import batch_flatten
from keras.optimizers import Adam
import gensim
import nltk
from task1.src.ClassifierBase import ClassifierBase
from task1.src.data_parser import Parser
from task1.src.Commons import HANDLES_DICT
import numpy as np
import tensorflow as tf
import os
from task1.src.Commons import OUT_DIR_PATH, split_training_validation_sets



class CNNClassifier(ClassifierBase):

    corpus = None
    models = []

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def zero_one_loss(y, y_hat):
        return np.count_nonzero(y - y_hat)

    def fit(self, X, y):
        self.corpus = Parser.tokenize_tweets(X)
        for stem in self.corpus:
            word_model = gensim.models.Word2Vec(list(stem), size=100, window=5,
                                           min_count=1, workers=4)
            inner_layers = word_model.wv.get_keras_embedding()
            inner_layers.trainable = False
            a = Input((None, None))
            b = inner_layers(a)
            gram_1 = Conv2D(3, (1, 1))(b)
            gram_2 = Conv2D(3, (2, 2))(b)
            gram_3 = Conv2D(3, (3, 3))(b)
            all = Concatenate()([gram_1, gram_2, gram_3])
            flatten = GlobalMaxPooling2D()(all) # psuedo flatten
            dense = Dense(300, activation=tf.nn.softmax)(flatten)
            out = Dense(10, activation=tf.nn.softmax)(dense)

            model = Model(inputs=a, outputs=out)
            model.summary()
            model.compile(optimizer='adam',
                          loss='hinge',
                          metrics=['accuracy'])
            model.fit(np.array(list(stem)), y, epochs=10, steps_per_epoch=10)
            self.models.append(model)

    def classify(self, X):
        pass

    def score(self, X, y):
        scores = []
        for model in self.models:
            scores.append(model.evaluate(X, y))
        return scores

if __name__ == "__main__":
    cnn = CNNClassifier()
    training_set_path = os.path.join(OUT_DIR_PATH, 'training_set.csv')
    X, y = Parser.load_csv_to_array(training_set_path)
    S, V = split_training_validation_sets(X, y, 0.8)
    cnn.fit(*S)
    cnn.score(*V)