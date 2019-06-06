from keras.layers import Input, Conv2D, Conv1D, Activation, Add, Dense, \
    Flatten, GlobalMaxPooling2D, GlobalMaxPooling1D, Concatenate, Embedding
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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class CNNClassifier(ClassifierBase):
    corpus = None
    models = []
    tokenizer = None

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def zero_one_loss(y, y_hat):
        return np.count_nonzero(y - y_hat)

    @staticmethod
    def canonize_y(y):
        ret = np.zeros((y.shape[0], 10))
        ret[np.arange(y.shape[0]), y.astype(int)] = 1
        return ret

    def fit(self, X, y):
        self.corpus = Parser.tokenize_tweets(X)
        tokenizer = Tokenizer()
        self.tokenizer = tokenizer
        tokenizer.fit_on_texts(X)
        length = max([len(t.split()) for t in X])
        size_of_vocabulary = len(tokenizer.word_index) + 1
        X_tokens = tokenizer.texts_to_sequences(X)
        X_train_pad = pad_sequences(X_tokens, maxlen=length, padding='post')
        a = Input((None,))
        b = Embedding(size_of_vocabulary, 120, input_length=length)(a)
        #         gram_1 = Conv1D(7, (1,), padding="same")(b)
        #         gram_1 = Conv1D(10, (1,), padding="same")(gram_1)
        gram_2 = Conv1D(7, (2,), padding="same")(b)
        gram_2 = Conv1D(10, (2,), padding="same")(gram_2)
        gram_3 = Conv1D(7, (3,), padding="same")(b)
        gram_3 = Conv1D(10, (3,), padding="same")(gram_3)
        all = Concatenate()([gram_3, gram_2])
        #         all = Activation('relu')(all)
        flatten = GlobalMaxPooling1D()(all)  # psuedo flatten
        dense = Dense(300, activation=tf.nn.sigmoid)(flatten)
        #         dense = Dense(300, activation=tf.nn.sigmoid)(dense)
        out = Dense(10, activation=tf.nn.sigmoid)(dense)

        model = Model(inputs=a, outputs=out)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_train_pad, CNNClassifier.canonize_y(y), epochs=6,
                  steps_per_epoch=21)
        self.models.append(model)


    def classify(self, X):
        pass

    def score(self, X, y):
        scores = []
        X_tokens = self.tokenizer.texts_to_sequences(X)
        length = max([len(t.split()) for t in X])
        X_train_pad = pad_sequences(X_tokens, maxlen=length, padding='post')
        for model in self.models:
            scores.append(model.evaluate(X_train_pad, CNNClassifier.canonize_y(y)))
        return scores

    def predict(self, X):
        X_tokens = self.tokenizer.texts_to_sequences(X)
        length = max([len(t.split()) for t in X])
        X_train_pad = pad_sequences(X_tokens, maxlen=length, padding='post')
        return self.models[0].predict(X_train_pad)


cnn = CNNClassifier()
training_set_path = os.path.join(OUT_DIR_PATH, 'training_set.csv')
X, y = Parser.load_csv_to_array(training_set_path)
S, V = split_training_validation_sets(X, y, 0.7)
cnn.fit(*S)
cnn.score(*V)
print(cnn.score(*V))
