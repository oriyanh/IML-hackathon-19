from keras.layers import Input, Conv1D, Dense, GlobalMaxPooling1D, \
    Concatenate, Embedding
from keras.models import Model, load_model
from task1.src.ClassifierBase import ClassifierBase
from task1.src.data_parser import Parser
import numpy as np
import tensorflow as tf
import os
from task1.src.Commons import OUT_DIR_PATH, split_training_validation_sets
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model
import pickle


# I will show you da wegh
MODEL_FILE_NAME = "model 0.75 acc"
TOKENIZER_FILE_NAME = "tokenizer.pickle"

class CNNClassifier(ClassifierBase):
    """
    Classifies using a convolutional network
    """
    corpus = None
    model = None
    tokenizer = None
    model_built = False
    model_trained = False

    def __init__(self) -> None:
        """
        Inits the object, loads model if already exists
        """
        super().__init__()
        if os.path.isfile(MODEL_FILE_NAME) and os.path.isfile(TOKENIZER_FILE_NAME):
            self.model = load_model(MODEL_FILE_NAME)
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
            self.model_trained = True


    @staticmethod
    def canonize_y(y):
        """
        Canonizes y vec to proper form
        :param y: Label vector
        :return: Proper form of y for the CNN
        """
        ret = np.zeros((y.shape[0], 10))
        ret[np.arange(y.shape[0]), y.astype(int)] = 1
        return ret

    def build_model(self):
        """
        Build the model
        """
        size_of_vocabulary = len(self.tokenizer.word_index) + 1
        a = Input((None,))
        b = Embedding(size_of_vocabulary, 120)(a)

        gram_1 = Conv1D(8, (1,), padding="same")(b)
        gram_1 = Conv1D(15, (5,), padding="same")(gram_1)

        gram_2 = Conv1D(8, (2,), padding="same")(b)
        gram_2 = Conv1D(15, (10,), padding="same")(gram_2)

        gram_3 = Conv1D(8, (3,), padding="same")(b)
        gram_3 = Conv1D(15, (15,), padding="same")(gram_3)

        flatten_first = GlobalMaxPooling1D()(b)  # psuedo flatten
        dense_first = Dense(400, activation=tf.nn.sigmoid)(flatten_first)
        all = Concatenate()([gram_3, gram_2, gram_1])
        flatten = GlobalMaxPooling1D()(all)  # psuedo flatten
        all = Concatenate()([flatten, dense_first])
        out = Dense(10, activation=tf.nn.sigmoid)(all)

        model = Model(inputs=a, outputs=out)
        model.summary()
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        plot_model(model, to_file='model.png')
        self.model_built = True

    def fit(self, X, y):
        """
        Fit the model
        :param X: Data
        :param y: Labels
        """
        self.corpus = Parser.tokenize_tweets(X)
        tokenizer = Tokenizer()
        self.tokenizer = tokenizer
        tokenizer.fit_on_texts(X)
        length = max([len(t.split()) for t in X])
        X_tokens = tokenizer.texts_to_sequences(self.corpus[0])
        X_train_pad = pad_sequences(X_tokens, maxlen=length, padding='post')
        self.build_model()
        self.model.fit(X_train_pad, CNNClassifier.canonize_y(y), epochs=8,
                       steps_per_epoch=15)
        self.model_trained = True
        with open(TOKENIZER_FILE_NAME, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def classify(self, X):
        """
        Classifies the given data using the CNN that has already been fitted
        :param X: The data
        :return: The prediction
        """
        Y = Parser.tokenize_tweets(X)
        X = Y[0]
        X_tokens = self.tokenizer.texts_to_sequences(X)
        length = max([len(t.split()) for t in X])
        X_train_pad = pad_sequences(X_tokens, maxlen=length, padding='post')

        return self.model.predict(X_train_pad)

    def score(self, X, y):
        Y = Parser.tokenize_tweets(X)
        X = Y[0]
        X_tokens = self.tokenizer.texts_to_sequences(X)
        length = max([len(t.split()) for t in X])
        X_train_pad = pad_sequences(X_tokens, maxlen=length, padding='post')
        return self.model.evaluate(X_train_pad, CNNClassifier.canonize_y(y))[1]


if __name__ == "__main__":
    cnn = CNNClassifier()
    training_set_path = os.path.join(OUT_DIR_PATH, 'training_set.csv')
    X, y = Parser.load_csv_to_array(training_set_path)
    S, V = split_training_validation_sets(X, y, 0.7)
    cnn.fit(*S)
    print("Evaluation on test")
    print(cnn.score(*V))
    print("Evaluation on train")
    print(cnn.score(*S))
