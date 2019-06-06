from functools import reduce

import nltk
import pandas as pd
import numpy as np
import os.path
from task1.src.Commons import *
import preprocessor as p

CLEAN_OPTION_SET = (p.OPT.URL, p.OPT.NUMBER)
PARSE_OPTION_SET = (p.OPT.RESERVED, p.OPT.MENTION)
TOKENIZING_OPTION_SET = (p.OPT.EMOJI, p.OPT.SMILEY)

SPECIAL_CHAR_SET = {"@": "AT_", "#": "HASH_", "$": "DOLLAR_", "...": " THREE_DOTS ", "!": " EXCLAMATION "}

class Parser:

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def preprocess_data(X):
        """

        :param X:
        :return:
        """

        X_new = np.array(X)
        for i, x in enumerate(X_new):
            # For each sample tweet x in the domain X_new, clean all URLs and numbers
            p.set_options(*CLEAN_OPTION_SET)
            s = p.clean(x)
            # Replace all caps words with placeholder ALL_CAPS_WORD
            split_tweet = s.split()
            for w in split_tweet:
                if w.isupper():
                    s = s.replace(w, "ALL_CAPS_WORD")
            p.set_options(*PARSE_OPTION_SET)
            # Figure out if the tweet is a retweet - if so, replace it with the person being retweeted
            tweet = s
            s = p.parse(tweet)
            p.set_options(*TOKENIZING_OPTION_SET)
            if s.reserved is not None and "RT" in s.reserved[0].match:
                tweet_modified = s.mentions[0].match
            else:
                tweet_modified = p.tokenize(tweet)
            # Replace all special characters with placeholders to make them unique
            for ch in SPECIAL_CHAR_SET:
                tweet_modified = tweet_modified.replace(ch, SPECIAL_CHAR_SET[ch])
            X_new[i] = tweet_modified
        return X_new

    @staticmethod
    def get_vocabulary(tweets):
        return np.unique(np.concatenate(np.char.split(tweets)))

    @staticmethod
    def get_all_vocabularies(tweets):
        return [(Parser.get_vocabulary(tokenized_tweets))
                for tokenized_tweets in Parser.tokenize_tweets(tweets)]

    @staticmethod
    def create_training_and_test_sets(tweet_csv_files, out_path=OUT_DIR_PATH):
        """
        Creates a training set and a validation set from sequence of CSV files, with columns 'user','tweet' .
        :param tweet_csv_files: Sequence of absolute paths to CSV files that contain tweet data.
        :param out_path: Path to output directory, to save valiation and training set CSVs. May be relative path.
        Defaults to '../../out/' . Results are saved to 'out_path/test_set.csv' and
        'out_path/training_set.csv' . OVERWRITES EXISTING FILES!
        """
        abs_out_path = os.path.abspath(out_path) if not os.path.isabs(
            out_path) else out_path
        csv_files = [os.path.join(tweet_csv_files, f) for f in CSV_FILE_NAMES]
        if not os.path.exists(abs_out_path):
            os.mkdir(abs_out_path)
        elif not os.path.isdir(OUT_DIR_PATH):
            raise Exception(
                "Output location is not a directory: [%s]" % out_path)
        val_set_path = os.path.join(abs_out_path, "test_set.csv")
        train_set_path = os.path.join(abs_out_path, "training_set.csv")
        if os.path.isfile(val_set_path):
            os.remove(val_set_path)
        if os.path.isfile(train_set_path):
            os.remove(train_set_path)
        tweets = {f[:-4]: pd.read_csv(f, sep=",", encoding='utf-8') for f in
                  csv_files}
        for f in tweets:
            shuffled_tweets = tweets[f].sample(frac=1)
            num_tweets = shuffled_tweets.shape[0]
            num_training_tweets = num_tweets - num_tweets // 10
            training_df = shuffled_tweets[:num_training_tweets]
            validation_df = shuffled_tweets[num_training_tweets:]
            Parser.csv_to_file(validation_df, val_set_path)
            Parser.csv_to_file(training_df, train_set_path)
        Parser.shuffle_csv(val_set_path, ['user', 'tweet'])
        Parser.shuffle_csv(train_set_path, ['user', 'tweet'])

    @staticmethod
    def shuffle_csv(path, columns=None):
        shuffled_df = pd.read_csv(path, sep=",", encoding='utf-8').sample(
            frac=1)
        Parser.csv_to_file(shuffled_df, path, "w", columns)

    @staticmethod
    def csv_to_file(df, path, mode="a", columns=None):
        """

        :param df:
        :param mode:
        :param path:
        :param columns:
        :return:
        """
        if columns is None:
            header = False
        else:
            header = True
            df.columns = columns
        df.to_csv(path, sep=",", mode=mode, header=header, index=False,
                  encoding='utf-8')

    @staticmethod
    def load_csv_to_array(path):
        Parser.shuffle_csv(path, ['user', 'tweet'])
        vals = pd.read_csv(path, sep=",", encoding='utf-8').values
        X = np.array([s.strip("\'\"[]") for s in vals[..., 1]])
        y = np.array(vals[..., 0])
        return X, y


if __name__ == "__main__":
    names_full_path = os.path.join(TWEETS_DATA_PATH, NAMES_FILE)
    names = pd.read_csv(names_full_path, sep=",")

    Parser.create_training_and_test_sets(TWEETS_DATA_PATH, OUT_DIR_PATH)
