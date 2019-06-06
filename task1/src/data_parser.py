import pandas as pd
import numpy as np
import os.path
from task1.src.Commons import *


def separate_training_and_validation_sets(tweet_csv_files, out_path=OUT_DIR_PATH):
	"""
	Creates a training set and a validation set from sequence of CSV files, with columns 'user','tweet' .
	:param tweet_csv_files: Sequence of absolute paths to CSV files that contain tweet data.
	:param out_path: Path to output directory, to save valiation and training set CSVs. May be relative path.
	Defaults to '../../out/' . Results are saved to 'out_path/validation_set.csv' and
	'out_path/training_set.csv' . OVERWRITES EXISTING FILES!
	"""
	abs_out_path = os.path.abspath(out_path) if not os.path.isabs(out_path) else out_path
	csv_files = [os.path.join(tweet_csv_files, f) for f in CSV_FILE_NAMES]
	if not os.path.exists(abs_out_path):
		os.mkdir(abs_out_path)
	elif not os.path.isdir(OUT_DIR_PATH):
		raise Exception("Output location is not a directory: [%s]" % out_path)
	val_set_path = os.path.join(abs_out_path, "validation_set.csv")
	train_set_path = os.path.join(abs_out_path, "training_set.csv")
	if os.path.isfile(val_set_path):
		os.remove(val_set_path)
	if os.path.isfile(train_set_path):
		os.remove(train_set_path)
	tweets = {f[:-4]: pd.read_csv(f, sep=",", encoding='utf-8') for f in csv_files}
	for f in tweets:
		shuffled_tweets = tweets[f].sample(frac=1)
		num_tweets = shuffled_tweets.shape[0]
		num_training_tweets = num_tweets - num_tweets // 10
		training_df = shuffled_tweets[:num_training_tweets]
		validation_df = shuffled_tweets[num_training_tweets:]
		csv_to_file(validation_df, val_set_path)
		csv_to_file(training_df, train_set_path)
	shuffle_csv(val_set_path, ['user', 'tweet'])
	shuffle_csv(train_set_path, ['user', 'tweet'])

def shuffle_csv(path, columns=None):
	shuffled_df = pd.read_csv(path, sep=",", encoding='utf-8').sample(frac=1)
	csv_to_file(shuffled_df, path, "w", columns)

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
	df.to_csv(path, sep=",", mode=mode, header=header, index=False, encoding='utf-8')

def get_word_counts(tweets, people):
	"""

	:param tweets:
	:type people: np.ndarray
	:param people:
	:type people: np.ndarray
	:return:
	"""
	for p in HANDLES_DICT.keys():
		p_indices = np.argwhere(people == p)
		p_tweets = tweets[p_indices]



if __name__ == "__main__":
	names_full_path = os.path.join(TWEETS_DATA_PATH, NAMES_FILE)
	names = pd.read_csv(names_full_path, sep=",")

	separate_training_and_validation_sets(TWEETS_DATA_PATH, OUT_DIR_PATH)
