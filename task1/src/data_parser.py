import pandas as pd
import numpy as np
import os.path

TWEETS_DATA_PATH = r"D:\Oriyan\School\HUJI PDF\67577 - Introduction to Machine Learning\Hackathon\tweets_data"
NAMES_FILE = r"names.txt"
dir_contents = np.array([file for file in os.listdir(TWEETS_DATA_PATH)])
CSV_FILES = list(filter(lambda f: f.endswith(".csv"), os.listdir(TWEETS_DATA_PATH)))

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


def separate_training_and_validation_sets(tweets):
	for f in tweets:
		shuffled_tweets = tweets[f].sample(frac=1).reset_index(drop=True)
		# np.random.shuffle(shuffled_tweets)

		num_tweets = shuffled_tweets.shape[0]
		num_training_tweets = num_tweets - num_tweets // 10
		training_df = shuffled_tweets[:num_training_tweets]
		validation_df = shuffled_tweets[num_training_tweets:]
		validation_df.to_csv(os.path.join(TWEETS_DATA_PATH, "validation_set.csv"), sep=",", mode="a", index=False)
		training_df.to_csv(os.path.join(TWEETS_DATA_PATH, "training_set.csv"), sep=",", mode="a", index=False)


def get_word_counts(tweets, people):
	pass


if __name__ == "__main__":
	csv_files = [os.path.join(TWEETS_DATA_PATH, f) for f in CSV_FILES]
	names_full_path = os.path.join(TWEETS_DATA_PATH, NAMES_FILE)
	names = pd.read_csv(names_full_path, sep=",")
	tweets = {f[:-4]: pd.read_csv(f, sep=",") for f in csv_files}
	separate_training_and_validation_sets(tweets)
	pass
