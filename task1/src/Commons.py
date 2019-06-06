import os

TWEETS_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tweets_data"))
OUT_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "out"))

NAMES_FILE = r"names.txt"
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