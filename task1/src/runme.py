import os

from task1.src.Commons import OUT_DIR_PATH
from task1.src.data_parser import Parser
from task1.src.naive_bayes import naive_bayes


def main():
	training_set_path = os.path.join(OUT_DIR_PATH, 'training_set.csv')
	X, y = Parser.load_csv_to_array(training_set_path)

	bayes_cls = naive_bayes()
	bayes_cls.fit(X, y)
