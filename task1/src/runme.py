import os

from task1.src.Commons import OUT_DIR_PATH, split_training_validation_sets
from task1.src.data_parser import Parser
from task1.src.naive_bayes import naive_bayes




def main():
	training_set_path = os.path.join(OUT_DIR_PATH, 'training_set.csv')
	X, y = Parser.load_csv_to_array(training_set_path)
	S, V = split_training_validation_sets(X, y, 0.8)
	bayes_cls = naive_bayes()
	bayes_cls.fit(*S)

	pass



if __name__ == "__main__":
	main()