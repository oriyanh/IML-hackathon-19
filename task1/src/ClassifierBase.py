import pandas as pd
import numpy as np


class ClassifierBase:

	def __init__(self) -> None:
		pass

	def fit(self, X, y):
		pass

	def classify(self, X):
		pass

	def score(self, X, y):
		pass
