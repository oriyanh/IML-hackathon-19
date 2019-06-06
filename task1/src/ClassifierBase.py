import pandas as pd
import numpy as np


class ClassifierBase:

	def __init__(self) -> None:
		super().__init__()

	def fit(self, X, y):
		label, handle = y

	def classify(self, X):
		pass

	def score(self, X, y):
		pass