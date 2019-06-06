from task1.src.ClassifierBase import ClassifierBase
from task1.src.CNNClassifier import CNNClassifier
from task1.src.General_classifier import General_classifier

class MegaClassifier(ClassifierBase):

    classifiers = [CNNClassifier, General_classifier]

    def __init__(self) -> None:
        for classifier in self.classifiers:
            classifier.init()

    def fit(self, X, y):
        pass

    def classify(self, X):
        pass

    def score(self, X, y):
        pass
