from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from .base_runner import AlgorithmRunner

class NaiveBayesRunner(AlgorithmRunner):
    @property
    def name(self):
        return "NaiveBayes"

    @property
    def data_preprocessor(self):
        return StandardScaler()

    @property
    def estimator(self):
        return GaussianNB()

    def evaluate(self, y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred)
        }
