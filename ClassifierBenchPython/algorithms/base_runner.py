from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
from typing import Dict
import numpy as np

class AlgorithmRunner(ABC):
    def __init__(self):
        self.pipeline: Pipeline | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def data_preprocessor(self):
        """Возвращает препроцессор sklearn, например ColumnTransformer."""
        pass

    @property
    @abstractmethod
    def estimator(self):
        """Возвращает классификатор sklearn, например LogisticRegression."""
        pass

    def prepare_data(self, X):
        return self.data_preprocessor.fit_transform(X)

    def train(self, X, y):
        self.pipeline = Pipeline([
            ("prep", self.data_preprocessor),  # например, StandardScaler или OneHotEncoder
            ("model", self.estimator),  # например, SVC или RandomForest
        ])
        self.pipeline.fit(X, y)

    def predict(self, X):
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        return self.pipeline.predict(X)

    @abstractmethod
    def evaluate(self, y_true, y_pred) -> Dict[str, float]:
        pass
