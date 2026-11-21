from sklearn.datasets import load_iris

from ml_framework.experiment import BaseExperiment
from ml_framework.models import ModelFactory


class IrisExperiment(BaseExperiment):
    def __init__(self, model_name: str, metric, observers=None):
        super().__init__(metric, observers)
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def load_data(self):
        dataset = load_iris()
        return dataset.data, dataset.target

    def create_model(self):
        return ModelFactory.create(self.model_name)
