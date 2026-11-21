from abc import ABC, abstractmethod
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from ml_framework.metrics import MetricStrategy
from ml_framework.timing import measure_time


@dataclass
class ExperimentResult:
    model_name: str
    metric_name: str
    metric_value: float
    train_size: int
    test_size: int


class BaseExperiment(ABC):
    def __init__(self, metric: MetricStrategy, observers=None):
        self.metric = metric
        self.observers = observers or []
        self.trajanje_sekundi = 0.0

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def create_model(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    def notify(self, event: str, details: str) -> None:
        for observer in self.observers:
            observer.update(event, details)

    @measure_time
    def run(self) -> ExperimentResult:
        # Template Method: redosljed koraka eksperimenta.
        features, labels = self.load_data()
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, random_state=42, stratify=labels
        )
        model = self.create_model()
        self.notify("training_started", self.model_name)
        model.fit(x_train, y_train)
        self.notify("training_finished", self.model_name)
        predictions = model.predict(x_test)
        score = self.metric.calculate(y_test, predictions)
        self.notify(
            "evaluation_finished", f"{self.metric.__class__.__name__}: {score:.4f}"
        )
        return ExperimentResult(
            self.model_name,
            self.metric.__class__.__name__,
            score,
            len(x_train),
            len(x_test),
        )
