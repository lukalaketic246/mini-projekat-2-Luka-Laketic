from typing import Protocol

from sklearn.metrics import accuracy_score, f1_score


class MetricStrategy(Protocol):
    def calculate(self, actual, predicted) -> float: ...


class AccuracyStrategy:
    def calculate(self, actual, predicted) -> float:
        return float(accuracy_score(actual, predicted))


class F1MacroStrategy:
    def calculate(self, actual, predicted) -> float:
        return float(f1_score(actual, predicted, average="macro"))
