import pytest

from ml_framework.iris_experiment import IrisExperiment
from ml_framework.metrics import AccuracyStrategy, F1MacroStrategy
from ml_framework.models import ModelFactory
from ml_framework.observers import ConsoleObserver


def test_factory_logistic():
    assert ModelFactory.create("logistic").__class__.__name__ == "LogisticRegression"


def test_factory_tree():
    assert ModelFactory.create("tree").__class__.__name__ == "DecisionTreeClassifier"


def test_factory_unknown_model():
    with pytest.raises(ValueError):
        ModelFactory.create("unknown")


def test_accuracy_strategy():
    assert AccuracyStrategy().calculate([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)


def test_f1_strategy():
    score = F1MacroStrategy().calculate([0, 1, 1], [0, 1, 0])
    assert 0 <= score <= 1


def test_iris_experiment():
    result = IrisExperiment("knn", AccuracyStrategy()).run()
    assert result.model_name == "knn"
    assert 0 <= result.metric_value <= 1
    assert result.train_size + result.test_size == 150


def test_observer(capsys):
    ConsoleObserver().update("test_event", "poruka")
    output = capsys.readouterr().out
    assert "test_event" in output
    assert "poruka" in output
