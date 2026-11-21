from ml_framework.iris_experiment import IrisExperiment
from ml_framework.metrics import AccuracyStrategy
from ml_framework.observers import ConsoleObserver


def main():
    experiment = IrisExperiment("logistic", AccuracyStrategy(), [ConsoleObserver()])
    result = experiment.run()
    print("\\nRezultat eksperimenta")
    print(f"Model: {result.model_name}")
    print(f"Metrika: {result.metric_name}")
    print(f"Vrijednost: {result.metric_value:.4f}")
    print(f"Trening uzorci: {result.train_size}")
    print(f"Test uzorci: {result.test_size}")
    print(f"Trajanje: {experiment.trajanje_sekundi:.4f} sekundi")


if __name__ == "__main__":
    main()
