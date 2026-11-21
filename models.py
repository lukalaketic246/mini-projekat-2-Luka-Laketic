from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class ModelFactory:
    """Factory: kreira klasifikacioni model po nazivu."""

    @staticmethod
    def create(name: str):
        name = name.lower()
        if name in ("logistic", "logistic_regression"):
            return LogisticRegression(max_iter=1000, random_state=42)
        if name in ("tree", "decision_tree"):
            return DecisionTreeClassifier(random_state=42)
        if name == "knn":
            return KNeighborsClassifier(n_neighbors=5)
        raise ValueError(f"Nepoznat model: {name}")
