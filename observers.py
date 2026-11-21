from typing import Protocol


class TrainingObserver(Protocol):
    def update(self, event: str, details: str) -> None: ...


class ConsoleObserver:
    def update(self, event: str, details: str) -> None:
        print(f"[{event}] {details}")
