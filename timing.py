from functools import wraps
from time import perf_counter


def measure_time(function):
    """Decorator koji cuva trajanje izvrsavanja metode."""

    @wraps(function)
    def wrapper(self, *args, **kwargs):
        start = perf_counter()
        try:
            return function(self, *args, **kwargs)
        finally:
            self.trajanje_sekundi = perf_counter() - start

    return wrapper
