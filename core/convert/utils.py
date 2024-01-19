from typing import Callable, Sequence


def multiple(conversion_fn: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        if isinstance(args[0], Sequence):
            return [
                conversion_fn(arg, **kwargs)
                for arg in args[0]
            ]
        else:
            return conversion_fn(*args, **kwargs)

    return wrapper
