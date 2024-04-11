import functools
import time
from typing import Any, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from sklearn.metrics.pairwise import cosine_similarity


def pairwise_cosine(x, y):
    return 1 - cosine_similarity(x, y)


def save_as_pickle(filename: str, obj):
    """
    Saves an object as a pickle file.
    Expects that the directory with the file exists.

    Parameters
    ----------
    filename : str
        Path to the file to write to.
    obj: object
        The object to save.
    """
    import pickle

    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_from_pickle(filename: str) -> Any:
    """
    Loads an object from a pickle file.
    Expects that the file exists.

    Parameters
    ----------
    filename : str
        Path to the file to read from.
    """
    import pickle

    with open(filename, "rb") as f:
        return pickle.load(f)


def log_runtime(level: int, message: str):
    """
    A decorator that prints the runtime of the decorated method.
    The decorated method must be a method of a class that has a logger property
    (e.g., a class that inherits from Logger).
    The message must contain a placeholder for the runtime.
    """

    def decorator_log_runtime(func):
        @functools.wraps(func)
        def wrapper_log_runtime(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            stop = time.time()

            self.logger.log(level, message.format(stop - start))

            return result

        return wrapper_log_runtime

    return decorator_log_runtime


def serialize(lst: List[Any]) -> str:
    """Serializes a list of objects into a string."""
    return ",".join(map(str, lst))


def filter_path_idxs(
    paths: npt.NDArray[Union[np.int32, np.int64]], path: Tuple
) -> npt.NDArray[np.int32]:
    """Returns the indexes of `paths` that match the given `path`."""
    return np.where(np.all(paths == np.array(path), axis=1))[0]
