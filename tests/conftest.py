import numpy as np
import pytest
from mlsampler import RandomSampler


@pytest.fixture
def X_mixed():
    """Dataset covering every dtype `setup()` can infer."""
    rng = np.random.default_rng(0)
    n = 50
    X = np.empty((n, 5), dtype=object)
    X[:, 0] = rng.uniform(0, 1, n)              # float
    X[:, 1] = rng.integers(1, 11, n)            # int
    X[:, 2] = rng.integers(0, 2, n)             # binary
    X[:, 3] = 5.0                               # constant
    X[:, 4] = rng.choice(["a", "b", "c"], n)    # categorical
    return X

@pytest.fixture
def X_numeric():
    """Five float columns plus one categorical, for exercising constraints."""
    rng = np.random.default_rng(0)
    n = 40
    X = np.empty((n, 6), dtype=object)
    for col in range(5):
        X[:, col] = rng.uniform(0, 10, n)
    X[:, 5] = rng.choice(["a", "b", "c"], n)
    return X

@pytest.fixture
def sampler(X_numeric):
    """Serial sampler over `X_numeric` with a fixed seed."""
    return RandomSampler.setup(X_numeric, random_state=0, n_jobs=1)
