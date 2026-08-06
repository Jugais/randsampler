import numpy as np
import pytest
from mlsampler import RandomSampler
from mlsampler.base import DtypeMeta as dm

def test_infers_one_feature_per_column(X_mixed):
    sampler = RandomSampler.setup(X_mixed)
    assert sampler.n_features == X_mixed.shape[1]
    assert [f.index for f in sampler.config.features] == list(range(X_mixed.shape[1]))

def test_dtype_inference_order(X_mixed):
    dtypes = [f.dtype for f in RandomSampler.setup(X_mixed).config.features]
    assert dtypes == [dm.float, dm.integer, dm.bin, dm.const, dm.cat]

def test_numeric_bounds_come_from_the_data(X_mixed):
    features = RandomSampler.setup(X_mixed).config.features
    for col in range(4):
        col_data = X_mixed[:, col].astype(float)
        assert features[col].low == col_data.min()
        assert features[col].high == col_data.max()

def test_categorical_carries_categories_and_no_bounds(X_mixed):
    f = RandomSampler.setup(X_mixed).config.features[4]
    assert f.categories == ["a", "b", "c"]
    assert f.low is None and f.high is None

def test_numeric_columns_carry_no_categories(X_mixed):
    features = RandomSampler.setup(X_mixed).config.features
    assert all(f.categories is None for f in features[:4])

@pytest.mark.parametrize(
    "values, dtype",
    [
        ([0, 1, 0, 1], dm.bin),
        ([0.0, 1.0, 0.0, 1.0], dm.bin),
        ([1, 2, 3, 4], dm.integer),
        ([1.5, 2.5, 3.5], dm.float),
        ([7, 7, 7, 7], dm.const),
        (["x", "y", "z"], dm.cat),
        (["x", "x", "x"], dm.const),
    ],
)
def test_single_column_dtype(values, dtype):
    """Constant short-circuits before binary, and a single-valued
    non-numeric column is constant rather than categorical."""
    X = np.array(values, dtype=object).reshape(-1, 1)
    assert RandomSampler.setup(X).config.features[0].dtype == dtype

@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_values_raise(missing):
    X = np.array([1.0, 2.0, missing], dtype=object).reshape(-1, 1)
    with pytest.raises(ValueError, match="missing values"):
        RandomSampler.setup(X)

def test_setup_forwards_config(X_mixed):
    sampler = RandomSampler.setup(X_mixed, random_state=7, n_jobs=1, max_retries=5)
    assert sampler.config.random_state == 7
    assert sampler.config.n_jobs == 1
    assert sampler.config.max_retries == 5
