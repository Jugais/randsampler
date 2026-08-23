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

class TestMissingValues:
    """`setup` drops missing values column by column and infers from what is left."""

    @staticmethod
    def _with_gap(values, gap):
        """`values` plus one missing cell, beside a column that is always present."""
        X = np.empty((len(values) + 1, 2), dtype=object)
        X[:, 0] = list(values) + [gap]
        X[:, 1] = np.arange(len(values) + 1, dtype=float)
        return X

    @pytest.mark.parametrize("gap", [None, np.nan, np.float32("nan")])
    @pytest.mark.parametrize(
        "values, dtype, low, high",
        [
            ([0.5, 2.5, 1.5], dm.float, 0.5, 2.5),
            ([1, 5, 3], dm.integer, 1.0, 5.0),
            ([0, 1, 1], dm.bin, 0.0, 1.0),
            ([7.0, 7.0, 7.0], dm.const, 7.0, 7.0),
        ],
    )
    def test_numeric_dtype_and_bounds_ignore_the_gap(self, values, dtype, low, high, gap):
        """np.float32 is included because it does not subclass Python's float."""
        f = RandomSampler.setup(self._with_gap(values, gap)).config.features[0]
        assert f.dtype == dtype
        assert f.low == low and f.high == high

    @pytest.mark.parametrize("gap", [None, np.nan, np.float32("nan")])
    def test_categories_never_include_the_gap(self, gap):
        f = RandomSampler.setup(self._with_gap(["a", "b", "a"], gap)).config.features[0]
        assert f.dtype == dm.cat
        assert f.categories == ["a", "b"]

    def test_a_single_valued_categorical_column_with_a_gap_stays_constant(self):
        f = RandomSampler.setup(self._with_gap(["A", "A", "A"], np.nan)).config.features[0]
        assert f.dtype == dm.const

    def test_gaps_are_independent_per_column(self):
        """The mask is read per column, so the positions need not line up."""
        X = np.array(
            [
                [1.0, "a", 10],
                [np.nan, "b", 20],
                [3.0, np.nan, 30],
                [4.0, "a", None],
            ],
            dtype=object,
        )
        features = RandomSampler.setup(X).config.features
        assert [f.dtype for f in features] == [dm.integer, dm.cat, dm.integer]
        assert (features[0].low, features[0].high) == (1.0, 4.0)
        assert features[1].categories == ["a", "b"]
        assert (features[2].low, features[2].high) == (10.0, 30.0)

    def test_an_input_without_gaps_is_unaffected(self, X_mixed):
        """The mask is all False, so inference is the same as before."""
        dtypes = [f.dtype for f in RandomSampler.setup(X_mixed).config.features]
        assert dtypes == [dm.float, dm.integer, dm.bin, dm.const, dm.cat]


class TestDegenerateMissing:
    """Two independent checks: a row with nothing in it, and a column with nothing
    to infer from. Either can hold without the other."""

    def test_an_all_missing_row_raises(self):
        X = np.array([[1.0, "a"], [np.nan, None], [3.0, "b"]], dtype=object)
        with pytest.raises(ValueError, match=r"Rows \[1\] are entirely missing"):
            RandomSampler.setup(X)

    def test_an_all_missing_column_raises_though_every_row_has_a_value(self):
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]], dtype=object)
        with pytest.raises(ValueError, match=r"Columns \[1\] are entirely missing"):
            RandomSampler.setup(X)

    def test_scattered_gaps_trigger_neither(self):
        X = np.array([[1.0, "a"], [np.nan, "b"], [3.0, None]], dtype=object)
        assert RandomSampler.setup(X).n_features == 2


class TestMissingNeverReachesTheOutput:
    """Accepting gaps on the way in does not put any on the way out."""

    X = np.array([[1.0, "a"], [np.nan, "b"], [3.0, None], [4.0, "a"]], dtype=object)

    def test_no_sampled_value_is_missing(self):
        from mlsampler.base import _is_missing

        out = RandomSampler.setup(self.X, random_state=0, n_jobs=1).sample(50)
        assert not any(_is_missing(val) for val in out.ravel())

    def test_the_output_round_trips_through_setup(self):
        out = RandomSampler.setup(self.X, random_state=0, n_jobs=1).sample(50)
        assert RandomSampler.setup(out).n_features == 2

def test_setup_forwards_config(X_mixed):
    sampler = RandomSampler.setup(X_mixed, random_state=7, n_jobs=1, max_retries=5)
    assert sampler.config.random_state == 7
    assert sampler.config.n_jobs == 1
    assert sampler.config.max_retries == 5
