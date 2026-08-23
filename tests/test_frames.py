# DataFrame support is duck-typed: `setup` looks for `.columns` and `.to_numpy`
# rather than importing anything, so pandas and polars go down one code path and
# are parametrized together. Cases that only one library can express (pandas alone
# allows integer and duplicate labels) are tested against that library directly.
import warnings

import numpy as np
import pytest

from mlsampler import RandomSampler, HyperGridSampler
from mlsampler.errors import ConstraintValidationError

pd = pytest.importorskip("pandas")
pl = pytest.importorskip("polars")

SAMPLERS = [RandomSampler, HyperGridSampler]
N = 12

DATA = {
    "temp": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    "n": [1, 2, 3, 4, 5, 6],
    "cat": ["x", "y", "z", "x", "y", "z"],
    "k": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
}


@pytest.fixture(params=["pandas", "polars"])
def frame(request):
    """The same table as a pandas and as a polars DataFrame."""
    build = pd.DataFrame if request.param == "pandas" else pl.DataFrame
    return request.param, build(DATA)


def columns_of(out):
    return list(out.columns)


def dtypes_of(out):
    return [str(dtype) for dtype in out.dtypes]


class TestEntry:
    def test_column_names_become_feature_names(self, frame):
        _, df = frame
        assert RandomSampler.setup(df).feature_names == list(DATA)

    def test_dtypes_are_still_inferred_through_the_conversion(self, frame):
        _, df = frame
        dtypes = [f.dtype for f in RandomSampler.setup(df).config.features]
        assert dtypes == ["float", "int", "categorical", "constant"]

    def test_an_array_keeps_positional_names(self):
        X = np.array([[1.5, "a"], [2.5, "b"]], dtype=object)
        sampler = RandomSampler.setup(X)
        assert sampler.feature_names == [0, 1]
        assert sampler.config.frame is None

    def test_setup_records_which_library_it_was_given(self, frame):
        name, df = frame
        assert RandomSampler.setup(df).config.frame == name

    def test_name_is_never_left_unset(self, frame):
        _, df = frame
        assert all(f.name is not None for f in RandomSampler.setup(df).config.features)

    @pytest.mark.parametrize("values", [DATA, {"a": [1.5, 2.5], "b": [1, 2]}])
    def test_numeric_only_and_mixed_frames_both_convert(self, values):
        """A numeric-only frame yields float64 from `to_numpy`, a mixed one object."""
        assert RandomSampler.setup(pd.DataFrame(values)).n_features == len(values)


class TestColumnLabels:
    """Labels must be all strings or exactly 0..n-1; anything else is ambiguous
    against the rule that an int always means a position."""

    def test_unnamed_pandas_frame_is_treated_as_positional(self):
        df = pd.DataFrame(np.array([[1.5, 2.5], [3.5, 4.5]]))
        assert RandomSampler.setup(df).feature_names == [0, 1]

    def test_explicit_range_labels_are_treated_as_positional(self):
        df = pd.DataFrame([[1.5, 2.5], [3.5, 4.5]], columns=[0, 1])
        assert RandomSampler.setup(df).feature_names == [0, 1]

    @pytest.mark.parametrize(
        "columns",
        [
            pytest.param([5, 3, 9], id="non_contiguous_ints"),
            pytest.param(["a", 1, "c"], id="mixed_types"),
            pytest.param(
                pd.MultiIndex.from_tuples([("x", "p"), ("x", "q"), ("x", "r")]),
                id="multiindex",
            ),
        ],
    )
    def test_rejected_labels(self, columns):
        df = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=columns)
        with pytest.raises(ValueError, match="Column labels"):
            RandomSampler.setup(df)

    def test_duplicate_names_are_rejected(self):
        """pandas allows them; a name would then identify two columns. polars
        refuses to build such a frame, so this case is pandas-only."""
        df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "a"])
        with pytest.raises(ValueError, match="Duplicate column names"):
            RandomSampler.setup(df)


class TestNameResolution:
    def test_a_name_selects_the_same_column_as_its_position(self, frame):
        _, df = frame

        def run(cols):
            sampler = RandomSampler.setup(df, random_state=0)
            sampler.set_constraints("range", cols=cols, low=0.0, high=1.0)
            return sampler.sample(N)

        assert run(["temp"]).equals(run([0]))

    def test_names_are_resolved_before_the_constraint_is_built(self, frame):
        _, df = frame
        sampler = RandomSampler.setup(df)
        sampler.set_constraints("range", cols=["temp", "n"], low=0.0, high=1.0)
        assert sampler.constraints[0].cols == [0, 1]

    def test_step_resolves_its_singular_col(self, frame):
        _, df = frame
        sampler = RandomSampler.setup(df)
        sampler.set_constraints("step", col="n", step=1.0, low=0.0, high=5.0)
        assert sampler.constraints[0].col == 1

    def test_positions_still_work_on_a_named_frame(self, frame):
        """Names are an added option, not a replacement."""
        _, df = frame
        sampler = RandomSampler.setup(df)
        sampler.set_constraints("range", cols=[0], low=0.0, high=1.0)
        assert sampler.constraints[0].cols == [0]

    def test_positions_and_names_can_be_mixed(self, frame):
        _, df = frame
        sampler = RandomSampler.setup(df)
        sampler.set_constraints("range", cols=[0, "n"], low=0.0, high=1.0)
        assert sampler.constraints[0].cols == [0, 1]

    def test_unknown_name_raises_and_lists_the_valid_ones(self, frame):
        _, df = frame
        sampler = RandomSampler.setup(df)
        with pytest.raises(ConstraintValidationError, match="Unknown column 'nope'"):
            sampler.set_constraints("range", cols=["nope"], low=0.0, high=1.0)

    def test_a_name_on_an_unnamed_sampler_raises(self):
        sampler = RandomSampler.setup(np.zeros((5, 3)))
        with pytest.raises(ConstraintValidationError, match="no column names"):
            sampler.set_constraints("range", cols=["temp"], low=0.0, high=1.0)

    def test_lows_and_highs_are_not_treated_as_column_references(self, frame):
        """`stepsum` takes per-column values there, not column names."""
        _, df = frame
        sampler = RandomSampler.setup(df)
        sampler.set_constraints(
            "stepsum", cols=["temp", "n"], sum_value=4, lows=[0, 0], highs=[4, 4], step=1
        )
        assert sampler.constraints[0].cols == [0, 1]


class TestOutput:
    @pytest.mark.parametrize("cls", SAMPLERS, ids=lambda c: c.__name__)
    def test_output_is_the_input_type(self, frame, cls):
        name, df = frame
        out = cls.setup(df, random_state=0).sample(N)
        assert type(out).__module__.split(".")[0] == name

    @pytest.mark.parametrize("cls", SAMPLERS, ids=lambda c: c.__name__)
    def test_column_names_and_order_survive(self, frame, cls):
        _, df = frame
        out = cls.setup(df, random_state=0).sample(N)
        assert columns_of(out) == list(DATA)

    @pytest.mark.parametrize("cls", SAMPLERS, ids=lambda c: c.__name__)
    def test_per_column_dtypes_are_restored(self, frame, cls):
        """Handing the object array over whole would leave every column object;
        a polars Object column cannot even be summed."""
        name, df = frame
        out = cls.setup(df, random_state=0).sample(N)
        expected = (
            ["float64", "int64", "object", "float64"]
            if name == "pandas"
            else ["Float64", "Int64", "String", "Float64"]
        )
        assert dtypes_of(out) == expected

    @pytest.mark.parametrize("cls", SAMPLERS, ids=lambda c: c.__name__)
    def test_row_count_is_exact(self, frame, cls):
        _, df = frame
        assert cls.setup(df, random_state=0).sample(N).shape[0] == N

    @pytest.mark.parametrize("cls", SAMPLERS, ids=lambda c: c.__name__)
    def test_an_array_still_returns_an_array(self, cls):
        X = np.array([[1.5, "a"], [2.5, "b"]], dtype=object)
        out = cls.setup(X, random_state=0).sample(N)
        assert isinstance(out, np.ndarray) and out.dtype == object

    def test_an_unnamed_frame_still_round_trips(self):
        df = pd.DataFrame(np.array([[1.5, 2.5], [3.5, 4.5]]))
        out = RandomSampler.setup(df, random_state=0).sample(N)
        assert isinstance(out, pd.DataFrame) and columns_of(out) == [0, 1]

    def test_constraints_hold_in_the_returned_frame(self, frame):
        _, df = frame
        sampler = RandomSampler.setup(df, random_state=0)
        sampler.set_constraints("range", cols=["temp"], low=2.0, high=3.0)
        temp = sampler.sample(N)["temp"]
        values = temp.to_numpy() if hasattr(temp, "to_numpy") else temp
        assert values.min() >= 2.0 and values.max() <= 3.0

    @pytest.mark.parametrize("cls", SAMPLERS, ids=lambda c: c.__name__)
    def test_a_constraint_may_widen_an_int_column_to_float(self, frame, cls):
        """`setup` inferred `n` as int, but a constraint can write floats into it.
        Forcing the inferred type here truncated silently (pandas) or raised
        (polars), so the rebuilt column follows the values, not the metadata."""
        _, df = frame
        sampler = RandomSampler.setup(df, random_state=0)
        sampler.set_constraints("range", cols=["n"], low=0.0, high=1.0)
        out = sampler.sample(N)
        values = out["n"].to_numpy()
        assert "int" not in dtypes_of(out)[1].lower()
        assert (values > 0).all() and (values < 1).all()

    def test_an_untouched_int_column_stays_int(self, frame):
        name, _ = frame
        _, df = frame
        out = RandomSampler.setup(df, random_state=0).sample(N)
        assert dtypes_of(out)[1] == ("int64" if name == "pandas" else "Int64")

    def test_numeric_columns_are_usable_without_casting(self, frame):
        """The point of restoring dtypes: arithmetic works on the result."""
        name, df = frame
        out = RandomSampler.setup(df, random_state=0).sample(N)
        total = out["temp"].sum() if name == "pandas" else out["temp"].sum()
        assert isinstance(total, float)


class TestMessages:
    """Errors and warnings address columns the way the user did."""

    def test_duplicate_column_warning_names_the_column(self, frame):
        _, df = frame
        sampler = RandomSampler.setup(df, random_state=0)
        sampler.set_constraints("range", cols=["temp"], low=0.0, high=1.0)
        sampler.set_constraints("sum", cols=["temp"], sum_value=1.0)
        with pytest.warns(UserWarning, match="Column 'temp'"):
            sampler.sample(N)

    def test_categorical_conflict_names_the_column(self, frame):
        _, df = frame
        sampler = RandomSampler.setup(df, random_state=0)
        sampler.set_constraints("range", cols=["cat"], low=0.0, high=1.0)
        with pytest.raises(ValueError, match="'cat'"):
            sampler.sample(N)

    def test_max_retries_diagnosis_names_the_column(self, frame):
        _, df = frame
        sampler = RandomSampler.setup(df, random_state=0, max_retries=2)
        sampler.set_constraints(lambda row: False, cols=["temp"])
        with pytest.raises(ValueError, match=r"cols=\['temp'\]"):
            sampler.sample(1)

    def test_an_unnamed_sampler_still_reports_positions(self, X_numeric):
        sampler = RandomSampler.setup(X_numeric, random_state=0)
        sampler.set_constraints("range", cols=[0], low=0.0, high=1.0)
        sampler.set_constraints("sum", cols=[0], sum_value=1.0)
        with pytest.warns(UserWarning, match="Column 0"):
            sampler.sample(N)


class TestMissingValues:
    """Both libraries hand `setup` the same thing after `.to_numpy()`: a numeric null
    becomes `nan`, a string null becomes `None`."""

    GAPPED = {
        "temp": [1.5, None, 3.5, 4.5],
        "n": [1, 2, None, 4],
        "cat": ["x", "y", "z", None],
    }

    @pytest.fixture(params=["pandas", "polars"])
    def gapped(self, request):
        build = pd.DataFrame if request.param == "pandas" else pl.DataFrame
        return build(self.GAPPED)

    def test_dtypes_are_inferred_from_the_present_values(self, gapped):
        features = RandomSampler.setup(gapped).config.features
        assert [f.dtype for f in features] == ["float", "int", "categorical"]

    def test_bounds_and_categories_skip_the_gaps(self, gapped):
        features = RandomSampler.setup(gapped).config.features
        assert (features[0].low, features[0].high) == (1.5, 4.5)
        assert (features[1].low, features[1].high) == (1.0, 4.0)
        assert features[2].categories == ["x", "y", "z"]

    @pytest.mark.parametrize("cls", SAMPLERS)
    def test_the_output_carries_no_gaps(self, gapped, cls):
        from mlsampler.base import _is_missing

        out = cls.setup(gapped, random_state=0).sample(N)
        assert not any(_is_missing(val) for val in out.to_numpy().ravel())

    def test_an_all_missing_column_is_named_not_numbered(self, frame):
        name, df = frame
        build = pd.DataFrame if name == "pandas" else pl.DataFrame
        empty = build({"good": [1.0, 2.0], "hollow": [None, None]})
        with pytest.raises(ValueError, match="'hollow'"):
            RandomSampler.setup(empty)


class TestNonFiniteRebuild:
    """`_column` casts an int column to int64 to check whether the values are still
    whole. A NaN made that cast emit RuntimeWarning even though the fallback to
    float was correct."""

    def test_a_callable_writing_nan_does_not_warn(self, frame):
        _, df = frame
        sampler = RandomSampler.setup(df, random_state=0, n_jobs=1)
        sampler.set_constraints(lambda row: np.array([np.nan]), cols=["n"])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            out = sampler.sample(N)
        assert str(out["n"].dtype).lower().startswith("float")

    def test_an_int_column_without_gaps_is_still_rebuilt_as_int(self, frame):
        _, df = frame
        out = RandomSampler.setup(df, random_state=0).sample(N)
        assert str(out["n"].dtype).lower().startswith("int")
