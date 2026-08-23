import numpy as np
import pytest
from mlsampler import RandomSampler
from mlsampler.errors import ConstraintValidationError, ConstraintViolationError

N = 30

def test_unconstrained_sample_respects_feature_bounds(sampler, X_numeric):
    out = sampler.sample(N)
    assert out.shape == (N, X_numeric.shape[1])
    for f in sampler.config.features[:5]:
        col = out[:, f.index].astype(float)
        assert col.min() >= f.low and col.max() <= f.high

def test_unconstrained_sample_draws_from_categories(sampler):
    out = sampler.sample(N)
    assert set(out[:, 5]) <= {"a", "b", "c"}

def test_sum(sampler):
    sampler.set_constraints("sum", cols=[0, 1, 2], sum_value=1.0)
    out = sampler.sample(N)[:, [0, 1, 2]].astype(float)
    assert np.allclose(out.sum(axis=1), 1.0)

def test_sumint(sampler):
    sampler.set_constraints("sumint", cols=[0, 1, 2], sum_value=100)
    out = sampler.sample(N)[:, [0, 1, 2]].astype(float)
    assert np.allclose(out.sum(axis=1), 100)
    assert np.all(out == out.astype(int))
    assert np.all(out >= 0)

def test_multihot(sampler):
    sampler.set_constraints("multihot", cols=[0, 1, 2], n_hot=2)
    out = sampler.sample(N)[:, [0, 1, 2]].astype(float)
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert np.all(out.sum(axis=1) == 2)


def test_random_select(sampler):
    """`max_used` is guaranteed: writing 0 can only lower the count of non-zero
    columns. `min_used` is not -- see TestRandomSelectLowerBound."""
    sampler.set_constraints("random", cols=[0, 1, 2], min_used=1, max_used=2)
    out = sampler.sample(N)[:, [0, 1, 2]].astype(float)
    assert np.all((out != 0).sum(axis=1) <= 2)


class TestRandomSelectLowerBound:
    """`random` writes nothing into the columns it selects, so a selected column
    whose base draw is already 0 is indistinguishable from an unselected one.
    `min_used` therefore counts columns selected, not columns that end up non-zero.

    This is invisible on float columns, where 0 has measure zero -- which is why
    the fixture above never caught it. The xfail pins the defect; when TODO.md A-5
    lands it will XPASS and force this to be rewritten."""

    @staticmethod
    def _binary(n_cols):
        rng = np.random.default_rng(0)
        X = np.empty((30, n_cols), dtype=object)
        for col in range(n_cols):
            X[:, col] = rng.integers(0, 2, 30)
        return X

    def _sample(self, n_cols=3, min_used=2):
        sampler = RandomSampler.setup(self._binary(n_cols), random_state=0, n_jobs=1)
        sampler.set_constraints(
            "random", cols=list(range(n_cols)), min_used=min_used, max_used=n_cols
        )
        return sampler.sample(300)[:, :n_cols].astype(float)

    def test_max_used_still_holds_on_discrete_columns(self):
        assert np.all((self._sample() != 0).sum(axis=1) <= 3)

    @pytest.mark.xfail(
        reason="min_used counts selected columns, not non-zero ones (TODO.md A-5)",
        strict=True,
    )
    def test_min_used_is_not_enforced_on_discrete_columns(self):
        assert np.all((self._sample() != 0).sum(axis=1) >= 2)


def test_range_overrides_the_inferred_bounds(sampler):
    sampler.set_constraints("range", cols=[0], low=2.0, high=3.0)
    out = sampler.sample(N)[:, 0].astype(float)
    assert out.min() >= 2.0 and out.max() <= 3.0


def test_step_lands_on_the_grid(sampler):
    """`step` takes `col` (singular) where every other constraint takes `cols`."""
    sampler.set_constraints("step", col=0, step=0.5, low=0.0, high=5.0)
    out = sampler.sample(N)[:, 0].astype(float)
    grid = np.arange(0.0, 5.5, 0.5)
    assert np.all([np.any(np.isclose(v, grid)) for v in out])


def test_stepsum(sampler):
    sampler.set_constraints(
        "stepsum", cols=[0, 1, 2], sum_value=10, lows=[0, 0, 0], highs=[10, 10, 10], step=1
    )
    out = sampler.sample(N)[:, [0, 1, 2]].astype(float)
    assert np.allclose(out.sum(axis=1), 10)
    assert np.allclose(out % 1, 0)
    assert np.all(out >= 0) and np.all(out <= 10)


class TestStepGrid:
    """The grid comes from `(high - low) / step`, which floating point rounds down:
    `(0.3 - 0.0) / 0.1` is 2.9999999999999996, so `high` used to fall off the end."""

    @staticmethod
    def _grid(sampler, low, high, step):
        sampler.set_constraints("step", col=0, step=step, low=low, high=high)
        return sampler.constraints[0].values

    @pytest.mark.parametrize(
        "low, high, step",
        [
            (0.0, 0.3, 0.1),        # regression: floor gave 2 steps, dropping 0.3
            (0.0, 0.7, 0.1),        # regression: floor gave 6 steps, dropping 0.7
            (0.0, 1.0, 0.1),
            (0.0, 1.0, 0.25),
            (1.0, 2.0, 0.1),
            (-5.0, 5.0, 0.1),
            (0.0, 10.0, 0.5),
        ],
    )
    def test_high_is_on_the_grid_when_it_is_a_multiple_of_step(self, sampler, low, high, step):
        assert np.isclose(self._grid(sampler, low, high, step)[-1], high)

    @pytest.mark.parametrize(
        "low, high, step", [(0.0, 1.0, 0.3), (0.0, 2.5, 1.0), (0.0, 10.0, 0.7)]
    )
    def test_the_grid_stops_below_high_when_it_is_not(self, sampler, low, high, step):
        values = self._grid(sampler, low, high, step)
        assert values[-1] <= high + 1e-9
        assert values[-1] + step > high

    def test_high_actually_gets_sampled(self, X_numeric):
        s = RandomSampler.setup(X_numeric, random_state=0, n_jobs=1)
        s.set_constraints("step", col=0, step=0.1, low=0.0, high=0.3)
        assert np.isclose(s.sample(200)[:, 0].astype(float), 0.3).any()

    def test_stepsum_is_unaffected(self, sampler):
        """`stepsum` inherits `StepConstraint` but overrides `_constrain` and never
        reads `self.values`."""
        sampler.set_constraints(
            "stepsum", cols=[0, 1, 2], sum_value=0.3,
            lows=[0, 0, 0], highs=[0.3, 0.3, 0.3], step=0.1,
        )
        assert np.allclose(sampler.sample(N)[:, [0, 1, 2]].astype(float).sum(axis=1), 0.3)


class TestStepSumValidation:
    """Feasibility follows from the arguments alone, so it is decided at construction
    instead of raised once per row from inside `sample()`."""

    def test_lows_exceeding_sum_value_is_rejected(self, sampler):
        with pytest.raises(ConstraintValidationError, match="exceeds sum_value"):
            sampler.set_constraints(
                "stepsum", cols=[0, 1, 2], sum_value=1,
                lows=[5, 5, 5], highs=[10, 10, 10], step=1,
            )

    def test_a_residual_off_the_step_grid_is_rejected(self, sampler):
        with pytest.raises(ConstraintValidationError, match="not a multiple of step"):
            sampler.set_constraints(
                "stepsum", cols=[0, 1, 2], sum_value=10,
                lows=[0, 0, 0], highs=[10, 10, 10], step=3,
            )

    def test_a_target_beyond_the_capacity_of_highs_is_rejected(self, sampler):
        with pytest.raises(ConstraintValidationError, match="unreachable within highs"):
            sampler.set_constraints(
                "stepsum", cols=[0, 1, 2], sum_value=100,
                lows=[0, 0, 0], highs=[1, 1, 1], step=1,
            )

    def test_the_message_carries_both_numbers(self, sampler):
        with pytest.raises(ConstraintValidationError, match="100 steps of 1"):
            sampler.set_constraints(
                "stepsum", cols=[0, 1, 2], sum_value=100,
                lows=[0, 0, 0], highs=[1, 1, 1], step=1,
            )

    @pytest.mark.parametrize("name", ["lows", "highs"])
    def test_bounds_shorter_than_cols_are_rejected(self, sampler, name):
        kwargs = dict(
            cols=[0, 1, 2], sum_value=3, lows=[0, 0, 0], highs=[3, 3, 3], step=1
        )
        kwargs[name] = kwargs[name][:2]
        with pytest.raises(ConstraintValidationError, match=f"{name} must hold one value"):
            sampler.set_constraints("stepsum", **kwargs)

    def test_empty_bounds_are_rejected_before_they_are_indexed(self, sampler):
        """`lows[0]` used to raise IndexError from the parent constructor."""
        with pytest.raises(ConstraintValidationError, match="lows must hold one value"):
            sampler.set_constraints(
                "stepsum", cols=[0, 1], sum_value=2, lows=[], highs=[2, 2], step=1
            )

    def test_a_feasible_set_still_builds_and_samples(self, sampler):
        sampler.set_constraints(
            "stepsum", cols=[0, 1, 2], sum_value=10,
            lows=[1, 1, 1], highs=[10, 10, 10], step=1,
        )
        out = sampler.sample(N)[:, [0, 1, 2]].astype(float)
        assert np.allclose(out.sum(axis=1), 10) and np.all(out >= 1)


def test_categories_soft(sampler):
    sampler.set_constraints("categories", cols=[5], values=[["a"], ["b"]], strength="soft")
    assert set(sampler.sample(N)[:, 5]) <= {"a", "b"}


def test_categories_rejects_duplicate_values(sampler):
    with pytest.raises(ConstraintViolationError, match="unique"):
        sampler.set_constraints("categories", cols=[5], values=[["a"], ["a"]])


class TestCategoriesSingleColumn:
    @pytest.mark.parametrize("strength", ["soft", "hard"])
    def test_a_flat_list_constrains_one_column(self, sampler, strength):
        sampler.set_constraints(
            "categories", cols=[5], values=["a", "b"], strength=strength
        )
        assert set(sampler.sample(N)[:, 5]) <= {"a", "b"}

    def test_flat_and_nested_are_equivalent(self, X_numeric):
        """Same seed, same shape of result: the flat form is only sugar."""
        def run(values):
            s = RandomSampler.setup(X_numeric, random_state=0, n_jobs=1)
            s.set_constraints("categories", cols=[5], values=values, strength="soft")
            return s.sample(N)

        assert np.array_equal(run(["a", "b"]), run([["a"], ["b"]]))

    def test_flat_values_are_normalised_at_construction(self, sampler):
        sampler.set_constraints("categories", cols=[5], values=["a", "b"], strength="soft")
        assert sampler.constraints[0].values.tolist() == [["a"], ["b"]]

    def test_duplicate_detection_still_applies_to_the_flat_form(self, sampler):
        with pytest.raises(ConstraintViolationError, match="unique"):
            sampler.set_constraints("categories", cols=[5], values=["a", "a"])


class TestCategoriesArity:
    """A mismatch between `cols` and `values` used to surface as an IndexError
    from inside `sample()`; it is now rejected at construction."""

    def test_flat_values_with_several_cols_is_rejected(self, sampler):
        with pytest.raises(ConstraintValidationError, match="Nest each entry"):
            sampler.set_constraints("categories", cols=[4, 5], values=["a", "b"])

    def test_short_nested_entry_is_rejected(self, sampler):
        with pytest.raises(ConstraintValidationError, match="must hold 2 value"):
            sampler.set_constraints("categories", cols=[4, 5], values=[["a", "b"], ["c"]])

    def test_non_sequence_entry_is_rejected(self, sampler):
        with pytest.raises(ConstraintValidationError, match="must hold 2 value"):
            sampler.set_constraints("categories", cols=[4, 5], values=[["a", "b"], 5])

    def test_empty_values_is_rejected(self, sampler):
        with pytest.raises(ConstraintValidationError, match="must not be empty"):
            sampler.set_constraints("categories", cols=[5], values=[])

    @pytest.mark.parametrize(
        "values",
        [
            pytest.param(np.array([["a", "b"], ["c", "a"]], dtype=object), id="ndarray_2d"),
            pytest.param((("a", "b"), ("c", "a")), id="tuple_of_tuples"),
        ],
    )
    def test_values_may_be_any_sequence(self, sampler, values):
        """The docs pass `df[[...]].to_numpy()`, so the checks must not assume a
        list: truthiness on a 2-D array raises."""
        sampler.set_constraints("categories", cols=[4, 5], values=values, strength="soft")
        out = sampler.sample(N)[:, [4, 5]]
        assert set(map(tuple, out)) <= {("a", "b"), ("c", "a")}

    def test_a_valid_multi_column_pattern_is_unaffected(self, sampler):
        sampler.set_constraints(
            "categories", cols=[4, 5], values=[["a", "b"], ["c", "a"]], strength="soft"
        )
        out = sampler.sample(N)[:, [4, 5]]
        assert set(map(tuple, out)) <= {("a", "b"), ("c", "a")}


def test_callable_constraint_filters(sampler):
    sampler.set_constraints(lambda row: float(row[0]) < 5.0, cols=[0])
    assert sampler.sample(N)[:, 0].astype(float).max() < 5.0


def test_callable_returning_an_array_rewrites_its_cols(sampler):
    sampler.set_constraints(lambda row: np.array([1.0]), cols=[0])
    assert np.allclose(sampler.sample(N)[:, 0].astype(float), 1.0)


def test_unsatisfiable_constraint_exhausts_retries(X_numeric):
    from mlsampler import RandomSampler

    sampler = RandomSampler.setup(X_numeric, random_state=0, n_jobs=1, max_retries=5)
    sampler.set_constraints(lambda row: False, cols=[0])
    with pytest.raises(ConstraintViolationError):
        sampler.sample(1)


def test_constructive_constraints_run_before_user_callables(sampler):
    """A user callable always sees a row every constructive constraint has shaped."""
    from mlsampler.errors import DuplicateColumnWarning

    seen = []
    sampler.set_constraints("range", cols=[0], low=2.0, high=3.0)
    sampler.set_constraints(lambda row: seen.append(float(row[0])) or True, cols=[0])
    with pytest.warns(DuplicateColumnWarning):
        sampler.sample(N)
    assert seen and all(2.0 <= v <= 3.0 for v in seen)


def test_unsupported_constraint_name_raises(sampler):
    from mlsampler.errors import ConstraintTypeError

    with pytest.raises(ConstraintTypeError):
        sampler.set_constraints("no_such_constraint", cols=[0])


def test_reset_constraints_also_clears_callables(sampler):
    """`_funcs` is a second list; leaving it populated kept filtering rows while
    `len(sampler.constraints)` reported none."""
    sampler.set_constraints(lambda row: float(row[0]) < 1.0, cols=[0])
    sampler.reset_constraints()
    assert len(sampler.constraints) == 0
    assert sampler._funcs == []
    assert sampler.sample(N)[:, 0].astype(float).max() > 1.0


def test_reset_flag_also_clears_callables(sampler):
    sampler.set_constraints(lambda row: float(row[0]) < 1.0, cols=[0])
    sampler.set_constraints("range", cols=[1], low=0.0, high=1.0, reset=True)
    assert sampler.sample(N)[:, 0].astype(float).max() > 1.0


def test_reset_clears_previously_registered_constraints(sampler):
    sampler.set_constraints("range", cols=[0], low=2.0, high=3.0)
    sampler.set_constraints("range", cols=[0], low=7.0, high=8.0, reset=True)
    assert len(sampler.constraints) == 1
    out = sampler.sample(N)[:, 0].astype(float)
    assert out.min() >= 7.0
