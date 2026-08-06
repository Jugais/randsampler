import numpy as np
import pytest
from mlsampler.errors import ConstraintViolationError

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
    sampler.set_constraints("random", cols=[0, 1, 2], min_used=1, max_used=2)
    out = sampler.sample(N)[:, [0, 1, 2]].astype(float)
    used = (out != 0).sum(axis=1)
    assert np.all(used <= 2) and np.all(used >= 1)


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


def test_categories_soft(sampler):
    sampler.set_constraints("categories", cols=[5], values=[["a"], ["b"]], strength="soft")
    assert set(sampler.sample(N)[:, 5]) <= {"a", "b"}


def test_categories_rejects_duplicate_values(sampler):
    with pytest.raises(ConstraintViolationError, match="unique"):
        sampler.set_constraints("categories", cols=[5], values=[["a"], ["a"]])


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
