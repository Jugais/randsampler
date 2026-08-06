# These encode the *intended* contract, not the behaviour at the time of writing.
# The guarantee is deliberately scoped: equal `random_state` under equal `n_jobs`
# produces equal output. Agreement *across* different `n_jobs` values is not
# guaranteed, so nothing here compares one against the other.
import warnings

import numpy as np
import pytest

from mlsampler import RandomSampler
from mlsampler.errors import ParallelRngWarning
from mlsampler.engine.random import _PARALLEL_MIN_SAMPLES

N = 20
PARALLEL_N = _PARALLEL_MIN_SAMPLES + 50  # above the threshold, so joblib really runs

# A constraint has to be registered for these tests to mean anything: the
# unconstrained path was already reproducible, so it cannot detect a regression.
CONSTRAINTS = [
    ("sum", dict(cols=[0, 1, 2], sum_value=1.0)),
    ("sumint", dict(cols=[0, 1, 2], sum_value=100)),
    ("multihot", dict(cols=[0, 1, 2], n_hot=2)),
    ("random", dict(cols=[0, 1, 2], min_used=1, max_used=2)),
    ("range", dict(cols=[0], low=2.0, high=3.0)),
    ("step", dict(col=0, step=0.5, low=0.0, high=5.0)),
    ("stepsum", dict(cols=[0, 1, 2], sum_value=10, lows=[0, 0, 0], highs=[10, 10, 10], step=1)),
    ("categories", dict(cols=[5], values=[["a"], ["b"]], strength="soft")),
]


def build(X, name, kwargs, random_state, n_jobs=1):
    sampler = RandomSampler.setup(X, random_state=random_state, n_jobs=n_jobs)
    sampler.set_constraints(name, **kwargs)
    return sampler


@pytest.mark.parametrize("name, kwargs", CONSTRAINTS, ids=[c[0] for c in CONSTRAINTS])
def test_same_seed_reproduces_serial(X_numeric, name, kwargs):
    a = build(X_numeric, name, kwargs, random_state=42).sample(N)
    b = build(X_numeric, name, kwargs, random_state=42).sample(N)
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name, kwargs", CONSTRAINTS, ids=[c[0] for c in CONSTRAINTS])
def test_zero_is_a_real_seed(X_numeric, name, kwargs):
    """`random_state=0` is falsy; it must still seed."""
    a = build(X_numeric, name, kwargs, random_state=0).sample(N)
    b = build(X_numeric, name, kwargs, random_state=0).sample(N)
    assert np.array_equal(a, b)


def test_zero_differs_from_another_seed(X_numeric):
    """Guards against `random_state=0` collapsing to the unseeded path."""
    name, kwargs = CONSTRAINTS[0]
    a = build(X_numeric, name, kwargs, random_state=0).sample(N)
    b = build(X_numeric, name, kwargs, random_state=1).sample(N)
    assert not np.array_equal(a, b)


def test_repeated_calls_on_one_sampler_reproduce(X_numeric):
    """A seeded sampler restarts its stream on every `sample()` call."""
    sampler = build(X_numeric, *CONSTRAINTS[0], random_state=42)
    assert np.array_equal(sampler.sample(N), sampler.sample(N))


def test_different_seeds_differ(X_numeric):
    name, kwargs = CONSTRAINTS[0]
    a = build(X_numeric, name, kwargs, random_state=1).sample(N)
    b = build(X_numeric, name, kwargs, random_state=2).sample(N)
    assert not np.array_equal(a, b)


def test_unseeded_is_not_reproducible(X_numeric):
    name, kwargs = CONSTRAINTS[0]
    a = build(X_numeric, name, kwargs, random_state=None).sample(N)
    b = build(X_numeric, name, kwargs, random_state=None).sample(N)
    assert not np.array_equal(a, b)


def test_unconstrained_is_reproducible(X_numeric):
    a = RandomSampler.setup(X_numeric, random_state=42, n_jobs=1).sample(N)
    b = RandomSampler.setup(X_numeric, random_state=42, n_jobs=1).sample(N)
    assert np.array_equal(a, b)


@pytest.mark.filterwarnings("ignore::mlsampler.errors.DuplicateColumnWarning")
def test_many_constraints_together_reproduce(X_numeric):
    def run():
        sampler = RandomSampler.setup(X_numeric, random_state=7, n_jobs=1)
        sampler.set_constraints("range", cols=[3], low=0.0, high=1.0)
        sampler.set_constraints("sum", cols=[0, 1, 2], sum_value=1.0)
        sampler.set_constraints("categories", cols=[5], values=[["a"], ["b"]], strength="soft")
        sampler.set_constraints(lambda row: float(row[3]) < 0.9, cols=[3])
        return sampler.sample(N)

    assert np.array_equal(run(), run())


def test_retries_do_not_resample_the_same_candidate(X_numeric):
    """The retry loop must advance one generator; re-seeding per attempt would
    regenerate the same rejected row until `max_retries` ran out."""
    sampler = RandomSampler.setup(X_numeric, random_state=42, n_jobs=1, max_retries=200)
    sampler.set_constraints(lambda row: float(row[0]) < 0.5, cols=[0])
    out = sampler.sample(N)[:, 0].astype(float)
    assert out.max() < 0.5


@pytest.mark.parametrize("n_jobs", [2, -1])
def test_same_seed_reproduces_parallel(X_numeric, n_jobs):
    name, kwargs = CONSTRAINTS[0]
    a = build(X_numeric, name, kwargs, random_state=42, n_jobs=n_jobs).sample(PARALLEL_N)
    b = build(X_numeric, name, kwargs, random_state=42, n_jobs=n_jobs).sample(PARALLEL_N)
    assert np.array_equal(a, b)
    assert a.shape == (PARALLEL_N, X_numeric.shape[1])


def test_parallel_chunks_reassemble_into_valid_rows(X_numeric):
    """Rows are generated per chunk and concatenated; every row must still
    satisfy the constraint and the count must be exact."""
    sampler = build(X_numeric, *CONSTRAINTS[0], random_state=42, n_jobs=2)
    out = sampler.sample(PARALLEL_N)
    assert out.shape[0] == PARALLEL_N
    assert np.allclose(out[:, [0, 1, 2]].astype(float).sum(axis=1), 1.0)


def test_parallel_chunks_are_independent_streams(X_numeric):
    """Each worker gets a spawned seed, so chunks must not repeat each other."""
    sampler = build(X_numeric, *CONSTRAINTS[0], random_state=42, n_jobs=2)
    out = sampler.sample(PARALLEL_N)
    assert len({tuple(row) for row in out}) == PARALLEL_N


def test_small_workloads_skip_the_worker_pool(X_numeric):
    """Below the threshold, n_jobs != 1 takes the serial branch. Equality with
    n_jobs=1 holds here by construction; it is not a general guarantee."""
    name, kwargs = CONSTRAINTS[0]
    n = _PARALLEL_MIN_SAMPLES - 1
    a = build(X_numeric, name, kwargs, random_state=42, n_jobs=1).sample(n)
    b = build(X_numeric, name, kwargs, random_state=42, n_jobs=-1).sample(n)
    assert np.array_equal(a, b)


def test_explicit_constraint_rng_warns_under_parallel(X_numeric):
    sampler = RandomSampler.setup(X_numeric, random_state=0, n_jobs=2)
    sampler.set_constraints("sum", cols=[0, 1, 2], sum_value=1.0, rng=np.random.default_rng(1))
    with pytest.warns(ParallelRngWarning):
        sampler.sample(PARALLEL_N)


def test_explicit_constraint_rng_is_silent_when_serial(X_numeric):
    sampler = RandomSampler.setup(X_numeric, random_state=0, n_jobs=1)
    sampler.set_constraints("sum", cols=[0, 1, 2], sum_value=1.0, rng=np.random.default_rng(1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParallelRngWarning)
        sampler.sample(N)


def test_no_warning_without_an_explicit_constraint_rng(X_numeric):
    sampler = RandomSampler.setup(X_numeric, random_state=0, n_jobs=2)
    sampler.set_constraints("sum", cols=[0, 1, 2], sum_value=1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ParallelRngWarning)
        sampler.sample(PARALLEL_N)


class TestConstraintLevelRng:
    """A generator handed to a constraint must survive construction and be used."""

    @pytest.mark.parametrize("name, kwargs", CONSTRAINTS, ids=[c[0] for c in CONSTRAINTS])
    def test_every_constraint_accepts_rng(self, X_numeric, name, kwargs):
        sampler = RandomSampler.setup(X_numeric, random_state=0, n_jobs=1)
        sampler.set_constraints(name, rng=np.random.default_rng(0), **kwargs)

    def run(self, X_numeric, seed):
        sampler = RandomSampler.setup(X_numeric, random_state=0, n_jobs=1)
        sampler.set_constraints(
            "sum", cols=[0, 1, 2], sum_value=1.0, rng=np.random.default_rng(seed)
        )
        return sampler.sample(N)

    def test_is_used(self, X_numeric):
        """Same sampler seed, different constraint seed -> different output."""
        assert not np.array_equal(self.run(X_numeric, 1), self.run(X_numeric, 2))

    def test_is_reproducible(self, X_numeric):
        assert np.array_equal(self.run(X_numeric, 1), self.run(X_numeric, 1))
