import numpy as np
import pandas as pd
import pytest
from mlsampler import HyperGridSampler, RandomSampler

N = 200

@pytest.fixture
def train():
    """The demo frame used throughout the tutorial."""
    return pd.DataFrame({
        "is_active": [1, 0, 1, 1, 0],
        "is_negative": [0, 1, 0, 0, 1],
        "score": [10, 0, 50, 100, 5],
        "temperature": [-5.5, 20.0, 36.6, -1.2, 15.8],
        "category": ["A", "B", "C", "D", "100"],
        "city": ["Tokyo", "Osaka", "Nagoya", "Fukuoka", "Sapporo"],
        "cost": [100] * 5,
    })


@pytest.fixture
def mix():
    """Three ratio columns, for the sum-style constraints."""
    return pd.DataFrame({
        "a": [0.2, 0.5, 0.1, 0.7],
        "b": [0.3, 0.2, 0.6, 0.1],
        "c": [0.5, 0.3, 0.3, 0.2],
    })

def test_quickstart(train):
    sampler = RandomSampler.setup(train, random_state=42)
    sampler.set_constraints("multihot", cols=["is_active", "is_negative"], n_hot=1)
    out = sampler.sample(N)

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == list(train.columns)
    assert len(out) == N
    assert (out[["is_active", "is_negative"]].sum(axis=1) == 1).all()


def test_quickstart_with_an_array(train):
    sampler = RandomSampler.setup(train.to_numpy(), random_state=42)
    assert sampler.feature_names == list(range(train.shape[1]))
    sampler.set_constraints("multihot", cols=[0, 1], n_hot=1)
    out = sampler.sample(N)

    assert isinstance(out, np.ndarray) and out.dtype == object
    assert out[:, [0, 1]].astype(int).sum(axis=1).max() == 1

def test_reproducibility_claim(train):
    """Documented: same random_state under the same n_jobs gives the same output."""
    def run():
        sampler = RandomSampler.setup(train.values, random_state=42, n_jobs=1)
        sampler.set_constraints("random", cols=[2, 3], max_used=1)
        return sampler.sample(50)

    assert np.array_equal(run(), run())

def test_sum(mix):
    sampler = RandomSampler.setup(mix.values, random_state=0)
    sampler.set_constraints("sum", cols=[0, 1, 2], sum_value=1.0)
    out = sampler.sample(N).astype(float)
    assert np.allclose(out.sum(axis=1), 1.0)

def test_sumint(mix):
    sampler = RandomSampler.setup(mix.values, random_state=0)
    sampler.set_constraints("sumint", cols=[0, 1, 2], sum_value=100)
    out = sampler.sample(N).astype(float)
    assert np.allclose(out.sum(axis=1), 100)
    assert np.all(out == out.astype(int))

def test_multihot(train):
    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints("multihot", cols=[0, 1], n_hot=1)
    out = sampler.sample(N)[:, [0, 1]].astype(int)
    assert (out.sum(axis=1) == 1).all()

def test_random_select(train):
    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints("random", cols=[2, 3], max_used=1)
    out = sampler.sample(N)[:, [2, 3]].astype(float)
    assert ((out != 0).sum(axis=1) <= 1).all()

def test_range(train):
    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints("range", cols=[3], low=-10.0, high=40.0)
    out = sampler.sample(N)[:, 3].astype(float)
    assert out.min() >= -10.0 and out.max() <= 40.0

def test_step_takes_a_singular_col(mix):
    """`step` is the only constraint whose argument is `col`, not `cols`."""
    sampler = RandomSampler.setup(mix.values, random_state=0)
    sampler.set_constraints("step", col=0, step=0.25, low=0.0, high=1.0)
    out = sampler.sample(N)[:, 0].astype(float)
    grid = np.arange(0.0, 1.25, 0.25)
    assert all(np.any(np.isclose(v, grid)) for v in out)

def test_stepsum(mix):
    sampler = RandomSampler.setup(mix.values, random_state=0)
    sampler.set_constraints(
        "stepsum", cols=[0, 1, 2], sum_value=10, lows=[0, 0, 0], highs=[6, 6, 6], step=1
    )
    out = sampler.sample(N).astype(float)
    assert np.allclose(out.sum(axis=1), 10)
    assert np.all(out <= 6)

def test_stepsum_infeasible_raises(mix):
    """Documented failure mode: the target cannot be reached within `highs`.
    Decided from the arguments, so it is rejected at registration."""
    from mlsampler.errors import ConstraintValidationError

    sampler = RandomSampler.setup(mix.values, random_state=0)
    with pytest.raises(ConstraintValidationError, match="unreachable within highs"):
        sampler.set_constraints(
            "stepsum", cols=[0, 1, 2], sum_value=100, lows=[0, 0, 0], highs=[1, 1, 1], step=1
        )

def test_categories_single_column(train):
    sampler = RandomSampler.setup(train, random_state=0)
    sampler.set_constraints(
        "categories", cols=["city"], values=["Tokyo", "Osaka"], strength="soft"
    )
    assert set(sampler.sample(N)["city"]) <= {"Tokyo", "Osaka"}

def test_categories_by_name(train):
    sampler = RandomSampler.setup(train, random_state=0)
    sampler.set_constraints(
        "categories",
        cols=["category", "city"],
        values=[["A", "Tokyo"], ["B", "Osaka"]],
        strength="soft",
    )
    out = sampler.sample(N)[["category", "city"]]
    assert set(map(tuple, out.to_numpy())) <= {("A", "Tokyo"), ("B", "Osaka")}

def test_categories_soft(train):
    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints(
        "categories",
        cols=[4, 5],
        values=train[["category", "city"]].to_numpy(),
        strength="soft",
    )
    out = sampler.sample(N)[:, [4, 5]]
    allowed = {tuple(v) for v in train[["category", "city"]].to_numpy()}
    assert {tuple(row) for row in out} <= allowed

def test_categories_hard(train):
    """Only the listed combinations are produced; other draws are rejected."""
    sampler = RandomSampler.setup(train.values, random_state=0, max_retries=500)
    sampler.set_constraints(
        "categories",
        cols=[4, 5],
        values=[["A", "Tokyo"], ["B", "Osaka"]],
        strength="hard",
    )
    out = sampler.sample(50)[:, [4, 5]]
    assert {tuple(row) for row in out} <= {("A", "Tokyo"), ("B", "Osaka")}

def test_callable_requires_cols(train):
    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints(lambda row: row[0] > 0 and row[2] < 80, cols=[0, 2])
    out = sampler.sample(N)
    assert (out[:, 0].astype(int) > 0).all()
    assert (out[:, 2].astype(float) < 80).all()

def test_combining_constraints(train):
    sampler = RandomSampler.setup(train.values, random_state=42)
    sampler.set_constraints("multihot", cols=[0, 1], n_hot=1)
    sampler.set_constraints("random", cols=[2, 3], max_used=1)
    sampler.set_constraints(
        "categories",
        cols=[4, 5],
        values=train[["category", "city"]].to_numpy(),
        strength="soft",
    )
    out = sampler.sample(N)
    assert out.shape == (N, len(train.columns))
    assert (out[:, [0, 1]].astype(int).sum(axis=1) == 1).all()

def test_reset_and_inspect(train):
    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints("multihot", cols=[0, 1], n_hot=1)
    sampler.set_constraints("range", cols=[3], low=0.0, high=1.0, reset=True)
    assert len(sampler.constraints) == 1

def test_hypergrid_example():
    df = pd.DataFrame(
        [
            [0.1, 0.9, 1, 0, "A"],
            [0.5, 0.4, 2, 1, "B"],
            [0.9, 0.75, 3, 0, "AB"],
            [0.2, 0.8, 4, 1, "O"],
        ],
        columns=["ratio1", "ratio2", "rank", "isOk", "bloodType"],
    )
    sampler = HyperGridSampler.setup(df.values, random_state=42)
    dtypes = [f.dtype for f in sampler.config.features]
    assert dtypes == ["float", "float", "int", "binary", "categorical"]

    samples = sampler.sample(N)
    assert samples.shape == (N, 5)
    assert set(samples[:, 4]) <= {"A", "B", "AB", "O"}

def test_hypergrid_rejects_constraints():
    """Documented limitation."""
    df = pd.DataFrame({"x": [0.1, 0.5, 0.9], "y": [1.0, 2.0, 3.0]})
    sampler = HyperGridSampler.setup(df.values)
    with pytest.raises(NotImplementedError):
        sampler.set_constraints("range", cols=[0], low=0, high=1)

def test_parallel_example(mix):
    """Documented: parallelism is opt-in and pays off under high rejection."""
    sampler = RandomSampler.setup(mix.values, random_state=42, n_jobs=2)
    sampler.set_constraints(lambda row: float(row[0]) < 0.5, cols=[0])
    out = sampler.sample(N)
    assert out[:, 0].astype(float).max() < 0.5
