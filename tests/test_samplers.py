import dataclasses
import numpy as np
import pytest
from mlsampler import HyperGridSampler, RandomSampler, SamplerConfig
from mlsampler.base import DtypeMeta as dm


class TestHyperGridSamplerRejectsConstraints:
    """4.2/4.3: it used to accept constraints silently and then fail on repr()."""

    @pytest.fixture
    def grid(self, X_mixed):
        return HyperGridSampler.setup(X_mixed, random_state=0)

    def test_set_constraints_raises_instead_of_no_op(self, grid):
        with pytest.raises(NotImplementedError, match="does not support constraints"):
            grid.set_constraints("range", cols=[0], low=0, high=1)

    def test_error_points_at_the_right_sampler(self, grid):
        with pytest.raises(NotImplementedError, match="RandomSampler"):
            grid.set_constraints("range", cols=[0], low=0, high=1)

    def test_apply_constraints_raises(self, grid):
        with pytest.raises(NotImplementedError):
            grid.apply_constraints(np.zeros(5, dtype=object))

    def test_repr_does_not_raise(self, grid):
        assert "HyperGridSampler" in repr(grid)

    def test_len_and_bool_do_not_raise(self, grid):
        assert len(grid) == 0
        assert not grid

    def test_still_samples(self, grid, X_mixed):
        out = grid.sample(20)
        assert out.shape == (20, X_mixed.shape[1])

    def test_sample_respects_dtypes(self, grid, X_mixed):
        out = grid.sample(20)
        assert set(out[:, 4]) <= {"a", "b", "c"}
        assert np.all(out[:, 3].astype(float) == 5.0)


class TestBatchSizeRemoved:
    """4.4: it was never read by anything."""

    def test_not_a_config_field(self):
        assert "batch_size" not in {f.name for f in dataclasses.fields(SamplerConfig)}

    def test_setup_rejects_it(self, X_mixed):
        with pytest.raises(TypeError):
            RandomSampler.setup(X_mixed, batch_size=10)

    def test_absent_from_the_setup_docstring(self):
        assert "batch_size" not in (RandomSampler.setup.__doc__ or "")


class TestCategoriesHard:
    """4.6: 'hard' raised on a mismatch instead of rejecting, making it unusable."""

    @pytest.fixture
    def X_cat(self):
        rng = np.random.default_rng(0)
        X = np.empty((40, 2), dtype=object)
        X[:, 0] = rng.choice(["a", "b", "c", "d", "e"], 40)
        X[:, 1] = rng.choice(["x", "y", "z"], 40)
        return X

    ALLOWED = [["a", "x"], ["b", "y"], ["c", "z"]]

    def sampler(self, X_cat):
        s = RandomSampler.setup(X_cat, random_state=0, n_jobs=1, max_retries=500)
        s.set_constraints("categories", cols=[0, 1], values=self.ALLOWED, strength="hard")
        return s

    def test_only_allowed_combinations_are_produced(self, X_cat):
        out = self.sampler(X_cat).sample(30)
        allowed = {tuple(v) for v in self.ALLOWED}
        assert {tuple(row) for row in out} <= allowed

    def test_every_allowed_combination_is_reachable(self, X_cat):
        """Rejection sampling must not collapse onto a single pattern."""
        out = self.sampler(X_cat).sample(60)
        assert len({tuple(row) for row in out}) == len(self.ALLOWED)

    def test_hard_is_reproducible(self, X_cat):
        assert np.array_equal(self.sampler(X_cat).sample(20), self.sampler(X_cat).sample(20))

    def test_soft_ignores_the_current_value(self, X_cat):
        s = RandomSampler.setup(X_cat, random_state=0, n_jobs=1)
        s.set_constraints("categories", cols=[0, 1], values=self.ALLOWED, strength="soft")
        allowed = {tuple(v) for v in self.ALLOWED}
        assert {tuple(row) for row in s.sample(30)} <= allowed


class TestHyperGridReproducibility:
    """The project invariant covers every sampler, not just RandomSampler."""

    def test_same_seed_reproduces(self, X_mixed):
        a = HyperGridSampler.setup(X_mixed, random_state=42).sample(20)
        b = HyperGridSampler.setup(X_mixed, random_state=42).sample(20)
        assert np.array_equal(a, b)

    def test_repeated_calls_reproduce(self, X_mixed):
        """Float and discrete columns must restart together; the LHS half used to
        repeat across calls while the grid half kept advancing."""
        grid = HyperGridSampler.setup(X_mixed, random_state=42)
        assert np.array_equal(grid.sample(20), grid.sample(20))

    def test_zero_is_a_real_seed(self, X_mixed):
        a = HyperGridSampler.setup(X_mixed, random_state=0).sample(20)
        b = HyperGridSampler.setup(X_mixed, random_state=0).sample(20)
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self, X_mixed):
        a = HyperGridSampler.setup(X_mixed, random_state=1).sample(20)
        b = HyperGridSampler.setup(X_mixed, random_state=2).sample(20)
        assert not np.array_equal(a, b)

    def test_unseeded_is_not_reproducible(self, X_mixed):
        a = HyperGridSampler.setup(X_mixed).sample(20)
        b = HyperGridSampler.setup(X_mixed).sample(20)
        assert not np.array_equal(a, b)


def test_setup_returns_the_concrete_subclass(X_mixed):
    """4.7: annotated with Self so RandomSampler.setup() is not widened to BaseSampler."""
    assert type(RandomSampler.setup(X_mixed)) is RandomSampler
    assert type(HyperGridSampler.setup(X_mixed)) is HyperGridSampler
