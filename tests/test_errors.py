import re
import numpy as np
import pytest

from mlsampler import RandomSampler
from mlsampler.errors import (
    ConstraintError,
    ConstraintTypeError,
    ConstraintValidationError,
    ConstraintViolationError,
)

# Every class name a user must never see in an error message.
INTERNAL_NAMES = re.compile(
    r"SumConstraint|SumIntConstraint|MultihotConstraint|RandomSelectConstraint|"
    r"RangeConstraint|CategoriesConstraint|StepConstraint|SumStepConstraint|"
    r"FunctionConstraint|SelectConstraint|__init__"
)

BAD_ARGS = [
    ("range", dict(low=0, high=1)),                                  # missing cols
    ("step", dict(cols=[0], step=1, low=0, high=5)),                 # cols= instead of col=
    ("multihot", dict(cols=[0, 1], n_hot=1, min_used=1)),            # arg it must not accept
    ("random", dict(cols=[0, 1], reset_cols=True)),                  # arg kept internal
    ("sum", dict(cols=[0, 1], sum_valu=5)),                          # typo
    ("sum", dict(cols=[0, 1], reset_cols=False)),                    # arg kept internal
]


class TestArgumentErrors:
    """`**kwargs` removal made bad arguments raise; 2.5.1 keeps class names out of the message."""

    @pytest.mark.parametrize("name, kwargs", BAD_ARGS)
    def test_raises_constraint_validation_error(self, sampler, name, kwargs):
        with pytest.raises(ConstraintValidationError):
            sampler.set_constraints(name, **kwargs)

    @pytest.mark.parametrize("name, kwargs", BAD_ARGS)
    def test_message_hides_internal_class_names(self, sampler, name, kwargs):
        with pytest.raises(ConstraintValidationError) as excinfo:
            sampler.set_constraints(name, **kwargs)
        assert not INTERNAL_NAMES.search(str(excinfo.value))

    @pytest.mark.parametrize("name, kwargs", BAD_ARGS)
    def test_message_names_the_string_key_and_valid_arguments(self, sampler, name, kwargs):
        with pytest.raises(ConstraintValidationError) as excinfo:
            sampler.set_constraints(name, **kwargs)
        message = str(excinfo.value)
        assert repr(name) in message
        assert "Valid arguments:" in message

    def test_step_error_points_at_the_singular_col(self, sampler):
        """`step` is the one constraint taking `col`, so its message must show it."""
        with pytest.raises(ConstraintValidationError, match=r"'col'"):
            sampler.set_constraints("step", cols=[0], step=1, low=0, high=5)


class TestConstructionTimeValidation:
    """Things that used to surface only once `sample()` was running."""

    def test_bad_method(self, sampler):
        with pytest.raises(ConstraintValidationError, match="method"):
            sampler.set_constraints("sum", cols=[0, 1], method="nope")

    def test_bad_strength(self, sampler):
        with pytest.raises(ConstraintValidationError, match="strength"):
            sampler.set_constraints("categories", cols=[5], values=[["a"]], strength="nope")

    def test_max_used_above_column_count(self, sampler):
        with pytest.raises(ConstraintValidationError, match="max_used"):
            sampler.set_constraints("random", cols=[0, 1], max_used=5)

    def test_zero_step_does_not_divide_by_zero(self, sampler):
        with pytest.raises(ConstraintValidationError, match="step must be positive"):
            sampler.set_constraints("step", col=0, step=0, low=0, high=5)

    def test_negative_step(self, sampler):
        with pytest.raises(ConstraintValidationError, match="step must be positive"):
            sampler.set_constraints("step", col=0, step=-1, low=0, high=5)

    def test_negative_sumint_is_a_constraint_error(self, sampler):
        """Was a bare TypeError, outside the ConstraintError hierarchy."""
        with pytest.raises(ConstraintValidationError, match="non-negative"):
            sampler.set_constraints("sumint", cols=[0, 1], sum_value=-5)

    def test_inverted_range(self, sampler):
        with pytest.raises(ConstraintValidationError, match="low must be <= high"):
            sampler.set_constraints("range", cols=[0], low=5, high=1)

    def test_non_numeric_range_bound(self, sampler):
        with pytest.raises(ConstraintValidationError, match="numeric"):
            sampler.set_constraints("range", cols=[0], low="a", high=1)


class TestUnsupportedName:
    def test_type_and_message(self, sampler):
        with pytest.raises(ConstraintTypeError) as excinfo:
            sampler.set_constraints("bogus", cols=[0])
        assert "bogus" in str(excinfo.value)

    def test_lists_every_registered_key(self, sampler):
        with pytest.raises(ConstraintTypeError) as excinfo:
            sampler.set_constraints("bogus", cols=[0])
        message = str(excinfo.value)
        for key in sampler._registry:
            assert key in message


class TestColumnChecks:
    def test_out_of_range_column(self, sampler):
        sampler.set_constraints("range", cols=[99], low=0, high=1)
        with pytest.raises(ConstraintValidationError, match="out of range"):
            sampler.sample(2)

    def test_numeric_constraint_on_a_categorical_column(self, sampler):
        """4.1: this guard existed but keyed off an attribute no class sets, so it never fired."""
        sampler.set_constraints("range", cols=[5], low=0, high=1)
        with pytest.raises(ConstraintViolationError, match="categorical"):
            sampler.sample(2)

    @pytest.mark.parametrize("name, kwargs", [
        ("sum", dict(cols=[5], sum_value=1.0)),
        ("sumint", dict(cols=[5], sum_value=10)),
    ])
    def test_sum_constraints_on_a_categorical_column(self, sampler, name, kwargs):
        sampler.set_constraints(name, **kwargs)
        with pytest.raises(ConstraintViolationError, match="categorical"):
            sampler.sample(2)


class TestMaxRetriesDiagnostics:
    def build(self, X_numeric):
        sampler = RandomSampler.setup(X_numeric, random_state=0, n_jobs=1, max_retries=3)
        sampler.set_constraints(lambda row: False, cols=[0])
        return sampler

    def test_reports_the_retry_budget(self, X_numeric):
        with pytest.raises(ConstraintViolationError, match="3 attempts"):
            self.build(X_numeric).sample(1)

    def test_names_the_rejecting_constraint(self, X_numeric):
        with pytest.raises(ConstraintViolationError, match="rejected by"):
            self.build(X_numeric).sample(1)

    def test_suggests_a_remedy(self, X_numeric):
        with pytest.raises(ConstraintViolationError, match="max_retries"):
            self.build(X_numeric).sample(1)

    def test_message_hides_internal_class_names(self, X_numeric):
        with pytest.raises(ConstraintViolationError) as excinfo:
            self.build(X_numeric).sample(1)
        assert not INTERNAL_NAMES.search(str(excinfo.value))


def test_parallel_rng_warning_hides_internal_class_names(X_numeric):
    from mlsampler.errors import ParallelRngWarning

    sampler = RandomSampler.setup(X_numeric, random_state=0, n_jobs=2)
    sampler.set_constraints("sum", cols=[0, 1, 2], sum_value=1.0, rng=np.random.default_rng(1))
    with pytest.warns(ParallelRngWarning) as record:
        sampler.sample(120)
    message = str(record[0].message)
    assert not INTERNAL_NAMES.search(message)
    assert "'sum'" in message


def test_duplicate_column_warning_names_constraints_by_key(X_numeric):
    """4.1 made the type set real; it used to render as {None}."""
    from mlsampler.errors import DuplicateColumnWarning

    sampler = RandomSampler.setup(X_numeric, random_state=0, n_jobs=1)
    sampler.set_constraints("range", cols=[0], low=0, high=1)
    sampler.set_constraints("sum", cols=[0, 1], sum_value=1.0)
    with pytest.warns(DuplicateColumnWarning) as record:
        sampler.sample(5)
    messages = " ".join(str(r.message) for r in record)
    assert "None" not in messages
    assert "'range'" in messages and "'sum'" in messages


def test_every_constraint_error_is_catchable_as_one_type(sampler):
    """All of these subclass ConstraintError, so one except clause covers the API."""
    for call in (
        lambda: sampler.set_constraints("bogus", cols=[0]),
        lambda: sampler.set_constraints("range", low=0, high=1),
        lambda: sampler.set_constraints("step", col=0, step=0, low=0, high=5),
    ):
        with pytest.raises(ConstraintError):
            call()
