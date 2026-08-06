import inspect
import typing
import pytest
from mlsampler import RandomSampler

@pytest.fixture(scope="module")
def overloads():
    found = typing.get_overloads(RandomSampler.set_constraints)
    assert found, "no overloads registered; @overload must run at import time"
    return found

def literal_keys(overload):
    """The constraint names an overload matches, or () for the callable one."""
    annotation = inspect.signature(overload).parameters["constraint_fn"].annotation
    return typing.get_args(annotation) if typing.get_origin(annotation) is typing.Literal else ()

def test_every_registry_key_has_an_overload(overloads, sampler):
    covered = {key for ov in overloads for key in literal_keys(ov)}
    assert set(sampler._registry) - covered == set()

def test_no_overload_names_an_unregistered_key(overloads, sampler):
    covered = {key for ov in overloads for key in literal_keys(ov)}
    assert covered - set(sampler._registry) == set()

def test_registry_and_overloads_match_exactly(overloads, sampler):
    covered = {key for ov in overloads for key in literal_keys(ov)}
    assert covered == set(sampler._registry)

def test_callable_overload_exists(overloads):
    assert any(not literal_keys(ov) for ov in overloads)

def test_each_overload_advertises_its_constraint_arguments(overloads, sampler):
    """An overload must expose exactly the arguments its constraint class takes,
    otherwise autocomplete offers something `set_constraints` will reject."""
    for ov in overloads:
        for key in literal_keys(ov):
            cls = sampler._registry[key]
            declared = set(inspect.signature(ov).parameters) - {"self", "constraint_fn", "reset"}
            actual = set(inspect.signature(cls).parameters) - {"self"}
            assert declared == actual, f"{key!r}: overload {declared} vs class {actual}"

def test_step_overload_uses_the_singular_col(overloads):
    for ov in overloads:
        if "step" in literal_keys(ov):
            params = inspect.signature(ov).parameters
            assert "col" in params and "cols" not in params

def test_multihot_overload_hides_the_usage_bounds(overloads):
    """`min_used`/`max_used` derive from `n_hot`; offering them would mislead."""
    for ov in overloads:
        if "multihot" in literal_keys(ov):
            params = inspect.signature(ov).parameters
            assert "n_hot" in params
            assert "min_used" not in params and "max_used" not in params

def test_no_overload_exposes_reset_cols(overloads):
    """Decided internal: it must not appear in any completion list."""
    assert all("reset_cols" not in inspect.signature(ov).parameters for ov in overloads)
