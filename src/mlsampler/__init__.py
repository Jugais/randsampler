"""
Constraint-aware sampling for design of experiments and reverse analysis.

`mlsampler` generates rows that satisfy constraints you declare, across continuous,
integer, binary and categorical columns at once. Hand it a sample of your data: it
infers each column's type and range, then samples within them.

Two samplers, deliberately not interchangeable:

- `RandomSampler` satisfies several constraints at the same time, by rejection
  sampling with retries. This is the one that takes constraints.
- `HyperGridSampler` covers a continuous space evenly (Latin Hypercube for the float
  columns, uniform grid for the rest). It takes no constraints and needs `scipy`.

Constraints are registered by name -- `sum`, `sumint`, `multihot`, `random`, `range`,
`categories`, `step`, `stepsum` -- or as any callable that accepts or rewrites a row.
Columns are addressed by name or by position.

>>> from mlsampler import RandomSampler
>>> sampler = RandomSampler.setup(df, random_state=0)
>>> sampler.set_constraints("range", cols=["temperature"], low=-10.0, high=40.0)
>>> sampler.sample(1000)

A pandas or polars DataFrame is accepted and returned in kind, carrying the input's
column names and per-column dtypes; neither library is a dependency. Equal
`random_state` under equal `n_jobs` produces equal output, with or without constraints.

Full documentation: https://jugais.github.io/randsampler/
"""

from importlib.metadata import version

from .base import BaseSampler, SamplerConfig, FeatureMeta
from .engine.random import RandomSampler
from .engine.hypergrid import HyperGridSampler

__version__ = version("mlsampler")

__all__ = [
    "BaseSampler",
    "SamplerConfig",
    "FeatureMeta",
    "RandomSampler",
    "HyperGridSampler",
    "__version__",
]

