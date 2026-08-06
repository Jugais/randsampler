<!-- [claude fixed] rewritten: added the constraint reference, a working example, the
     reproducibility guarantee and the known limitations (TASK.md Phase 5.4) -->
# mlsampler

A constraint-aware sampling library for reverse analysis and design of experiments.

`mlsampler` generates synthetic rows that satisfy constraints you declare, across
continuous, integer, binary and categorical columns at the same time. Give it a sample
of your data; it infers each column's type and range, and samples within them.

## Installation

```bash
pip install mlsampler
```

`HyperGridSampler` additionally needs `scipy`:

```bash
pip install mlsampler scipy
```

Requires Python 3.11 or newer.

## Quick start

```python
import pandas as pd
from mlsampler import RandomSampler

train = pd.DataFrame({
    "is_active":   [1, 0, 1, 1, 0],
    "is_negative": [0, 1, 0, 0, 1],
    "score":       [10, 0, 50, 100, 5],
    "temperature": [-5.5, 20.0, 36.6, -1.2, 15.8],
    "category":    ["A", "B", "C", "D", "100"],
    "city":        ["Tokyo", "Osaka", "Nagoya", "Fukuoka", "Sapporo"],
})

sampler = RandomSampler.setup(train.values, random_state=42)

# exactly one of the two flags is on
sampler.set_constraints("multihot", cols=[0, 1], n_hot=1)
# keep temperature in a range the training data never covered
sampler.set_constraints("range", cols=[3], low=-10.0, high=40.0)
# category and city must stay a combination that actually occurs
sampler.set_constraints(
    "categories",
    cols=[4, 5],
    values=train[["category", "city"]].to_numpy(),
    strength="soft",
)

result = sampler.sample(1000)
print(pd.DataFrame(result, columns=train.columns))
```

Columns are addressed by **integer position**, not by name.

## Which sampler?

| | `RandomSampler` | `HyperGridSampler` |
|---|---|---|
| Purpose | satisfy several constraints at once | cover a continuous space evenly (DoE) |
| Constraints | yes | **no** |
| Method | rejection sampling with retries | Latin Hypercube + uniform grid |
| Needs `scipy` | no | yes |

## Constraint reference

Register constraints with `set_constraints(name, **kwargs)`.

| Name | Effect | Arguments |
|---|---|---|
| `"sum"` | distributes a total across the columns | `cols`, `sum_value`, `method`, `alpha`, `min_used`, `max_used` |
| `"sumint"` | as `sum`, but partitions an integer total into whole numbers | `cols`, `sum_value`, `min_used`, `max_used` |
| `"multihot"` | sets exactly `n_hot` columns to 1, the rest to 0 | `cols`, `n_hot` |
| `"random"` | keeps a random subset, zeroes the others | `cols`, `min_used`, `max_used` |
| `"range"` | draws uniformly from `[low, high]` | `cols`, `low`, `high` |
| `"categories"` | restricts columns to allowed combinations | `cols`, `values`, `strength` |
| `"step"` | snaps onto `low + k * step` | **`col`**, `step`, `low`, `high` |
| `"stepsum"` | distributes a total in fixed increments | `cols`, `sum_value`, `lows`, `highs`, `step` |
| callable | accepts/rejects a row, or rewrites its columns | `cols` |

> **`"step"` takes `col` (a single integer). Every other constraint takes `cols` (a
> list).** This is the one inconsistency in the API; passing `cols=` to `"step"` raises
> a `ConstraintValidationError` telling you so.

All constraints also accept `rng`, honoured only when `n_jobs=1`.

A callable receives the row after every other constraint has shaped it. Return a bool to
accept or reject, or a numpy array to write into `cols`:

```python
sampler.set_constraints(lambda row: float(row[3]) < 30.0, cols=[3])
```

Rows are `dtype=object` arrays, since one row mixes numbers with category strings — cast
before doing arithmetic.

## Reproducibility

**Equal `random_state` under equal `n_jobs` produces equal output**, with or without
constraints.

```python
a = RandomSampler.setup(X, random_state=42, n_jobs=1).sample(100)
b = RandomSampler.setup(X, random_state=42, n_jobs=1).sample(100)
(a == b).all()   # True
```

Three things to know:

- A seeded sampler restarts its stream on each `sample()` call, so calling `sample()`
  twice on one sampler returns the same rows. Use a different seed, or
  `random_state=None`, for fresh draws.
- Output is **not** guaranteed to match across different `n_jobs` values.
- `n_jobs=-1` resolves to the machine's core count, so it does not reproduce across
  machines with different core counts. Pass an explicit integer if you need that.

## Parallel execution

`n_jobs` defaults to `1`. Enabling parallelism only pays off when constraints reject
most candidates, since starting worker processes costs more than generating a row.
Measured speedup against serial:

| Rejection rate | n=200 | n=1000 | n=5000 |
|---|---|---|---|
| 0% | 0.00x | 0.22x | 0.69x |
| 80% | 0.30x | 0.67x | 1.72x |
| 95% | 1.21x | 2.15x | 4.25x |

Below 100 requested rows the serial path is used regardless of `n_jobs`.

## Limitations

- `HyperGridSampler` does not support constraints; `set_constraints()` raises
  `NotImplementedError`.
- `HyperGridSampler` requires `scipy`.
- Constraints are applied per row, so very large draws are bounded by Python-level
  iteration rather than vectorised numpy.
- Input must not contain missing values; `setup()` raises on `NaN`/`None`.

## Upgrading to 0.4.0

This release fixes reproducibility, which was broken whenever a constraint was
registered. It contains breaking changes.

**Output changes for the same seed.** Constraints are now seeded from the sampler
instead of drawing from fresh entropy, and one generator serves a whole `sample()` call
instead of one per row. Rows generated by 0.3.x cannot be reproduced by 0.4.0. In
exchange, `random_state` now actually works with constraints — before, it did not.

| Change | What breaks | What to do |
|---|---|---|
| `n_jobs` now defaults to `1` | sampling is no longer parallel by default | pass `n_jobs=-1` explicitly if your constraints reject most candidates |
| constraint arguments are now explicit | a misspelled or inapplicable argument raises `ConstraintValidationError` instead of being ignored | fix the argument — it was never taking effect |
| `SamplerConfig.batch_size` removed | code passing `batch_size` raises `TypeError` | drop it; nothing ever read it |

Two things that used to fail silently now raise:

- `HyperGridSampler.set_constraints(...)` accepted constraints and ignored them. It now
  raises `NotImplementedError`.
- `set_constraints("multihot", ..., min_used=1)` and similar were accepted for
  constraints that cannot use them. They are now rejected at registration.

One thing that used to raise now works: `categories` with `strength="hard"` rejects and
redraws instead of aborting with `ConstraintViolationError`, which is what makes it
usable at all.

## Documentation

https://jugais.github.io/randsampler/

## Use cases

- Reverse analysis — finding inputs that satisfy target conditions
- Candidate generation for optimization
- Design of experiments over a mixed continuous/categorical space
