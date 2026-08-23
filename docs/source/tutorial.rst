Tutorial
========

Installation
------------

.. code-block:: bash

    pip install mlsampler

``HyperGridSampler`` additionally requires ``scipy``:

.. code-block:: bash

    pip install mlsampler scipy


Quick start
-----------

``RandomSampler.setup()`` inspects your data and infers a type and range for every
column. You then register constraints and draw samples.

.. code-block:: python

    import pandas as pd
    from mlsampler import RandomSampler

    train = pd.DataFrame({
        "is_active":   [1, 0, 1, 1, 0],
        "is_negative": [0, 1, 0, 0, 1],
        "score":       [10, 0, 50, 100, 5],
        "temperature": [-5.5, 20.0, 36.6, -1.2, 15.8],
        "category":    ["A", "B", "C", "D", "100"],
        "city":        ["Tokyo", "Osaka", "Nagoya", "Fukuoka", "Sapporo"],
        "cost":        [100] * 5,
    })

    sampler = RandomSampler.setup(train, random_state=42)
    sampler.set_constraints("multihot", cols=["is_active", "is_negative"], n_hot=1)

    print(sampler.sample(200))
    

Columns are addressed by **name or by integer position**. An ``int`` always means a
position, so ``cols=[0]`` is the first column whatever it is called.

DataFrames round-trip
^^^^^^^^^^^^^^^^^^^^^

``setup()`` accepts a pandas or polars DataFrame as well as a NumPy array, and
``sample()`` returns the same type it was given, with the input's column names, column
order and per-column dtypes. Neither library is a dependency: a DataFrame is recognised
by duck typing, and the import needed to rebuild the result only runs once you have
handed that library's object in.

.. code-block:: python

    sampler = RandomSampler.setup(train.to_numpy(), random_state=42)
    sampler.feature_names          # [0, 1, 2, 3, 4, 5, 6] -- no names, so positions
    sampler.set_constraints("multihot", cols=[0, 1], n_hot=1)
    sampler.sample(200)            # an object array, as before

Column labels must be either all strings or exactly ``0..n_features-1``. Non-contiguous
integer labels, a mix of names and numbers, a ``MultiIndex``, or duplicate names raise
``ValueError``, since any of those would make ``cols`` ambiguous.

The dtype of a returned column follows its **values**, not what ``setup()`` inferred. A
constraint that writes floats into a column inferred as integer produces a float column.

Inferred types are available on the config:

.. code-block:: python

    for i, f in enumerate(sampler.config.features):
        print(f"col {i}: {f.dtype}")

A column becomes ``constant`` when every value is the same, ``binary`` when the values
are a subset of ``{0, 1}``, ``int`` when every value is integral, ``float`` otherwise,
and ``categorical`` when the values are not numeric at all.


Reproducibility
---------------

Passing ``random_state`` makes sampling repeatable:

.. code-block:: python

    def run():
        sampler = RandomSampler.setup(train.values, random_state=42, n_jobs=1)
        sampler.set_constraints("random", cols=[2, 3], max_used=1)
        return sampler.sample(50)

    (run() == run()).all()   # True

The guarantee is: **equal** ``random_state`` **under equal** ``n_jobs`` **produces equal
output.** Three details are worth knowing.

- A seeded sampler restarts its stream on every ``sample()`` call, so calling
  ``sample()`` twice on one sampler returns the same rows. Use a different seed, or
  ``random_state=None``, when you want fresh draws.
- Output is **not** guaranteed to match across different ``n_jobs`` values, because work
  is split into one chunk per worker and each chunk gets its own stream.
- ``n_jobs=-1`` resolves to the number of cores on the machine, so it is **not**
  reproducible across machines with different core counts. Pass an explicit integer if
  you need that.

You can also give a single constraint its own generator. This is honoured only when
``n_jobs=1``; otherwise the constraint is copied into a worker process and a
``ParallelRngWarning`` is raised.

.. code-block:: python

    import numpy as np

    sampler.set_constraints("sum", cols=[0, 1, 2], sum_value=1.0,
                            rng=np.random.default_rng(7))


Constraints
-----------

Constraints come in two kinds, and both are registered with ``set_constraints``:

- **Constructive** constraints rewrite a row so that it complies.
- **Validation** constraints (any callable returning a bool) can only accept or reject
  a row; rejected rows are redrawn, up to ``max_retries``.

Constructive constraints always run first, so a callable always sees a row that every
other constraint has already shaped.

.. note::

   Every constraint takes ``cols`` (a list) **except** ``step``, which takes ``col``
   (a single integer).

``sum``
^^^^^^^

Distributes a total across the selected columns.

.. code-block:: python

    mix = pd.DataFrame({
        "a": [0.2, 0.5, 0.1, 0.7],
        "b": [0.3, 0.2, 0.6, 0.1],
        "c": [0.5, 0.3, 0.3, 0.2],
    })

    sampler = RandomSampler.setup(mix.values, random_state=0)
    sampler.set_constraints("sum", cols=[0, 1, 2], sum_value=1.0)

    out = sampler.sample(200).astype(float)
    out.sum(axis=1)     # every row sums to 1.0

Accepts ``method`` (``'uniform'`` or ``'proportional'``), ``alpha``, and ``min_used`` /
``max_used`` to limit how many of ``cols`` are used at once.

``sumint``
^^^^^^^^^^

Like ``sum``, but partitions an integer total into whole numbers.

.. code-block:: python

    sampler = RandomSampler.setup(mix.values, random_state=0)
    sampler.set_constraints("sumint", cols=[0, 1, 2], sum_value=100)

    out = sampler.sample(200).astype(float)
    # every row sums to exactly 100, and every value is a whole number

``sum_value`` must be non-negative.

``multihot``
^^^^^^^^^^^^

Sets exactly ``n_hot`` of the selected columns to 1 and the rest to 0.

.. code-block:: python

    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints("multihot", cols=[0, 1], n_hot=1)

``min_used`` and ``max_used`` are not accepted here; both follow from ``n_hot``.

``random``
^^^^^^^^^^

Keeps a random subset of the columns and zeroes the others, leaving the kept values as
sampled.

.. code-block:: python

    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints("random", cols=[2, 3], max_used=1)

``max_used`` must not exceed ``len(cols)``.

``range``
^^^^^^^^^

Overrides the range inferred from the data.

.. code-block:: python

    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints("range", cols=[3], low=-10.0, high=40.0)

    out = sampler.sample(200)[:, 3].astype(float)
    out.min(), out.max()    # within [-10.0, 40.0]

``step``
^^^^^^^^

Snaps a column onto a discrete grid. **This is the one constraint that takes**
``col`` **(singular).**

.. code-block:: python

    sampler = RandomSampler.setup(mix.values, random_state=0)
    sampler.set_constraints("step", col=0, step=0.25, low=0.0, high=1.0)

    # values are drawn from 0.0, 0.25, 0.50, 0.75, 1.00

``step`` must be positive.

``stepsum``
^^^^^^^^^^^

Distributes a total in fixed increments, honouring a lower and upper bound per column.

.. code-block:: python

    sampler = RandomSampler.setup(mix.values, random_state=0)
    sampler.set_constraints(
        "stepsum", cols=[0, 1, 2], sum_value=10,
        lows=[0, 0, 0], highs=[6, 6, 6], step=1,
    )

    out = sampler.sample(200).astype(float)
    # every row sums to 10, and no value exceeds 6

.. [claude fixed] feasibility moved to registration time; error type changed

The target must be reachable: ``sum(lows)`` must not exceed ``sum_value``, the residual
must be a whole number of steps, and ``sum(highs)`` must be able to absorb it. All three
follow from the arguments, so an unreachable target raises ``ConstraintValidationError``
when the constraint is registered rather than during ``sample()`` — for example
``sum_value=100`` with ``highs=[1, 1, 1]`` cannot succeed. ``lows`` and ``highs`` must
also hold one value per column in ``cols``.

``categories``
^^^^^^^^^^^^^^

Restricts a group of columns to a list of allowed combinations.

For a single column, list the allowed values directly:

.. code-block:: python

    sampler = RandomSampler.setup(train, random_state=0)
    sampler.set_constraints(
        "categories", cols=["city"], values=["Tokyo", "Osaka"], strength="soft"
    )

For several columns, each entry of ``values`` gives one value per column, in the order
of ``cols``. A mismatch is reported when the constraint is registered, not while
sampling.

With ``strength='soft'`` the sampler picks one of the allowed combinations directly:

.. code-block:: python

    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints(
        "categories",
        cols=[4, 5],
        values=train[["category", "city"]].to_numpy(),
        strength="soft",
    )

With ``strength='hard'`` the drawn values are *kept only if* they already form an
allowed combination, and the row is otherwise rejected and redrawn:

.. code-block:: python

    sampler = RandomSampler.setup(train.values, random_state=0, max_retries=500)
    sampler.set_constraints(
        "categories",
        cols=[4, 5],
        values=[["A", "Tokyo"], ["B", "Osaka"]],
        strength="hard",
    )

``'hard'`` is rejection sampling, so its acceptance rate is roughly
``len(values) / (total combinations)``. Raise ``max_retries`` when the allowed set is
small relative to the space.

Callable constraints
^^^^^^^^^^^^^^^^^^^^

Any callable taking a row works. ``cols`` **is required.**

.. code-block:: python

    sampler = RandomSampler.setup(train.values, random_state=0)
    sampler.set_constraints(lambda row: row[0] > 0 and row[2] < 80, cols=[0, 2])

Returning a bool accepts or rejects the row. Returning a NumPy array instead writes
that array into ``cols``.

Rows are ``dtype=object`` arrays, because one row mixes numbers with category strings.
Cast before doing arithmetic:

.. code-block:: python

    sampler.set_constraints(lambda row: float(row[3]) < 30.0, cols=[3])


Combining constraints
---------------------

Constraints are applied in registration order.

.. code-block:: python

    sampler = RandomSampler.setup(train.values, random_state=42)
    sampler.set_constraints("multihot", cols=[0, 1], n_hot=1)
    sampler.set_constraints("random", cols=[2, 3], max_used=1)
    sampler.set_constraints(
        "categories",
        cols=[4, 5],
        values=train[["category", "city"]].to_numpy(),
        strength="soft",
    )

    result = sampler.sample(200)

Registering two constraints on the same column raises a ``DuplicateColumnWarning``: the
later one overwrites the earlier one's work.

Pass ``reset=True`` to replace the registered set, or call ``reset_constraints()``:

.. code-block:: python

    sampler.set_constraints("range", cols=[3], low=0.0, high=1.0, reset=True)
    len(sampler.constraints)    # 1

If no row satisfies every constraint within ``max_retries``, a
``ConstraintViolationError`` reports which constraint rejected the last candidate.


HyperGrid sampling
------------------

``HyperGridSampler`` covers the space evenly rather than satisfying constraints. Float
columns are drawn by Latin Hypercube Sampling and scaled to their range; every other
type is drawn uniformly from its own value set.

.. code-block:: python

    from mlsampler import HyperGridSampler

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
    samples = sampler.sample(1000)

    print(pd.DataFrame(samples, columns=df.columns))

.. warning::

   ``HyperGridSampler`` **does not support constraints.** Calling ``set_constraints()``
   on it raises ``NotImplementedError``. It also requires ``scipy``.


Parallel execution
------------------

``n_jobs`` defaults to ``1``. Parallelism is worth enabling only when constraints reject
most candidates, because starting worker processes costs more than generating a row.

.. code-block:: python

    sampler = RandomSampler.setup(mix.values, random_state=42, n_jobs=2)
    sampler.set_constraints(lambda row: float(row[0]) < 0.5, cols=[0])
    result = sampler.sample(200)

Measured speedup against serial, by rejection rate:

============  =========  =========  =========
Rejection     n=200      n=1000     n=5000
============  =========  =========  =========
0%            0.00x      0.22x      0.69x
80%           0.30x      0.67x      1.72x
95%           1.21x      2.15x      4.25x
============  =========  =========  =========

Below 100 requested rows the serial path is used regardless of ``n_jobs``.


Choosing a sampler
------------------

- ``RandomSampler`` — when you need several constraints satisfied at once, or the
  relationships between variables are hard to express analytically.
- ``HyperGridSampler`` — when you want even coverage of a continuous space, such as for
  design of experiments (DoE), and you need no constraints.
