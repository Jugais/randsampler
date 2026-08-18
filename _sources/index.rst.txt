Documentation of mlsampler
==========================

**Constraint-aware sampling for design of experiments and reverse analysis.**

``mlsampler`` generates rows that satisfy constraints you declare, across continuous,
integer, binary and categorical columns at once. Hand it a sample of your data: it
infers each column's type and range, then samples within them.

.. code-block:: python

    from mlsampler import RandomSampler

    sampler = RandomSampler.setup(train, random_state=42)
    sampler.set_constraints("multihot", cols=["is_active", "is_negative"], n_hot=1)
    sampler.set_constraints("range", cols=["temperature"], low=-10.0, high=40.0)

    sampler.sample(1000)

Which sampler
-------------

The two samplers are **not** interchangeable.

===============  ===================================  ===============================
                 ``RandomSampler``                    ``HyperGridSampler``
===============  ===================================  ===============================
Purpose          satisfy several constraints at once  cover a continuous space evenly
Constraints      yes                                  **no**
Method           rejection sampling with retries      Latin Hypercube + uniform grid
Needs ``scipy``  no                                   yes
===============  ===================================  ===============================

What it gives you
-----------------

- **Eight constraints by name** -- ``sum``, ``sumint``, ``multihot``, ``random``,
  ``range``, ``categories``, ``step``, ``stepsum`` -- plus any callable that accepts or
  rewrites a row. Several constraints apply to one draw at the same time.
- **Types inferred from the data.** A column becomes constant, binary, int, float or
  categorical without you declaring anything.
- **Columns by name or by position.** A ``str`` is a name, an ``int`` is always a
  position.
- **DataFrames round-trip.** A pandas or polars frame is accepted and returned in kind,
  with the input's column names and per-column dtypes. Neither library is a dependency.
- **Reproducibility as a guarantee.** Equal ``random_state`` under equal ``n_jobs``
  produces equal output, with or without constraints.

Install with ``pip install mlsampler``. Runtime requirements are ``numpy`` and
``joblib``; ``scipy`` is needed only for ``HyperGridSampler``.

.. toctree::
   :maxdepth: 2

   RandomSampler
   HyperGridSampler
   tutorial
