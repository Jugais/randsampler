Hybrid Grid Sampler
===================

A hybrid constraint-based sampler combining LHS and grid sampling.

Overview
--------

- Uses Latin Hypercube Sampling (LHS) for continuous features
- Uses grid-like random sampling for discrete features (integer, binary, categorical)
- Allows custom constraints
- Parallel sampling supported

API Reference
-------------

.. automodule:: mlsampler.engine.hybrid
   :members:
   :undoc-members:
   :show-inheritance: