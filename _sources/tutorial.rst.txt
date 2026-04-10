Tutorial
========

Basic usage:

.. code-block:: python

    import pandas as pd
    from mlsampler import RandomSampler, HybridSampler


Random Sampling
---------------

.. code-block:: python
    
    def create_demo_df():
        data = {
            "is_active": [1, 0, 1, 1, 0],
            "is_negative": [0, 1, 0, 0, 1],
            "score": [10, 0, 50, 100, 5],
            "temperature": [-5.5, 20.0, 36.6, -1.2, 15.8],
            "category": ["A", "B", "C", "D", "100"],
            "city": ["Tokyo", "Osaka", "Nagoya", "Fukuoka", "Sapporo"],
            "cost": [100]*5
        }
        df = pd.DataFrame(data)
        return df

    train = create_demo_df()
    sampler = RandomSampler.setup(train.values)
    print(type(sampler).__qualname__)

    sampler.reset_constraints()
    sampler.set_constraints('multihot', cols=[0,1])
    sampler.set_constraints('random', cols=[2,3], max_used=1)
    sampler.set_constraints(
        'categories', 
        cols=[4, 5], 
        values=train[['category', 'city']].to_numpy(), 
        strength='soft'
    )

    result = sampler.sample(20)
    print(pd.DataFrame(result, columns=train.columns))


HyperGrid Sampling
------------------

.. code-block:: python

    df = pd.DataFrame([
            [0.1, 0.9, 1, 0, "A"],
            [0.5, 0.4, 2, 1, "B"],
            [0.9, 0.75, 3, 0, "AB"],
            [0.2, 0.8, 4, 1, "O"],
        ], 
        columns=['ratio1', 'ratio2', 'rank', 'isOk', 'bloodType']
    )

    sampler = HyperGridSampler.setup(df.values, random_state=42)

    print("\n=== dtype check ===")
    for i, f in enumerate(sampler.config.features):
        print(f"col {i}: {f.dtype}")

    samples = sampler.sample(1000)

    print(pd.DataFrame(samples, columns=df.columns))


Notes
-----

- ``RandomSampler`` is recommended when:
  - You have multiple constraints to satisfy simultaneously
  - The relationships between variables are complex or hard to express analytically 

- ``HybridSampler`` is recommended when:

  - You want better coverage of continuous feature space
  - You are performing design of experiments (DoE)