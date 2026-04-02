Tutorial
========

Basic usage:

.. code-block:: python

    import pandas as pd
    from mlsampler import RandomSampler

    def create_demo_df():
        data = {
            "is_active": [1, 0, 1, 1, 0],
            "is_negative": [0, 1, 0, 0, 1],
            "score": [10, 0, 50, 100, 5],
            "temperature": [-5.5, 20.0, 36.6, -1.2, 15.8],
            "category": ["A", "B", "C", "D", "100"],
            "city": ["Tokyo", "Osaka", "Nagoya", "Fukuoka", "Sapporo"]
        }
        df = pd.DataFrame(data)
        return df

    train = create_demo_df()

    sampler = RandomSampler.setup(train.values)
    print(type(sampler).__qualname__)

    sampler.reset_constraints()
    sampler.set_constraints('multihot', cols=[0,1])
    sampler.set_constraints('random', cols=[2,3], max_used=1)
    sampler.set_constraints('categories', 
        cols=[4, 5], 
        values = train[['category', 'city']].to_numpy(), 
        strength = 'soft'
    )
    print(sampler.constraints)

    result = sampler.sample(20)
    print(pd.DataFrame(result, columns=train.columns))