import numpy as np
from typing import Optional
from ..base import BaseSampler, DtypeMeta as dm
from ..errors import ConstraintViolationError
from ..terminals import spinning
try:
    from scipy.stats import qmc
except:
    qmc = None

class HyperGridSampler(BaseSampler):
    """
    Hybrid grid and LHS-based constraint sampler.

    This sampler generates samples based on feature metadata inferred
    from training data. Continuous features are sampled using Latin
    Hypercube Sampling (LHS), while discrete features (e.g., integer,
    binary, categorical) are sampled using grid random selection
    over their respective value spaces.

    Parameters
    ----------
    config : SamplerConfig
        Configuration object containing feature metadata and sampling settings.

    Notes
    -----
    Sampling strategy depends on feature data types:

    - Float features:
        Sampled using Latin Hypercube Sampling (LHS), then scaled to
        their respective [low, high] ranges.

    - Integer features:
        Sampled uniformly from the inclusive range [low, high].

    - Binary features:
        Sampled uniformly from {0, 1}.

    - Categorical features:
        Sampled uniformly from the provided category list.

    - Constant features:
        Always return the same fixed value.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.rng = np.random.default_rng(config.random_state)

    def _lhs(self, n_samples: int, float_features: list):
        """LHS sampling for float columns"""
        if qmc is None:
            raise ImportError("scipy is required for HyperGridSampler")
        
        if not float_features:
            return np.empty((n_samples, 0))

        dim = len(float_features)
        
        sampler = qmc.LatinHypercube(d=dim, seed=self.config.random_state)
        sample = sampler.random(n=n_samples)

        lows = np.array([f.low for f in float_features])
        highs = np.array([f.high for f in float_features])

        scaled = qmc.scale(sample, lows, highs)
        return scaled

    def _discrete(self, n_samples: int, features: list):
        """Grid random sampling for discrete columns"""
        cols = []

        for f in features:
            if f.dtype == dm.bin:
                vals = np.array([0, 1])
            elif f.dtype == dm.integer:
                vals = np.arange(int(f.low), int(f.high) + 1)
            elif f.dtype == dm.cat:
                vals = np.array(f.categories)
            elif f.dtype == dm.const:
                vals = np.array([f.low])
            else:
                raise ConstraintViolationError(f"Unsupported dtype: {f.dtype}")

            sampled = self.rng.choice(vals, size=n_samples)
            cols.append(sampled)

        return np.column_stack(cols) if cols else np.empty((n_samples, 0))

    def _sample(self, n_samples: int) -> np.ndarray:
        float_feats = [f for f in self.config.features if f.dtype == dm.float]
        discrete_feats = [f for f in self.config.features if f.dtype != dm.float]

        float_part = self._lhs(n_samples, float_feats)
        discrete_part = self._discrete(n_samples, discrete_feats)

        result = np.zeros((n_samples, self.n_features), dtype=object)

        float_idx = 0
        disc_idx = 0

        for i, f in enumerate(self.config.features):
            if f.dtype == dm.float:
                result[:, i] = float_part[:, float_idx]
                float_idx += 1
            else:
                result[:, i] = discrete_part[:, disc_idx]
                disc_idx += 1

        return result

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Parameters
        ----------
        n_samples : int
            Total number of samples to generate.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_features) containing the generated samples.
            The column order matches the input feature configuration.

        Notes
        -----
        - Float features are scaled to their respective [low, high] ranges.
        - Integer features are sampled from the inclusive range [low, high].
        - Categorical features are sampled from the provided category list.
        - Constant features return the same value for all samples.
        """
        
        with spinning():
            samples = self._sample(n_samples)
        return samples

