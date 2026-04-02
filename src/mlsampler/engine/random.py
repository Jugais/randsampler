import numpy as np

from ..base import BaseSampler, SamplerConfig, DtypeMeta as dm
from ..constraints import *
from ..errors import (
    ConstraintViolationError, 
    ConstraintTypeError, 
    DuplicateColumnWarning
)

from typing import Optional, Callable
from types import MappingProxyType
from collections import defaultdict
from joblib import Parallel, delayed

import warnings

class RandomSampler(BaseSampler):
    """
    Random constraint-based sampler.

    This sampler generates samples based on feature metadata
    inferred from training data. Users can register constraints
    that are applied during sample generation.

    Parameters
    ----------
    config : SamplerConfig
        Configuration object containing feature metadata and sampling settings.

    Notes
    -----
    Two types of constraints are supported:

    - Validation constraints:
        Return a boolean. If False, the sample is rejected.

    - Constructive constraints:
        Return a modified numpy array.

    Internally, constraints are unified to return either:
    - np.ndarray (valid sample)
    - None (invalid sample)

    Sampling is performed with retry logic up to `max_retries`.
    Parallel generation is supported via joblib.
    """
    
    __qualname__ = "RandomSampler"
    
    def __init__(self, config: SamplerConfig):
        super().__init__(config)
        self.rng = np.random.default_rng(config.random_state)
        self.features = config.features
        self.batch_size = config.batch_size
        self.n_jobs = config.n_jobs
        self.seed = config.random_state
        self.max_retries = config.max_retries

        self.dim = len(self.features)
        self.constraints:list[Constraints] = []
        self._registry = MappingProxyType({
            "sum": SumConstraint,
            "sumint": SumIntConstraint,
            "multihot": MultihotConstraint,
            "random": RandomSelectConstraint,
            "range": RangeConstraint,
            "categories": CategoriesConstraint,
            "step": StepConstraint,
            "stepsum": StepSumConstraint
        })

    def get_constraint(self, name: str):
        """Get a registered constraint class by name."""
        return self._registry.get(name)
    
    def reset_constraints(self):
        """Clear all registered constraints."""
        self.constraints = []

    def set_constraints(
            self,
            constraint_fn: str | Callable = "sum",
            replace=False,
            **kwargs
        ):
        """
        Set a constraint to the sampler.
        Parameters
        ----------
        constraint_fn : str or callable, default="sum"
            Type of constraint. Supported types:
            - callable: user-defined function that takes a row and returns a boolean. Required `fn` parameter.
            - "sum": constraint based on the sum of selected columns. `sum_value` and `cols` must be provided in kwargs.
            - "sumint": similar to "sum" but ensures the sum is an integer. `sum_value` and `cols` must be provided in kwargs.
            - "multihot": constraint ensuring a specified number of columns in a set are active.`n_hot` and `cols` must be provided in kwargs.
            - "random": constraint selecting a random subset of columns. `cols`, `min_used`, and `max_used` can be provided in kwargs.
            - "range": constraint setting values within a specified range. `cols`, `low`, and `high` must be provided in kwargs.
            - "categories": constraint selecting from a list of categorical values. `cols` and `values` must be provided in kwargs.
            - "step": constraint selecting values that are multiples of a step. `cols` and `step`, `low`, and `high` must be provided in kwargs.
            - "stepsum": constraint ensuring the sum of values is a multiple of a step.
        replace : bool, default=True
            If True, clears existing constraints before adding the new one.
        **kwargs
            Additional parameters specific to the constraint type.
            
        Examples
        --------
        >>> from mlsampler import RandomSampler
        >>> sampler = RandomSampler.setup(X_train)
        >>> # Custom function constraint
        >>> sampler.set_constraints(lambda x: (0 < x[0] < 1) and (0 < x[1] < 1))
        >>> # Sum constraint
        >>> sampler.set_constraints("sum", sum_value=1, cols=[2, 3, 4], max_used=3)
        """

        if replace:
            self.constraints = []
        
        if callable(constraint_fn):
            self.constraints.append(FunctionConstraint(fn=constraint_fn))
        elif isinstance(constraint_fn, str) and constraint_fn in self._registry:
            self.constraints.append(self._registry[constraint_fn](**kwargs))
        else:
            raise ConstraintTypeError(
                    f"Unsupported constraint type: {constraint_fn}"
                )
        
    def _base_sample(self, rng: np.random.Generator):
        # x = np.empty(self.dim, dtype=float)
        x = np.empty(len(self.config.features), dtype=object)

        for i, f in enumerate(self.features):
            if f.dtype == dm.bin:
                x[i] = rng.integers(0, 2)
            elif f.dtype == dm.integer and f.low is not None and f.high is not None:
                x[i] = rng.integers(int(f.low), int(f.high) + 1)
            elif f.dtype == dm.flt and f.low is not None and f.high is not None:
                x[i] = rng.uniform(f.low, f.high)
            else:
                x[i] = 0

        return x
    
    def _fill_categorical_features(self, x, rng: np.random.Generator):
        for i, f in enumerate(self.features):
            if f.dtype == dm.cat:
                if f.categories:
                    x[i] = rng.choice(f.categories)
                else:
                    x[i] = "unknown"
        return x

    def _apply_constraints(self, row: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply all registered constraints sequentially.

        Parameters
        ----------
        row : np.ndarray Input sample.

        Returns
        -------
        np.ndarray or None
            - np.ndarray: if all constraints succeed
            - None: if any constraint fails

        Notes
        -----
        Each constraint must return either:
        - np.ndarray (possibly modified)
        - None (indicating failure)
        """
        for constraint in self:
            row = constraint(row)
            if row is None:
                return None  # Constraint violation
        return row
    
    def _detect_conflicts(self):
        col_usage = defaultdict(list)

        for i, c in enumerate(self.constraints):
            for col in getattr(c, "cols", []):
                col_usage[col].append(i)

        for col, ids in col_usage.items():
            if len(ids) > 1:
                warnings.warn(
                    f"Column {col} used in multiple constraints {ids}",
                    DuplicateColumnWarning
                )
    
    def _generate_one(self, seed_offset: int = 0):
        base_seed = self.seed if self.seed is not None else None
        rng = np.random.default_rng(
            None if base_seed is None else base_seed + seed_offset
        )

        for _ in range(self.max_retries):
            x = self._base_sample(rng)
            x = self._fill_categorical_features(x, rng)
            x = self._apply_constraints(x) 
            if x is not None:
                return x
        
        raise ConstraintViolationError("Max retries exceeded.")
    
    def _generate_batch(self, batch_size):
        return np.array([self._generate_one() for _ in range(batch_size)])

    def sample(self, n_samples: int):
        """
        Generate samples satisfying all registered constraints.

        Parameters
        ----------
        n_samples : int
            Total number of samples to generate.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_features).

        """

        self._detect_conflicts()

        batches = []
        remaining = n_samples

        print("sampling...", flush=True)

        while remaining > 0:
            current_batch = min(self.batch_size, remaining)

            if self.n_jobs == 1:
                batch = self._generate_batch(current_batch)
            else:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._generate_one)()
                    for _ in range(current_batch)
                )
                batch = np.array(results)

            batches.append(batch)

            remaining -= current_batch

            print(f"{n_samples - remaining}/{n_samples}", flush=True)

        return np.vstack(batches)
    