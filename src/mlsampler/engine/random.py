import numpy as np
from ..terminals import spinning
from ..base import BaseSampler, SamplerConfig, DtypeMeta as dm
from ..constraints import *
from ..errors import (
    ConstraintViolationError, 
    ConstraintTypeError, 
    DuplicateColumnWarning
)

from typing import Optional, Callable, Any
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
        self.n_jobs = config.n_jobs
        self.seed = config.random_state
        self.max_retries = config.max_retries

        self.dim = len(self.features)
        self._constraints:list[Constraints] = []
        self._funcs: list[FunctionConstraint] = []
        self._registry = MappingProxyType({
            "sum": SumConstraint,
            "sumint": SumIntConstraint,
            "multihot": MultihotConstraint,
            "random": RandomSelectConstraint,
            "range": RangeConstraint,
            "categories": CategoriesConstraint,
            "step": StepConstraint,
            "stepsum": SumStepConstraint,
            
        })

    @property
    def constraints(self):
        """Get a registered constraint class"""
        return tuple(self._constraints.copy())

    def reset_constraints(self):
        """Clear all registered constraints."""
        self._constraints = []

    def set_constraints(
            self,
            constraint_fn: str | Callable[[np.ndarray], bool],
            reset=False,
            **kwargs
        ):
        """
        Set a constraint to the sampler.
        Parameters
        ----------
        constraint_fn : str or callable, default="sum"
            Type of constraint. Supported types:
            - callable: user-defined function that takes a row and returns a boolean or a new row. `cols` must be provided in kwargs.
            - "sum": constraint based on the sum of selected columns. `sum_value` and `cols` must be provided in kwargs.
            - "sumint": similar to "sum" but ensures the sum is an integer. `sum_value` and `cols` must be provided in kwargs.
            - "multihot": constraint ensuring a specified number of columns in a set are active.`n_hot` and `cols` must be provided in kwargs.
            - "random": constraint selecting a random subset of columns. `cols`, `min_used`, and `max_used` can be provided in kwargs.
            - "range": constraint setting values within a specified range. `cols`, `low`, and `high` must be provided in kwargs.
            - "categories": constraint selecting from a list of categorical values. `cols` and `values` must be provided in kwargs.
            - "step": constraint selecting values that are multiples of a step. `cols` and `step`, `low`, and `high` must be provided in kwargs.
            - "stepsum": constraint ensuring the sum of values is a multiple of a step.
        reset : bool, default=True
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
        >>> sampler.set_constraints("sum", sum_value=1, cols=[2, 3, 4], max_used=2)
        """

        if reset:
            self._constraints = []
            self._funcs = []

        if callable(constraint_fn):
            self._constraints.append(FunctionConstraint(fn=constraint_fn, **kwargs))
            self._funcs.append(FunctionConstraint(fn=constraint_fn, **kwargs))
        elif isinstance(constraint_fn, str) and constraint_fn in self._registry:
            self._constraints.append(self._registry[constraint_fn](**kwargs))
        else:
            raise ConstraintTypeError(
                f"Unsupported constraint type: {constraint_fn}"
            )
        
    def _base_sample(self, rng: np.random.Generator):
        x = np.empty(self.dim, dtype=object)
        for i, f in enumerate(self.features):
            if f.dtype == dm.const:
                x[i] = f.low
            elif f.dtype == dm.bin:
                x[i] = rng.integers(0, 2)
            elif f.dtype == dm.integer and f.low is not None and f.high is not None:
                x[i] = rng.integers(int(f.low), int(f.high) + 1)
            elif f.dtype == dm.float and f.low is not None and f.high is not None:
                x[i] = rng.uniform(f.low, f.high)
            else:
                x[i] = 0

        return x
    
    def _fill_categoricals(self, x, rng: np.random.Generator):
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
            if isinstance(constraint, FunctionConstraint):
                continue

            result = constraint(row)
            if result is None:
                return None  # Constraint violation
            row = result
        return row
    
    def _apply_funcs(self, row: np.ndarray) -> Optional[np.ndarray]:
        for constraint in self._funcs:
            result = constraint(row)
            if result is None:
                return None
        return row

    def _detect_conflicts(self):
        col_usage = defaultdict(list)
        constraints_by_col = defaultdict(list)

        for i, c in enumerate(self._constraints):
            for col in getattr(c, "cols", []):
                col_usage[col].append(i)
                constraints_by_col[col].append(c)

        for col, ids in col_usage.items():
            constraints = constraints_by_col[col]
            types = {getattr(c, "constraint_fn", None) for c in constraints}

            meta = self.config.features[col]
            
            if meta.dtype == dm.cat:
                invalid = types & {"sum", "sumint", "range"}

                if invalid:
                    raise ConstraintViolationError(
                        f"{invalid} cannot be applied to categorical column: {col}"
                    )

            if meta.dtype == dm.const:
                if len(ids) > 1:
                    warnings.warn(
                        f"Const column {col} has multiple constraints {ids} (types={types})",
                        DuplicateColumnWarning
                    )

            if len(ids) > 1:
                warnings.warn(
                    f"Column {col} used in multiple constraints {ids}",
                    DuplicateColumnWarning
                )
    
    def _generate_one(self, seed_offset: int = 0):
        base_seed = self.seed if self.seed is not None else None
        rng = np.random.default_rng(
            (base_seed + seed_offset) if base_seed else None
        )

        for _ in range(self.max_retries):
            x = self._base_sample(rng)
            x = self._fill_categoricals(x, rng)
            x = self._apply_constraints(x)
            if x is None:
                continue

            x = self._apply_funcs(x)
            if x is not None:
                return x

        raise ConstraintViolationError("Max retries exceeded.")

    def _sample(self, n_samples):
        if self.n_jobs == 1:
            samples = [
                self._generate_one(i) for i in range(n_samples)
            ]
        else:
            samples = Parallel(n_jobs=self.n_jobs)(
                delayed(self._generate_one)(i)
                for i in range(n_samples)
            )
        return samples

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

        with spinning():
            samples = self._sample(n_samples)
            
        return np.array(samples)


    