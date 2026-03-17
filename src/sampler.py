import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional
from joblib import Parallel, delayed
from collections import defaultdict
from .constraint import Constraints

from .errors import (
    ConstraintViolationError, 
    DuplicateColumnWarning
)
import warnings

@dataclass
class FeatureMeta:
    low: Optional[float] = None
    high: Optional[float] = None
    dtype: str = "float"  # "float", "int", "binary"


@dataclass
class SamplerConfig:
    features: list[FeatureMeta]
    batch_size: int = 1000
    random_state: Optional[int] = None
    n_jobs: int = -1
    max_retries: int = 1000


class RandomSampler:
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

    - Validation constraints: return a boolean.
    - Constructive constraints: return a modified numpy array.

    Sampling is performed with retry logic up to `max_retries`.
    Parallel generation is supported via joblib.
    """

    def __init__(self, config: SamplerConfig):
        self.features = config.features
        self.batch_size = config.batch_size
        self.n_jobs = config.n_jobs
        self.seed = config.random_state
        self.max_retries = config.max_retries

        self.dim = len(self.features)
        self.constraints:list[dict] = []
        self.constraint_registry = {
            "sum": Constraints.sum,
            "sumint": Constraints.sum_int,
            "multihot": Constraints.multihot,
            "random": Constraints.random,
            "range": Constraints.range,
            "categories": Constraints.categories,
        }

    @classmethod
    def setup(
        cls,
        X: np.ndarray,
        batch_size: int = 1000,
        random_state: Optional[int] = None,
        n_jobs: int = -1,
        max_retries: int = 1000,
        ):
        """
        Create a RandomSampler from provided training data.

        Parameters
        ----------
        X : np.ndarray
            Input dataset used to infer feature ranges and types.
        batch_size : int, default=1000
            Number of samples generated per batch.
        random_state : int or None, default=None
            Seed for reproducible sampling.
        n_jobs : int, default=-1
            Number of parallel jobs.
        max_retries : int, default=1000
            Maximum retries for satisfying constraints.

        Returns
        -------
        RandomSampler
        """

        features = []

        for col in range(X.shape[1]):
            col_data = X[:, col]
            low = col_data.min()
            high = col_data.max()
            unique_vals = np.unique(col_data)

            if set(unique_vals).issubset({0, 1}):
                dtype = "binary"
            elif np.all(col_data.astype(int) == col_data):
                dtype = "int"
            else:
                dtype = "float"

            features.append(
                FeatureMeta(low=low, high=high, dtype=dtype)
            )

        config = SamplerConfig(
            features = features,
            batch_size = batch_size,
            random_state = random_state,
            n_jobs = n_jobs,
            max_retries = max_retries,
        )

        return cls(config)

    def reset_constraints(self):
        """Clear all registered constraints."""
        self.constraints = []

    def set_constraints(
            self,
            constraint_type: str = "func",
            fn: Optional[Callable] = None,
            replace=False,
            **kwargs
        ):

        """
        Set a constraint to the sampler.

        Parameters
        ----------
        constraint_type : str, default="func"
            Type of constraint. Supported types:
            - "func": user-defined function that takes a row and returns a boolean. Required `fn` parameter.
            - "sum": constraint based on the sum of selected columns. `sum_value` and `cols` must be provided in kwargs.
            - "sumint": similar to "sum" but ensures the sum is an integer. `sum_value` and `cols` must be provided in kwargs.
            - "multihot": constraint ensuring a specified number of columns in a set are active.`n_hot` and `cols` must be provided in kwargs.
            - "random": constraint selecting a random subset of columns. `cols`, `min_used`, and `max_used` can be provided in kwargs.
            - "range": constraint setting values within a specified range. `cols`, `min_val`, and `max_val` must be provided in kwargs.
            - "categories": constraint selecting from a list of categorical values. `col` and `values` must be provided in kwargs.
        fn : callable, optional
            Function that receives a row (np.ndarray) and returns a boolean.
            Required when `constraint_type="func"`.
        replace : bool, default=True
            If True, clears existing constraints before adding the new one.
        **kwargs
            Additional parameters specific to the constraint type.
        
        Examples
        --------
        sampler = RandomSampler.setup(X_train)

        # Custom function constraint
        sampler.set_constraints("func", fn=lambda x: (0 < x[0] < 1) and (0 < x[1] < 1))
        
        # Sum constraint
        sampler.set_constraints("sum", sum_value=1, cols=[2, 3, 4], max_used=3)
        
        """
            
        if replace:
            self.reset_constraints()

        if constraint_type == "func":
            if fn is None:
                raise ConstraintViolationError("fn must be provided for func constraint")

            self.constraints.append({
                "type": "func",
                "fn": fn
            })
            return

        if constraint_type not in self.constraint_registry:
            raise ConstraintViolationError(f"Unsupported constraint type: {constraint_type}")

        self.constraints.append({
            "type": constraint_type,
            "params": kwargs
        })

    def _base_sample(self, rng: np.random.Generator):
        x = np.empty(self.dim, dtype=float)

        for i, f in enumerate(self.features):
            if f.dtype == "binary":
                x[i] = rng.integers(0, 2)
            elif f.dtype == "int":
                x[i] = rng.integers(int(f.low), int(f.high) + 1)
            elif f.dtype == "float":
                x[i] = rng.uniform(f.low, f.high)

        return x

    def _apply_constraints(self, row):

        for constraint in self.constraints:
            if constraint["type"] == "func":
                result = constraint["fn"](row)
            else:
                handler = self.constraint_registry[constraint["type"]]
                result = handler(row, **constraint["params"])
            
            # constructive
            if isinstance(result, np.ndarray):
                row = result
            # validation
            elif np.isscalar(result) and isinstance(result, (bool, np.bool_)):
                if not result:
                    return None
            else:
                raise ConstraintViolationError(
                        f"Constraint returned invalid type: {type(result)}. Must be bool or np.ndarray."
                    )

        return row

    def _detect_conflicts(self):
        col_usage = defaultdict(list)

        for i, c in enumerate(self.constraints):
            if c["type"] == "func":
                continue

            params = c.get("params", {})
            if "cols" in params:
                for col in params["cols"]:
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
            x = self._apply_constraints(row=x)
            if x is not None:
                return x

        raise RuntimeError("Max retries exceeded.")
    
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