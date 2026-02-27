import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, List, Union
from joblib import Parallel, delayed
from collections import defaultdict

@dataclass
class FeatureMeta:
    low: Optional[float] = None
    high: Optional[float] = None
    dtype: str = "float"  # "float", "int", "binary"


@dataclass
class SamplerConfig:
    features: List[FeatureMeta]
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
        self.constraints:List[dict] = []
        self.constraint_registry = {
            "sum": self._constraint_sum,
            "sumint":self._constraint_sum_int,
            "onehot": self._constraint_onehot,
            "range": self._constraint_range,
            "categories": self._constraint_categories,
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
            features=features,
            batch_size=batch_size,
            random_state=random_state,
            n_jobs=n_jobs,
            max_retries=max_retries,
        )

        return cls(config)

    def set_constraints(
        self,
        constraint_type: str = "func",
        fn: Optional[Callable] = None,
        replace=True,
        **kwargs
        ):

        """
        Set a constraint to the sampler.

        Parameters
        ----------
        constraint_type : str, default="func"
            Type of constraint. Supported types:
            - "func"
            - "sum"
            - "onehot"
            - "range"
            - "categories"
        fn : callable, optional
            Function that receives a row (np.ndarray) and returns a boolean.
            Required when `constraint_type="func"`.
        replace : bool, default=True
            If True, clears existing constraints before adding the new one.

        """

        if replace:
            self.constraints = []

        if constraint_type == "func":
            if fn is None:
                raise ValueError("fn must be provided for func constraint")

            self.constraints.append({
                "type": "func",
                "fn": fn
            })
            return

        if constraint_type not in self.constraint_registry:
            raise ValueError(f"Unknown constraint type: {constraint_type}")

        self.constraints.append({
            "type": constraint_type,
            "params": kwargs
        })

    def _constraint_sum_int(self, row, value, cols, max_used, min_used=1):

        for c in cols:
            row[c] = 0

        k = np.random.randint(min_used, max_used + 1)
        selected = np.random.choice(cols, size=k, replace=False)

        if k == 1:
            row[selected[0]] = value
            return row
        
        cuts = np.sort(np.random.choice(range(value + k - 1), k - 1, replace=False))
        parts = np.diff(np.concatenate(([-1], cuts, [value + k - 1]))) - 1

        for c, v in zip(selected, parts):
            row[c] = v

        return row
    
    def _constraint_sum(self, row, value, cols, min_used=1, max_used=None, alpha=None, rng=None):

        if rng is None:
            rng = np.random.default_rng()

        k = len(cols)
        if max_used is None:
            max_used = k
        
        used_k = rng.integers(min_used, max_used + 1)
        selected = rng.choice(cols, size=used_k, replace=False)

        for c in cols:
            row[c] = 0.0

        if alpha is None:
            alpha = np.ones(used_k)
        weights = rng.dirichlet(alpha)

        for c, w in zip(selected, weights):
            row[c] = w * value

        return row

    def _constraint_onehot(self, row, cols, min_used=1, max_used=1):
        for c in cols:
            row[c] = 0

        k = np.random.randint(min_used, max_used + 1)
        selected = np.random.choice(cols, size=k, replace=False)

        for c in selected:
            row[c] = 1

        return row

    def _constraint_range(self, row, col, min_val, max_val):
        row[col] = np.random.uniform(min_val, max_val)
        return row
    
    def _constraint_categories(self, row, col, values):
        rng = np.random.default_rng()
        row[col] = rng.choice(values)
        return row

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
                raise ValueError("Constraint must return numpy array or bool")

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
                print(f"Warning: Column {col} used in multiple constraints {ids}")
    
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