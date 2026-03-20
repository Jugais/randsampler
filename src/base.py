import numpy as np
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod
from .constraints import Constraints

@dataclass(frozen=True)
class FeatureMeta:
    index: Optional[int] = None
    low: Optional[float] = None
    high: Optional[float] = None
    dtype: str = "float"  # "float", "int", "binary", "categorical"
    categories:Optional[list] = None

@dataclass(frozen=True)
class DtypeMeta:
    flt: str = "float"
    integer: str = "int"
    bin:str = "binary"
    cat:str = "categorical"

@dataclass
class SamplerConfig:
    features: list[FeatureMeta]
    batch_size: int = 1000
    random_state: Optional[int] = None
    n_jobs: int = -1
    max_retries: int = 1000


class BaseSampler(ABC):
    """
    Base class for constraint-based samplers.

    This class provides common functionality for sampling based on feature metadata
    and registered constraints. It is designed to be extended by specific sampler implementations.

    Parameters
    ----------
    config : SamplerConfig
        Configuration object containing feature metadata and sampling settings.
    """
    constraints: list[Constraints]

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self.constraints = []
        
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
            
            if any(val is None or (
                    isinstance(val, float) and np.isnan(val)
                ) for val in col_data):
                raise ValueError(f"Column {col} contains missing values (NaN/None).")

            low, high = None, None
            categories = None
            dtype = None
            
            try:
                numeric_data = col_data.astype(float)
                low = numeric_data.min()
                high = numeric_data.max()
                unique_vals = np.unique(col_data)

                if set(unique_vals).issubset({0, 1}):
                    dtype = DtypeMeta.bin
                elif np.all(numeric_data.astype(int) == numeric_data):
                    dtype = DtypeMeta.integer
                else:
                    dtype = DtypeMeta.flt
            except (ValueError, TypeError):
                dtype = DtypeMeta.cat
                categories=np.unique(col_data).tolist()

            features.append(
                FeatureMeta(
                    index=col, 
                    low=low, 
                    high=high, 
                    dtype=dtype, 
                    categories=categories
                )
            )

        config = SamplerConfig(
            features = features,
            batch_size = batch_size,
            random_state = random_state,
            n_jobs = n_jobs,
            max_retries = max_retries,
        )

        return cls(config)
    
    def set_constraints(self, constraint_fn: str, **kwargs) -> None:
        pass

    def apply_constraints(self, row: np.ndarray) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        pass

    @property
    def n_features(self) -> int:
        return len(self.config.features)
    
    def __len__(self):
        return len(self.constraints)
    
    def __getitem__(self, key):
        return self.constraints[key]

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self)} constraints>"
    
    def __iter__(self):
        return iter(self.constraints)
    
    def __bool__(self):
        return bool(self.constraints)
    