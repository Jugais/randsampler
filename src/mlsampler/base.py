import numpy as np
from dataclasses import dataclass
from typing import Optional, Self
from abc import ABC, abstractmethod
from .constraints import Constraints
from .types import ColumnRef, SampleOutput, SetupInput, DataFrameLike

@dataclass(frozen=True)
class FeatureMeta:
    index: Optional[int] = None
    low: Optional[float] = None
    high: Optional[float] = None
    dtype: str = "float"  # "float", "int", "binary", "categorical"
    categories:Optional[list] = None
    name: Optional[ColumnRef] = None

@dataclass(frozen=True)
class DtypeMeta:
    float: str = "float"
    integer: str = "int"
    bin:str = "binary"
    cat:str = "categorical"
    const:str = "constant"

def _column(f: FeatureMeta, values: np.ndarray) -> np.ndarray:
    if f.dtype == DtypeMeta.cat or (f.dtype == DtypeMeta.const and f.low is None):
        return values.astype(str)

    numeric = values.astype(float)
    if f.dtype in (DtypeMeta.integer, DtypeMeta.bin):
        whole = numeric.astype(np.int64)
        if np.array_equal(numeric, whole):
            return whole
    return numeric

@dataclass
class SamplerConfig:
    features: list[FeatureMeta]
    random_state: Optional[int] = None
    n_jobs: int = 1
    max_retries: int = 1000
    frame: Optional[str] = None


class BaseSampler(ABC):
    """
    Base class for samplers.

    Provides the feature-metadata setup shared by every sampler. Constraint support is
    not part of the base — `set_constraints` and `apply_constraints` raise
    NotImplementedError unless a subclass overrides them.

    Parameters
    ----------
    config : SamplerConfig
        Configuration object containing feature metadata and sampling settings.
    """

    def __init__(self, config: SamplerConfig) -> None:
        self.config = config
        self._constraints: list[Constraints] = []

    @staticmethod
    def _read_input(X: SetupInput) -> tuple[np.ndarray, list, Optional[str]]:
        """Split an input into its array, its column labels, and its library."""
        if not isinstance(X, DataFrameLike):
            return X, list(range(X.shape[1])), None

        labels = list(X.columns)
        positions = list(range(len(labels)))

        if labels == positions:
            labels = positions
        elif not all(isinstance(label, str) for label in labels):
            raise ValueError(
                f"Column labels must be all strings or exactly 0-{len(labels) - 1}, "
                f"got {labels}. Rename the columns, or pass the array itself to "
                "address columns by position."
            )
        elif len(set(labels)) != len(labels):
            duplicated = sorted({label for label in labels if labels.count(label) > 1})
            raise ValueError(
                f"Duplicate column names {duplicated}: a name cannot identify one column. "
                "Rename them, or pass the array itself to address columns by position."
            )

        return X.to_numpy(), labels, type(X).__module__.split(".")[0]

    @classmethod
    def setup(
            cls,
            X: SetupInput,
            *,
            random_state: Optional[int] = None,
            n_jobs: int = 1,
            max_retries: int = 1000,
        ) -> Self:
        """
        Create a sampler from provided training data.

        Parameters
        ----------
        X : np.ndarray or DataFrame
            Input dataset used to infer feature ranges and types. A pandas or polars
            DataFrame is accepted; its column names become the feature names, and
            `sample` returns the same type. Labels must be all strings or exactly
            0-(n_features - 1).
        random_state : int or None, default=None
            Seed for reproducible sampling.
        n_jobs : int, default=1
            Number of parallel jobs. 
            Parallelism only pays off when constraints reject most candidates.
            It is slower than serial otherwise.
        max_retries : int, default=1000
            Maximum retries for satisfying constraints.

        Returns
        -------
        BaseSampler
            An instance of the class `setup` was called on.
        """

        X, feature_names, frame = cls._read_input(X)

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
            unique_vals = np.unique(col_data)
            try:
                numeric_data = col_data.astype(float)
                low = numeric_data.min()
                high = numeric_data.max()

                if np.isclose(low, high, equal_nan=True):
                    dtype = DtypeMeta.const
                elif set(unique_vals).issubset({0, 1}):
                    dtype = DtypeMeta.bin
                elif np.all(numeric_data.astype(int) == numeric_data):
                    dtype = DtypeMeta.integer
                else:
                    dtype = DtypeMeta.float
            except (ValueError, TypeError):
                if len(unique_vals) == 1:
                    dtype = DtypeMeta.const
                else:    
                    dtype = DtypeMeta.cat
                categories=np.unique(col_data).tolist()

            features.append(
                FeatureMeta(
                    index=col,
                    low=low,
                    high=high,
                    dtype=dtype,
                    categories=categories,
                    name=feature_names[col],
                )
            )

        config = SamplerConfig(
            features = features,
            random_state = random_state,
            n_jobs = n_jobs,
            max_retries = max_retries,
            frame = frame,
        )

        return cls(config)
    
    def set_constraints(self, constraint_fn: str, **kwargs) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support constraints. "
            "Use RandomSampler for constraint-based sampling."
        )

    def apply_constraints(self, row: np.ndarray) -> Optional[np.ndarray]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support constraints."
        )

    @abstractmethod
    def sample(self, n_samples: int) -> SampleOutput:  # [claude fixed]
        pass

    def _to_frame(self, samples: np.ndarray) -> SampleOutput:
        """Rebuild the container `setup` was given, restoring per-column dtypes."""
        if self.config.frame is None:
            return samples

        columns = [_column(f, samples[:, i]) for i, f in enumerate(self.config.features)]

        if self.config.frame == "polars":
            import polars as pl

            return pl.DataFrame([
                pl.Series(str(f.name), values)
                for f, values in zip(self.config.features, columns)
            ])

        import pandas as pd

        return pd.DataFrame(dict(zip(self.feature_names, columns)))

    @property
    def n_features(self) -> int:
        return len(self.config.features)

    @property
    def feature_names(self) -> list:
        return [f.name for f in self.config.features]
    
    def __len__(self):
        return len(self._constraints)
    
    def __getitem__(self, key):
        return self._constraints[key]

    def __repr__(self):
        return f"<{self.__class__.__name__} with {len(self)} constraints>"
    
    def __iter__(self):
        return iter(self._constraints)
    
    def __bool__(self):
        return bool(self._constraints)
    