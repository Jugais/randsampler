import numpy as np
from typing import TypeAlias, Union, Callable, Any, Protocol, runtime_checkable

Numeric: TypeAlias = Union[int, float, np.integer, np.floating]
ArrayLike: TypeAlias = Union[np.ndarray, list, tuple]
Bool: TypeAlias = Union[bool, np.bool_]
ConstraintFn: TypeAlias = Callable[[np.ndarray], Union[Bool, np.ndarray]]
ColumnRef: TypeAlias = Union[int, str]

@runtime_checkable
class DataFrameLike(Protocol):
    @property
    def columns(self) -> Any: ...
    def to_numpy(self) -> np.ndarray: ...

SetupInput: TypeAlias = Union[np.ndarray, DataFrameLike]  # [claude fixed]

SampleOutput: TypeAlias = Any