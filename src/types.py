import numpy as np
from typing import TypeAlias, Union, Callable

Numeric: TypeAlias = Union[int, float, np.integer, np.floating]
ArrayLike: TypeAlias = Union[np.ndarray, list, tuple]
Bool: TypeAlias = Union[bool, np.bool_]
ConstraintFn: TypeAlias = Callable[[np.ndarray], Union[Bool, np.ndarray]]