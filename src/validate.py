import numpy as np
from typing import Optional, TypeAlias

Numeric: TypeAlias = int | float | np.integer | np.floating
ArrayLike: TypeAlias = np.ndarray | list | tuple

class Validator:
    @staticmethod
    def cols(cols):
        if not isinstance(cols, ArrayLike):
            raise TypeError("cols must be an array-like of column indices")
        
        if len(set(cols)) != len(cols):
            raise ValueError("Column indices must be unique")
        
        for c in cols:
            if not isinstance(c, (int, np.integer)) or c < 0:
                raise ValueError("Column indices must be non-negative integers")
        
    @staticmethod
    def usage(min_used: int, max_used: Optional[int]):
        if min_used < 0:
            raise ValueError("min_used must be non-negative")
        
        if max_used is None:
            raise ValueError("max_used must be specified")
        
        if max_used is not None and max_used < 0:
            raise ValueError("max_used must be non-negative")
        
        if min_used > max_used:
            raise ValueError("min_used must be <= max_used")
        
    @staticmethod
    def value(value: Numeric):
        if not isinstance(value, Numeric):
            raise TypeError("Value must be a numeric")
        
        if value < 0:
            raise ValueError("Value must be non-negative")
        
    @staticmethod
    def range(min_val: Numeric, max_val: Numeric):
        if not isinstance(min_val, Numeric):
            raise TypeError("min_val must be a numeric")
        
        if not isinstance(max_val, Numeric):
            raise TypeError("max_val must be a numeric")
        
        if min_val > max_val:
            raise ValueError("min_val must be <= max_val")
        