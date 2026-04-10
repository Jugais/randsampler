import numpy as np
from typing import Optional
from .errors import *
from .types import Numeric, ArrayLike
import warnings

def validate_cols(cols):
    if not isinstance(cols, ArrayLike):
        raise ConstraintValidationError("cols must be an array-like of column indices")
    
    if len(set(cols)) != len(cols):
        raise ConstraintValidationError("Column indices must be unique")
    
    for c in cols:
        if not isinstance(c, (int, np.integer)) or c < 0:
            raise ConstraintValidationError("Column indices must be non-negative integers")
        
def validate_usage(min_used: int, max_used: Optional[int]):
    if min_used < 0:
        raise ConstraintValidationError("min_used must be non-negative")
        
    if max_used is None:
        raise ConstraintValidationError("max_used must be specified")
    
    if max_used is not None and max_used < 0:
        raise ConstraintValidationError("max_used must be non-negative")
    
    if min_used > max_used:
        raise ConstraintValidationError("min_used must be <= max_used")
    
def validate_values(value: Numeric):
    if not isinstance(value, Numeric):
        raise ConstraintValidationError("Value must be a numeric")
    
    if value < 0:
        warnings.warn("Value should be non-negative", ConstraintWarning)
        
def validate_range(low: Numeric, high: Numeric):
    if not isinstance(low, Numeric):
        raise ConstraintTypeError("low must be a numeric")
    
    if not isinstance(high, Numeric):
        raise ConstraintTypeError("high must be a numeric")
    
    if low > high:
        raise ConstraintValidationError("low must be <= high")
    
        