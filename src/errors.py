class ConstraintError(ValueError):
    """Custom exception for constraint violations"""
    pass

class ConstraintViolationError(ConstraintError):
    """Raised when a constraint is violated during data generation"""
    pass

class ConstraintValidationError(ConstraintError):
    """Raised when a constraint definition is invalid."""
    pass

class ConstraintWarning(UserWarning):
    """Base warning for constraint-related issues."""
    pass

class DuplicateColumnWarning(ConstraintWarning):
    """Column is used in multiple constraints."""
    pass