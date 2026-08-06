class ConstraintError(ValueError):
    """Base class for every constraint error."""
    pass

class ConstraintTypeError(ConstraintError):
    """Raised when an unsupported constraint type is specified."""
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

class ParallelRngWarning(ConstraintWarning):
    """A generator given to a constraint is ignored when n_jobs != 1."""
    pass