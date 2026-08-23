import numpy as np
from numpy.random import Generator
from .base import Constraints, SelectConstraint
from .. import validate as v
from ..types import Numeric, ArrayLike, Bool, ConstraintFn
from typing import Optional
from ..errors import (
    ConstraintViolationError,
    ConstraintError,
    ConstraintTypeError,
    ConstraintValidationError
)


class MultihotConstraint(SelectConstraint):
    def __init__(
            self, 
            cols: list[int],
            n_hot: int = 1,
            rng: Optional[np.random.Generator] = None,
        ):
        super().__init__(
            cols,
            min_used=n_hot,
            max_used=n_hot,
            rng=rng
        )
        self.n_hot = n_hot

    def _constrain_selected(
            self,
            row: np.ndarray,
            selected: np.ndarray,
            rng: np.random.Generator,  # passed by the caller
        ) -> np.ndarray:
        row[selected] = 1
        return row

class RandomSelectConstraint(SelectConstraint):
    def __init__(
            self, 
            cols: list[int],
            min_used: int = 1,
            max_used: Optional[int] = None,
            rng: Optional[np.random.Generator] = None,
        ):
        super().__init__(
            cols, 
            min_used=min_used,
            max_used=max_used,
            reset_cols=False,
            rng=rng
        )

    def _constrain_selected(
            self,
            row: np.ndarray,
            selected: np.ndarray,
            rng: np.random.Generator  # unified signature
        ) -> np.ndarray:

        not_selected = np.setdiff1d(self.cols, selected)
        row[not_selected] = 0
        return row
    
class SumConstraint(SelectConstraint):
    def __init__(self, 
            cols: list[int], 
            sum_value: Numeric = 1,
            method: str = 'uniform',
            alpha: Optional[np.ndarray] = None,
            min_used: int = 1,
            max_used: Optional[int] = None,
            rng: Optional[np.random.Generator] = None,
        ):
        v.validate_values(sum_value)
        v.validate_choice(method, ('uniform', 'proportional'), 'method')

        super().__init__(
            cols, 
            min_used=min_used,
            max_used=max_used,
            rng=rng
        )
        self.sum_value = sum_value
        self.method = method
        self.alpha = alpha

    def _constrain_selected(self,
            row: np.ndarray,
            selected: np.ndarray,
            rng: np.random.Generator
        ) -> np.ndarray:
        """Distribute `sum_value` across the selected columns."""
        
        if self.method == 'uniform':
            if self.alpha is None:
                alpha = np.ones(len(selected))
            else:
                alpha = self.alpha

            weights = rng.dirichlet(alpha)
            row[selected] = weights * self.sum_value
        elif self.method == 'proportional':
            row[selected] = row / np.sum(row) * self.sum_value
        else:
            raise ConstraintTypeError(f'Method: {self.method} not implemented')

        return row

class SumIntConstraint(SumConstraint):
    def __init__(
            self,
            cols: list[int],
            sum_value: int = 100,
            min_used: int = 1,
            max_used: Optional[int] = None,
            rng: Generator | None = None,
        ):
        if sum_value < 0:
            raise ConstraintValidationError("sum_value must be a non-negative integer")
        super().__init__(
            cols,
            sum_value=sum_value,
            min_used=min_used,
            max_used=max_used,
            rng=rng
        )

        # the distribution settings the parent stores are never read
        del self.method, self.alpha

    def _constrain_selected(
            self,
            row: np.ndarray,
            selected: np.ndarray,
            rng: np.random.Generator
        ) -> np.ndarray:
        k = len(selected)
        if k == 1:
            row[selected[0]] = self.sum_value
            return row

        cuts = np.sort(rng.choice((self.sum_value + k - 1), k - 1, replace=False))
        parts = np.diff(
                    np.concatenate((
                        np.array([-1], dtype=cuts.dtype),
                        cuts,
                        np.array([self.sum_value + k - 1], dtype=cuts.dtype)
                    ))
                ) - 1

        row[selected] = parts
        return row

class CategoriesConstraint(Constraints):
    def __init__(
            self,
            cols: list[int],
            values: list[list],
            strength:str = "hard",
            rng: Optional[np.random.Generator] = None,
        ):
        v.validate_choice(strength, ('hard', 'soft'), 'strength')
        super().__init__(cols, rng)
        self.strength = strength

        if len(values) == 0:
            raise ConstraintValidationError(
                "values must not be empty: list the allowed value of every column "
                "in cols, or the allowed combinations across them."
            )

        flat = not any(isinstance(val, ArrayLike) for val in values)
        if flat:
            values = [[val] for val in values]

        # check arity at construction; a mismatch used to surface as an
        # IndexError from `self.values[:, i]` in the middle of sample()
        wrong = None
        for val in values:
            if not isinstance(val, ArrayLike) or len(val) != len(cols):
                wrong = val
                break

        if wrong is not None:
            if flat:
                detail = (
                    f"values holds single values, but cols names {len(cols)} columns. "
                    "Nest each entry to give one value per column, "
                    "e.g. [[a1, b1], [a2, b2]]."
                )
            else:
                detail = (
                    f"Each entry of values must hold {len(cols)} value(s), one per "
                    f"column in cols; got {wrong!r}."
                )
            raise ConstraintValidationError(detail)

        val_tuples = [tuple(v) for v in values]
        if len(set(val_tuples)) != len(values):
            raise ConstraintViolationError("values must be unique")

        self.values = np.array(values, dtype=object)

    def _constrain(
            self, 
            row: np.ndarray, 
            rng: Optional[np.random.Generator] = None,
        ) -> Optional[np.ndarray]:
        rng = self._rng(rng)

        current_row = row[self.cols]
        mask = np.ones(len(self.values), dtype=bool)
        if self.strength == "hard":
            for i, col in enumerate(self.cols):
                if current_row[i] is not None:
                    mask &= (self.values[:, i] == current_row[i])
        elif self.strength == "soft":
            mask = mask
        else:
            raise ConstraintError(f"{self.strength} was not supported.")
        
        valid_patterns = self.values[mask]
        if len(valid_patterns) == 0:
            return None

        idx = rng.integers(len(valid_patterns))
        selected_pattern = valid_patterns[idx]
        
        row[self.cols] = selected_pattern
        return row
    
class RangeConstraint(Constraints):
    def __init__(
            self,
            cols: list[int],
            low: float = 0,
            high: float = 1,
            rng: Optional[np.random.Generator] = None,
        ):
        super().__init__(cols, rng)
        self.low = low
        self.high = high
        v.validate_range(low, high)

    def _constrain(
            self, 
            row: np.ndarray, 
            rng: Optional[np.random.Generator] = None
        ) -> np.ndarray:
        # No need to reset cols here
        rng = self._rng(rng)
        row[self.cols] = rng.uniform(self.low, self.high, size=len(self.cols))
        return row

class StepConstraint(Constraints):
    def __init__(
            self,
            col:int,
            step: float,
            low: float,
            high: float,
            rng: Optional[np.random.Generator] = None,
        ):
        super().__init__(cols=[col], rng=rng)
        self.col = col

        v.validate_range(low, high, step)

        # (0.3 - 0.0) / 0.1 is 2.9999999999999996,
        n_grid = (high - low) / step
        n_steps = int(np.floor(n_grid + 1e-9 * max(1.0, abs(n_grid))))
        self.values = low + np.arange(n_steps + 1) * step

        self.low = low
        self.high = high
        self.step = step

    def _constrain(
            self,
            row: np.ndarray,
            rng: Optional[np.random.Generator] = None
        ) -> np.ndarray:
        rng = self._rng(rng)
        
        if row[self.col] is not None:
            if np.any(np.isclose(row[self.col], self.values)):
                return row
        
        row[self.col] = rng.choice(self.values)
        return row

class SumStepConstraint(StepConstraint):
    def __init__(
            self, 
            cols: list[int], 
            sum_value: float, 
            lows: Optional[ArrayLike] = None, 
            highs: Optional[ArrayLike] = None, 
            step: float = 1,
            rng: Optional[np.random.Generator] = None,
        ):
        # Initialize parent with the first column's range
        if lows is not None:
            lows = lows 
        else:
            lows = np.zeros(len(cols))

        if highs is not None:
            highs = highs
        else:
            highs = np.ones(len(cols))*100

        # a short lows/highs used to surface as an IndexError,
        # either here on lows[0] or later on highs[i] inside sample()
        for name, bounds in (("lows", lows), ("highs", highs)):
            if len(bounds) != len(cols):
                raise ConstraintValidationError(
                    f"{name} must hold one value per column in cols: expected "
                    f"{len(cols)}, got {len(bounds)}."
                )

        #　parent col/low/high stay internal
        super().__init__(
            col=cols[0],
            low=lows[0],
            high=highs[0],
            step=step,
            rng=rng
        )
        self.cols = cols

        self.lows = np.array(lows)
        self.highs = np.array(highs)
        self.sum_value = sum_value
        self.step = step

        # feasibility follows from the arguments, so decide it here
        # rather than raising once per row from inside a joblib worker
        total_lows = np.sum(self.lows)
        residual = sum_value - total_lows
        if residual < -1e-9:
            raise ConstraintValidationError(
                f"Sum of lows ({total_lows}) exceeds sum_value ({sum_value})."
            )

        n_grid = residual / step
        if not np.isclose(n_grid, round(n_grid)):
            raise ConstraintValidationError(
                f"sum_value ({sum_value}) minus the sum of lows ({total_lows}) "
                f"is {residual}, which is not a multiple of step ({step})."
            )

        capacity = np.sum(self.highs - self.lows) / step
        if round(n_grid) > capacity + 1e-9:
            raise ConstraintValidationError(
                f"sum_value ({sum_value}) is unreachable within highs: it needs "
                f"{round(n_grid)} steps of {step}, but only {capacity} fit "
                "between lows and highs."
            )

    def _constrain(
            self,
            row: np.ndarray,
            rng: Optional[np.random.Generator] = None
        ) -> np.ndarray:
        rng = self._rng(rng)
        
        current_values = self.lows.copy().astype(float)
        current_sum = np.sum(current_values)
        residual = self.sum_value - current_sum

        # Randomly distribute the residual in 'step' increments
        num_steps = int(round(residual / self.step))
        
        for _ in range(num_steps):
            # Find indices where adding a step won't exceed the column's high limit
            eligible_indices = [
                i
                for i, val in enumerate(current_values)
                if val + self.step <= self.highs[i] + 1e-9
            ]

            # unreachable once __init__ checks the capacity; kept so a
            # floating-point edge case names itself instead of failing in rng.choice
            if not eligible_indices:
                raise ConstraintViolationError(
                    "Target sum_value is unreachable within defined highs."
                )

            target_idx = rng.choice(eligible_indices)
            current_values[target_idx] += self.step
            
        row[self.cols] = current_values
        return row


class FunctionConstraint(Constraints):
    def __init__(self, fn: ConstraintFn, cols:list[int]):
        super().__init__(cols=cols)
        self.fn = fn
    
    def _constrain(
            self, 
            row: np.ndarray, 
            rng: Optional[np.random.Generator] = None
        ) -> Optional[np.ndarray]:
        
        result = self.fn(row)

        if isinstance(result, Bool):
            return row if result else None
        elif isinstance(result, np.ndarray):
            row[self.cols] = result
            return row
        else:
            raise ConstraintTypeError(
                "Constraint function must return either a boolean or a numpy array"
            )