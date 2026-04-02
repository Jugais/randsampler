import numpy as np
from numpy.random import Generator
from .base import Constraints, SelectConstraint
from .. import validate as v
from ..types import Numeric, ArrayLike, Bool, ConstraintFn
from typing import Optional, Callable
from ..errors import ConstraintViolationError, ConstraintError

class MultihotConstraint(SelectConstraint):
    def __init__(
            self, 
            cols: list[int], 
            n_hot: int = 1,
            **kwargs
        ):
        super().__init__(
            cols, 
            min_used=n_hot, 
            max_used=n_hot, 
            **kwargs
        )
        self.n_hot = n_hot

    def _constrain_selected(
            self, 
            row: np.ndarray, 
            selected: np.ndarray,
            rng: Optional[np.random.Generator] = None,
        ) -> np.ndarray:
        row[selected] = 1
        return row
    
class RandomSelectConstraint(SelectConstraint):
    def __init__(
            self, 
            cols: list[int], 
            rng: Optional[np.random.Generator] = None,
            **kwargs
        ):
        super().__init__(cols, reset_cols=False, **kwargs)
        self.rng = rng
    
    def _constrain_selected(
            self, 
            row: np.ndarray, 
            selected: np.ndarray
        ) -> np.ndarray:
        
        not_selected = np.setdiff1d(self.cols, selected)
        row[not_selected] = 0
        return row
    
class SumConstraint(SelectConstraint):
    def __init__(self, 
            cols: list[int], 
            sum_value: Numeric = 1, 
            alpha: Optional[np.ndarray] = None,
            rng: Optional[np.random.Generator] = None,
            **kwargs
        ):
        v.validate_values(sum_value)

        super().__init__(cols, **kwargs)
        self.sum_value = sum_value
        self.alpha = alpha
        self.rng = self._rng(rng)

    def _constrain_selected(self, 
            row: np.ndarray, 
            selected: np.ndarray
        ) -> np.ndarray:
        """
        Args:
            row (np.ndarray): numpy array representing the row to be modified
            selected (np.ndarray): numpy array of selected column indices
            sum_value (float, optional): The desired sum of the selected columns. Defaults to 1.
            alpha (Optional[np.ndarray], optional): The concentration parameters for the Dirichlet distribution. Defaults to None.
            rng (Optional[np.random.Generator], optional): The random number generator. Defaults to None.

        Returns:
            np.ndarray: The modified row
        """
        if self.alpha is None:
            alpha = np.ones(len(selected))
        else:
            alpha = self.alpha

        weights = self.rng.dirichlet(alpha)

        row[selected] = weights * self.sum_value
        return row

class SumIntConstraint(SumConstraint):
    def __init__(self, cols: list[int], sum_value: int = 100, rng: Generator | None = None, **kwargs):
        if sum_value < 0:
            raise TypeError("sum_value must be a non-negative integer")
        super().__init__(cols, sum_value=sum_value, rng=rng, **kwargs)

    def _constrain_selected(self, row: np.ndarray, selected: np.ndarray) -> np.ndarray:
        k = len(selected)
        if k == 1:
            row[selected[0]] = self.sum_value
            return row

        cuts = np.sort(self.rng.choice((self.sum_value + k - 1), k - 1, replace=False))
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
    def __init__(self, cols: list[int], values: list[list], strength:str = "hard", **kwargs):
        super().__init__(cols, **kwargs)
        self.strength = strength
        val_tuples = [tuple(v) for v in values]
        if len(set(val_tuples)) != len(values):
            raise ConstraintViolationError("values must be unique")
        
        self.values = np.array(values, dtype=object)

    def _constrain(
            self, 
            row: np.ndarray, 
            rng: Optional[np.random.Generator] = None,
        ) -> np.ndarray:
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
            raise ConstraintError(f"{self.strength} was not supported")
        
        valid_patterns = self.values[mask]
        if len(valid_patterns) == 0:
            raise ConstraintViolationError("No patterns found in categories")

        idx = rng.integers(len(valid_patterns))
        selected_pattern = valid_patterns[idx]
        
        row[self.cols] = selected_pattern
        return row
    
class RangeConstraint(Constraints):
    def __init__(self, cols: list[int], low: float = 0, high: float = 1, **kwargs):
        super().__init__(cols, **kwargs)
        self.low = low
        self.high = high

    def _constrain(self, row: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        # No need to reset cols here
        rng = self._rng(rng)
        row[self.cols] = rng.uniform(self.low, self.high, size=len(self.cols))
        return row

class StepConstraint(Constraints):
    def __init__(self, col:int, step: float, low: float, high: float,  **kwargs):
        super().__init__(cols=[col], **kwargs)
        self.values = np.arange(low, high + step, step)
        self.col = col
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

class StepSumConstraint(StepConstraint):
    def __init__(
            self, 
            cols: list[int], 
            sum_value: float, 
            lows: Optional[ArrayLike] = None, 
            highs: Optional[ArrayLike] = None, 
            step: float = 1, 
            **kwargs
        ):
        # Initialize parent with the first column's range
        lows = lows if lows is not None else np.zeros(len(cols))
        highs = highs if highs is not None else np.ones(len(cols))*100
        
        super().__init__(
            col=cols[0], 
            low=lows[0], 
            high=highs[0], 
            step=step, 
            **kwargs
        )
        self.cols = cols

        self.lows = np.array(lows)
        self.highs = np.array(highs)
        self.sum_value = sum_value
        self.step = step

    def _constrain(self, row: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = self._rng(rng)
        
        # 1. Initialize all target columns with their respective minimum (low) values
        current_values = self.lows.copy().astype(float)
        
        # 2. Calculate the difference to reach the target sum_value
        current_sum = np.sum(current_values)
        residual = self.sum_value - current_sum
        
        # Basic validation for feasibility
        if residual < -1e-9:
            raise ConstraintViolationError(f"Sum of lows ({current_sum}) exceeds sum_value ({self.sum_value}).")
        
        if not np.isclose(residual % self.step, 0) and not np.isclose(residual % self.step, self.step):
            raise ConstraintViolationError(f"Residual ({residual}) is not a multiple of step ({self.step}).")

        # 3. Randomly distribute the residual in 'step' increments
        num_steps = int(round(residual / self.step))
        
        for _ in range(num_steps):
            # Find indices where adding a step won't exceed the specific column's high limit
            eligible_indices = [
                i for i, val in enumerate(current_values)
                if val + self.step <= self.highs[i] + 1e-9
            ]
            
            if not eligible_indices:
                raise ConstraintViolationError("Target sum_value is unreachable within defined highs.")
            
            # Pick a random column and increment it
            target_idx = rng.choice(eligible_indices)
            current_values[target_idx] += self.step
            
        # 4. Final assignment to the row
        row[self.cols] = current_values
        return row


class FunctionConstraint(Constraints):
    def __init__(self, fn: ConstraintFn):
        super().__init__(cols=[])
        self.fn = fn
    
    def _constrain(
            self, 
            row: np.ndarray, 
            rng: Optional[np.random.Generator] = None
        ) -> np.ndarray:
        result = self.fn(row)

        if isinstance(result, Bool):
            if not result:
                raise ConstraintViolationError("Constraint failed for row: " + str(row))
            else:
                return row
        elif isinstance(result, np.ndarray):
            return result
        else:
            from ..errors import ConstraintTypeError
            raise ConstraintTypeError("Constraint function must return either a boolean or a numpy array")
