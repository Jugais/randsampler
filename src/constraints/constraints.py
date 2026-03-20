import numpy as np
from numpy.random import Generator
from .base import Constraints, SelectConstraint
from .. import validate as v
from ..types import Numeric, ArrayLike, Bool, ConstraintFn
from typing import Optional, Callable

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
        parts = np.diff(np.concatenate(([-1], cuts, [self.sum_value + k - 1]))) - 1

        row[selected] = parts
        return row

class CategoriesConstraint(Constraints):
    def __init__(self, col: int, values: list[float], **kwargs):
        super().__init__([col], **kwargs)
        self.values = values

    def _constrain(self, row: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = self._rng(rng)
        row[self.cols[0]] = rng.choice(self.values)
        return row

class RangeConstraint(Constraints):
    def __init__(self, cols: list[int], lb: float = 0, ub: float = 1, **kwargs):
        super().__init__(cols, **kwargs)
        self.lb = lb
        self.ub = ub

    def _constrain(self, row: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        # No need to reset cols here
        rng = self._rng(rng)
        row[self.cols] = rng.uniform(self.lb, self.ub, size=len(self.cols))
        return row

class FunctionConstraint(Constraints):
    def __init__(self, fn: ConstraintFn):
        super().__init__(cols=[])
        self.fn = fn
    
    def _constrain(
            self, 
            row: np.ndarray, 
            rng: Optional[np.random.Generator] = None
        ) -> Optional[np.ndarray]:
        result = self.fn(row)

        if isinstance(result, Bool):
            if not result:
                return None
            else:
                return row
        elif isinstance(result, np.ndarray):
            return result
        else:
            from ..errors import ConstraintTypeError
            raise ConstraintTypeError("Constraint function must return either a boolean or a numpy array")