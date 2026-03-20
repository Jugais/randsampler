import numpy as np
from typing import Optional
from . import validate as V
from .validate import Numeric

class Constraints:
    @staticmethod
    def _reset_cols(row:np.ndarray, cols:list, value: Numeric=0):
        row[cols] = value
        return row
        
    @staticmethod
    def _rng(rng):
        return rng if rng is not None else np.random.default_rng()

    @staticmethod
    def _select_cols(
            cols:list, 
            min_used:int = 1, 
            max_used:Optional[int] = None, 
            rng:Optional[np.random.Generator] = None
        ) -> np.ndarray:

        rng = Constraints._rng(rng)
        k = len(cols)
        if max_used is None:
            max_used = k
        
        used_k = rng.integers(min_used, max_used + 1)
        selected = rng.choice(cols, size=used_k, replace=False)

        return selected

    @staticmethod
    def multihot(
            row:np.ndarray, 
            cols:list, 
            n_hot:int = 1,
            rng: Optional[np.random.Generator] = None
        ) -> np.ndarray:
        """
        Args:
            row (np.ndarray): numpy array representing the row to be modified
            cols (list): list of column indices to apply the multi-hot constraint on
            min_used (int, optional): minimum number of columns to select. Defaults to 1.
            max_used (int, optional): maximum number of columns to select. Defaults to 1.
            rng (Optional[np.random.Generator], optional): random number generator. Defaults to None.

        Returns:
            np.ndarray: modified row with multi-hot constraint applied
        """

        V.cols(cols)
        V.value(n_hot)
        min_used = max_used = n_hot

        rng = Constraints._rng(rng)
        row = Constraints._reset_cols(row, cols)
        selected = Constraints._select_cols(cols, min_used, max_used, rng)
        row[selected] = 1
        return row
    
    @staticmethod
    def random(
            row:np.ndarray, 
            cols:list, 
            min_used:int = 1, 
            max_used:Optional[int] = None, 
            rng:Optional[np.random.Generator] = None
        ) -> np.ndarray:
        """
        Apply a random constraint to the row.

        Args:
            row (np.ndarray): numpy array representing the row to be modified
            cols (list): list of column indices to apply the random constraint on
            min_used (int, optional): minimum number of columns to select. Defaults to 1.
            max_used (Optional[int], optional): maximum number of columns to select. Defaults to None.
            rng (Optional[np.random.Generator], optional): random number generator. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        
        if max_used is None:
            max_used = len(cols)
        
        
        V.cols(cols)
        V.usage(min_used, max_used)
        
        rng = Constraints._rng(rng)
        # row = Constraints._reset_cols(row, cols)
        selected = Constraints._select_cols(cols, min_used, max_used, rng)
        
        mask = np.ones(len(row), dtype=bool)
        mask[selected] = False
        row[mask] = 0
        return row
    
    @staticmethod
    def sum(
            row:np.ndarray, 
            sum_value:float, 
            cols:list, 
            min_used:int = 1, 
            max_used:Optional[int] = None, 
            alpha:Optional[np.ndarray] = None, 
            rng:Optional[np.random.Generator] = None
        ) -> np.ndarray:
        """
        Args:
            row (np.ndarray): numpy array representing the row to be modified
            sum_value (float): the value that the selected columns should sum up to
            cols (list): list of column indices to apply the sum constraint on
            min_used (int, optional): minimum number of columns to select. Defaults to 1.
            max_used (Optional[int], optional): maximum number of columns to select. Defaults to None.
            alpha (Optional[np.ndarray], optional): weights for the Dirichlet distribution. Defaults to None.
            rng (Optional[np.random.Generator], optional): random number generator. Defaults to None.

        Returns:
            np.ndarray: modified row with sum constraint applied
        """

        if max_used is None:
            max_used = len(cols)

        V.cols(cols)
        V.usage(min_used, max_used)
        V.value(sum_value)

        rng = Constraints._rng(rng)
        row = Constraints._reset_cols(row, cols)
        selected = Constraints._select_cols(cols, min_used, max_used, rng)

        if alpha is None:
            alpha = np.ones(len(selected))
        weights = rng.dirichlet(alpha)

        row[selected] = weights * sum_value
        return row

    @staticmethod
    def sum_int(
            row:np.ndarray, 
            sum_value:int, 
            cols:list, 
            min_used:int = 1,
            max_used:Optional[int] = None, 
            rng:Optional[np.random.Generator]=None
        ) -> np.ndarray:
        
        # stars and bars method for integer partitioning

        if max_used is None:
            max_used = len(cols)
        
        V.cols(cols)
        V.usage(min_used, max_used)
        V.value(sum_value)

        rng = Constraints._rng(rng)
        row = Constraints._reset_cols(row, cols)
        selected = Constraints._select_cols(cols, min_used, max_used, rng)

        k = len(selected)
        if k == 1:
            row[selected[0]] = sum_value
            return row

        cuts = np.sort(rng.choice((sum_value + k - 1), k - 1, replace=False))
        parts = np.diff(np.concatenate(([-1], cuts, [sum_value + k - 1]))) - 1

        row[selected] = parts
        return row
    
    @staticmethod
    def range(
            row: np.ndarray, 
            cols: list, 
            lb: float, 
            ub: float, 
            rng: Optional[np.random.Generator] = None
        ) -> np.ndarray:

        V.cols(cols)
        V.range(lb, ub)
        
        rng = Constraints._rng(rng)
        row[cols] = rng.uniform(lb, ub)
        return row
    
    @staticmethod
    def categories(
            row: np.ndarray, 
            col: int, 
            values: list, 
            rng: Optional[np.random.Generator] = None
        ) -> np.ndarray:
        
        V.value(col)
        V.cols([col])
        
        rng = Constraints._rng(rng)
        row[col] = rng.choice(values)
        return row
