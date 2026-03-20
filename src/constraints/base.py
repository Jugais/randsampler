from abc import ABC, abstractmethod
from typing import Optional
from .. import validate as v
import numpy as np

class Constraints(ABC):
    def __init__(self, cols: list[int]):
        v.validate_cols(cols)
        self.cols = cols

    def _rng(
            self, 
            rng: Optional[np.random.Generator] = None
        ) -> np.random.Generator:
        return rng if rng is not None else np.random.default_rng()
    
    def __call__(self, row: np.ndarray, rng: Optional[np.random.Generator] = None):
        return self._constrain(row, rng)
    
    def __repr__(self):
        if not self.cols:
            return f"<{self.__class__.__name__}>"
        return f"<{self.__class__.__name__}(cols={self.cols})>"
    
    @abstractmethod
    def _constrain(self, row: np.ndarray, rng: Optional[np.random.Generator] = None):
        pass


class SelectConstraint(Constraints):
    def __init__(
            self, 
            cols: list[int], 
            min_used: int = 1, 
            max_used: Optional[int] = None,
            reset_cols: bool = True,
            **kwargs
        ):
        
        if max_used is None:
            max_used = len(cols)
        v.validate_usage(min_used, max_used)

        super().__init__(cols)
        self.min_used = min_used
        self.max_used = max_used 
        self.reset_cols = reset_cols

    def _reset_cols(self, row:np.ndarray, cols:list, value: float = 0):
        row[cols] = value
        return row
    
    def _constrain(self, 
            row: np.ndarray, 
            rng: Optional[np.random.Generator] = None
        ) -> np.ndarray:
        rng = self._rng(rng)
        self.rng = rng
        if self.reset_cols:
            row = self._reset_cols(row, self.cols)
        used = rng.integers(self.min_used, self.max_used + 1)
        selected = rng.choice(self.cols, size=used, replace=False)

        return self._constrain_selected(row, selected)

    @abstractmethod
    def _constrain_selected(self, row: np.ndarray, selected: np.ndarray) -> np.ndarray:
        pass


