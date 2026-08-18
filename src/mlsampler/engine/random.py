import numpy as np
from ..terminals import spinning
from ..base import BaseSampler, SamplerConfig, DtypeMeta as dm
from ..constraints import *
from ..errors import (
    ConstraintViolationError,
    ConstraintTypeError,
    ConstraintValidationError,
    DuplicateColumnWarning,
    ParallelRngWarning
)

from typing import Optional, overload, Literal, Any
from ..types import Numeric, ArrayLike, ConstraintFn, ColumnRef, SampleOutput
from types import MappingProxyType
from collections import defaultdict
from collections.abc import Sequence
from joblib import Parallel, delayed, effective_n_jobs

import warnings
import inspect  # list a constraint's valid arguments in error messages


_PARALLEL_MIN_SAMPLES = 100

class RandomSampler(BaseSampler):
    """
    Random constraint-based sampler.

    This sampler generates samples based on feature metadata
    inferred from training data. Users can register constraints
    that are applied during sample generation.

    Parameters
    ----------
    config : SamplerConfig
        Configuration object containing feature metadata and sampling settings.

    Notes
    -----
    Two types of constraints are supported:

    - Validation constraints:
        Return a boolean. If False, the sample is rejected.

    - Constructive constraints:
        Return a modified numpy array.

    Internally, constraints are unified to return either:
    - np.ndarray (valid sample)
    - None (invalid sample)

    Sampling is performed with retry logic up to `max_retries`.
    """
    
    __qualname__ = "RandomSampler"
    
    def __init__(self, config: SamplerConfig):
        super().__init__(config)
        self.rng = np.random.default_rng(config.random_state)
        self.features = config.features
        self.n_jobs = config.n_jobs
        self.seed = config.random_state
        self.max_retries = config.max_retries

        self.dim = len(self.features)
        self._constraints:list[Constraints] = []
        self._funcs: list[FunctionConstraint] = []
        self._registry = MappingProxyType({
            "sum": SumConstraint,
            "sumint": SumIntConstraint,
            "multihot": MultihotConstraint,
            "random": RandomSelectConstraint,
            "range": RangeConstraint,
            "categories": CategoriesConstraint,
            "step": StepConstraint,
            "stepsum": SumStepConstraint,
        })

    @property
    def constraints(self):
        return tuple(self._constraints.copy())

    def reset_constraints(self):
        self._constraints = []
        self._funcs = []

    @overload
    def set_constraints(
            self,
            constraint_fn: Literal["sum"],
            reset: bool = ...,
            *,
            cols: Sequence[ColumnRef],
            sum_value: Numeric = ...,
            method: str = ...,
            alpha: Optional[np.ndarray] = ...,
            min_used: int = ...,
            max_used: Optional[int] = ...,
            rng: Optional[np.random.Generator] = ...,
        ) -> None: ...

    @overload
    def set_constraints(
            self,
            constraint_fn: Literal["sumint"],
            reset: bool = ...,
            *,
            cols: Sequence[ColumnRef],
            sum_value: int = ...,
            min_used: int = ...,
            max_used: Optional[int] = ...,
            rng: Optional[np.random.Generator] = ...,
        ) -> None: ...

    @overload
    def set_constraints(
            self,
            constraint_fn: Literal["multihot"],
            reset: bool = ...,
            *,
            cols: Sequence[ColumnRef],
            n_hot: int = ...,
            rng: Optional[np.random.Generator] = ...,
        ) -> None: ...

    @overload
    def set_constraints(
            self,
            constraint_fn: Literal["random"],
            reset: bool = ...,
            *,
            cols: Sequence[ColumnRef],
            min_used: int = ...,
            max_used: Optional[int] = ...,
            rng: Optional[np.random.Generator] = ...,
        ) -> None: ...

    @overload
    def set_constraints(
            self,
            constraint_fn: Literal["range"],
            reset: bool = ...,
            *,
            cols: Sequence[ColumnRef],
            low: float = ...,
            high: float = ...,
            rng: Optional[np.random.Generator] = ...,
        ) -> None: ...

    @overload
    def set_constraints(
            self,
            constraint_fn: Literal["categories"],
            reset: bool = ...,
            *,
            cols: Sequence[ColumnRef],
            values: Sequence[Any],
            strength: str = ...,
            rng: Optional[np.random.Generator] = ...,
        ) -> None: ...

    @overload
    def set_constraints(
            self,
            constraint_fn: Literal["step"],
            reset: bool = ...,
            *,
            col: ColumnRef,
            step: float,
            low: float,
            high: float,
            rng: Optional[np.random.Generator] = ...,
        ) -> None: ...

    @overload
    def set_constraints(
            self,
            constraint_fn: Literal["stepsum"],
            reset: bool = ...,
            *,
            cols: Sequence[ColumnRef],
            sum_value: float,
            lows: Optional[ArrayLike] = ...,
            highs: Optional[ArrayLike] = ...,
            step: float = ...,
            rng: Optional[np.random.Generator] = ...,
        ) -> None: ...

    @overload
    def set_constraints(
            self,
            constraint_fn: ConstraintFn,
            reset: bool = ...,
            *,
            cols: Sequence[ColumnRef],
        ) -> None: ...

    def set_constraints(
            self,
            constraint_fn: str | ConstraintFn,
            reset=False,
            **kwargs
        ):
        """
        Register a constraint on the sampler.

        Every constraint takes ``cols`` (a list of column indices) except ``"step"``,
        which takes ``col`` (a single index). All of them also accept ``rng``, which is
        honoured only when ``n_jobs=1``.

        Supported constraints, with the arguments each one accepts:

        - ``"sum"`` : spreads ``sum_value`` over ``cols``; also ``method``, ``alpha``, ``min_used``, ``max_used``.
        - ``"sumint"`` : as ``"sum"``, but splits an integer total; also ``min_used``, ``max_used``.
        - ``"multihot"`` : sets exactly ``n_hot`` of ``cols`` to 1 and the rest to 0.
        - ``"random"`` : keeps a random subset of ``cols``, zeroes the rest; also ``min_used``, ``max_used``.
        - ``"range"`` : draws ``cols`` uniformly from ``[low, high]``.
        - ``"categories"`` : restricts ``cols`` to the combinations in ``values``; ``strength`` is ``"soft"`` or ``"hard"``.
        - ``"step"`` : snaps ``col`` onto ``low + k * step`` inside ``[low, high]``.
        - ``"stepsum"`` : spreads ``sum_value`` over ``cols`` in ``step`` units, bounded by ``lows`` and ``highs``.
        - a callable : takes a row, returns a bool to accept or reject it, or an array to write into ``cols``.

        Parameters
        ----------
        constraint_fn : str or callable
            The constraint to register, chosen from the list above.
        reset : bool, default=False
            If True, clears existing constraints before adding the new one.
        **kwargs : Any
            Arguments for the chosen constraint, as listed above.

        Raises
        ------
        ConstraintTypeError
            The name is not a registered constraint.
        ConstraintValidationError
            The arguments do not match the chosen constraint.

        Examples
        --------
        >>> from mlsampler import RandomSampler
        >>> sampler = RandomSampler.setup(X_train)
        >>> sampler.set_constraints(
        ...     lambda row: (0 < row[0] < 1) and (0 < row[1] < 1), cols=[0, 1]
        ... )
        >>> sampler.set_constraints("sum", sum_value=1, cols=[2, 3, 4], max_used=2)
        >>> # Step constraint takes col, not cols
        >>> sampler.set_constraints("step", col=2, step=0.5, low=0.0, high=5.0)
        """

        if reset:
            self._constraints = []
            self._funcs = []

        cols = kwargs.get("cols")
        if isinstance(cols, ArrayLike):
            kwargs["cols"] = [self._resolve(ref) for ref in cols]
        if "col" in kwargs:
            kwargs["col"] = self._resolve(kwargs["col"])

        if callable(constraint_fn):
            self._constraints.append(self._build("callable", FunctionConstraint, fn=constraint_fn, **kwargs))
            self._funcs.append(self._build("callable", FunctionConstraint, fn=constraint_fn, **kwargs))
        elif isinstance(constraint_fn, str) and constraint_fn in self._registry:
            self._constraints.append(self._build(constraint_fn, self._registry[constraint_fn], **kwargs))
        else:
            raise ConstraintTypeError(
                f"Unsupported constraint type: {constraint_fn!r}. "
                f"Valid types: {sorted(self._registry)} or a callable."
            )

    # a str is a name, an int is always a position
    def _resolve(self, ref: ColumnRef) -> ColumnRef:
        if not isinstance(ref, str):
            return ref

        names = self.feature_names
        if ref in names:
            return names.index(ref)

        if all(isinstance(name, int) for name in names):
            raise ConstraintValidationError(
                f"Cannot look up column {ref!r}: this sampler has no column names "
                f"(it was set up without them). Use a position in 0-{self.n_features - 1}."
            )
        raise ConstraintValidationError(
            f"Unknown column {ref!r}. Valid names: {names}"
        )

    # address columns the way the user did. `name` holds the position
    # when the input carried no names, so this needs no branch on that
    def _labels(self, cols) -> list:
        features = self.config.features
        return [features[c].name if c < len(features) else c for c in cols]

    def _name(self, constraint) -> str:
        cols = self._labels(getattr(constraint, "cols", []))
        for key, cls in self._registry.items():
            if type(constraint) is cls:
                return f"{key!r} on cols={cols}"
        return f"callable on cols={cols}"

    def _build(self, constraint_fn: str, cls, **kwargs):
        try:
            return cls(**kwargs)
        except TypeError as e:
            detail = str(e).split("() ", 1)[-1]
            valid = [p for p in inspect.signature(cls).parameters if p != "self"]
            raise ConstraintValidationError(
                f"{constraint_fn!r} {detail}. Valid arguments: {valid}"
            ) from None

    def _base_sample(self, rng: np.random.Generator):
        x = np.empty(self.dim, dtype=object)
        for i, f in enumerate(self.features):
            if f.dtype == dm.const:
                x[i] = f.low
            elif f.dtype == dm.bin:
                x[i] = rng.integers(0, 2)
            elif f.dtype == dm.integer and f.low is not None and f.high is not None:
                x[i] = rng.integers(int(f.low), int(f.high) + 1)
            elif f.dtype == dm.float and f.low is not None and f.high is not None:
                x[i] = rng.uniform(f.low, f.high)
            else:
                x[i] = 0

        return x
    
    def _fill_categoricals(self, x, rng: np.random.Generator):
        for i, f in enumerate(self.features):
            if f.dtype == dm.cat:
                if f.categories:
                    x[i] = rng.choice(f.categories)
                else:
                    x[i] = "unknown"
        return x

    def _apply_constraints(self, row: np.ndarray, rng: np.random.Generator) -> Optional[np.ndarray]:
        """
        Apply all registered constraints sequentially.

        Parameters
        ----------
        row : np.ndarray Input sample.
        rng : np.random.Generator
            Generator for this `sample()` call, handed to every constraint.

        Returns
        -------
        np.ndarray or None
            - np.ndarray: if all constraints succeed
            - None: if any constraint fails

        Notes
        -----
        Each constraint must return either:
        - np.ndarray (possibly modified)
        - None (indicating failure)
        """
        for constraint in self:
            if isinstance(constraint, FunctionConstraint):
                continue

            result = constraint(row, rng)
            if result is None:
                return None  # Constraint violation
            row = result
        return row

    def _apply_funcs(self, row: np.ndarray, rng: np.random.Generator) -> Optional[np.ndarray]:
        for constraint in self._funcs:
            result = constraint(row, rng)
            if result is None:
                return None
        return row

    def _detect_conflicts(self):
        col_usage = defaultdict(list)
        constraints_by_col = defaultdict(list)

        for i, c in enumerate(self._constraints):
            for col in getattr(c, "cols", []):
                col_usage[col].append(i)
                constraints_by_col[col].append(c)

        for col, ids in col_usage.items():
            constraints = constraints_by_col[col]
            types = {
                key for c in constraints
                for key, cls in self._registry.items() if isinstance(c, cls)
            }

            if col >= self.n_features:
                raise ConstraintValidationError(
                    f"Column {col} is out of range: the data has {self.n_features} "
                    f"columns (valid indices 0-{self.n_features - 1})."
                )

            meta = self.config.features[col]

            if meta.dtype == dm.cat:
                invalid = types & {"sum", "sumint", "range"}

                if invalid:
                    raise ConstraintViolationError(
                        f"{invalid} cannot be applied to categorical column: {meta.name!r}"
                    )

            if meta.dtype == dm.const:
                if len(ids) > 1:
                    warnings.warn(
                        f"Const column {meta.name!r} has multiple constraints {ids} (types={types})",
                        DuplicateColumnWarning
                    )

            if len(ids) > 1:
                warnings.warn(
                    f"Column {meta.name!r} used in multiple constraints {sorted(types)} "
                    f"(registered at positions {ids})",
                    DuplicateColumnWarning
                )
    
    # a generator handed to a constraint is single-process only:
    # joblib pickles the constraint, so the worker's copy advances in isolation
    def _detect_parallel_rng(self):
        if self.n_jobs == 1:
            return

        named = [
            self._name(c) for c in self._constraints if getattr(c, "rng", None) is not None
        ]
        if named:
            warnings.warn(
                f"Constraints {named} were given an explicit rng, which is ignored "
                f"when n_jobs != 1 (n_jobs={self.n_jobs}). Use n_jobs=1 to keep it, "
                "or set random_state on the sampler instead.",
                ParallelRngWarning
            )

    # the retry loop must keep advancing a single stream or every attempt reproduces the rejected row
    def _generate_one(self, rng: np.random.Generator):
        for _ in range(self.max_retries):
            x = self._base_sample(rng)
            x = self._fill_categoricals(x, rng)
            x = self._apply_constraints(x, rng)
            if x is None:
                continue

            x = self._apply_funcs(x, rng)
            if x is not None:
                return x

        raise ConstraintViolationError(
            f"Max retries exceeded: no row satisfied every constraint in "
            f"{self.max_retries} attempts. {self._diagnose(rng)} "
            "Constraints may be mutually infeasible. relax them or raise max_retries."
        )

    # names the constraint that rejected one more candidate
    # Mirrors the two phases of the real pipeline
    # runs only on the failure path, so sampling pays nothing
    def _diagnose(self, rng: np.random.Generator) -> str:
        row = self._fill_categoricals(self._base_sample(rng), rng)

        for constraint in self:
            if isinstance(constraint, FunctionConstraint):
                continue
            result = constraint(row, rng)
            if result is None:
                return f"A further candidate was rejected by {self._name(constraint)}."
            row = result

        for constraint in self._funcs:
            if constraint(row, rng) is None:
                return f"A further candidate was rejected by {self._name(constraint)}."

        return f"Registered constraints: {[self._name(c) for c in self._constraints]}."

    # unit of parallel dispatch: one task per worker
    def _generate_chunk(
            self, 
            n_samples: int, 
            rng: np.random.Generator
        ) -> list:
        return [self._generate_one(rng) for _ in range(n_samples)]

    def _sample(self, n_samples: int) -> np.ndarray:
        self.rng = np.random.default_rng(self.seed)

        if self.n_jobs == 1 or n_samples < _PARALLEL_MIN_SAMPLES:
            samples = self._generate_chunk(n_samples, self.rng)
        else:
            n_chunks = min(effective_n_jobs(self.n_jobs), n_samples)
            sizes = [len(c) for c in np.array_split(np.arange(n_samples), n_chunks)]
            seeds = np.random.SeedSequence(self.seed).spawn(n_chunks)
            chunks: list[list] = Parallel(n_jobs=self.n_jobs)(
                delayed(self._generate_chunk)(size, np.random.default_rng(seed))
                for size, seed in zip(sizes, seeds)
            )  # type: ignore[assignment]
            samples = [row for chunk in chunks for row in chunk]
        return np.array(samples)


    def sample(self, n_samples: int) -> SampleOutput:
        """
        Generate samples satisfying all registered constraints.

        Parameters
        ----------
        n_samples : int
            Total number of samples to generate.

        Returns
        -------
        np.ndarray or DataFrame
            Shape (n_samples, n_features). The same type `setup` was given: an
            object array for an array, or a pandas/polars DataFrame carrying the
            input's column names and per-column dtypes.

        Raises
        ------
        ConstraintValidationError
            A constraint refers to a column the data does not have.
        ConstraintViolationError
            A constraint cannot apply to a column's dtype, or no row satisfied every
            constraint within `max_retries`.

        Warns
        -----
        DuplicateColumnWarning
            A column is used by more than one constraint.
        ParallelRngWarning
            A constraint was given an explicit `rng` while `n_jobs != 1`.

        Notes
        -----
        The generator is reseeded from `random_state` at the start of every call, so a
        seeded sampler returns the same rows each time. Pass a different seed, or
        `random_state=None`, for fresh draws.
        """
        
        self._detect_conflicts()
        self._detect_parallel_rng()

        with spinning():
            samples = self._sample(n_samples)

        return self._to_frame(samples)


    