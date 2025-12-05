"""Worker state management for parallel execution.

This module provides infrastructure for managing worker state in
concurrent.futures executors (ProcessPoolExecutor and ThreadPoolExecutor).

Module-level globals are required because:
- concurrent.futures requires worker functions to be picklable (module-level)
- Executor initializers run once per worker to set up shared state efficiently
- Process isolation: Each process has separate memory, so process-level globals are safe
- Thread-local storage: threading.local() provides thread-safe state for worker threads

State variables (initialized in RoutingApp.run() when executors are created):
- _WORKER_STATE: Per-process state for ProcessPoolExecutor workers
- _THREAD_LOCAL_STATE: Per-thread state for ThreadPoolExecutor workers
- _SHARED_FORCING: Shared forcing data for threads only (avoids serialization overhead)
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import ForcingData, HyperParams


# ========== Module-level Worker State ==========

_WORKER_STATE = None
_THREAD_LOCAL_STATE = None
_SHARED_FORCING = None


@dataclass
class WorkerState:
    """State shared across worker function calls in a single process.

    Attributes
    ----------
    forcing : ForcingData
        Ocean forcing data (currents, winds, waves)
    rng : np.random.Generator
        Random number generator for this worker
    params : HyperParams
        Algorithm hyperparameters
    """

    forcing: ForcingData
    rng: np.random.Generator
    params: HyperParams


def _initialize_worker_process(
    forcing: ForcingData, seed: int, params: HyperParams
) -> None:
    """Initialize worker process with shared state.

    Called once per worker process at creation time by ProcessPoolExecutor.
    Avoids expensive serialization of forcing data and RNG initialization on every task.

    Parameters
    ----------
    forcing : ForcingData
        Ocean forcing data to share across tasks
    seed : int
        Random seed for this worker's RNG
    params : HyperParams
        Algorithm hyperparameters to share across tasks
    """
    global _WORKER_STATE
    _WORKER_STATE = WorkerState(
        forcing=forcing,
        rng=np.random.default_rng(seed),
        params=params,
    )


def _initialize_worker_thread(seed: int, params: HyperParams) -> None:
    """Initialize worker thread with thread-local state.

    Called once per worker thread at creation time by ThreadPoolExecutor.
    Threads share memory, so forcing data is accessed from module-level _SHARED_FORCING.

    Parameters
    ----------
    seed : int
        Random seed base for this worker's RNG
    params : HyperParams
        Algorithm hyperparameters to share across tasks
    """
    _THREAD_LOCAL_STATE.state = WorkerState(
        forcing=_SHARED_FORCING,
        rng=np.random.default_rng(seed + threading.get_ident()),
        params=params,
    )


def _initialize_sequential(
    forcing: ForcingData, seed: int, params: HyperParams
) -> None:
    """Initialize worker state for sequential execution.

    Called once at setup when running in sequential mode (no executor).
    Sets up global state that will be reused across all stages.

    Parameters
    ----------
    forcing : ForcingData
        Ocean forcing data to share across tasks
    seed : int
        Random seed for the sequential executor's RNG
    params : HyperParams
        Algorithm hyperparameters to share across tasks
    """
    global _WORKER_STATE
    _WORKER_STATE = WorkerState(
        forcing=forcing,
        rng=np.random.default_rng(seed),
        params=params,
    )


def _get_state() -> WorkerState:
    """Get the worker state for current process/thread.

    Returns thread-local state if running in a worker thread,
    otherwise returns global process state.

    Returns
    -------
    WorkerState
        The worker state for this process or thread
    """
    if hasattr(_THREAD_LOCAL_STATE, "state"):
        return _THREAD_LOCAL_STATE.state
    return _WORKER_STATE


class SequentialExecutor:
    """Sequential executor that mimics concurrent.futures.Executor interface.

    Provides a `.map()` method for compatibility with ThreadPoolExecutor
    and ProcessPoolExecutor, but executes tasks sequentially in the main thread.
    This allows for a unified executor interface across all execution modes,
    eliminating conditional branching in stage methods.
    """

    def __init__(self, initializer=None, initargs=()):
        """Initialize the sequential executor.

        Parameters
        ----------
        initializer : callable, optional
            Function to call before executing tasks
        initargs : tuple, optional
            Arguments to pass to the initializer
        """
        if initializer is not None:
            initializer(*initargs)

    def map(self, func, *iterables):
        """Map a function over iterables sequentially.

        Compatible with concurrent.futures.Executor.map() interface.

        Parameters
        ----------
        func : callable
            Function to apply to each element
        *iterables : iterables
            One or more iterables to map over

        Yields
        ------
        Results of applying func to elements
        """
        return map(func, *iterables)

    def shutdown(self, wait=True):
        """Shutdown the executor (no-op for sequential execution).

        Parameters
        ----------
        wait : bool, optional
            Ignored for sequential execution (exists for interface compatibility)
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
        return False
