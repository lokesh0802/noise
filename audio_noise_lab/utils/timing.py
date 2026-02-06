"""
Timing utilities for measuring processing latency.

Provides context manager and decorator patterns for accurate
timing measurements with sub-millisecond precision.
"""

import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, cast

F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class TimingResult:
    """Container for timing measurement results."""
    
    elapsed_seconds: float
    elapsed_ms: float = field(init=False)
    
    def __post_init__(self) -> None:
        self.elapsed_ms = self.elapsed_seconds * 1000


class Timer:
    """
    High-precision timer for measuring code execution time.
    
    Can be used as a context manager or via start/stop methods.
    Uses perf_counter for monotonic, high-resolution timing.
    
    Examples
    --------
    >>> timer = Timer()
    >>> with timer:
    ...     # code to measure
    ...     pass
    >>> print(f"Elapsed: {timer.elapsed_ms:.2f} ms")
    
    >>> timer.start()
    >>> # code to measure
    >>> timer.stop()
    >>> print(timer.elapsed_ms)
    """
    
    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._measurements: List[float] = []
    
    def start(self) -> "Timer":
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None
        return self
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds."""
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        
        self._end_time = time.perf_counter()
        elapsed = self._end_time - self._start_time
        self._measurements.append(elapsed)
        return elapsed
    
    def reset(self) -> None:
        """Reset the timer state."""
        self._start_time = None
        self._end_time = None
    
    def clear_measurements(self) -> None:
        """Clear all recorded measurements."""
        self._measurements.clear()
    
    @property
    def elapsed_seconds(self) -> float:
        """Elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        
        end = self._end_time if self._end_time is not None else time.perf_counter()
        return end - self._start_time
    
    @property
    def elapsed_ms(self) -> float:
        """Elapsed time in milliseconds."""
        return self.elapsed_seconds * 1000
    
    @property
    def measurements(self) -> List[float]:
        """List of all recorded measurements in seconds."""
        return self._measurements.copy()
    
    @property
    def average_ms(self) -> float:
        """Average of all measurements in milliseconds."""
        if not self._measurements:
            return 0.0
        return (sum(self._measurements) / len(self._measurements)) * 1000
    
    def __enter__(self) -> "Timer":
        self.start()
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.stop()


@contextmanager
def measure_time():
    """
    Context manager for timing code blocks.
    
    Yields a TimingResult-like object that stores elapsed time.
    
    Examples
    --------
    >>> with measure_time() as t:
    ...     # code to measure
    ...     pass
    >>> print(f"Elapsed: {t.elapsed_ms:.2f} ms")
    """
    result = {"elapsed_seconds": 0.0, "elapsed_ms": 0.0}
    start = time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - start
        result["elapsed_seconds"] = elapsed
        result["elapsed_ms"] = elapsed * 1000


def timed(func: F) -> F:
    """
    Decorator to measure function execution time.
    
    Adds 'elapsed_ms' attribute to function after each call.
    
    Examples
    --------
    >>> @timed
    ... def process_audio(data):
    ...     return data * 2
    >>> result = process_audio(audio)
    >>> print(f"Processing took {process_audio.elapsed_ms:.2f} ms")
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        wrapper.elapsed_seconds = elapsed  # type: ignore
        wrapper.elapsed_ms = elapsed * 1000  # type: ignore
        return result
    
    wrapper.elapsed_seconds = 0.0  # type: ignore
    wrapper.elapsed_ms = 0.0  # type: ignore
    return cast(F, wrapper)


def benchmark(
    func: Callable[..., Any],
    *args: Any,
    iterations: int = 10,
    warmup: int = 2,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Benchmark a function with multiple iterations.
    
    Parameters
    ----------
    func : Callable
        Function to benchmark
    *args : Any
        Positional arguments for the function
    iterations : int
        Number of timed iterations (default: 10)
    warmup : int
        Number of warmup iterations (not timed, default: 2)
    **kwargs : Any
        Keyword arguments for the function
        
    Returns
    -------
    dict
        Benchmark statistics including mean, min, max, std in milliseconds
    """
    import statistics
    
    # Warmup runs
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    return {
        "mean_ms": statistics.mean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "iterations": iterations,
    }
