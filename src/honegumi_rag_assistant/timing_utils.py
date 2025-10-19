"""
Timing utilities for measuring node execution time.

This module provides a simple decorator to time node execution
and print the results in a user-friendly format.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

from .app_config import settings


def time_node(node_name: str) -> Callable:
    """Decorator to time a node's execution and print the result.
    
    Only prints timing information when debug mode is enabled.
    
    Parameters
    ----------
    node_name : str
        Human-readable name of the node (e.g., "Parameter Selector")
    
    Returns
    -------
    Callable
        Decorated function that prints timing information
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            
            # Only print timing in debug mode
            if settings.debug:
                print(f"⏱️  {node_name}: {elapsed:.2f}s")
            
            return result
        return wrapper
    return decorator
