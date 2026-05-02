# lorasdrpy/gray/__init__.py
"""
Gray code conversion utilities.

Provides functions to convert between standard binary representations
and Gray-coded representations.
"""

from .gray2nongray import gray2nongray
from .nongray2gray import nongray2gray

__all__ = [
    "gray2nongray",
    "nongray2gray",
]