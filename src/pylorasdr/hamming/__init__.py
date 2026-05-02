# lorasdrpy/hamming/__init__.py
"""
Hamming coding utilities.

Implements Hamming encoding/decoding functions (the 'custom'  LoRa way).
"""

from .encode import encode
from .decode import decode

__all__ = [
    "encode",
    "decode",
]