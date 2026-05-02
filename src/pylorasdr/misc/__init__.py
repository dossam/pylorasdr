# lorasdrpy/misc/__init__.py
"""
Miscellaneous utilities and helper components.

This module groups together various supporting functions, data structures,
enumerations, and constants used across the library.

Includes:
- Conversion utilities (bin/dec)
- CRC computation
- Circular buffer implementation
- State enums
- Predefined encoding/decoding matrices
"""

# Conversion / utility functions
from .bin2dec import bin2dec
from .dec2bin import dec2bin
from .ccit16 import ccit16

# Data structures
from .circ_buffer import CircBuffer

# Enums
from .enums import ReceiverState, SyncState, LDROMode

# Matrices (assuming these are constants)
from .matrices import (
    G_1,
    G_234,
    H_1,
    H_234,
    CB_1,
    CB_234,
    whitn_seq,
)

__all__ = [
    # Functions
    "bin2dec",
    "dec2bin",
    "ccit16",

    # Classes
    "CircBuffer",

    # Enums
    "ReceiverState",
    "SyncState",
    "LDROMode",

    # Matrices
    "G_1",
    "G_234",
    "H_1",
    "H_234",
    "CB_1",
    "CB_234",
    "whitn_seq",
]