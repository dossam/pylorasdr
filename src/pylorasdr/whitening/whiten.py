import numpy as np
from pylorasdr.misc import whitn_seq

def whiten(byte_seq: np.ndarray, offset: np.uint8 = 0) -> np.ndarray:
    """
    Apply a whitening sequence to an input byte sequence.

    Parameters
    ----------
    byte_seq : np.ndarray
        Input array of bytes to be whitened. Expected to be a one-dimensional
        NumPy array of integer type (e.g., np.uint8).

    offset : np.uint8, optional
        Starting index in the whitening sequence. Default is 0.

    Returns
    -------
    np.ndarray
        Whitened byte sequence as a NumPy array of the same shape and dtype
        as the input.

    Raises
    ------
    ValueError
        Intended to be raised if the input sequence length exceeds the maximum
        allowed size (255 - offset). Currently not implemented.

    Notes
    -----
    The whitening sequence is defined in `pylorasdr.misc.matrices.whitn_seq`.

    TODO:
    - Replace the current print statement with a proper exception
      (e.g., ValueError) when the input length exceeds the allowed limit.
    - Validate that `byte_seq` has dtype np.uint8.
    - Validate that `offset` stays within valid bounds of the whitening sequence.
    """
    byte_seq_len = len(byte_seq)
    if byte_seq_len > (255 - offset):
        # TODO: implement an exception here
        print("Byte sequence longer than maximum 255. Should not happen")
        return
    
    return byte_seq ^ whitn_seq[offset: offset + byte_seq_len]