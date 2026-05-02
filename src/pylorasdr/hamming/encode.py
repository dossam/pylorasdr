import numpy as np
from pylorasdr.misc import G_1, G_234

def encode(nibbles: np.ndarray, cr: int) -> np.ndarray:
    """
    Encode binary nibbles 'LoRa-like' Hamming codewords.

    This function encodes input binary information bits into codewords 
    using predefined generator matrices selected according to
    the coding rate.

    Parameters
    ----------
    nibbles : np.ndarray
        Input binary array of shape (n_samples, k), where k depends on
        the coding configuration.

    cr : int
        Coding rate selector. Determines which generator matrix is used:
        - 1: uses G_1 generator matrix
        - 2, 3, 4: uses G_234 generator matrix with truncation

    Returns
    -------
    np.ndarray
        Encoded binary codeword matrix.

    Raises
    ------
    ValueError
        If `cr` is not in {1, 2, 3, 4} (not currently enforced).
    """
    if cr <= 0 or cr > 4:
        print("Invalid CR provided. Valid values are 1, 2, 3 and 4")
        # TODO: implement an exception raise here
        return
    
    if cr == 1:
        cw = np.mod(nibbles @ G_1, 2)
    else:
        # cr 2, 3 or 4
        cw = np.mod(nibbles @ G_234, 2)
        cw = cw[:, 0: 4 + cr]

    return cw