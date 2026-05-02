import numpy as np

def dec2bin(dec_vect: np.ndarray, width: int, order: str = 'left-msb') -> np.ndarray:
    """
    Convert integer values into binary representation.

    Each integer in the input vector is converted into a fixed-width binary
    representation, returned as a 2D array where each row corresponds to one
    integer.

    Parameters
    ----------
    dec_vect : np.ndarray
        Input vector of integers to convert.

    width : int
        Number of bits for the binary representation of each integer.

    order : str, optional
        Bit ordering convention:
        - 'left-msb': Most significant bit is placed at the left (default).
        - 'right-msb': Most significant bit is placed at the right.

    Returns
    -------
    np.ndarray
        2D integer array of shape (n_samples, width), where each row is the
        binary representation of the corresponding input integer.

    Raises
    ------
    ValueError
        Intended to be raised if:
        - Input integers exceed the representable range for the given width.
        - Unsupported `order` is provided.
        (Not currently enforced in the implementation.)

    Notes
    -----
    TODO:
    - Replace silent assumptions with proper exceptions for invalid `order`.
    """

    m_dec_vect = np.atleast_1d(dec_vect)
    n_rows = len(m_dec_vect)
    bin_mat = np.zeros([n_rows, width], dtype=int)
    
    for row_idx in range(n_rows):
        bin_tmp = np.binary_repr(m_dec_vect[row_idx], width)
        for bit_idx in range(width):
            bin_mat[row_idx, bit_idx] = int(bin_tmp[bit_idx])
        
    # Default conversion above is 'left-msb'
    if order == 'right-msb':
        bin_mat = np.flip(bin_mat, 1)
            
    return bin_mat