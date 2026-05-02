import numpy as np

def bin2dec(bin_vect: np.ndarray, order: str = 'left-msb') -> np.ndarray:
    """
    Convert binary vectors to their decimal (unsigned integer) representation.

    Each row of the input array is interpreted as a binary number and converted
    into its corresponding unsigned integer value.

    Parameters
    ----------
    bin_vect : np.ndarray
        Input binary array. Can be either 1D or 2D. Each row represents a
        binary number with up to 16 bits.

    order : str, optional
        Bit ordering convention:
        - 'left-msb': Most significant bit is on the left (default).
        - 'right-msb': Most significant bit is on the right.

    Returns
    -------
    np.ndarray
        1D array of type np.uint16 containing the decimal representation of
        each input binary vector.

    Raises
    ------
    ValueError
        Intended to be raised if `order` is not 'left-msb' or 'right-msb'
        (not currently enforced in implementation).

    Notes
    -----
    TODO:
    - Replace print statement with proper exception handling for invalid `order`.
    - Validate that input contains only binary values (0 or 1).
    - Enforce maximum supported bit width if required.
    """
    pow2vect = np.array([[1, 2, 4, 8,
                         16, 32, 64, 128,
                         256, 512, 1024, 2048,
                         4096, 8192, 16384, 32768]])

    m_bin_vect = np.atleast_1d(bin_vect)          # safeguard if input is pure python list
    if m_bin_vect.ndim == 1:
        width = m_bin_vect.shape[0]
    else:
        width = m_bin_vect.shape[1]
        
    pow2vect = pow2vect[0, 0:width]
 
    if order == 'left-msb':
        pow2vect = np.flip(pow2vect)
    elif order != 'right-msb':
        # TODO: implement an exception here instead
        print(f"Unsupported order {order}. Use 'left-msb' or 'right-msb'!")
        return
    
    return np.uint16(np.dot(m_bin_vect, pow2vect)).squeeze()