import numpy as np
from pylorasdr.misc import dec2bin

def nongray2gray(symbols: np.ndarray, bitwidth: int, binary_out: bool = True) -> np.ndarray:
    """
    Convert binary (natural) integer symbols to Gray-coded representation.

    This function converts integer symbols into their Gray-coded. The output can be returned
    either as binary vector representation or as integer Gray-coded symbols.

    Parameters
    ----------
    symbols : np.ndarray
        Input array of integer symbols (non-Gray encoded).

    bitwidth : int
        Number of bits used to represent each symbol.

    binary_out : bool, optional
        If True, returns a binary matrix representation of the Gray-coded
        symbols with shape (n_symbols, bitwidth). If False, returns a 1D
        array of integer Gray-coded symbols. Default is True.

    Returns
    -------
    np.ndarray
        Either:
        - Binary matrix of shape (n_symbols, bitwidth) if `binary_out=True`, or
        - 1D array of Gray-coded integer symbols if `binary_out=False`.
    """
    m_symbols = np.atleast_1d(symbols)
    n_symb = m_symbols.shape[0]

    symbols_gray = np.zeros(n_symb, dtype=int)
    for symb_idx in range(n_symb):
        symbols_gray[symb_idx] = m_symbols[symb_idx] ^ (m_symbols[symb_idx] >> 1)

    if binary_out:    
        return dec2bin(symbols_gray, bitwidth, 'left-msb')
    
    return symbols_gray