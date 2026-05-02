import numpy as np

def gray2nongray(symbols: np.ndarray, bitwidth: int) -> np.ndarray:
    """
    Convert Gray-coded symbols to binary (natural) representation.

    This function converts an array of Gray-coded integers into their
    corresponding binary values.

    Parameters
    ----------
    symbols : np.ndarray
        Input array of Gray-coded integer symbols.

    bitwidth : int
        Number of bits used to represent each symbol.

    Returns
    -------
    np.ndarray
        Array of decoded binary (non-Gray) integer symbols.
    """
    m_symbols = np.atleast_1d(symbols)
    n_symb = len(m_symbols)
    out_symbols = np.zeros_like(m_symbols, dtype=np.uint16)

    for symb_idx in range(n_symb):
        out_symbols[symb_idx] = m_symbols[symb_idx]
        for bit_idx in range(1, bitwidth):
            out_symbols[symb_idx] = out_symbols[symb_idx] ^ (
                m_symbols[symb_idx] >> bit_idx
            )

    return out_symbols