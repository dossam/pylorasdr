import numpy as np

def modulate(sf: int, S: np.ndarray, os_factor: int = 1, upchirp: bool = True) -> np.ndarray:
    """
    Generate LoRa chirp samples from one or more input symbols.

    This function constructs complex baseband chirps corresponding to the given
    LoRa symbol(s), based on the specified spreading factor and oversampling factor.
    It supports both upchirps and downchirps.

    Parameters
    ----------
    sf : int
        Spreading factor. Determines the number of samples per symbol
        as number_of_bins = 2**sf.

    S : np.ndarray
        Input symbol or sequence of symbols. Can be a scalar or a
        one-dimensional NumPy array of integers in the range [0, 2**sf - 1].

    os_factor : int, optional
        Oversampling factor. Defines how many samples are used per chip.
        Default is 1.

    upchirp : bool, optional
        If True, generates an upchirp (increasing frequency).
        If False, generates a downchirp (decreasing frequency).
        Default is True.

    Returns
    -------
    np.ndarray
        2D array of complex64 samples with shape
        (number_of_bins * os_factor, number_of_symbols), where each column
        corresponds to one modulated symbol.

    Raises
    ------
    ValueError
        Intended to be raised if:
        - `sf` is not a positive integer.
        - Symbols in `S` fall outside the valid range [0, 2**sf - 1].
        - `os_factor` is less than 1.
        (Not currently enforced in the implementation.)

    Notes
    -----
    TODO:
    - Add input validation for `sf`, `S`, and `os_factor`.
    - Ensure symbol values are integers within the valid range.
    - Consider supporting different output dtypes if needed.
    """
    m_S = np.atleast_1d(S)
    number_of_bins = 2**sf
    n = np.arange(0, 2**sf*os_factor)                         # vector
    n_symb = len(m_S)                                     # number of symbols
    samples = np.zeros((number_of_bins*os_factor, n_symb), dtype=np.complex64, order='F')

    for symb_idx in range(n_symb):
        n_f = (number_of_bins - m_S[symb_idx])*os_factor
        if upchirp:
            samples[0:n_f, symb_idx] = np.exp(2j*np.pi*(n[0:n_f]**2 / (2*number_of_bins * os_factor**2) + (m_S[symb_idx]/number_of_bins - 1/2)*n[0:n_f]/os_factor))
            samples[n_f:, symb_idx] = np.exp(2j*np.pi*(n[n_f:]**2 / (2*number_of_bins * os_factor**2) + (m_S[symb_idx]/number_of_bins - 3/2)*n[n_f:]/os_factor))
        else:
            samples[0:n_f, symb_idx] = np.exp(-2j*np.pi*(n[0:n_f]**2 / (2*number_of_bins * os_factor**2) + (m_S[symb_idx]/number_of_bins - 1/2)*n[0:n_f]/os_factor))
            samples[n_f:, symb_idx] = np.exp(-2j*np.pi*(n[n_f:]**2 / (2*number_of_bins * os_factor**2) + (m_S[symb_idx]/number_of_bins - 3/2)*n[n_f:]/os_factor))

    return samples