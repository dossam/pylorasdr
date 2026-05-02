import numpy as np

def estimate_cfo_frac(samples: np.ndarray, downchirp: np.ndarray, sf: int) -> float:
    """
    Estimate the fractional Carrier Frequency Offset (CFO) from a LoRa preamble.

    Parameters
    ----------
    samples : np.ndarray
        Input complex baseband samples corresponding to concatenated preamble
        symbols. Length should be a multiple of 2**sf.

    downchirp : np.ndarray
        Reference downchirp used for dechirping. Must have length 2**sf.

    sf : int
        Spreading factor, defining the FFT size as number_of_bins = 2**sf.

    Returns
    -------
    float
        Estimated fractional CFO expressed in frequency bin units
        (i.e., normalized with respect to FFT bin spacing).

    Raises
    ------
    ValueError
        Intended to be raised if:
        - The length of `samples` is not a multiple of 2**sf.
        - The length of `downchirp` does not match 2**sf.
        (Not currently enforced in the implementation.)

    Notes
    -----
    TODO:
    - Add input validation for array lengths and consistency.
    - Optionally return CFO-corrected samples if needed.
    - Clarify expected dtype (e.g., complex64 vs complex128).
    """
    number_of_bins = 2 ** sf
    n_symb = len(samples) // number_of_bins
        

    dechirped = np.reshape(samples[:n_symb * number_of_bins], (number_of_bins, -1), order='F') * np.reshape(downchirp, (number_of_bins, 1))
    fft_val = np.fft.fft(dechirped, axis=0)
    mag_sq = np.abs(fft_val)**2

    # find the main bin across symbols
    idx_max = np.argmax(mag_sq.flatten(order='F'))
    idx_max = np.mod(idx_max, number_of_bins)

    # Phase accumulation and cfo_frac estimate
    four_cum = np.sum(fft_val[idx_max, :-1] * np.conj(fft_val[idx_max, 1:]))
    cfo_frac = -np.angle(four_cum) / (2 * np.pi)

    return cfo_frac