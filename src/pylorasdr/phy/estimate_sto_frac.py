import numpy as np

def estimate_sto_frac(samples: np.ndarray, sf: int, ref_downchirp: np.ndarray) -> float:
    """
    Estimate the fractional Symbol Timing Offset (STO) from LoRa preamble samples.

    Parameters
    ----------
    samples : np.ndarray
        Input complex baseband samples corresponding to concatenated preamble
        symbols. Length should be a multiple of 2**sf.

    sf : int
        Spreading factor, defining the base FFT size as number_of_bins = 2**sf.

    ref_downchirp : np.ndarray
        Reference downchirp used for dechirping. Must have length 2**sf.

    Returns
    -------
    float
        Estimated fractional STO, in the range ]-0.5, 0.5].

    Raises
    ------
    ValueError
        Intended to be raised if:
        - The length of `samples` is not a multiple of 2**sf.
        - The length of `ref_downchirp` does not match 2**sf.
        (Not currently enforced in the implementation.)

    Notes
    -----
    TODO:
    - Add input validation for array sizes and consistency.
    - add reference to the origin of the estimation algorithm.
    - Clarify expected dtype (e.g., complex64 vs complex128).
    """
    number_of_bins = 2**sf
    n_symb = len(samples)//number_of_bins
    
    fft_mag_sq = np.zeros(2*number_of_bins, dtype=np.float32)

    dechirped = np.reshape(samples[0:n_symb*number_of_bins], (number_of_bins, -1), order='F') * np.reshape(ref_downchirp, (number_of_bins, 1))
    fft_in = np.zeros((2*number_of_bins, n_symb), dtype=np.complex64)
    fft_in[:number_of_bins, :] = dechirped
    fft_out = np.fft.fft(fft_in, axis=0)

    fft_mag_sq = np.sum(np.abs(fft_out)**2, axis=1)

    k0 = np.argmax(fft_mag_sq)
    
    Y_1 = fft_mag_sq[np.mod(k0 - 1, 2*number_of_bins)]
    Y0 = fft_mag_sq[k0]
    Y1 = fft_mag_sq[np.mod(k0 + 1, 2*number_of_bins)]
    
    u = 64*number_of_bins/406.5506497
    v = u * 2.4674
    
    wa = (Y1 - Y_1)/(u*(Y1 + Y_1) + v*Y0)
    ka = wa*number_of_bins/np.pi
    k_residual = np.fmod((k0 + ka)/2, 1)
    sto_frac = k_residual - (1 if k_residual > 0.5 else 0)
    return sto_frac