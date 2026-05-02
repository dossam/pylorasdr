import numpy as np

def fft_mag(samples: np.ndarray, ref_chirp: np.ndarray) -> np.ndarray:
    """
    Dechirp, FFT, and compute the FFT magnitude of input LoRa samples.

    Parameters
    ----------
    samples : np.ndarray
        Input complex baseband samples. The total length must be a multiple
        of the reference chirp length.

    ref_chirp : np.ndarray
        Reference chirp used for dechirping. Can be either an upchirp or a
        downchirp. Its length defines the FFT size.

    Returns
    -------
    np.ndarray
        2D array containing the FFT magnitudes with shape
        (len(ref_chirp), n_symbols), where each column corresponds to one symbol.

    Raises
    ------
    ValueError
        Intended to be raised if:
        - The length of `samples` is not a multiple of `ref_chirp.size`.
        - `ref_chirp` is empty.
        (Not currently enforced in the implementation.)

    Notes
    -----
    TODO:
    - Add input validation for shape consistency.
    - Validate dtype (e.g., complex64 or complex128).
    """
     
    samples_tmp = np.reshape(samples, (ref_chirp.size, -1), order='F')
    return np.abs(np.fft.fft(samples_tmp * np.reshape(ref_chirp, (ref_chirp.size, 1)), axis=0))