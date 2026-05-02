import numpy as np

def demodulate(
    samples: np.ndarray,
    ref_chirp: np.ndarray,
    sf: int,
    soft_decoding: bool = False,
    noise_sigma_sq: float = 1,
    gray_mat: np.ndarray = None
) -> np.ndarray:
    """
    Demodulate LoRa symbols from received samples using a reference chirp.

    This function performs LoRa demodulation as dechirp -> FFT -> argmax.

    Parameters
    ----------
    samples : np.ndarray
        Input complex baseband samples. Expected to be a one-dimensional array
        whose length is a multiple of 2**sf.

    ref_chirp : np.ndarray
        Reference chirp used for dechirping. Must have length 2**sf.

    sf : int
        Spreading factor. Determines the FFT size (number_of_bins = 2**sf).

    soft_decoding : bool, optional
        If True, computes log-likelihood ratios (LLRs) for each bit.
        If False, performs hard symbol detection. Default is False.

    noise_sigma_sq : float, optional
        Noise variance per dimension. Used only for soft decoding.
        Default is 1.

    gray_mat : np.ndarray, optional
        Gray mapping matrix of shape (number_of_bins, sf), where
        number_of_bins = 2**sf. Required if `soft_decoding=True`.

    Returns
    -------
    np.ndarray
        - If `soft_decoding=False`: 1D array of detected symbol indices
          with shape (n_symbols,).
        - If `soft_decoding=True`: 2D array of LLR values with shape
          (n_symbols, sf), where each row corresponds to one symbol.

    Raises
    ------
    ValueError
        Intended to be raised if:
        - `gray_mat` is not provided when `soft_decoding=True`.
        - Input dimensions are inconsistent (e.g., sample length not divisible by 2**sf).
        - `ref_chirp` length does not match 2**sf.
        (Not currently enforced in the implementation.)

    Notes
    -----
    TODO:
    - Replace the print statement with a proper exception when `gray_mat` is missing.
    - Validate input shapes and consistency between parameters.
    - Clarify and enforce expected dtype (e.g., complex64 for samples).
    """

    if gray_mat is None and soft_decoding:
        # TODO: raise an exception here
        print("gray_mat is required for soft-decoding!")
        return
    
    number_of_bins = 2**sf

    # dechirp and fft
    dechirped = np.reshape(samples, (number_of_bins, -1), order='F')  * np.reshape(ref_chirp, (number_of_bins, 1))
    spectr = np.fft.fft(dechirped, axis=0)

    if soft_decoding:
        mags = np.abs(spectr)/np.sqrt(number_of_bins)
        v = np.max(mags, axis=0)
        Lambda_comb = v * mags / noise_sigma_sq - np.log(mags + 1e-10) / 2
        Lambda_comb_rep = np.tile(Lambda_comb[:, :, np.newaxis], (1, 1, sf))
        gray_rep = np.tile(gray_mat[:, :, np.newaxis], (1, 1, Lambda_comb.shape[1]))
        gray_rep_join = np.transpose(gray_rep, (0, 2, 1))

        Lambda_flat = np.reshape(Lambda_comb_rep, (-1, 1), order='F')
        gray_flat = np.reshape(gray_rep_join, (-1, 1), order='F')

        lambda_1 = Lambda_flat[gray_flat > 0]
        lambda_1 = lambda_1.reshape((number_of_bins // 2, -1, sf), order='F')
        num = np.max(lambda_1, axis=0)

        lambda_0 = Lambda_flat[gray_flat < 1]
        lambda_0 = lambda_0.reshape((number_of_bins // 2, -1, sf), order='F')
        den = np.max(lambda_0, axis=0)

        llr = num - den
        
        return np.squeeze(llr)

    else:
        # hard-decoding
        return np.squeeze(np.argmax(np.abs(spectr), axis=0))