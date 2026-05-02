import numpy as np

def deinterleave(
    symb_bin_seq: np.ndarray,
    cr: np.uint8,
    sf: np.uint16,
    soft_decoding: bool = False,
    ldro: bool = False
) -> np.ndarray:
    """
    Deinterleave a sequence of LoRa symbols into codewords.

    This function reverses the interleaving process by reconstructing the
    original codeword sequence from diagonnaly-interleaved symbols.

    Parameters
    ----------
    symb_bin_seq : np.ndarray
        Input symbol matrix, where each row is a binary vector representing
        one demodulated symbol.

    cr : np.uint8
        Coding rate parameter defining redundancy structure.

    sf : np.uint16
        Spreading factor defining symbol size (number of bits per symbol).

    soft_decoding : bool, optional
        If True, uses LLR bits values.
        If False, uses integer binary values. Default is False.

    ldro : bool, optional
        Low Data Rate Optimization flag affecting deinterleaving structure.
        Default is False.

    Returns
    -------
    np.ndarray
        Deinterleaved codeword matrix of shape (n_codewords, bits_per_codeword),
        with dtype depending on `soft_decoding`.
    """
    n_symb = symb_bin_seq.shape[0]

    # number of symbols and codewords per block
    symb_per_blk = 4 + cr
    cw_per_blk = sf - 2 * ldro

    # number of interleaving blocks
    n_blk = int(np.ceil(n_symb / symb_per_blk))

    # -- pad the symbol sequence
    if n_symb < symb_per_blk * n_blk:
        pad_len = symb_per_blk * n_blk - n_symb
        m_symb_bin_seq = np.pad(
            symb_bin_seq,
            ((0, pad_len), (0, 0)),
            mode='constant',
            constant_values=0
        )
    else:
        m_symb_bin_seq = symb_bin_seq

    if soft_decoding:
        m_dtype = np.float32
    else:
        m_dtype = int

    out_blk = np.zeros([cw_per_blk, symb_per_blk], dtype=m_dtype)
    cw_seq = np.zeros([n_blk * cw_per_blk, symb_per_blk], dtype=m_dtype)

    for blk_idx in range(n_blk):
        in_blk = m_symb_bin_seq[
            blk_idx * symb_per_blk:(blk_idx + 1) * symb_per_blk, :
        ]
        for i in range(symb_per_blk):
            for j in range(cw_per_blk):
                out_blk[(i - j - 1) % cw_per_blk, i] = in_blk[i, j]

        cw_seq[
            blk_idx * cw_per_blk:(blk_idx + 1) * cw_per_blk, :
        ] = out_blk

    return cw_seq