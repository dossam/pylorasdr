import numpy as np

def interleave(cw_seq: np.ndarray, cr: np.uint8, sf: np.uint16, ldro: bool = False) -> np.ndarray:
    """
    Interleave a sequence of codewords into LoRa symbols.

    This function rearranges input codewords into diagonally-interleaved symbol blocks.

    Parameters
    ----------
    cw_seq : np.ndarray
        Input sequence of codewords (binary rows).

    cr : np.uint8
        Coding rate parameter defining redundancy structure.

    sf : np.uint16
        Spreading factor defining symbol size.

    ldro : bool, optional
        Low Data Rate Optimization flag affecting interleaving structure.
        Default is False.

    Returns
    -------
    np.ndarray
        Interleaved binary symbol matrix of shape (n_symbols, sf), where each
        row corresponds to one interleaved symbol.
    """
    # number of symbols and codewords per block
    n_symb_per_blk = 4 + cr
    n_cw_per_blk = sf - 2 * ldro

    # number of interleaving blocks
    n_blk = int(np.ceil(cw_seq.shape[0] / (n_cw_per_blk)))

    # Convert codewords to binary rep. and perform interleaving
    # -- pad input sequence to match block size if necessary
    if cw_seq.shape[0] < n_cw_per_blk * n_blk:
        m_seq_bin = np.pad(
            cw_seq,
            ((0, n_cw_per_blk * n_blk - cw_seq.shape[0]), (0, 0)),
            mode='constant'
        )
    else:
        m_seq_bin = cw_seq

    out_blk = np.zeros([n_symb_per_blk, n_cw_per_blk])
    symb_seq_bin = np.zeros([n_blk * (n_symb_per_blk), sf])

    for blk_idx in range(n_blk):
        in_blk = m_seq_bin[blk_idx * n_cw_per_blk:(blk_idx + 1) * n_cw_per_blk, :]
        for i in range(n_symb_per_blk):
            for j in range(n_cw_per_blk):
                out_blk[i, j] = in_blk[(i - (j + 1)) % n_cw_per_blk, i]

        symb_seq_bin[
            blk_idx * n_symb_per_blk:(blk_idx + 1) * n_symb_per_blk,
            0:n_cw_per_blk
        ] = out_blk

    return symb_seq_bin