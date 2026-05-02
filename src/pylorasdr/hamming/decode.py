import numpy as np
from pylorasdr.misc import H_1, H_234, CB_1, CB_234

def decode(cw_in: np.ndarray, cr: int, soft_decoding: bool = False) -> np.ndarray:
    """
    Decode LoRa-like codewords.

    This function decodes a sequence of received codewords into a sequence of information nibbles bits.

    Parameters
    ----------
    cw_in : np.ndarray
        Input codewords. Format depends on decoding mode:
        - Hard decoding: binary codewords
        - Soft decoding: bit LLRs

    cr : int
        Coding rate selector defining the codeword structure and redundancy.

    soft_decoding : bool, optional
        If True, performs soft-decision decoding.
        If False, performs syndrome-based hard-decision decoding.
        Default is False.

    Returns
    -------
    np.ndarray
        Decoded information bits (binary nibbles).

    Raises
    ------
    ValueError
        If an unsupported coding rate is provided or if the decoding
        configuration is invalid (not currently enforced in implementation).
    """
    cw_in = np.atleast_2d(cw_in)
    n_cw = np.size(cw_in, 0)
    cw_len = 4 + cr

    if not soft_decoding:
        # Hard-decoding
        if cr == 1 or cr == 2:
            return cw_in[:, 0:4]
        
        if cr == 3 or cr == 4:
            H = H_234

            # -- compute syndrome vector
            s = np.transpose(
                np.mod(
                    H[0:cr, 0:cw_len] @ np.transpose(cw_in),
                    2
                )
            )
        else:
            print(f'Codeword {cr} not supported')    
            return
            ## TODO Implement exception for unsupported code rates
        
        H_tmp = np.tile(H[0:cr, 0:cw_len, None], (1, 1, n_cw))
        s_tmp = np.reshape(s.transpose(), (cr, 1, n_cw))

        pos_mat = np.squeeze(np.logical_not(np.any(H_tmp ^ s_tmp, 0)))
        cw_tmp = np.transpose(cw_in).copy()

        if cr == 4:
            flip = np.mod(np.sum(cw_in, 1), 2)
            pos_mat[:, flip == 0] = 0
        
        # -- flip detected erroneous bits
        cw_tmp[pos_mat == 1] = cw_tmp[pos_mat == 1] == 0

        return np.transpose(cw_tmp)[:, 0:4]

    else:
        # Soft-decoding
        if cr == 1:
            CB = CB_1
        else:
            CB = CB_234
        
        CB_map = CB * 2 - 1
        cw_tmp = np.zeros(cw_in.shape, dtype=np.int8)
        n_cw = np.size(cw_in, 0)

        for cw_idx in range(n_cw):
            P_max_idx = np.argmax(
                np.dot(
                    cw_in[cw_idx, :],
                    np.transpose(CB_map[:, 0:cw_len])
                )
            )
            cw_tmp[cw_idx, :] = CB[P_max_idx, 0:cw_len]
        
        return cw_tmp[:, 0:4]