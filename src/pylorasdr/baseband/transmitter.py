import numpy as np
from pylorasdr.phy import modulate
from pylorasdr.gray import gray2nongray
from pylorasdr.interleaving import interleave
from pylorasdr.hamming import encode as hamming_enc
from pylorasdr.whitening import whiten
from pylorasdr.misc import ccit16, dec2bin, bin2dec
from pylorasdr.misc import LDROMode

class Transmitter:
    def __init__(self, bw:np.uint32, sf:np.uint16, os_factor:np.uint8,
                 preamble_len:np.uint16, netid:np.uint8, cr:np.uint8,
                 implicit_hdr:bool = False, has_crc:bool = True):
        """
        Initialize the Transmitter.

        Parameters
        ----------
        bw : np.uint32
            Signal bandwidth.

        sf : np.uint16
            Spreading factor.

        os_factor : np.uint8
            Oversampling factor.

        preamble_len : np.uint16
            Number of preamble symbols.

        netid : np.uint8
            Network identifier used to derive the synchronization word.

        cr : np.uint8
            Coding rate.

        implicit_hdr : bool, optional
            If True, uses implicit header mode. Default is False.

        has_crc : bool, optional
            If True, enables CRC in the payload. Default is True.
        """
        self.bw = bw
        self.sf = sf
        self.os_factor = os_factor
        self.preamble_len = preamble_len
        self.sync_word = [(netid & 0xF0) >> 1, (netid & 0x0F) << 3]
        self.cr = cr
        self.implicit_hdr = implicit_hdr
        self.has_crc = has_crc

        self.number_of_bins = 2**self.sf
        self.ldro_mode = LDROMode.Auto              # ldro auto by default

    def set_bw(self, bw):
        """
        Set the signal bandwidth.

        Parameters
        ----------
        bw : int
            New bandwidth value.
        """
        self.bw = bw

    def set_sf(self, sf):
        """
        Set the spreading factor.

        Parameters
        ----------
        sf : int
            New spreading factor.
        """
        self.sf = sf
        self.number_of_bins = 2**self.sf

    def set_os_factor(self, os_factor):
        """
        Set the oversampling factor.

        Parameters
        ----------
        os_factor : int
            Oversampling factor.
        """
        self.os_factor = os_factor
    
    def set_preamble_length(self, preamble_len):
        """
        Set the preamble length.

        Parameters
        ----------
        preamble_len : int
            Number of preamble symbols.
        """
        self.preamble_len = preamble_len

    
    def set_netid(self, netid):
        """
        Set the network identifier and update the synchronization word.

        Parameters
        ----------
        netid : int
            Network identifier.
        """
        self.sync_word = [(netid & 0xF0) >> 1, (netid & 0x0F) << 3]

    def set_cr(self, cr):
        """
        Set the coding rate.

        Parameters
        ----------
        cr : int
            Coding rate.
        """
        self.cr = cr

    def set_implicit_header(self, implicit_hdr:bool = True):
        """
        Enable or disable implicit header mode.

        Parameters
        ----------
        implicit_hdr : bool, optional
            If True, enables implicit header mode. Default is True.
        """
        self.implicit_hdr = implicit_hdr
    
    def enable_crc(self):
        """
        Enable payload CRC.
        """
        self.has_crc = True
    
    def disable_crc(self):
        """
        Disable CRC in the payload.
        """
        self.has_crc = False

    def enable_ldro(self):
        """
        Enable Low Data Rate Optimization (LDRO).
        """
        self.ldro_mode = LDROMode.Enabled

    def disable_ldro(self):
        """
        Disable Low Data Rate Optimization (LDRO).
        """
        self.ldro_mode = LDROMode.Disabled

    def set_ldro_auto(self):
        """
        Set Low Data Rate Optimization (LDRO) mode to automatic.
        """
        self.ldro_mode = LDROMode.Auto
    
    def build_frame(self, payload):
        """
        Build LoRa frame and generate IQ samples from an input payload.

        Parameters
        ----------
        payload : np.ndarray
            Input payload as a sequence of bytes.

        Returns
        -------
        np.ndarray
            Complex IQ samples representing the modulated frame.
        """
        payload_len = len(payload)
        if payload_len > 255:
            print(f"{payload_len} bytes provided. Only the first 255 are processed")
            payload_len = 255
        
        m_payload = payload[:payload_len]
        payload_white = whiten(m_payload)

        # compute and append crc if configured
        if self.has_crc:
            payload_crc = np.bitwise_xor(ccit16(m_payload[:-2]), (m_payload[-2]<<8) + m_payload[-1])                  # 16 bit int
            payload_crc = np.uint8([payload_crc & 0x00FF, payload_crc >> 8])                                              # 2 bytes
            payload_white = np.append(payload_white, payload_crc)
            
        frm_bits = dec2bin(payload_white, 8, 'right-msb')
        frm_nib = np.reshape(frm_bits, (-1, 4), order='C')                                      # 2*payload_anc_crc_size nibbles

        # add header if explicit header mode
        if not self.implicit_hdr:
            hdr_nib = np.zeros((5, 4), dtype=np.uint8)       # 5 header nibbles: dtype is actually bool
            hdr_pl_len = np.squeeze(dec2bin(payload_len, 8, 'right-msb'))
            

            hdr_nib[0, :] = hdr_pl_len[4:]
            hdr_nib[1, :] = hdr_pl_len[0:4]
            hdr_nib[2, :] = dec2bin(self.has_crc | self.cr << 1, 4, 'right-msb')                # (has_crc, cr0, cr1, cr2) lsb first

            # -- compute and append header crc
            c4 = hdr_nib[0, 3] ^ hdr_nib[0, 2] ^ hdr_nib[0, 1] ^ hdr_nib[0, 0]
            c3 = hdr_nib[0, 3] ^ hdr_nib[1, 3] ^ hdr_nib[1, 2] ^ hdr_nib[1, 1] ^ hdr_nib[2, 0]
            c2 = hdr_nib[0, 2] ^ hdr_nib[1, 3] ^ hdr_nib[1, 0] ^ hdr_nib[2, 3] ^ hdr_nib[2, 1]
            c1 = hdr_nib[0, 1] ^ hdr_nib[1, 2] ^ hdr_nib[1, 0] ^ hdr_nib[2, 2] ^ hdr_nib[2, 1] ^ hdr_nib[2, 0]
            c0 = hdr_nib[0, 0] ^ hdr_nib[1, 1] ^ hdr_nib[2, 3] ^ hdr_nib[2, 2] ^ hdr_nib[2, 1] ^ hdr_nib[2, 0]
            hdr_nib[3, 0] = c4
            hdr_nib[4, :] = [c0, c1, c2, c3]

            frm_nib = np.append(hdr_nib, frm_nib, axis=0)
        
        # create 'header' and payload blocks: each is encoded separately
        hdr_blk_nib = frm_nib[:self.sf - 2, :]                                                 # 'header' block, encoded in reduced rate mode
        pay_blk_nib = frm_nib[self.sf - 2:, :]                                                 # payload block, encoded with configured cr             
        
        # Hamming encoding
        hdr_blk_cw = hamming_enc(hdr_blk_nib, 4)
        pay_blk_cw = hamming_enc(pay_blk_nib, self.cr)

        # interleaving
        hdr_blk_symb = interleave(hdr_blk_cw, 4, self.sf, True)                                # cr 4 and reduced rate

        # in auto mode, enable LDRO if symbol duration > 16 ms
        if self.ldro_mode == LDROMode.Auto:
            symb_duration_ms = 1e3*self.number_of_bins/self.bw            # symbol duration (ms)
            ldro = symb_duration_ms > 16
        else:
            ldro = LDROMode(self.ldro_mode)

        pay_blk_symb = interleave(pay_blk_cw, self.cr, self.sf, ldro)                          # padding is handled by the interleaver

        frm_symb = np.append(hdr_blk_symb, pay_blk_symb, axis=0)
        
        frm_symb_bytes = bin2dec(frm_symb, 'left-msb')
        
        tx_symb = gray2nongray(frm_symb_bytes, self.sf)
        tx_symb = np.mod(tx_symb + 1, self.number_of_bins)

        # chirp modulation
        upchirps = np.tile(modulate(self.sf, 0, self.os_factor), (self.preamble_len, 1))
        downchirps = np.tile(modulate(self.sf, 0, self.os_factor, False), (2, 1))
        downchirps = np.append(downchirps, downchirps[:self.os_factor*self.number_of_bins//4])
        sw_samples = modulate(self.sf, self.sync_word, self.os_factor)                      # sync_word samples
        sw_samples = np.reshape(sw_samples, (-1, 1), order='F')
        pl_samples = modulate(self.sf, tx_symb, self.os_factor)
        pl_samples = np.reshape(pl_samples, (-1, 1), order='F')

        # frame assembly
        frm_samples = np.append(upchirps, sw_samples)
        frm_samples = np.append(frm_samples, downchirps)
        frm_samples = np.append(frm_samples, pl_samples)
        return frm_samples