import numpy as np
import time
import traceback
from pylorasdr.phy import modulate, demodulate
from pylorasdr.misc import CircBuffer
from pylorasdr.misc import ReceiverState, SyncState, LDROMode
from pylorasdr.misc import bin2dec, ccit16
from pylorasdr.gray import nongray2gray
from pylorasdr.interleaving import deinterleave
from pylorasdr.hamming import decode as hamming_dec
from pylorasdr.whitening import whiten
from pylorasdr.phy import estimate_sto_frac
from pylorasdr.phy import estimate_cfo_frac

class Receiver:
    def __init__(self, center_freq, bw, sf, os_factor, netid, preamble_len, soft_decoding = False):
        """
        Initialize the Receiver for a LoRa-like digital RF transceiver.

        Parameters
        ----------
        center_freq : float
            Center frequency of the received signal (Hz).

        bw : float
            Signal bandwidth (Hz).

        sf : int
            Spreading factor. Determines symbol resolution (2^sf bins)
            and symbol duration.

        os_factor : int
            Oversampling factor (samples per chip).

        netid : int
            Network identifier used to derive the synchronization word.

        preamble_len : int
            Number of preamble symbols.

        soft_decoding : bool, optional
            If True, enables soft-decision decoding (LLR-based).
            Default is False.

        Notes
        -----
        - The receiver operates in explicit header mode by default.
        - Synchronization (timing and frequency offsets) is estimated
        during preamble and sync processing.
        - Internal buffers and state variables are initialized for
        frame processing and updated dynamically during runtime.
        """
        self.center_freq = center_freq
        self.bw = bw
        self.sf = sf
        self.os_factor = os_factor
        self._sync_word = [(netid & 0xF0) >> 1, (netid & 0x0F) << 3]
        self.preamble_len = preamble_len
        self.soft_decoding = soft_decoding

        self.number_of_bins = 2**self.sf
        self.samples_per_symbol = self.number_of_bins*os_factor
        self.samp_rate = bw*os_factor

        self.implicit_hdr = False           # explicit header mode by default        
        self.ldro_mode = LDROMode.Auto  # ldro auto by default
        self._ldro = False               # controls the use of ldro for payload symbol decoding: placeholder; set during frame_decode
        self._buffer = None
        self.one_shot = False
        self.on_success = None              # function to call on successful decoding: only if one_shot = False 

        self._n_up_count = 0                  # number of consecutive upchirps detected 
        self._ref_symb = 0                    # reference upchirp value: placeholder; real value defined by _detect_preamble
        self._to_consume = 0                  # number of samples to consume after a processing round: placeholder; real value defined by processing functions

        # estimated frame offsets, and correction vectors
        self._cfo_frac_est = 0
        self._sto_frac_est = 0
        self._cfo_int_est = 0
        self._sto_int_est = 0
        self._cfo_frac_corr_vect = None      # placeholder: initialized after cfo estimation
        self._cfo_int_corr_vect = None

        self._rx_state = ReceiverState.PREAMB_DET    # receiver state
        self._sync_state = SyncState.COARSE  # synchronization state
        self._symb_count = 0                 # number of frame symbols processed
        self._extra_up = 0                   # number of extra upchirps after preamble detection
        self._down_val = 0                   # demodulated downchirp value (during sync)

        self._noise_sigma_sq = 0             # noise variance, used in soft-decoding (estimated during sync)

        # header symbols array
        if self.soft_decoding:
            self._hdr_symbols = np.zeros((8, self.sf), dtype=np.float32)         # 8*self.sf header LLR bits
        else:
            self._hdr_symbols = np.zeros((8, 1), dtype=np.int16)                 # 8 header symbols

        self._hdr_nib = np.ndarray                                           # header nibbles: placeholder; initialized during header decoding
        self._pl_symbols = np.ndarray                                            # payload symbols: placeholder; initialized after header decoding

        # header fields: placeholder; computed during header decoding
        self._rx_pl_len = 0
        self._rx_has_crc = True
        self._rx_cr = 1

        self._n_symb = 0             # number of symbols (header + payload): placeholder; computed after header decoding
        self._symb_duration_ms = 1e3*self.number_of_bins/bw            # symbol duration (ms)

        # compute gray encoding matrix
        self._gray_mat = np.zeros((self.number_of_bins, self.sf), dtype=np.int8)
        for symbol in range(self.number_of_bins):
            self._gray_mat[symbol, :] = nongray2gray((symbol - 1)%self.number_of_bins, self.sf)

        self._ref_downchirp = modulate(self.sf, 0, 1, upchirp=False)
        self._ref_upchirp = np.conjugate(self._ref_downchirp)
        self._n_up_req = self.preamble_len - 3   # number of consecutive upchirps for detection
        self._n_up_to_use = self._n_up_req - 1    # number of preamble upchirp usable after coarse sync
        self._prb_raw = np.zeros(self._n_up_req*self.number_of_bins, dtype=np.complex64)      # preamble upchirps samples
        self._prb_raw_sync = None                # placeholder for synchronized preamble upchirps samples: allocated during SYNC
        self._prb_raw_up = np.zeros((self.preamble_len)*self.samples_per_symbol + self.os_factor, dtype=np.complex64)   # upsampled preamble upchirps samples (+os_factor samples as a downsampling safeguard negative sto_frac)
        self._up_val = np.zeros(self._n_up_req, dtype=np.uint16)                                # consecutive preamble upchirps values
        self._extra_symbol_samp = np.zeros(2*self.samples_per_symbol, dtype=np.complex64)      # holds 2 symbols samples from the second downchirp
        self._netid_samp = np.zeros(np.int32(2.5*self.samples_per_symbol), dtype=np.complex64) # netid samples

        self._frame_info = {}                    # holds info about the frame and status: initialized on run

    def set_implicit_header(self, implicit_hdr = True, cr = 1, payload_len = 0, crc_enable = True):
        """
        Configure the header mode of the receiver.

        In implicit header mode, payload parameters are fixed and must be
        provided (coding rate, payload length, and CRC presence).
        In explicit mode, these parameters are extracted from the received header.

        Parameters
        ----------
        implicit_hdr : bool, optional
            If True, enables implicit header mode. If False, explicit header
            mode is used. Default is True.

        cr : int, optional
            Coding rate to use in implicit mode. Ignored in explicit mode.
            Default is 1.

        payload_len : int, optional
            Expected payload length in bytes for implicit mode.
            Ignored in explicit mode. Default is 0.

        crc_enable : bool, optional
            Whether CRC is enabled for the payload in implicit mode.
            Ignored in explicit mode. Default is True.
        """
        self.implicit_hdr = implicit_hdr
        self.cr = cr
        self.payload_len = payload_len
        self.crc_enable = crc_enable
    
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

    def set_buffer(self, buffer:CircBuffer):
        """
        Attach an input circular buffer to the receiver.

        The provided buffer is used as the source of incoming complex
        samples for frame detection and demodulation.

        Parameters
        ----------
        buffer : CircBuffer
            Circular buffer containing received IQ samples.
        """
        self._buffer = buffer

    def set_sf(self, sf):
        """
        Update the spreading factor (SF) and reinitialize dependent receiver parameters.

        Parameters
        ----------
        sf : int
            New spreading factor. Determines number of frequency bins (2^sf)
            and symbol resolution.
        """
        self.sf = sf

        # re-allocate buffers and ref. chirps
        self._ref_downchirp = modulate(self.sf, 0, 1, upchirp=False)
        self._ref_upchirp = np.conjugate(self._ref_downchirp)
        self._n_up_req = self.preamble_len - 3   # number of consecutive upchirps for detection
        self._n_up_to_use = self._n_up_req - 1    # number of preamble upchirp usable after coarse sync
        self._prb_raw = np.zeros(self._n_up_req*self.number_of_bins, dtype=np.complex64)      # preamble upchirps samples
        self._prb_raw_sync = None                # placeholder for synchronized preamble upchirps samples: allocated during SYNC
        self._prb_raw_up = np.zeros((self.preamble_len)*self.samples_per_symbol + self.os_factor, dtype=np.complex64)   # upsampled preamble upchirps samples
        self._extra_symbol_samp = np.zeros(2*self.samples_per_symbol, dtype=np.complex64)      # holds 2 symbols samples from the second downchirp
        self._netid_samp = np.zeros(np.int32(2.5*self.samples_per_symbol), dtype=np.complex64) # netid samples

        # recompute gray encoding matrix and resize hdr symbols array
        self._gray_mat = np.zeros((self.number_of_bins, self.sf), dtype=np.int8)
        for symbol in range(self.number_of_bins):
            self._gray_mat[symbol, :] = nongray2gray((symbol - 1)%self.number_of_bins, self.sf)
            
        if self.soft_decoding:
            self._hdr_symbols = np.zeros((8, self.sf), dtype=np.float32)         # 8*self.sf header LLR bits
        else:
            self._hdr_symbols = np.zeros((8, 1), dtype=np.int16)                 # 8 header symbols
    
    def set_soft_decoding(self, soft_decoding = True):
        """
        Enable or disable soft-decision decoding for header processing.

        Parameters
        ----------
        soft_decoding : bool, optional
            Default is True.
        """
        self.soft_decoding = soft_decoding

        if self.soft_decoding:
            self._hdr_symbols = np.zeros((8, self.sf), dtype=np.float32)         # 8*sf header LLR bits
        else:
            self._hdr_symbols = np.zeros((8, 1), dtype=np.uint16)                 # 8 header symbols

    def set_one_shot(self, one_shot: bool):
        """
        Enable or disable one-shot processing mode.

        In one-shot mode, the receiver processes a single buffer or frame
        and then exits/returns once processing is complete. This includes
        termination on successful decoding, buffer exhaustion, or upon
        encountering a processing exception.

        Parameters
        ----------
        one_shot : bool
            If True, enables one-shot mode (process one frame and stop): convenient for simulation.
            If False, the receiver runs in continuous streaming mode.
        """
        self.one_shot = one_shot 
    
    # Register a callback for successful frames (in continuous stream mode)
    def register_frame_handler(self, handler):
        """
        Register a callback function to be invoked upon successful frame decoding.

        In continuous streaming mode, this handler is called whenever the receiver
        successfully decodes a complete frame. The callback receives a dictionary
        containing all relevant frame information.

        Parameters
        ----------
        handler : callable
            Function to be executed on successful frame decoding. It must accept
            a single argument:

            frame_info : dict
                Dictionary containing decoded frame information, including:
                - payload : decoded payload data
                - cr : coding rate
                - snr: the signal-to-noise ratio
        """
        self.on_success = handler

    # Reset the receiver state back to preamble detection and enable buffer overwrite
    def _reset_rx(self):
        """
        Reset the receiver state machine to the initial preamble detection state.
        """
        self._rx_state = ReceiverState.PREAMB_DET
        self._sync_state = SyncState.COARSE
        self._n_up_count = 0
        self._symb_count = 0
        self._buffer.enable_overwrite()
        
    def _detect_preamble(self, samples: np.ndarray) -> None:
        """
        Detect the presence of a preamble in the incoming signal samples.

        This method processes one full symbol's worth of samples at a time and
        updates the internal receiver state machine.

        Parameters
        ----------
        samples : np.ndarray
            Complex IQ samples corresponding to a single symbol duration and a margin.
        """
        symb = demodulate(samples[self.os_factor//2:self.os_factor//2 + self.samples_per_symbol:self.os_factor], self._ref_downchirp, self.sf, False)
        self._to_consume = self.samples_per_symbol

        # look for n_up_req consecutive upchirps within a +/1 range
        if not self._n_up_count:
            # -- very first symbol
            self._ref_symb = symb
            self._up_val[0] = symb
            self._prb_raw_up[0:self.samples_per_symbol] = samples[:self.samples_per_symbol]
            self._prb_raw[0:self.number_of_bins] = samples[self.os_factor//2:self.os_factor//2 + self.samples_per_symbol:self.os_factor]
            self._n_up_count += 1
        else:
            if symb >= self._ref_symb - 1 and symb <= self._ref_symb + 1:
                self._prb_raw_up[self._n_up_count*self.samples_per_symbol:(self._n_up_count + 1)*self.samples_per_symbol] = samples[:self.samples_per_symbol]
                self._prb_raw[self._n_up_count*self.number_of_bins: (self._n_up_count + 1)*self.number_of_bins] = samples[self.os_factor//2:self.os_factor//2 + self.samples_per_symbol:self.os_factor]
                self._up_val[self._n_up_count] = symb
                self._ref_symb = symb
                self._n_up_count += 1
            else:
                self._prb_raw_up[0:self.samples_per_symbol] = samples[:self.samples_per_symbol]
                self._prb_raw[0:self.number_of_bins] = samples[self.os_factor//2:self.os_factor//2 + self.samples_per_symbol:self.os_factor]
                self._ref_symb = symb
                self._up_val[0] = symb
                self._n_up_count = 1
                
        # preamble detected                    
        if self._n_up_count == self._n_up_req:
            # -- copy last +os_factor samples as safeguard (picture an sto compensation situation with sto_frac*os_factor<-0.5)
            self._prb_raw_up[self._n_up_req*self.samples_per_symbol: self._n_up_req*self.samples_per_symbol + self.os_factor] = samples[self.samples_per_symbol: self.samples_per_symbol + self.os_factor]
            
            # -- update receiver state and disbale buffer overwrite
            self._rx_state = ReceiverState.FRAME_SYNC
            self._buffer.disable_overwrite()
        
    def _sync_frame(self, samples: np.ndarray) -> None:
        """
        Perform frame synchronization using incoming symbol samples.

        This method refines timing and frequency alignment, validates the network
        identifier (netID), and updates internal receiver state.

        Parameters
        ----------
        samples : np.ndarray
            Complex IQ samples corresponding to a single symbol duration
            (samples_per_symbol + os_factor).
        """
        # coarse sync, with cfo_frac and sto_frac estimation
        if self._sync_state == SyncState.COARSE:
            values_, counts = np.unique(self._up_val, return_counts=True)
            k_hat = values_[np.argmax(counts)]

            # perform coarse sync: only n_up_req - 1 full symbols remaining after that
            coarse_shift = int(self.number_of_bins - k_hat)
            self._prb_raw_sync = np.roll(self._prb_raw, -coarse_shift)
            self._prb_raw_up[: self._n_up_req*self.samples_per_symbol - self.os_factor*coarse_shift] = self._prb_raw_up[self.os_factor*coarse_shift: self._n_up_req*self.samples_per_symbol]
            
            # estimate cfo_frac
            self._cfo_frac_est = estimate_cfo_frac(self._prb_raw_sync[0: (self._n_up_req - 1)*self.number_of_bins], self._ref_downchirp, self.sf)
            n = np.arange(0, (self._n_up_req)*self.number_of_bins)
            self._cfo_frac_corr_vect = np.exp(-2j*np.pi*self._cfo_frac_est*n/self.number_of_bins)

            # compensate for cfo_frac in the preamble upchirp and estimate sto_frac
            self._prb_raw_sync = self._prb_raw_sync * self._cfo_frac_corr_vect
            self._sto_frac_est = estimate_sto_frac(self._prb_raw_sync[0:(self._n_up_req - 1)*self.number_of_bins], self.sf, self._ref_downchirp)

            self._sync_state = SyncState.NETID1
            self._to_consume = (self.number_of_bins - k_hat)*self.os_factor
            return
        
        samples_dwn = samples[self.os_factor//2 - np.int32(np.round(self.os_factor*self._sto_frac_est)):self.os_factor//2 - np.int32(np.round(self.os_factor*self._sto_frac_est)) + self.samples_per_symbol:self.os_factor]
        symb = demodulate(samples_dwn, self._ref_downchirp, self.sf, False)
        
        # 'samples_per_symbol' samples consumed by default in all states below, except in QUARTER_DOWN if matching netids
        self._to_consume = self.samples_per_symbol

        # expecting netid1 samples
        if self._sync_state == SyncState.NETID1:
            if symb == 0 or symb == 1 or symb == self.number_of_bins - 1:
                self._netid_samp[0:np.int32(0.25*self.samples_per_symbol)] = samples[np.int32(0.75*self.samples_per_symbol):self.samples_per_symbol]

                # extra upchirps: symbol index are offset by -1 because of coarse sync            
                if self._extra_up >= 3:
                    self._prb_raw_up[: (self._n_up_req + 1)*self.samples_per_symbol] = self._prb_raw_up[self.samples_per_symbol: (self._n_up_req + 2)*self.samples_per_symbol]
                    self._prb_raw_up[(self._n_up_req + 1)*self.samples_per_symbol: (self._n_up_req + 2)*self.samples_per_symbol + self.os_factor] = samples[:self.samples_per_symbol + self.os_factor]
                else:
                    self._prb_raw_up[(self._n_up_req + self._extra_up - 1)*self.samples_per_symbol: (self._n_up_req + self._extra_up)*self.samples_per_symbol + self.os_factor] = samples[:self.samples_per_symbol + self.os_factor]
                    self._extra_up += 1
            else:
                # netid 1
                self._netid_samp[np.int32(0.25*self.samples_per_symbol):np.int32(1.25*self.samples_per_symbol)] = samples[0:self.samples_per_symbol]
                self._sync_state = SyncState.NETID2
            
            return
        
        # expecting netid2 samples
        if self._sync_state == SyncState.NETID2:
            self._netid_samp[np.int32(1.25*self.samples_per_symbol):np.int32(2.25*self.samples_per_symbol)] = samples[0:self.samples_per_symbol]
            self._sync_state = SyncState.DOWNCHIRP1
            return
        
        # expecting first downchirp samples
        if self._sync_state == SyncState.DOWNCHIRP1:
            self._netid_samp[np.int32(2.25*self.samples_per_symbol):np.int32(2.5*self.samples_per_symbol)] = samples[0:np.int32(0.25*self.samples_per_symbol)]
            self._sync_state = SyncState.DOWNCHIRP2
            return
        
        # expecting second downchirp samples
        if self._sync_state == SyncState.DOWNCHIRP2:
            self._down_val = demodulate(samples_dwn, self._ref_upchirp, self.sf, False)
            self._extra_symbol_samp[0:self.samples_per_symbol] = samples[0:self.number_of_bins*self.os_factor]
            self._sync_state = SyncState.QUARTER_DOWN
            return
        
        # expecting quarter downchirp samples: estimate cfo_int, sto_int, final synchronization and check netids
        if self._sync_state == SyncState.QUARTER_DOWN:
            self._extra_symbol_samp[self.samples_per_symbol:2*self.samples_per_symbol] = samples[0:self.number_of_bins*self.os_factor]

            if self._down_val < self.number_of_bins/2:
                self._cfo_int_est = np.int16(np.floor(self._down_val/2))
            else:
                self._cfo_int_est = np.int16(np.floor((self._down_val - self.number_of_bins)/2))

            self._sto_int_est = -(self._cfo_int_est % self.number_of_bins)

            # compensate for sto_int in prb_raw_sync. Only self._n_up_req - 1 upchirps usable after that
            self._prb_raw_sync = np.roll(self._prb_raw_sync[0:self._n_up_req*self.number_of_bins], self._sto_int_est)

            # compensate cfo_int in prb_raw_sync
            self._cfo_int_corr_vect = np.exp(-2j*np.pi*self._cfo_int_est/self.number_of_bins*np.arange(0, (self._n_up_req + self._extra_up)*self.number_of_bins))
            self._prb_raw_sync[0:(self._n_up_req - 1)*self.number_of_bins] = self._prb_raw_sync[0:(self._n_up_req - 1)*self.number_of_bins] * self._cfo_int_corr_vect[0:(self._n_up_req - 1)*self.number_of_bins]

            # TODO: estimate and correct SFO in prb_raw_sync
            sto_frac_est2 = estimate_sto_frac(self._prb_raw_sync[0:(self._n_up_req - 1)*self.number_of_bins], self.sf, self._ref_downchirp)
            if abs(sto_frac_est2 - self._sto_frac_est) <= (self.os_factor - 1)/self.os_factor:
                self._sto_frac_est = sto_frac_est2
            
            # compensate for sto (and downsample) : only n_up_req - 1 + extra_up symbols usable after that
            prb_raw_corr = self._prb_raw_up[self.os_factor//2 - np.int32(np.round(self.os_factor*self._sto_frac_est)): self.os_factor//2 - np.int32(np.round(self.os_factor*self._sto_frac_est)) + self.samples_per_symbol*(self._n_up_req - 1 + self._extra_up): self.os_factor]
            prb_raw_corr = np.roll(prb_raw_corr, self._sto_int_est)
                                    
            # compensate for cfo_frac
            for idx in range(self._n_up_req - 1 + self._extra_up):
                prb_raw_corr[idx*self.number_of_bins: (idx + 1)*self.number_of_bins] *= self._cfo_frac_corr_vect[0:self.number_of_bins]

            # compensate for cfo_int
            prb_raw_corr = prb_raw_corr[0: (self._n_up_req - 1 + self._extra_up)*self.number_of_bins] * self._cfo_int_corr_vect[0:(self._n_up_req - 1 + self._extra_up)*self.number_of_bins]

            snr = 0
            dechirped_tmp = np.reshape(prb_raw_corr[0:(self._n_up_req - 1 + self._extra_up)*self.number_of_bins], (self.number_of_bins, -1), order='F') * self._ref_downchirp
            spectr_tmp = np.fft.fft(dechirped_tmp, axis=0)
            spectr_pow = np.abs(spectr_tmp)**2
            noise_bins = np.zeros((self.number_of_bins - 3, self._n_up_req - 1 + self._extra_up), dtype=np.complex64)
            for idx in range(self._n_up_req - 1 + self._extra_up):
                bin_mask = np.ones(self.number_of_bins, dtype=bool)
                main_bin = np.argmax(np.abs(spectr_tmp[:, idx]))
                bin_mask[[(main_bin - 1) % self.number_of_bins, main_bin, (main_bin + 1) % self.number_of_bins]] = False
                noise_bins[:, idx] = spectr_tmp[bin_mask, idx]

                sig_pow = np.sum(spectr_pow[~bin_mask, idx])
                noise_pow = np.sum(spectr_pow[bin_mask, idx])
                noise_pow = noise_pow*self.number_of_bins/(self.number_of_bins - 3)
                sig_pow -= 3*noise_pow/self.number_of_bins

                snr += sig_pow/noise_pow

            self._noise_sigma_sq = (np.mean(np.std(noise_bins, axis=0))**2)/self.number_of_bins                            # noise variance for 2 dimensions
            snr /= (self._n_up_req - 1 + self._extra_up)
            self._frame_info["snr"] = snr

            # TODO: compensate SFO in prb_raw_corr
            # TODO: update sto_frac across payload symbols according to sfo

            # sync and demodulate the netid
            netid_start_off = self.os_factor//2 - np.int32(np.round(self._sto_frac_est*self.os_factor)) + self.os_factor*(self.number_of_bins//4 + self._cfo_int_est)     # -- netid starting offset in netid_samp
            netid_samp_dec = self._netid_samp[netid_start_off: netid_start_off + self.samples_per_symbol*2: self.os_factor] 
            netid_samp_dec = netid_samp_dec * self._cfo_int_corr_vect[0: 2*self.number_of_bins]
            netid_samp_dec[0: self.number_of_bins] *= self._cfo_frac_corr_vect[0:self.number_of_bins]
            netid_samp_dec[self.number_of_bins: 2*self.number_of_bins] *= self._cfo_frac_corr_vect[0:self.number_of_bins]
            netids =  demodulate(netid_samp_dec, self._ref_downchirp, self.sf, False)
            
            netid_match = False
            netid_off = 0
            if abs(self._sync_word[0] - netids[0]) > 2:       # wrong netid1
                if abs(self._sync_word[1] - netids[0]) <= 2:  # correct netid2 with symbol offset
                    netid_off = netids[0] - self._sync_word[1]
                    # look for first sync_word in upchirps.
                    for idx in range(self.preamble_len - 2, self._n_up_req + self._extra_up - 1):
                        if demodulate(prb_raw_corr[idx*self.number_of_bins: (idx + 1)*self.number_of_bins], self._ref_downchirp, self.sf, False) + netid_off == self._sync_word[0]:     # found netid 1
                            netid_match = True
                            # the first header-payload symbol was mistaken for the end of downchirp: correct and output it
                            start_off = self.os_factor//2 - np.int32(np.round(self._sto_frac_est*self.os_factor)) + self.os_factor*(self.self.number_of_bins//4 + self._cfo_int_est)
                            if self.soft_decoding:
                                self._hdr_symbols[0] = demodulate(self._extra_symbol_samp[start_off: start_off + self.samples_per_symbol: self.os_factor], self._ref_downchirp, self.sf, self.soft_decoding, self._noise_sigma_sq/2, self._gray_mat)
                            else:
                                self._hdr_symbols[0] = demodulate(self._extra_symbol_samp[start_off: start_off + self.samples_per_symbol: self.os_factor], self._ref_downchirp, self.sf, False)

                            self._symb_count = 1
            else:                                       # netid1 valid
                netid_off = netids[0] - self._sync_word[0]
                if (netids[1] - netid_off)%self.number_of_bins == self._sync_word[1]:   # netid2 valid
                    netid_match = True

            self._sync_state = SyncState.END

            # check netid_match and update receiver state, or reset to preamble detection
            if netid_match:
                self._to_consume = self.samples_per_symbol//4 - self.os_factor*netid_off + self.os_factor*self._cfo_int_est
                self._frame_info["detected"] = True
                
                # -- update receiver state
                self._rx_state = ReceiverState.FRAME_DECODE
            else:                
                # -- reset receiver
                self._reset_rx()            

    def _decode_frame(self, samples: np.ndarray) -> None:
        """
        Decode frame header and payload from incoming symbol samples.

        The method updates the internal receiver state depending on decoding
        outcome:
        - If decoding succeeds, the receiver state is set to END.
        - If header or payload CRC validation fails, the receiver is reset to
        PREAMB_DET.

        Parameters
        ----------
        samples : np.ndarray
            Complex IQ samples corresponding to a single symbol duration
            (samples_per_symbol + os_factor), provided sequentially for the
            entire frame.
        """
        # downsample, compensate for cfo, and demodulate
        samples_dwn = samples[self.os_factor//2 - np.int32(np.round(self.os_factor*self._sto_frac_est)): self.os_factor//2 - np.int32(np.round(self.os_factor*self._sto_frac_est)) + self.samples_per_symbol: self.os_factor]
        samples_dwn *= self._cfo_frac_corr_vect[0:self.number_of_bins] * self._cfo_int_corr_vect[0:self.number_of_bins]
        
        # demodulate and accumulate header symbols
        if self._symb_count < 8:
            if self.soft_decoding:
                self._hdr_symbols[self._symb_count] = demodulate(samples_dwn, self._ref_downchirp, self.sf, self.soft_decoding, self._noise_sigma_sq/2, self._gray_mat)
            else:
                self._hdr_symbols[self._symb_count] = demodulate(samples_dwn, self._ref_downchirp, self.sf, False)
        
        # demodulate payload symbols
        else:
            if self.soft_decoding:
                self._pl_symbols[self._symb_count - 8] = demodulate(samples_dwn, self._ref_downchirp, self.sf, self.soft_decoding, self._noise_sigma_sq/2, self._gray_mat)
            else:
                self._pl_symbols[self._symb_count - 8] = demodulate(samples_dwn, self._ref_downchirp, self.sf, False)

        self._symb_count += 1
        self._to_consume = self.samples_per_symbol

        # decode 'header' block in reduced rate mode with cr4
        if self._symb_count == 8:
            if self.soft_decoding:
                hdr_symbols_deintd = deinterleave(self._hdr_symbols, 4, self.sf, self.soft_decoding, True)       # deinterleaving; gray mapping performed during hamming decoding
            else:
                hdr_symbols_gray = nongray2gray(np.mod(self._hdr_symbols - 1, self.number_of_bins), self.sf)            # gray mapping, then deinterleaving
                hdr_symbols_deintd = deinterleave(hdr_symbols_gray, 4, self.sf, self.soft_decoding, True)
                
            self._hdr_nib = hamming_dec(hdr_symbols_deintd, 4, self.soft_decoding)

            if not self.implicit_hdr:
                # -- compute and check header crc
                rx_hdr_crc = (self._hdr_nib[3, 0] << 4) + bin2dec(self._hdr_nib[4, :], "right-msb")

                c4 = self._hdr_nib[0, 3] ^ self._hdr_nib[0, 2] ^ self._hdr_nib[0, 1] ^ self._hdr_nib[0, 0]
                c3 = self._hdr_nib[0, 3] ^ self._hdr_nib[1, 3] ^ self._hdr_nib[1, 2] ^ self._hdr_nib[1, 1] ^ self._hdr_nib[2, 0]
                c2 = self._hdr_nib[0, 2] ^ self._hdr_nib[1, 3] ^ self._hdr_nib[1, 0] ^ self._hdr_nib[2, 3] ^ self._hdr_nib[2, 1]
                c1 = self._hdr_nib[0, 1] ^ self._hdr_nib[1, 2] ^ self._hdr_nib[1, 0] ^ self._hdr_nib[2, 2] ^ self._hdr_nib[2, 1] ^ self._hdr_nib[2, 0]
                c0 = self._hdr_nib[0, 0] ^ self._hdr_nib[1, 1] ^ self._hdr_nib[2, 3] ^ self._hdr_nib[2, 2] ^ self._hdr_nib[2, 1] ^ self._hdr_nib[2, 0]
                hdr_crc = bin2dec([c0, c1, c2, c3, c4], 'right-msb')
                hdr_err = rx_hdr_crc - hdr_crc
                
                if hdr_err:
                    # -- header error: set header_info and reset receiver state
                    self._frame_info["header"] = False
                    self._reset_rx()
                    
                else:
                    # -- get header fields
                    self._rx_pl_len = bin2dec(np.reshape(self._hdr_nib[1::-1, :], (1, -1), order='C'), "right-msb")
                    self._rx_has_crc = self._hdr_nib[2, 0]
                    self._rx_cr = bin2dec(self._hdr_nib[2, 1:4], "right-msb")
            else:
                self._rx_pl_len = self.payload_len
                self._rx_has_crc = self.crc_enable
                self._rx_cr = self.cr

            self._frame_info["header"] = True
            self._frame_info["cr"] = self._rx_cr
            self._frame_info["paylen"] = self._rx_pl_len

            # -- in auto mode, enable LDRO if symbol duration > 16 ms
            if self.ldro_mode == LDROMode.Auto:
                self._ldro = self._symb_duration_ms > 16
            else:
                self._ldro = LDROMode(self.ldro_mode)

            # -- compute the number of symbols (header + payload)
            n_blk = np.ceil((8*self._rx_pl_len - 4*self.sf + 28 + 16*self._rx_has_crc - 20*self.implicit_hdr)/(4*(self.sf - 2*self._ldro)))
            self._n_symb = np.uint32(8 + max(n_blk*(self._rx_cr + 4), 0))

            # -- allocated payload symbols array
            if self.soft_decoding:
                self._pl_symbols = np.zeros((self._n_symb - 8, self.sf), dtype=np.float32)              # sf LLR bits per symbols in soft-decoding 
            else:
                self._pl_symbols = np.zeros(self._n_symb - 8, dtype=np.uint16)

        # decode payload block
        elif self._symb_count == self._n_symb:
            if self.soft_decoding:
                pl_symbols_deintd = deinterleave(self._pl_symbols, self._rx_cr, self.sf, self.soft_decoding, self._ldro)
            else:
                pl_symbols_gray = nongray2gray(np.mod(self._pl_symbols - 1, self.number_of_bins), self.sf)
                pl_symbols_deintd = deinterleave(pl_symbols_gray, self._rx_cr, self.sf, self.soft_decoding, self._ldro)

            pl_nibbles = hamming_dec(pl_symbols_deintd, self._rx_cr, self.soft_decoding)

            # -- concatenate payload nibbles from header block if any
            if self.implicit_hdr:
                pl_nibbles = np.concatenate((self._hdr_nib, pl_nibbles), axis=0)
            else:
                pl_nibbles = np.concatenate((self._hdr_nib[5:self.sf-2, :], pl_nibbles), axis=0)
            pl_nibbles = pl_nibbles[:2*(self._rx_pl_len + 2*self._rx_has_crc), :]                             # remove padding nibbles if any

            pl_bytes = bin2dec(np.reshape(pl_nibbles[:-4, :], (-1, 8), order='C'), "right-msb")
            payload =  whiten(pl_bytes)

            # -- check payload crc if enabled
            if self._rx_has_crc:
                rx_crc = bin2dec(np.reshape(pl_nibbles[-4:, :], (1, 16), order='C'), "right-msb")
                crc_tmp = ccit16(payload[:-2])
                crc = np.bitwise_xor(crc_tmp, (payload[-2]<<8) + payload[-1])
                frame_success = crc == rx_crc
            else:
                frame_success = True
            
            self._frame_info["payload"] = payload

            if frame_success:
                # -- set frame_info and update receiver state
                self._frame_info["success"] = True
                self._rx_state = ReceiverState.END
            else:
                # -- set frame_info and reset receiver state
                self._frame_info["success"] = False
                self._reset_rx()

    def start(self) -> dict:
        """
        Main receiver processing loop.

        This function reads incoming IQ samples from the attached buffer and
        processing to the appropriate stage based on the current receiver state 
        (preamble detection synchronization, or frame decoding).

        It also provides top-level exception handling: any exception raised
        during processing is caught and recorded in 'frame_info'.

        Operation Modes
        ---------------
        - One-shot mode:
            The function returns the decoded 'frame_info' dictionary once a
            frame is successfully decoded or terminates on failure/end of buffer.

        - Continuous mode:
            The function invokes the registered frame callback (`on_success`)
            upon successful frame decoding and continues processing subsequent
            frames.

        Returns
        -------
        dict
            Dictionary containing decoded frame information and processing
            metadata (`frame_info`). Returned only in one-shot mode.
        """
        # initialize frame_info
        self._frame_info = {
            "detected": False,     # if netID match
            "header": False,       # if header is correct
            "success": False,      # if payload CRC is correct
            "eof": False,          # reached end of file (buffer): used for one-shot simulation
            "exception": False,    # exited on exception
            "extb": "",            # exception traceback if any
            "cr": 1,               # detected CR (or configured if implicit header)
            "payload": "",         # payload
            "paylen": 0,           # payload length
            "snr": None            # snr computed on the preamble
        }

        # ensure a buffer is provided
        if self._buffer is None:
            self._frame_info["exception"] = True
            self._frame_info["extb"] = "[run] Buffer is not initialized"
            print("Buffer is not initialized")
            return self._frame_info
        
        # ensure a frame handling callback is provided, or one-shot is set
        if not self.one_shot and self.on_success is None:
            self._frame_info["exception"] = True
            self._frame_info["extb"] = "[run] Please set a callback for successful frames"
            print("Please register a callback for successful frames")
            return self._frame_info
        
        # main loop
        counter = 0
        while True:
            try:
                # wait for a full symbol samples if possible
                counter += 1
                while self._buffer.available() < self.samples_per_symbol + self.os_factor:
                    # -- stop with EOF if one-shot configured
                    if self.one_shot:
                        self._frame_info["eof"] = True
                        return self._frame_info
                    else:
                        time.sleep(self._symb_duration_ms/2/1000)         # sleep for half of a symbol duration
                
                samples = self._buffer.peek(self.samples_per_symbol + self.os_factor)
                
                # preamble detection
                if self._rx_state == ReceiverState.PREAMB_DET:
                    ret = self._detect_preamble(samples)
                                    
                # frame synchronization
                elif self._rx_state == ReceiverState.FRAME_SYNC:
                    ret = self._sync_frame(samples)

                # frame decoding
                elif self._rx_state == ReceiverState.FRAME_DECODE:
                    ret = self._decode_frame(samples)

                # payload processing (successful frame decoding)
                if self._rx_state == ReceiverState.END:
                    # -- return frame info if one-shot, or invoke frame handling callback
                    if self.one_shot:
                        return self._frame_info
                    else:
                        self.on_success(self._frame_info)
                        self._reset_rx()
                
                # consume the samples processed
                self._buffer.consume(self._to_consume)

            # exception caught in main loop
            except Exception as e:
                # -- return frame_info if exception caught in main loop, or just reset the receiver 
                if self.one_shot:
                    self._frame_info["exception"] = True
                    self._frame_info["extb"] = traceback.format_exc()
                    self._reset_rx()
                    return self._frame_info
                else:
                    # -- reset receiver
                    self._reset_rx()
                    self._buffer.consume(self.samples_per_symbol)
                    continue