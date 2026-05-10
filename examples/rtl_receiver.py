# A LoRa receiver based on RTL-SDR dongle
# -- Requires a functional SoapySDR library (which includes of course an rtlsdr driver)

from pylorasdr import CircBuffer
from pylorasdr import Receiver
import numpy as np
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import time
import threading
import signal
import sys

# frame handler
def print_payload_hex(frame_info:dict):
    print(f"Received frame with payload: {[format(x, '02x') for x in frame_info['payload']]}")

# signal handler
def shutdown(signum, frame):
    print("Shutting down...")
    rx_thread.stop()
    rx_thread.join()
    sys.exit(0)

# ===============================
# SDR Receiver Thread
# ===============================

class SDRReceiver(threading.Thread):
    def __init__(self, circ_buffer):
        super().__init__()
        self.buffer = circ_buffer
        self.running = False

        # Setup SDR
        self.sdr = SoapySDR.Device(dict(driver="rtlsdr"))
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, samp_rate)
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
        self.sdr.setGain(SOAPY_SDR_RX, 0, 40)
        self.sdr.setBandwidth(SOAPY_SDR_RX, 0, bw)

        self.rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.sdr.activateStream(self.rxStream)

    def run(self):
        self.running = True
        buff = np.zeros(4096, dtype=np.complex64)

        while self.running:
            sr = self.sdr.readStream(self.rxStream, [buff], len(buff), timeoutUs=int(1e6))
            if sr.ret > 0:
                self.buffer.write(buff[:sr.ret])
            elif sr.ret != SOAPY_SDR_TIMEOUT:
                print("SDR read error:", sr.ret)

    def stop(self):
        self.running = False
        self.sdr.deactivateStream(self.rxStream)
        self.sdr.closeStream(self.rxStream)

    
# frame args
# -- explicit header mode by default (use set_implicit_header() on the receiver to change)
center_freq = 868.1e6           # center frequency
bw = 125e3                      # bandwidth
osF = 2                         # oversampling factor (at least 2)
samp_rate = osF*bw              # sampling rate (for the SDR)
sf = 7                          # spreading factor
n_up = 8                        # number of preamble upchirps
netid = 0x12                    # netID
soft_decoding = True            # enable/disable soft-decoding

# initialize receiver
lora_rx = Receiver(center_freq, bw, sf, osF, netid, n_up, soft_decoding)
lora_rx.register_frame_handler(print_payload_hex)

# create and attach buffer to receiver
buf_size = 2**18
circ_buf = CircBuffer(buf_size)
lora_rx.attach_buffer(circ_buf)

# initialize SDR rx (samples rx thread) 
rx_thread = SDRReceiver(circ_buf)

# register signal for stopping on Ctrl-C
# -- currently the only way to stop (poor logics, I know. Will fix that :)
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# start samples reception and baseband receiver
rx_thread.start()
fram_info = lora_rx.start()

