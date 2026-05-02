# Simulate frame error rate in function of SNR
# -- Requires matplotlib for the FER plot

from datetime import datetime
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
from pylorasdr import CircBuffer
from pylorasdr import Receiver
from pylorasdr import Transmitter

center_freq = 868.1e6
bw = 125e3
osF = 2
samp_rate = osF*bw
sf = 7
N = 2**sf
cr = 1
n_up = 8
samples_per_symbol = N*osF
netid = 0x12
implicit_hdr = False
payload_len = 7
has_crc = True
soft_decoding = True
one_shot = True

n_epochs = 100000
payload_len = 7
snr_min_dB = -15
snr_max_dB = 0
snr_step_dB  = 1
snr_dB = np.arange(snr_min_dB, snr_max_dB + snr_step_dB, snr_step_dB)

rng = np.random.default_rng()
lora_tx = Transmitter(bw, sf, osF, n_up, netid, cr)
lora_rx = Receiver(center_freq, bw, sf, osF, netid, n_up, soft_decoding)
lora_rx.set_one_shot(True)

buf_exists = False

fer = np.zeros(len(snr_dB))
for snr_idx in range(len(snr_dB)):
    print(f"{snr_dB[snr_idx]} dB SNR at {datetime.now()}")
    noise_pow = 1/10**(snr_dB[snr_idx]/10)
    for epoch in range(n_epochs):
        payload = np.random.randint(0, N, size=payload_len)
        samples = lora_tx.build_frame(payload)
        samples = np.append(samples, np.zeros(osF*2**(sf - 1), dtype=np.complex64))     # pad the samples with few 0
        sigma = np.sqrt(noise_pow/2)
        noise_samples = sigma*(rng.standard_normal(len(samples)) + 1j*rng.standard_normal(len(samples)))    
        samples_noisy = samples + noise_samples

        # -- create buffer on first run
        if not buf_exists:
            buf_size = len(samples_noisy)
            circ_buf = CircBuffer(buf_size)

        # -- reset for other runs
        else:
            circ_buf.clear()

        circ_buf.write(samples_noisy)                
        lora_rx.set_buffer(circ_buf)

        frame = lora_rx.start()
        if not frame['success']:
            fer[snr_idx] += 1

# compute fer and save
print(f"End of simulation at {datetime.now()}")
fer /= n_epochs

# -- create "results" folder inside script parent
base_dir = Path(__file__).resolve().parent
results_dir = base_dir / "results"
results_dir.mkdir(exist_ok=True)

# -- make filenames and save
snr_file = results_dir / f"snr_dB_sf-{sf}_cr-{cr}_{n_epochs}.txt"
fer_file = results_dir / f"fer_sf-{sf}_cr-{cr}_{n_epochs}.txt"
np.savetxt(snr_file, snr_dB)
np.savetxt(fer_file, fer)

print(f"Results saved as {snr_file} and {fer_file}")

# show time
plt.plot(snr_dB, fer)
plt.xlabel("SNR (dB)")
plt.ylabel("Frame error rate")
plt.show()

