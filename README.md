# PyLoRaSDR

## Description

This is a Python library for generating and processing baseband LoRa signals. It features a transmitter and a receiver modules that can be easily interfaced with SDR RF frontend, or used for simulation.

The public API exposes:
- `CircBuffer`: a circular buffer implementation. It holds the samples to be processed by the receiver.
- `Receiver`: a baseband receiver implementation. Requires a buffer attached with the `Receiver.attach_buffer()` method. It processes the samples contained in the buffer to detect and decode potential LoRa frames, according to provided frame parameters (`bw`, `sf`, `cr`).
- `Transmitter`: a baseband transmitter implementation. Builds a full LoRa frame from a provided payload and to frame parameters (`bw`, `sf`, `cr`).

The `Receiver` operates in two modes:
- A continuous mode (default), where it continuously looks for and decodes incoming frames, reports decoded frame through a registered callback, waits for more samples if empty-buffer is reached, and starts over. This is the recommended mode for interfacing with SDR.
- A one-shot mode. It processes the samples until it reached empty-buffer, and returns the frame decoded if any. This mode is recommended for simulation, and is set with `Receiver.set_one()`.

The receiver currently supports both uplink and downlink frames, which are automatically detected and decoded.

For more details, check the [frame error rate (FER) simulation](examples/fer_simulation.py) and the [RTL-SDR receiver](examples/rtl_receiver.py) examples.

## Notes
Although all the current modules are tested and work 'as-is', this is still a work in progress. For a fully-tested alternative, you might want to check the C++ GNU Radio implementation at https://github.com/tapparelj/gr-lora_sdr.

## Installation
- Clone this repo:
```bash 
    git clone [text](https://github.com/dossam/pylorasdr.git)
```

- Move into the repo folder:
```bash
    cd pylorasdr
```

- Install with `pip`
```bash
    pip install .
```

- And that's it. Enjoy.
You can start by exploring the examples.

## Dependencies:
The library itself only depends on `numpy`. In function of your use case, additional libraries might be required:
- The FER simulation example requires `matplotlib` for FER plot.
- Interfacing with SDR requires `SoapySDR`, including of course a functioal driver for the target SDR platform.

## Changelog
- Added downlink frame Rx support

## TODO:
Incoming changes and improvements are still on the way, including:
- A support for downlink frames (Inverted IQ) transmission; Reception is already implemented
- SFO compensation in the receiver (currently not implemented)
- Robust input arguments validation for the underlying building blocks (only if you are thinkering with those blocks)

## References:
For more details about the LoRa modulation and underlying coding/decoding mechanisms, you might want to check this useful reverse-engineering report at https://www.epfl.ch/labs/tcl/wp-content/uploads/2020/02/Reverse_Eng_Report.pdf (I am not the author; just sharing).

## Credit
This work is inspired from the C++ GNU Radio implementation at https://github.com/tapparelj/gr-lora_sdr.