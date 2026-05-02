from enum import IntEnum, auto

class ReceiverState(IntEnum):
    """
    Operating state machine for the receiver.

    Represents the differnt stages of frame reception: preamble detection, 
    frame synchronization, frame decoding, and end of reception (payload to process).
    """
    PREAMB_DET = auto()
    FRAME_SYNC = auto()
    FRAME_DECODE = auto()
    END = auto()

class SyncState(IntEnum):
    """
    Synchronization state machine for the LoRa receiver.

    Represents the different stages of packet synchronization during reception,
    from network ID detection to final alignment completion.
    """
    COARSE = auto()
    NETID1 = auto()
    NETID2 = auto()
    DOWNCHIRP1 = auto()
    DOWNCHIRP2 = auto()
    QUARTER_DOWN = auto()
    END = auto()

class LDROMode(IntEnum):
    """
    Low Data Rate Optimization (LDRO) operating modes.

    Controls how LDRO is applied in the receiver chain.
    """
    Disabled = 0
    Enabled = 1
    Auto = 2