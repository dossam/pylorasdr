def ccit16(data: bytes) -> int:
    """
    Compute the CRC-16-CCITT checksum of a byte sequence.

    This function implements the standard CRC-16-CCITT algorithm using the
    polynomial 0x1021 and an initial value of 0x0000.

    Parameters
    ----------
    data : bytes
        Input byte sequence for which the CRC is computed.

    Returns
    -------
    int
        an unsigned 16-bit integer CRC.
    """
    poly = 0x1021
    crc = 0x0000

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) if (crc & 0x8000) else (crc << 1)
            crc &= 0xFFFF

    return crc