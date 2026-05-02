import numpy as np

class CircBuffer:
    """
    Circular buffer for complex IQ samples.

    Provides FIFO-like behavior with optional overwrite mode and supports
    writing, reading, peeking, and manual pointer control.
    """

    def __init__(self, size: int):
        """
        Circular buffer for complex IQ samples.

        Provides FIFO-like behavior with optional overwrite mode and supports
        writing, reading, peeking, and manual pointer control.
        """
        self.size = size
        self.buf = np.zeros(size + 1, dtype=np.complex64)  # +1 full-buffer indicator
        self.head = 0
        self.tail = 0
        self.overwrite_enabled = True  # enable/disable overwrite on full buffer

    def write(self, samples: np.ndarray) -> None:
        """
        Write one or more samples into the buffer.

        Parameters
        ----------
        samples : np.ndarray
            Complex samples to write (np.complex64 array).
        """
        n_samples = samples.size
        if self.overwrite_enabled:
            to_write = n_samples
        else:
            free_size = self.size - (self.head - self.tail) % (self.size + 1)
            to_write = min(n_samples, free_size)

        n_until_wrap = self.size - self.head + 1
        if to_write <= n_until_wrap:
            self.buf[self.head: self.head + to_write] = samples[0:to_write]
        else:
            self.buf[self.head: self.head + n_until_wrap] = samples[0:n_until_wrap]
            self.buf[0: to_write - n_until_wrap] = samples[n_until_wrap:to_write]

        self.head = self.head + to_write
        if self.head >= self.size + 1:
            self.head = self.head % (self.size + 1)

    def read(self, out_size: int) -> np.ndarray | None:
        """
        Read and remove samples from the buffer.

        Parameters
        ----------
        out_size : int
            Number of samples to read.

        Returns
        -------
        np.ndarray | None
            Array of samples if enough data is available, otherwise None.
        """
        n_avail = (self.head - self.tail) % (self.size + 1)
        if n_avail < out_size:
            return None

        out = np.zeros(out_size, dtype=np.complex64)
        n_until_wrap = self.size - self.tail + 1

        if out_size <= n_until_wrap:
            out[0: out_size] = self.buf[self.tail: self.tail + out_size]
        else:
            out[0: n_until_wrap] = self.buf[self.tail: self.tail + n_until_wrap]
            out[n_until_wrap: out_size] = self.buf[0: out_size - n_until_wrap]

        self.tail = self.tail + out_size
        if self.tail >= self.size + 1:
            self.tail = self.tail % (self.size + 1)

        return out

    def peek(self, out_size: int) -> np.ndarray | None:
        """
        Read samples without advancing the buffer tail.

        Parameters
        ----------
        out_size : int
            Number of samples to peek.

        Returns
        -------
        np.ndarray | None
            Array of samples if available, otherwise None.
        """
        n_avail = (self.head - self.tail) % (self.size + 1)
        if n_avail < out_size:
            return None

        out = np.zeros(out_size, dtype=np.complex64)
        n_until_wrap = self.size - self.tail + 1

        if out_size <= n_until_wrap:
            out[0: out_size] = self.buf[self.tail: self.tail + out_size]
        else:
            out[0: n_until_wrap] = self.buf[self.tail: self.tail + n_until_wrap]
            out[n_until_wrap: out_size] = self.buf[0: out_size - n_until_wrap]

        return out

    def consume(self, n_samples: int) -> int:
        """
        Advance the tail pointer without reading data.

        Parameters
        ----------
        n_samples : int
            Number of samples to skip.

        Returns
        -------
        int
            Actual number of samples skipped.
        """
        n_avail = (self.head - self.tail) % (self.size + 1)
        n_move = np.min((n_avail, n_samples))

        self.tail = self.tail + n_move
        if self.tail >= self.size + 1:
            self.tail = self.tail % (self.size + 1)

        return n_move

    def available(self) -> int:
        """
        Get number of available samples in the buffer.

        Returns
        -------
        int
            Number of readable samples.
        """
        return (self.head - self.tail) % (self.size + 1)

    def enable_overwrite(self) -> None:
        """
        Enable overwrite mode when buffer is full.
        """
        self.overwrite_enabled = True

    def disable_overwrite(self) -> None:
        """
        Disable overwrite mode when buffer is full.
        """
        self.overwrite_enabled = False

    def overwrite_enabled(self) -> bool:
        """
        Check whether overwrite mode is enabled.

        Returns
        -------
        bool
            True if overwrite is enabled, False otherwise.
        """
        return self.overwrite_enabled

    def clear(self) -> None:
        """
        Reset buffer pointers .
        """
        self.head = 0
        self.tail = 0