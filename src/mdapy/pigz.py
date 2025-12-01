# Copyright (c) 2022-2025, Yongchao Wu in Aalto University
# This file is from the mdapy project, released under the BSD 3-Clause License.
"""
Parallel gzip compression using multiprocessing.

Simple, fast, and reliable:
- Uses multiprocessing (true parallelism, no GIL)
- Always uses maximum compression (level 9)
- Focuses on: speed for large files, correctness for all files
- Clean code, proper resource management
"""

import os
import sys
import time
import zlib
from multiprocessing import Pool, cpu_count
from pathlib import Path
from queue import PriorityQueue
from threading import Lock, Thread


__all__ = ["compress_file"]  # 只导出这个函数给Sphinx


def compress_file(input_file, output_file=None):
    """
    Compress a file to .gz format using multiprocessing.

    This function provides parallel gzip compression for faster processing
    of large files. It automatically uses all available CPU cores and
    maintains full gzip format compatibility.

    Args:
        input_file (str): Path to input file to compress
        output_file (str, optional): Path to output file.
            If not specified, adds .gz to input filename.

    Returns:
        str: Path to the created compressed file

    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input file already has .gz extension
        Exception: If compression fails for any reason

    Examples:
        >>> # Compress with automatic output name
        >>> compress_file("data.txt")
        'data.txt.gz'

        >>> # Compress with custom output name
        >>> compress_file("input.txt", "output.gz")
        'output.gz'

    Note:
        - Small files (<5MB) use single-process compression
        - Large files automatically use all CPU cores
        - Uses 512KB chunks for optimal parallelism
    """
    compressor = _ParallelGzip(input_file, output_file)
    return compressor.compress()


# ============================================================================
# Internal implementation (not exposed to Sphinx)
# ============================================================================


class _ParallelGzip:
    """
    Internal parallel gzip compressor implementation.

    This class handles the multiprocessing coordination for parallel
    compression. It's not meant to be used directly - use compress_file()
    instead.
    """

    # Constants
    DEFAULT_BLOCKSIZE_KB = 512
    SMALL_FILE_THRESHOLD_MB = 5
    GZIP_MAGIC = b"\x1f\x8b"
    GZIP_METHOD_DEFLATE = b"\x08"
    GZIP_FLAG_FNAME = b"\x08"
    GZIP_EXTRA_MAX_COMPRESSION = b"\x02"

    def __init__(self, input_file, output_file=None, blocksize_kb=None, workers=None):
        """Initialize compressor with validated parameters."""
        # Validate input
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if input_file.endswith(".gz"):
            raise ValueError("Input file is already .gz")

        self.input_file = input_file
        self.output_file = output_file or (input_file + ".gz")
        self.blocksize = (blocksize_kb or self.DEFAULT_BLOCKSIZE_KB) * 1024
        self.workers = workers or cpu_count()

        # Optimize for small files
        file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        if file_size_mb < self.SMALL_FILE_THRESHOLD_MB:
            self.workers = 1

        # Internal state
        self._chunk_queue = PriorityQueue()
        self._last_chunk = -1
        self._last_chunk_lock = Lock()
        self._input_size = 0
        self._crc32 = 0
        self._error = None

        # Resources to cleanup
        self._output_file = None
        self._pool = None
        self._reader_thread = None
        self._writer_thread = None

    def compress(self):
        """
        Execute the compression process.

        Returns:
            str: Path to compressed file

        Raises:
            Exception: If compression fails
        """
        try:
            self._output_file = open(self.output_file, "wb")
            self._write_header()

            # Create process pool
            self._pool = Pool(processes=self.workers)

            # Start worker threads
            self._reader_thread = Thread(target=self._read_and_dispatch, daemon=True)
            self._writer_thread = Thread(target=self._collect_and_write, daemon=True)

            self._writer_thread.start()
            self._reader_thread.start()

            # Wait for completion
            self._writer_thread.join()
            self._reader_thread.join(timeout=5)

            # Check for errors from worker threads
            if self._error:
                raise self._error

            return self.output_file

        except Exception as e:
            # Clean up failed compression attempt
            self._cleanup()
            if os.path.exists(self.output_file):
                try:
                    os.remove(self.output_file)
                except OSError:
                    pass  # Ignore errors during cleanup
            raise Exception(f"Compression failed: {e}") from e

        finally:
            self._cleanup()

    def _write_header(self):
        """Write gzip header according to RFC 1952."""
        # Magic number + compression method
        self._output_file.write(self.GZIP_MAGIC)
        self._output_file.write(self.GZIP_METHOD_DEFLATE)

        # Flags (include filename)
        self._output_file.write(self.GZIP_FLAG_FNAME)

        # Modification time
        try:
            mtime = int(os.path.getmtime(self.input_file))
        except OSError:
            mtime = int(time.time())
        self._output_file.write(mtime.to_bytes(4, "little"))

        # Extra flags (max compression) + OS
        self._output_file.write(self.GZIP_EXTRA_MAX_COMPRESSION)
        os_flag = 3 if sys.platform != "win32" else 0
        self._output_file.write(bytes([os_flag]))

        # Original filename (optional)
        self._write_filename()

    def _write_filename(self):
        """Write original filename to gzip header."""
        try:
            fname = Path(self.input_file).name.encode("latin-1")
            if fname.endswith(b".gz"):
                fname = fname[:-3]
            self._output_file.write(fname + b"\x00")
        except (UnicodeEncodeError, OSError):
            # If filename can't be encoded, skip it
            pass

    def _write_trailer(self):
        """Write gzip trailer (CRC32 and original size)."""
        self._output_file.write(self._crc32.to_bytes(4, "little"))
        # Original size modulo 2^32
        self._output_file.write((self._input_size & 0xFFFFFFFF).to_bytes(4, "little"))

    def _read_and_dispatch(self):
        """Read input file in chunks and dispatch to process pool."""
        chunks_to_compress = []

        try:
            # Read all chunks
            with open(self.input_file, "rb") as f:
                chunk_num = 0
                while True:
                    chunk = f.read(self.blocksize)
                    if not chunk:
                        break

                    chunk_num += 1
                    self._input_size += len(chunk)
                    chunks_to_compress.append((chunk_num, chunk))

            # Mark the last chunk
            with self._last_chunk_lock:
                self._last_chunk = len(chunks_to_compress)

            # Dispatch all chunks to process pool
            for idx, (chunk_num, chunk) in enumerate(chunks_to_compress):
                is_last = idx == len(chunks_to_compress) - 1
                self._pool.apply_async(
                    _compress_chunk,
                    args=(chunk_num, chunk, is_last),
                    callback=self._chunk_queue.put,
                    error_callback=self._handle_compression_error,
                )

        except (IOError, OSError) as e:
            self._error = e
            # Ensure last_chunk is set so writer thread can exit
            with self._last_chunk_lock:
                if self._last_chunk < 0:
                    self._last_chunk = 0

    def _handle_compression_error(self, error):
        """Callback for handling errors from worker processes."""
        self._error = error

    def _collect_and_write(self):
        """Collect compressed chunks in order and write to output."""
        next_chunk = 1

        try:
            while True:
                # Check if we're done
                with self._last_chunk_lock:
                    if self._last_chunk > 0 and next_chunk > self._last_chunk:
                        break

                # Wait for data
                if self._chunk_queue.empty():
                    time.sleep(0.01)
                    continue

                # Get next chunk
                chunk_num, original_data, compressed_data = self._chunk_queue.get()

                # If not the next chunk in sequence, put it back
                if chunk_num != next_chunk:
                    self._chunk_queue.put((chunk_num, original_data, compressed_data))
                    time.sleep(0.01)
                    continue

                # Write compressed data
                self._output_file.write(compressed_data)

                # Update CRC32 with original data
                self._crc32 = zlib.crc32(original_data, self._crc32)

                # Check if this was the last chunk
                with self._last_chunk_lock:
                    if chunk_num == self._last_chunk:
                        break

                next_chunk += 1

            # Write gzip trailer
            self._write_trailer()

        except Exception as e:
            self._error = e

    def _cleanup(self):
        """Clean up all resources."""
        # Close output file
        if self._output_file:
            try:
                self._output_file.flush()
                self._output_file.close()
            except (IOError, OSError):
                pass

        # Terminate process pool
        if self._pool:
            try:
                self._pool.close()
                self._pool.join(timeout=3)
                self._pool.terminate()
            except Exception:
                # Ignore all errors during pool cleanup
                pass


def _compress_chunk(chunk_num, data, is_last=False):
    """
    Compress a single chunk of data.

    This function runs in a separate process for parallel compression.

    Args:
        chunk_num: Chunk number for ordering
        data: Raw data bytes to compress
        is_last: Whether this is the last chunk

    Returns:
        tuple: (chunk_num, original_data, compressed_data)
    """
    compressor = zlib.compressobj(
        level=9,  # Maximum compression
        method=zlib.DEFLATED,
        wbits=-zlib.MAX_WBITS,  # Raw deflate
        memLevel=zlib.DEF_MEM_LEVEL,
        strategy=zlib.Z_DEFAULT_STRATEGY,
    )

    compressed = compressor.compress(data)

    # Use appropriate flush mode
    if is_last:
        compressed += compressor.flush(zlib.Z_FINISH)
    else:
        compressed += compressor.flush(zlib.Z_SYNC_FLUSH)

    return (chunk_num, data, compressed)


if __name__ == "__main__":
    # Simple test
    start = time.time()
    result = compress_file("cu.xyz")
    end = time.time()
    print(f"Compressed to: {result}")
    print(f"Time: {end - start:.2f} seconds")
