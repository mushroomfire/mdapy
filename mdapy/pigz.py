"""
Functions and classes to speed up compression of files by utilizing
multiple cores on a system.
"""
"""
The original file comes from https://github.com/bguise987/pigz-python/tree/master.
mdapy changes some details to compress the dump to dump.gz.
"""
import os
import sys
import time
import zlib
from multiprocessing.dummy import Pool
from pathlib import Path
from queue import PriorityQueue
from threading import Lock, Thread

CPU_COUNT = os.cpu_count()
DEFAULT_BLOCK_SIZE_KB = 128

# 1 is fastest but worst, 9 is slowest but best
GZIP_COMPRESS_OPTIONS = list(range(1, 9 + 1))
_COMPRESS_LEVEL_BEST = max(GZIP_COMPRESS_OPTIONS)

# FLG bits
FTEXT = 0x1
FHCRC = 0x2
FEXTRA = 0x4
FNAME = 0x8
FCOMMENT = 0x10


class PigzFile:  # pylint: disable=too-many-instance-attributes
    """Class to implement Pigz functionality in Python"""

    def __init__(
        self,
        compression_target,
        output_filename=None,
        compresslevel=_COMPRESS_LEVEL_BEST,
        blocksize=DEFAULT_BLOCK_SIZE_KB,
        workers=CPU_COUNT,
    ):
        """
        Take in a file or directory and gzip using multiple system cores.
        """
        assert isinstance(compression_target, str)
        assert (
            compression_target.split(".")[-1] != "gz"
        ), "Can not compress the .gz file."
        assert os.path.exists(
            compression_target
        ), f"No such file or directory: {compression_target}"
        self.compression_target = compression_target
        self.compression_level = compresslevel
        self.blocksize = blocksize * 1000
        self.workers = workers

        self.output_file = None
        if output_filename is None:
            output_filename = self.compression_target + ".gz"
        self.output_filename = output_filename

        # This is how we know if we're done reading, compressing, & writing the file
        self._last_chunk = -1
        self._last_chunk_lock = Lock()
        # This is calculated as data is written out
        self.checksum = 0
        # This is calculated as data is read in
        self.input_size = 0
        self.chunk_queue = PriorityQueue()

        # Setup the system threads for compression
        self.pool = Pool(processes=self.workers)
        # Setup read thread
        self.read_thread = Thread(target=self._read_file)
        # Setup write thread
        self.write_thread = Thread(target=self._write_file)

    def process_compression_target(self):
        """
        Setup output file.
        Start read and write threads.
        Join to write thread.
        """
        self._setup_output_file()

        # Start the write thread first so it's ready to accept data
        self.write_thread.start()
        # Start the read thread
        self.read_thread.start()

        # Block until writing is complete
        # This prevents us from returning prior to the work being done
        self.write_thread.join()

    def _write_output_header(self):
        """
        Write gzip header to file
        See RFC documentation: http://www.zlib.org/rfc-gzip.html#header-trailer
        """
        self._write_header_id()
        self._write_header_cm()

        # We must first figure out if we can write out the filename before writing FLG
        fname = self._determine_fname(self.compression_target)
        flags = 0x0
        if fname:
            flags = flags | FNAME

        self._write_header_flg(flags)

        self._write_header_mtime()
        self._write_header_xfl()
        self._write_header_os()

        # After this point, content of flags (FLG) determines what (if anything)
        # we write to header
        if flags & FNAME:
            # Write the FNAME
            self.output_file.write(fname)

    def _write_header_id(self):
        """
        Write ID (IDentification) ID1, then ID2 to file header
        These denote the file as being gzip format
        """
        self.output_file.write((0x1F).to_bytes(1, sys.byteorder))
        self.output_file.write((0x8B).to_bytes(1, sys.byteorder))

    def _write_header_cm(self):
        """Write the CM (compression method) to file header"""
        self.output_file.write((8).to_bytes(1, sys.byteorder))

    def _write_header_flg(self, flags):
        """Write FLG (FLaGs)"""
        self.output_file.write((flags).to_bytes(1, sys.byteorder))

    def _write_header_mtime(self):
        """Write MTIME (Modification time)"""
        mtime = self._determine_mtime()
        self.output_file.write((mtime).to_bytes(4, sys.byteorder))

    def _write_header_xfl(self):
        """Write XFL (eXtra FLags)"""
        extra_flags = self._determine_extra_flags(self.compression_level)
        self.output_file.write((extra_flags).to_bytes(1, sys.byteorder))

    def _write_header_os(self):
        """Write OS"""
        os_number = self._determine_operating_system()
        self.output_file.write((os_number).to_bytes(1, sys.byteorder))

    def _setup_output_file(self):
        """
        Setup the output file
        """

        self.output_file = open(self.output_filename, "wb")
        self._write_output_header()

    def _determine_mtime(self):
        """
        Determine MTIME to write out in Unix format (seconds since Unix epoch).
        From http://www.zlib.org/rfc-gzip.html#header-trailer:
        If the compressed data did not come from a file, MTIME is set to the time at
        which compression started.
        MTIME = 0 means no time stamp is available.
        """
        try:
            return int(os.stat(self.compression_target).st_mtime)
        except Exception:  # pylint: disable=broad-except
            return int(time.time())

    @staticmethod
    def _determine_extra_flags(compression_level):
        """
        Determine the XFL or eXtra FLags value based on compression level.
        Note this is copied from the pigz implementation.
        """
        return 2 if compression_level >= 9 else 4 if compression_level == 1 else 0

    @staticmethod
    def _determine_operating_system():
        """
        Return appropriate number based on OS format.
        0 - FAT filesystem (MS-DOS, OS/2, NT/Win32)
        1 - Amiga
        2 - VMS (or OpenVMS)
        3 - Unix
        4 - VM/CMS
        5 - Atari TOS
        6 - HPFS filesystem (OS/2, NT)
        7 - Macintosh
        8 - Z-System
        9 - CP/M
        10 - TOPS-20
        11 - NTFS filesystem (NT)
        12 - QDOS
        13 - Acorn RISCOS
        255 - unknown
        """
        if sys.platform.startswith(("freebsd", "linux", "aix", "darwin")):
            return 3
        if sys.platform.startswith(("win32")):
            return 0

        return 255

    @staticmethod
    def _determine_fname(input_filename):
        """
        Determine the FNAME (filename) of the source file to the output
        """
        try:
            # RFC 1952 requires the FNAME field to be Latin-1. Do not
            # include filenames that cannot be represented that way.
            fname = Path(input_filename).name
            if not isinstance(fname, bytes):
                fname = fname.encode("latin-1")
            if fname.endswith(b".gz"):
                fname = fname[:-3]
            # Terminate with zero byte
            fname += b"\0"
        except UnicodeEncodeError:
            fname = b""

        return fname

    def _read_file(self):
        """
        Read {filename} in {blocksize} chunks.
        This method is run on the read thread.
        """
        # Initialize this to 0 so our increment sets first chunk to 1
        chunk_num = 0
        with open(self.compression_target, "rb") as input_file:
            while True:
                chunk = input_file.read(self.blocksize)
                # Break out of the loop if we didn't read anything
                if not chunk:
                    with self._last_chunk_lock:
                        self._last_chunk = chunk_num
                    break

                self.input_size += len(chunk)
                chunk_num += 1
                # Apply this chunk to the pool
                self.pool.apply_async(self._process_chunk, (chunk_num, chunk))

    def _process_chunk(self, chunk_num: int, chunk: bytes):
        """
        Overall method to handle the chunk and pass it back to the write thread.
        This method is run on the pool.
        """
        with self._last_chunk_lock:
            last_chunk = chunk_num == self._last_chunk
        compressed_chunk = self._compress_chunk(chunk, last_chunk)
        self.chunk_queue.put((chunk_num, chunk, compressed_chunk))

    def _compress_chunk(self, chunk: bytes, is_last_chunk: bool):
        """
        Compress the chunk.
        """
        compressor = zlib.compressobj(
            level=self.compression_level,
            method=zlib.DEFLATED,
            wbits=-zlib.MAX_WBITS,
            memLevel=zlib.DEF_MEM_LEVEL,
            strategy=zlib.Z_DEFAULT_STRATEGY,
        )
        compressed_data = compressor.compress(chunk)
        if is_last_chunk:
            compressed_data += compressor.flush(zlib.Z_FINISH)
        else:
            compressed_data += compressor.flush(zlib.Z_SYNC_FLUSH)

        return compressed_data

    def _write_file(self):
        """
        Write compressed data to disk.
        Read chunks off of the priority queue.
        Priority is the chunk number, so we can keep track of which chunk to get next.
        This is run from the write thread.
        """
        next_chunk_num = 1
        while True:
            if not self.chunk_queue.empty():
                chunk_num, chunk, compressed_chunk = self.chunk_queue.get()

                if chunk_num != next_chunk_num:
                    # If this isn't the next chunk we're looking for,
                    # place it back on the queue and sleep
                    self.chunk_queue.put((chunk_num, chunk, compressed_chunk))
                    time.sleep(0.5)
                else:
                    # Calculate running checksum
                    self.calculate_chunk_check(chunk)
                    # Write chunk to file, advance next chunk we're looking for
                    self.output_file.write(compressed_chunk)
                    # If this was the last chunk,
                    # we can break the loop and close the file
                    if chunk_num == self._last_chunk:
                        break
                    next_chunk_num += 1
            else:
                # If the queue is empty, we're likely waiting for data.
                time.sleep(0.5)
        # Loop breaks out if we've received the final chunk
        self.clean_up()

    def calculate_chunk_check(self, chunk: bytes):
        """
        Calculate the check value for the chunk.
        """
        self.checksum = zlib.crc32(chunk, self.checksum)

    def clean_up(self):
        """
        Close the output file.
        Clean up the processing pool.
        """
        self.write_file_trailer()

        # Flush internal buffers
        self.output_file.flush()
        self.output_file.close()

        self._close_workers()

    def write_file_trailer(self):
        """
        Write the trailer for the compressed data.
        """
        # Write CRC32
        self.output_file.write((self.checksum).to_bytes(4, sys.byteorder))
        # Write ISIZE (Input SIZE)
        # This contains the size of the original (uncompressed) input data modulo 2^32.
        self.output_file.write(
            (self.input_size & 0xFFFFFFFF).to_bytes(4, sys.byteorder)
        )

    def _close_workers(self):
        """
        Close compression thread pool.
        """
        self.pool.close()
        self.pool.join()


def compress_file(
    source_file,
    output_file=None,
    inplace=False,
    compresslevel=_COMPRESS_LEVEL_BEST,
    blocksize=DEFAULT_BLOCK_SIZE_KB,
    workers=CPU_COUNT,
):
    """This function provides a interface to compress a file to .gz format parallelly.

    Args:
        source_file (str): filename you want to compress.
        output_file (str, optional): output compressed filename. If not give, mdapy will automatically append a .gz postfix.
        inplace (bool, optional): whether inplace the original file. Defaults to False.
        compresslevel (int, optional): 1 is fastest but worst, 9 is slowest but best. Defaults to 9.
        blocksize (int, optional): blocksize, generally do not need change. Defaults to 128 KB.
        workers (int, optional): number of threads to be used to do compression. Defaults to all your CPU cores.
    """
    assert isinstance(source_file, str)
    file_size = os.path.getsize(source_file) / 1024**2  # M
    if file_size < 4:
        workers = 1
    pigz_file = PigzFile(source_file, output_file, compresslevel, blocksize, workers)
    pigz_file.process_compression_target()
    if inplace:
        try:
            os.remove(source_file)
        except Exception:
            pass


if __name__ == "__main__":
    start = time.time()
    compress_file(r"E:/MyPackage/mdapy/test.dump")
    end = time.time()
    print(f"Time cost {end-start} s.")
