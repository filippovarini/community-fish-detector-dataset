import io
import requests
import zipfile
import tarfile
import time
from pathlib import Path
from enum import Enum
from tqdm import tqdm


class CompressionType(Enum):
    ZIP = "zip"
    TAR = "tar"


def download_file(url: str, save_path: Path):
    print(f"Downloading {url} to {save_path}...")
    # Use a session for connection pooling
    with requests.Session() as session:
        response = session.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Get file size for progress bar
        total_size = int(response.headers.get("content-length", 0))

        # Initialize tqdm with total file size
        with (
            open(save_path, "wb") as f,
            tqdm(
                desc=save_path.name,
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar,
        ):
            # Increased chunk size from 8192 to 1MB for faster downloads
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Download complete: {save_path}")


def download_file_resumable(url: str, save_path: Path, max_attempts: int = 20):
    """
    Download url to save_path through a save_path.part file. A broken connection is
    resumed from the end of the .part file with an HTTP Range request (up to max_attempts
    times, and also across reruns); save_path only appears once the download is complete.
    """
    if save_path.exists():
        print(f"Already downloaded: {save_path}")
        return

    partial = save_path.with_name(save_path.name + ".part")
    for attempt in range(1, max_attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        print(f"Downloading {url} to {save_path} (attempt {attempt}, from byte {offset})")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60) as response:
                if response.status_code == 416:  # the .part file is already complete
                    break
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError(f"Server ignored the Range header; delete {partial} and rerun")
                total = offset + int(response.headers.get("content-length", 0))
                with (
                    open(partial, "ab") as f,
                    tqdm(desc=save_path.name, total=total, initial=offset, unit="B", unit_scale=True, unit_divisor=1024) as pbar,
                ):
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        pbar.update(len(chunk))
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            print(f"Connection broken: {e}; retrying in 30 s")
            time.sleep(30)
            continue
        if partial.stat().st_size == total:
            break
    else:
        raise RuntimeError(f"Download of {url} incomplete after {max_attempts} attempts; rerun to resume")

    partial.rename(save_path)


def extract_downloaded_file(
    download_path: Path,
    extract_to: Path,
    compression_type: CompressionType = CompressionType.ZIP,
):
    print(f"Extracting {download_path} to {extract_to}")

    if not download_path.exists():
        print("Compressed file not found")
        return

    match compression_type:
        case CompressionType.ZIP:
            with zipfile.ZipFile(download_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)
            print(f"Extraction complete: {download_path}")
        case CompressionType.TAR:
            with tarfile.open(download_path, "r") as tar_ref:
                tar_ref.extractall(extract_to)
            print(f"Extraction complete: {download_path}")
        case _:
            raise ValueError(f"Unsupported compression type: {compression_type}")

    print("Removing Zipped files...")
    download_path.unlink()


def download_and_extract(
    data_dir: Path,
    data_url: str,
    dataset_shortname: str,
    compression_type: CompressionType = CompressionType.ZIP,
):
    """
    Download and extract a dataset from a URL.
    """
    download_path = data_dir / f"{dataset_shortname}.{compression_type.value}"

    print("Downloading data...")
    download_file(data_url, download_path)
    print("Extracting data...")
    extract_downloaded_file(download_path, data_dir, compression_type)

    return data_dir


class HttpRangeFile(io.RawIOBase):
    """
    Read-only, seekable file over HTTP Range requests, so zipfile.ZipFile can list a remote
    zip and extract single members without downloading the whole archive. Reads are served
    from a small cache of block_size blocks, so extracting members in archive order costs
    about one request per block.

    get_request returns (url, headers) and is called again when the server answers 401/403,
    e.g. because a signed URL or bearer token has expired.
    """

    def __init__(self, get_request, block_size: int = 16 * 1024 * 1024, cache_blocks: int = 4, max_attempts: int = 10):
        self.get_request = get_request
        self.block_size = block_size
        self.cache_blocks = cache_blocks
        self.max_attempts = max_attempts
        self.session = requests.Session()
        self.url, self.headers = get_request()
        self.cache: dict[int, bytes] = {}
        self.position = 0
        response = self._get_range(0, 0)
        self.size = int(response.headers["Content-Range"].split("/")[-1])

    def _get_range(self, start: int, end: int) -> requests.Response:
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.get(
                    self.url, headers={**self.headers, "Range": f"bytes={start}-{end}"}, timeout=120
                )
                if response.status_code in (401, 403):
                    self.url, self.headers = self.get_request()
                    continue
                response.raise_for_status()
                if response.status_code != 206:
                    raise RuntimeError(f"Server ignored the Range header for {self.url}")
                return response
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
                print(f"Connection broken ({e}); retrying in 30 s")
                time.sleep(30)
        raise RuntimeError(f"Range request {start}-{end} failed after {self.max_attempts} attempts")

    def _block(self, index: int) -> bytes:
        if index not in self.cache:
            if len(self.cache) >= self.cache_blocks:
                self.cache.pop(next(iter(self.cache)))
            start = index * self.block_size
            end = min(start + self.block_size, self.size) - 1
            self.cache[index] = self._get_range(start, end).content
        return self.cache[index]

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.position
        n = min(n, self.size - self.position)
        chunks = []
        while n > 0:
            index, offset = divmod(self.position, self.block_size)
            chunk = self._block(index)[offset : offset + n]
            chunks.append(chunk)
            self.position += len(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)
