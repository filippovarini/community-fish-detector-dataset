import requests
import zipfile
import tarfile
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
