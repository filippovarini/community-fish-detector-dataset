"""
PomerFish Dataset
Source: https://zenodo.org/records/17432128
Split logic: By video/deployment ID (extracted from filename pattern dataset_<id>_frame_<n>.PNG)
Categories kept: All (all 10 categories are freshwater fish species)
"""

import json
import re
import subprocess
from pathlib import Path

from sklearn.model_selection import train_test_split

from datasets.settings import Settings
from datasets.utils import (
    compress_annotations_to_single_category,
    add_dataset_shortname_prefix_to_image_names,
    split_coco_dataset_into_train_validation,
    copy_images_to_processing,
    save_preview_image,
    download_file,
)

DATASET_SHORTNAME = "pomerfish"
CATEGORIES_FILTER = None  # All categories are freshwater fish species

DATA_URL = "https://zenodo.org/records/17432128/files/PomerFish.rar?download=1"
RAR_FILENAME = "PomerFish.rar"

settings = Settings()


def download_data(download_path: Path):
    """Download and extract the PomerFish RAR archive."""
    rar_path = download_path / RAR_FILENAME
    extracted_dir = download_path / Path(RAR_FILENAME).stem

    if extracted_dir.exists():
        print(f"Dataset already extracted at {extracted_dir}")
        return extracted_dir / "PomerFishObj"

    print(f"Downloading {RAR_FILENAME}...")
    download_file(DATA_URL, rar_path)

    print("Extracting RAR archive with unar...")
    subprocess.run(
        ["unar", "-o", str(download_path), "-f", str(rar_path)],
        check=True,
    )

    return extracted_dir / "PomerFishObj"


def get_train_video_ids(images_path: Path) -> set:
    """Split by unique video/deployment IDs (80/20)."""
    video_ids = set()
    for img in images_path.glob("*"):
        if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            name = img.stem.replace(f"{DATASET_SHORTNAME}_", "", 1)
            m = re.match(r"dataset_(\d+)_frame", name)
            if m:
                video_ids.add(m.group(1))

    train_ids, _ = train_test_split(
        sorted(video_ids),
        test_size=settings.train_val_split_ratio,
        random_state=settings.random_state,
    )
    return set(train_ids)


def main():
    # 1. DOWNLOAD
    raw_download_path = settings.raw_dir / DATASET_SHORTNAME
    raw_download_path.mkdir(parents=True, exist_ok=True)
    pomerfish_obj_dir = download_data(raw_download_path)

    raw_images_path = pomerfish_obj_dir / "images"
    raw_annotations_path = pomerfish_obj_dir / "combined_standard.json"

    # 2. PROCESS
    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    images_path = copy_images_to_processing(DATASET_SHORTNAME, raw_images_path)

    compressed_annotations_path = processing_dir / "annotations_coco_compressed.json"
    if not compressed_annotations_path.exists():
        compress_annotations_to_single_category(
            raw_annotations_path, CATEGORIES_FILTER, compressed_annotations_path
        )

    add_dataset_shortname_prefix_to_image_names(
        images_path,
        compressed_annotations_path,
        DATASET_SHORTNAME,
    )

    # 3. PREVIEW
    save_preview_image(images_path, compressed_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_video_ids = get_train_video_ids(images_path)

    def is_train_image(image_filename: str) -> bool:
        name = Path(image_filename).stem.replace(f"{DATASET_SHORTNAME}_", "", 1)
        m = re.match(r"dataset_(\d+)_frame", name)
        return m.group(1) in train_video_ids if m else False

    train_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    )
    val_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    )
    train_dataset_path.mkdir(parents=True, exist_ok=True)
    val_dataset_path.mkdir(parents=True, exist_ok=True)

    split_coco_dataset_into_train_validation(
        images_path,
        compressed_annotations_path,
        train_dataset_path,
        val_dataset_path,
        is_train_image,
    )


if __name__ == "__main__":
    main()
