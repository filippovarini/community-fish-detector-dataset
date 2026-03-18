"""
VIAME FishTrack Dataset
Source: https://viame.kitware.com/
Split logic: Pre-split by URL (train and val downloaded separately)
Categories kept: All fish (non_fish categories excluded at download time)

Converts VIAME CSV annotations to COCO format. Extracts frames from videos.
No split function needed as train/val are separate downloads.
"""

import json
import random
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import pandas as pd
import supervision as sv

from datasets.settings import Settings
from datasets.utils import download_and_extract


DATASET_SHORTNAME = "viame_fishtrack"
TESTING = False
all_species = set()

settings = Settings()


def _is_non_fish(species: str) -> bool:
    return species.startswith("non_fish")


def build_image_id(video_path: Path, frame_id: str) -> str:
    """Generate a unique identifier for a video frame."""
    return f"{video_path.stem}_{frame_id}"


def timestamp_to_milliseconds(timestamp_str: str) -> int:
    """Convert a timestamp string in format HH:MM:SS.ffffff to milliseconds."""
    hours, minutes, seconds = timestamp_str.split(":")
    seconds, microseconds = seconds.split(".")

    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    microseconds = int(microseconds.ljust(6, "0"))

    total_milliseconds = (
        hours * 3600 * 1000
        + minutes * 60 * 1000
        + seconds * 1000
        + microseconds // 1000
    )
    return total_milliseconds


def extract_frame(
    frames_path: Path, video_path: Path, timestamp_str: str, frame_id: str
):
    """Extract a frame from a video at the specified timestamp."""
    filename = f"{frame_id}.jpg"
    output_path = frames_path / filename
    frames_path.mkdir(parents=True, exist_ok=True)

    milliseconds = timestamp_to_milliseconds(timestamp_str)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, milliseconds)
    ret, frame = cap.read()
    if not ret:
        raise ValueError(
            f"Failed to read frame {frame_id} from {video_path} at {timestamp_str}"
        )

    height, width, _ = frame.shape
    if not output_path.exists():
        cv2.imwrite(str(output_path), frame)

    cap.release()
    return filename, height, width


def get_frame_from_video(
    row: pd.Series,
    video_path: Path,
    output_frames_path: Path,
    coco_data: dict,
    image_ids: set,
):
    """Extract frame from video and add to COCO dataset."""
    frame_timestamp = row["2: Video or Image Identifier"]
    frame_id = build_image_id(video_path, frame_timestamp)

    if frame_id not in image_ids:
        image_ids.add(frame_id)
        image_filename, image_height, image_width = extract_frame(
            output_frames_path, video_path, frame_timestamp, frame_id
        )
        coco_data["images"].append(
            {
                "id": frame_id,
                "file_name": str(image_filename),
                "height": image_height,
                "width": image_width,
            }
        )
    return frame_id


def get_frame_from_images(
    row: pd.Series,
    camera_path: Path,
    output_frames_path: Path,
    coco_data: dict,
    image_ids: set,
):
    """Extract frame from images and add to COCO dataset."""
    annotation_frame_id = row["2: Video or Image Identifier"]
    frame_id = build_image_id(camera_path, annotation_frame_id)

    if frame_id not in image_ids:
        frame_path = camera_path / annotation_frame_id
        assert frame_path.exists(), f"Frame not found: {frame_path}"

        height, width, _ = cv2.imread(str(frame_path)).shape

        image_ids.add(frame_id)
        new_frame_path = output_frames_path / frame_id
        shutil.copy(frame_path, new_frame_path)

        coco_data["images"].append(
            {
                "id": frame_id,
                "file_name": str(new_frame_path),
                "height": height,
                "width": width,
            }
        )
    return frame_id


def viame_to_coco(camera_path: Path, images_dir: Path, coco_data: dict):
    """Converts VIAME annotations to COCO format."""
    global all_species

    csv_path = camera_path / "annotations.viame.csv"
    assert csv_path.exists(), f"CSV file not found: {csv_path}"

    is_video = True
    video_path = camera_path / f"{camera_path.name}.mp4"
    if not video_path.exists():
        print(f"Video file not found: {video_path}, trying to use png images instead")
        images_available = len(list(camera_path.glob("*.png")))
        assert images_available > 0, f"No png images found in {camera_path}"
        is_video = False
        print(f"Using {images_available} png images instead of video")

    df = pd.read_csv(csv_path, skiprows=lambda x: x in [1])

    output_frames_path = images_dir
    output_frames_path.mkdir(parents=True, exist_ok=True)

    image_ids = set()
    annotation_id = 1

    for index, row in df.iterrows():
        if TESTING and index > 100:
            break

        species = row["10-11+: Repeated Species"]
        if species not in all_species:
            all_species.add(species)
            print(f"Processing species: {species}")
        if not pd.notna(species) or _is_non_fish(species):
            print(f"Skipping row because of non-fish category: {species}")
            continue

        try:
            if is_video:
                frame_id = get_frame_from_video(
                    row, video_path, output_frames_path, coco_data, image_ids
                )
            else:
                frame_id = get_frame_from_images(
                    row, camera_path, output_frames_path, coco_data, image_ids
                )
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

        bbox_cols = ["4-7: Img-bbox(TL_x", "TL_y", "BR_x", "BR_y)"]
        assert all(col in row.index for col in bbox_cols)
        assert not any(pd.isna(row[col]) for col in bbox_cols)

        xmin = float(row["4-7: Img-bbox(TL_x"])
        ymin = float(row["TL_y"])
        xmax = float(row["BR_x"])
        ymax = float(row["BR_y)"])

        width = xmax - xmin
        height = ymax - ymin

        coco_data["annotations"].append(
            {
                "id": annotation_id,
                "image_id": frame_id,
                "category_id": 1,
                "bbox": [xmin, ymin, width, height],
                "area": width * height,
                "iscrowd": 0,
            }
        )
        annotation_id += 1


def download_data_and_build_coco_dataset(
    raw_data_download_path: Path, coco_dataset_path: Path, data_url: str
) -> Tuple[Path, Path]:
    """Downloads VIAME dataset and builds a COCO dataset."""
    downloaded_data_path = download_and_extract(
        raw_data_download_path, data_url, DATASET_SHORTNAME
    )

    fish_category = {"id": 1, "name": "fish"}
    coco_data = {"images": [], "annotations": [], "categories": [fish_category]}

    images_output_path = coco_dataset_path / "JPEGImages"
    images_output_path.mkdir()

    for camera_path in downloaded_data_path.glob("*"):
        print(f"Processing camera: {camera_path}...")
        if not camera_path.is_dir():
            continue
        viame_to_coco(camera_path, images_output_path, coco_data)
        print(f"Images loaded in coco dataset: {len(coco_data['images'])}")

    annotations_path = coco_dataset_path / "annotations_coco.json"
    with open(annotations_path, "w") as f:
        json.dump(coco_data, f)

    return images_output_path, annotations_path


def setup_raw_processed_directories_for_dataset(dataset_name: str) -> tuple[Path, Path]:
    raw_data_path = settings.raw_dir / dataset_name
    processed_data_path = settings.processed_dir / dataset_name
    return raw_data_path, processed_data_path


def main():
    """
    Downloads the VIAME FishTrack data.
    No need to split in train and val, as the VIAME FishTrack data is already split.
    No need to compress the annotations into fish only, as the
    download_data_and_build_coco_dataset function already does this.
    """
    # Download the train data
    train_data_name = f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    train_raw_data_path, train_coco_dataset_path = (
        setup_raw_processed_directories_for_dataset(train_data_name)
    )
    train_raw_data_path.mkdir(parents=True, exist_ok=True)
    train_coco_dataset_path.mkdir(parents=True, exist_ok=True)

    train_data_url = "https://viame.kitware.com/api/v1/dive_dataset/export?folderIds=[%2265a19f85cf5a99794ea9ccfb%22,%2265a1a15fcf5a99794eaaa790%22,%2265a1a028cf5a99794eaa2419%22,%2265a19f70cf5a99794ea9c1f7%22,%2265a19f59cf5a99794ea9b5b4%22,%2265a19f70cf5a99794ea9c20c%22,%2265a1a160cf5a99794eaaa7e7%22,%2265a1a123cf5a99794eaa925a%22,%2265a19f85cf5a99794ea9cd00%22,%2265a1a040cf5a99794eaa3185%22,%2265a19f9bcf5a99794ea9d8e3%22,%2265a1a13acf5a99794eaa9c17%22,%2265a1a16dcf5a99794eaaabd2%22,%2265a1a160cf5a99794eaaa7db%22,%2265a1a162cf5a99794eaaa858%22,%2265a1a11bcf5a99794eaa8dbb%22,%2265a19f83cf5a99794ea9cc04%22,%2265a19fcecf5a99794ea9f433%22,%2265a1a144cf5a99794eaa9f0c%22,%2265a1a0dccf5a99794eaa7ac8%22]"
    download_data_and_build_coco_dataset(
        raw_data_download_path=train_raw_data_path,
        coco_dataset_path=train_coco_dataset_path,
        data_url=train_data_url,
    )

    # Download the val data
    val_data_name = f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    val_raw_data_path, val_coco_dataset_path = (
        setup_raw_processed_directories_for_dataset(val_data_name)
    )
    val_raw_data_path.mkdir(parents=True, exist_ok=True)
    val_coco_dataset_path.mkdir(parents=True, exist_ok=True)

    val_data_url = "https://viame.kitware.com/api/v1/dive_dataset/export?folderIds=[%2265a1a1d1cf5a99794eaacb57%22,%2265a1a291cf5a99794eab01fb%22,%2265a1a205cf5a99794eaadbb6%22,%2265a1a223cf5a99794eaae509%22,%2265a1a20ccf5a99794eaadddd%22,%2265a1a1d1cf5a99794eaacb3d%22,%2265a1a23ecf5a99794eaaed79%22,%2265a1a20ccf5a99794eaadde0%22,%2265a1a223cf5a99794eaae50e%22,%2265a1a1d1cf5a99794eaacb52%22,%2265a1a28fcf5a99794eab01b2%22,%2265a1a22fcf5a99794eaae8c1%22,%2265a1a205cf5a99794eaadbbb%22,%2265a1a1ffcf5a99794eaad9c8%22,%2265a1a1d8cf5a99794eaacd93%22,%2265a1a1f1cf5a99794eaad548%22,%2265a1a1d1cf5a99794eaacb67%22,%2265a1a23ecf5a99794eaaed82%22,%2265a1a230cf5a99794eaae92a%22,%2265a1a244cf5a99794eaaef6b%22]"
    download_data_and_build_coco_dataset(
        raw_data_download_path=val_raw_data_path,
        coco_dataset_path=val_coco_dataset_path,
        data_url=val_data_url,
    )


if __name__ == "__main__":
    main()
