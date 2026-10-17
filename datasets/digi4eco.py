"""
DIGI4ECO Underwater Fish Tracking Dataset
Source: https://doi.org/10.5281/zenodo.20701225 (DIGI4ECO_tracking_dataset.zip, CVAT XML video tracks)
Split logic: By site. The Ancona sequence (Adriatic Sea) goes to val; the OBSEA sequence and
    the two SmartBay sequences (Galway Bay, two cameras) go to train.
Category mapping:
  fish: Fish, Apogon sp (2 tracks in SmartBay1, cardinalfish; crops checked)
  non-fish: none
  discard: none

The dataset has multi-object tracking annotations of 4 videos; only the boxes are used. Every
annotated frame is extracted (all of them have fish) and every box with outside="0" and
occluded="0" becomes a COCO box; track ids are dropped. Boxes with occluded="1" (9.4% of the
visible boxes, ~95% of them interpolated by CVAT) are dropped: in 96 random crops most of them
show no fish, only the pillar or rope the fish is behind, open water, or a closer fish that
hides it. Each COCO image entry has extra fields so that consecutive frames can be dropped
later: video (sequence name), frame_index (CVAT frame number) and site.

Frame alignment: Ancona, SmartBay1 and SmartBay2 decode to exactly the number of annotated
frames, so CVAT frame k is the k-th decoded frame. The OBSEA video decodes to 148 frames for
150 annotated ones: its timestamps lie on a 1/30 s grid with two gaps (at 33 ms and 4,933 ms),
so each decoded frame is matched to the CVAT frame at the same time on that grid, and CVAT
frames 1 and 148, which have no video frame, are skipped.
"""

import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

import cv2
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "digi4eco"
DATA_URL = "https://zenodo.org/records/20701225/files/DIGI4ECO_tracking_dataset.zip?download=1"
SOURCE_ZIP_NAME = "DIGI4ECO_tracking_dataset.zip"
SOURCE_ZIP_MD5 = "e91d8df3e9f6a18527bcb854f98df962"
SOURCE_FOLDER_NAME = "DIGI4ECO_tracking_dataset"

# Every source category must be listed here explicitly; unlisted categories are discarded.
CATEGORIES_FILTER = {
    "Fish": "fish",
    "Apogon sp": "fish",  # cardinalfish
}

# Sequence folder -> site
SEQUENCE_SITES = {
    "Ancona": "Ancona",
    "OBSEA": "OBSEA",
    "SmartBay1": "SmartBay",
    "SmartBay2": "SmartBay",
}
VALIDATION_SITES = {"Ancona"}

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
source_root = raw_data_dir / SOURCE_FOLDER_NAME
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
source_annotations_path = processing_dir / "digi4eco_source_classes_coco.json"
mapped_annotations_path = processing_dir / settings.coco_file_name


def md5sum(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_data():
    """Download (resuming broken downloads), check and extract the zip from Zenodo."""
    extracted_marker = raw_data_dir / f".{SOURCE_FOLDER_NAME}_extracted"
    if extracted_marker.exists():
        print(f"Dataset already extracted at {source_root}")
        return

    raw_data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_data_dir / SOURCE_ZIP_NAME
    download_file_resumable(DATA_URL, zip_path)

    print(f"Checking MD5 of {zip_path}...")
    actual_md5 = md5sum(zip_path)
    if actual_md5 != SOURCE_ZIP_MD5:
        raise ValueError(
            f"MD5 mismatch for {zip_path}: expected {SOURCE_ZIP_MD5}, got {actual_md5}. "
            "Delete it and rerun."
        )

    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path) as zipf:
        zipf.extractall(raw_data_dir)
    extracted_marker.touch()


def read_cvat_tracks(xml_path: Path) -> tuple[int, dict[int, list[tuple]]]:
    """
    Read a CVAT for video 1.1 file. Returns the number of annotated frames and
    {frame number: [(label, xtl, ytl, xbr, ybr)]} with the boxes where the fish is visible and
    not occluded.
    """
    root = ET.parse(xml_path).getroot()
    frame_count = int(root.findtext("meta/job/size"))
    expected_box_count = int(root.findtext("meta/sequence_info/annotation_stats/num_bounding_boxes"))

    boxes_by_frame = {}
    visible_box_count = 0
    for track in root.findall("track"):
        label = track.get("label")
        for box in track.findall("box"):
            if box.get("outside") == "1":
                continue
            visible_box_count += 1
            if box.get("occluded") == "1":
                continue
            frame_number = int(box.get("frame"))
            if not 0 <= frame_number < frame_count:
                raise ValueError(f"Frame {frame_number} outside [0, {frame_count - 1}] in {xml_path}")
            coords = tuple(float(box.get(key)) for key in ("xtl", "ytl", "xbr", "ybr"))
            boxes_by_frame.setdefault(frame_number, []).append((label, *coords))

    # annotation_stats counts the occluded boxes too
    if visible_box_count != expected_box_count:
        raise ValueError(
            f"{xml_path} has {visible_box_count} visible boxes, its annotation_stats say {expected_box_count}"
        )
    kept_box_count = sum(len(boxes) for boxes in boxes_by_frame.values())
    print(
        f"{xml_path.parent.name}: dropped {visible_box_count - kept_box_count} occluded "
        f"of {visible_box_count} visible boxes"
    )
    return frame_count, boxes_by_frame


def get_cvat_frame_number_of_each_video_frame(video_path: Path, cvat_frame_count: int) -> list[int]:
    """
    The CVAT frame number of each decoded frame of the video (see "Frame alignment" above).
    """
    cap = cv2.VideoCapture(str(video_path))
    timestamps_ms = []
    while cap.grab():
        timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    cap.release()

    if len(timestamps_ms) == cvat_frame_count:
        return list(range(cvat_frame_count))
    if len(timestamps_ms) > cvat_frame_count:
        raise ValueError(f"{video_path} has {len(timestamps_ms)} frames, more than the {cvat_frame_count} annotated")

    # Missing frames: place each decoded frame on the video's frame-time grid
    frame_interval_ms = statistics.median(b - a for a, b in zip(timestamps_ms, timestamps_ms[1:]))
    grid_positions = [timestamp / frame_interval_ms for timestamp in timestamps_ms]
    frame_numbers = [round(position) for position in grid_positions]
    if (
        any(abs(position - number) > 0.1 for position, number in zip(grid_positions, frame_numbers))
        or len(set(frame_numbers)) != len(frame_numbers)
        or frame_numbers[-1] >= cvat_frame_count
    ):
        raise ValueError(f"Cannot match the frames of {video_path} to its {cvat_frame_count} annotated frames")
    return frame_numbers


def convert_digi4eco_to_coco():
    """
    Extract the frames of each video and convert the visible track boxes to COCO with the source
    labels as categories, writing the images with the dataset prefix.
    """
    if source_annotations_path.exists():
        print(f"Source-class COCO file already exists: {source_annotations_path}")
        return

    images_path.mkdir(parents=True, exist_ok=True)
    sequences = {}
    for sequence in SEQUENCE_SITES:
        sequences[sequence] = read_cvat_tracks(source_root / sequence / "annotations.xml")

    labels = sorted(
        {
            box[0]
            for _, boxes_by_frame in sequences.values()
            for boxes in boxes_by_frame.values()
            for box in boxes
        }
    )
    category_ids = {label: category_id for category_id, label in enumerate(labels, start=1)}
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": category_id, "name": label} for label, category_id in category_ids.items()],
    }
    box_counts = Counter()

    for sequence, (cvat_frame_count, boxes_by_frame) in sequences.items():
        video_path = source_root / sequence / "video.mp4"
        frame_numbers = get_cvat_frame_number_of_each_video_frame(video_path, cvat_frame_count)
        skipped = sorted(set(range(cvat_frame_count)) - set(frame_numbers))
        if skipped:
            print(f"{sequence}: annotated frames without a video frame, skipped: {skipped}")

        cap = cv2.VideoCapture(str(video_path))
        for frame_number in tqdm(frame_numbers, desc=f"Extracting {sequence} frames"):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Failed to read frame {frame_number} of {video_path}")
            image_height, image_width = frame.shape[:2]

            file_name = f"{DATASET_SHORTNAME}_{sequence}_frame_{frame_number:05d}.jpg"
            cv2.imwrite(str(images_path / file_name), frame)
            image_id = len(coco_data["images"]) + 1
            coco_data["images"].append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": image_width,
                    "height": image_height,
                    "video": sequence,
                    "frame_index": frame_number,
                    "site": SEQUENCE_SITES[sequence],
                }
            )

            for label, xtl, ytl, xbr, ybr in boxes_by_frame.get(frame_number, []):
                x_min, y_min = max(0.0, xtl), max(0.0, ytl)
                x_max, y_max = min(float(image_width), xbr), min(float(image_height), ybr)
                if x_max <= x_min or y_max <= y_min:
                    raise ValueError(f"Empty box in {sequence} frame {frame_number}")

                box_counts[label] += 1
                coco_data["annotations"].append(
                    {
                        "id": len(coco_data["annotations"]) + 1,
                        "image_id": image_id,
                        "category_id": category_ids[label],
                        "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "area": (x_max - x_min) * (y_max - y_min),
                        "iscrowd": 0,
                    }
                )
        cap.release()

    print(f"Images: {len(coco_data['images'])}, boxes per source label: {dict(box_counts)}")
    partial = source_annotations_path.with_name(source_annotations_path.name + ".part")
    with open(partial, "w") as f:
        json.dump(coco_data, f)
    partial.rename(source_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    convert_digi4eco_to_coco()
    map_annotations_to_fish_and_non_fish(
        source_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )


def get_site(image_filename: str) -> str:
    """Site of a frame, from its sequence name: "digi4eco_SmartBay1_frame_00042.jpg" -> "SmartBay"."""
    stem = Path(image_filename).stem.removeprefix(f"{DATASET_SHORTNAME}_")
    return SEQUENCE_SITES[stem.rsplit("_frame_", 1)[0]]


def dataset_splitting():
    should_the_image_be_included_in_train_set = (
        lambda image_path: get_site(image_path) not in VALIDATION_SITES
    )

    train_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    )
    val_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    )
    train_dataset_path.mkdir(parents=True)
    val_dataset_path.mkdir(parents=True)

    split_coco_dataset_into_train_validation(
        images_path,
        mapped_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )


def main():
    # 1. DOWNLOAD
    download_data()

    # 2. PROCESS
    processing()

    # 3. PREVIEW
    save_preview_image(images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
