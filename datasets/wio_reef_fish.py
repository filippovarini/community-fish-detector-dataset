"""
WIO-ReefFish Dataset
Source: https://doi.org/10.5281/zenodo.21360120 (WIO-ReefFish_v1.0.0.zip, YOLO boxes)
Split logic: By location. Each image's video is looked up in the authors' transect split
    file (alternative_splits/transect/VIDEO_ASSIGNMENT.csv), which gives the location of
    every video. Diani (Kenya) and Mikindani (Tanzania) go to val; the other 8 locations
    go to train. The authors' own train/valid/test split is random by frame, so frames of
    the same video end up on both sides and it isn't used.
    Videos GX011330, GX011332 and GX011334 (12 images) are missing from the split files.
    Their GoPro file numbers fall between the Kunduchi and Kilwa videos (both Tanzania),
    so they go to train, and neither Kunduchi nor Kilwa is a val location.
Category mapping:
  fish: all 24 classes (23 reef fish families and the parrotfish subfamily Scarinae)
  non-fish: none
  discard: none

The YOLO class ids follow the order of the names in data.yaml; the conversion checks the
box count per class against metadata/annotations_summary.csv.
"""

import csv
import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "wio_reef_fish"
DATA_URL = "https://zenodo.org/records/21360120/files/WIO-ReefFish_v1.0.0.zip?download=1"
SOURCE_ZIP_NAME = "WIO-ReefFish_v1.0.0.zip"
SOURCE_ZIP_MD5 = "629c6b7250bcd01c2b03707db5d449c8"

# YOLO class ids 0-23, in data.yaml order
SOURCE_CLASSES = [
    "Acanthuridae",  # surgeonfishes
    "Anthiadidae",  # anthias
    "Aulostomidae",  # trumpetfishes
    "Balistidae",  # triggerfishes
    "Caesionidae",  # fusiliers
    "Chaetodontidae",  # butterflyfishes
    "Epinephelidae",  # groupers
    "Haemulidae",  # grunts, sweetlips
    "Holocentridae",  # squirrelfishes, soldierfishes
    "Kyphosidae",  # sea chubs
    "Labridae",  # wrasses
    "Lethrinidae",  # emperors
    "Lutjanidae",  # snappers
    "Microdesmidae",  # dartfishes
    "Monacanthidae",  # filefishes
    "Mullidae",  # goatfishes
    "Pempheridae",  # sweepers
    "Pomacanthidae",  # angelfishes
    "Pomacentridae",  # damselfishes
    "Priacanthidae",  # bigeyes
    "Scarinae",  # parrotfishes
    "Siganidae",  # rabbitfishes
    "Tetraodontidae",  # pufferfishes
    "Zanclidae",  # Moorish idol
]
CATEGORIES_FILTER = {name: "fish" for name in SOURCE_CLASSES}

# Locations assigned to the validation set
VALIDATION_LOCATIONS = {"Diani", "Mikindani"}

SOURCE_SPLITS = ["train", "valid", "test"]

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
source_root = raw_data_dir / "WIO-ReefFish_v1.0.0" / "WIO-ReefFish"
video_assignment_path = (
    source_root / "alternative_splits" / "transect" / "VIDEO_ASSIGNMENT.csv"
)
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
source_annotations_path = processing_dir / "wio_reef_fish_source_classes_coco.json"
mapped_annotations_path = processing_dir / settings.coco_file_name


def md5sum(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_data():
    """Download (resuming broken downloads), check and extract the zip from Zenodo."""
    extracted_marker = raw_data_dir / ".extracted"
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


def get_video_id(image_filename: str) -> str:
    """
    Video id of an image, e.g. GX010034_frame_044_jpg.rf.<hash>.jpg -> GX010034
    (same rule as the authors' scripts/apply_alternative_split.py).
    """
    stem = Path(image_filename).stem
    if stem.startswith(f"{DATASET_SHORTNAME}_"):
        stem = stem[len(f"{DATASET_SHORTNAME}_") :]
    stem = re.sub(r"_jpg\.rf\.[a-f0-9]+$", "", stem)
    match = re.match(r"^(.+)_frame_\d+$", stem)
    if match is None:
        raise ValueError(f"Can't read the video id from {image_filename}")
    return match.group(1)


def load_video_locations() -> dict[str, str]:
    """{video_id: location} from the authors' transect split file."""
    with open(video_assignment_path, encoding="utf-8") as f:
        return {row["video_id"]: row["location"] for row in csv.DictReader(f)}


def load_expected_box_counts() -> dict[str, int]:
    """{class_name: total boxes} from metadata/annotations_summary.csv."""
    with open(source_root / "metadata" / "annotations_summary.csv") as f:
        return {row["class_name"]: int(row["n_total"]) for row in csv.DictReader(f)}


def parse_yolo_label_file(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """
    Read a list of [class_id, x_center, y_center, box_width, box_height] bounding
    boxes from the .txt file [label_path].
    """
    rows = []
    with open(label_path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            fields = line.split()
            if len(fields) != 5:
                raise ValueError(
                    f"Malformed YOLO row at {label_path}:{line_number}: {line}"
                )

            class_id = int(fields[0])
            x_center, y_center, box_width, box_height = [
                float(value) for value in fields[1:]
            ]
            if not 0 <= class_id < len(SOURCE_CLASSES):
                raise ValueError(
                    f"Class id {class_id} outside [0, {len(SOURCE_CLASSES) - 1}] at "
                    f"{label_path}:{line_number}"
                )

            rows.append((class_id, x_center, y_center, box_width, box_height))

    return rows


def yolo_box_to_coco_box(
    x_center: float,
    y_center: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> list[float]:
    """
    Convert a YOLO box (normalized x_center, y_center, w, h) to COCO (x_min, y_min, w, h)
    in pixels, clipped to the image.
    """
    x_min = max(0.0, min(float(image_width), (x_center - box_width / 2) * image_width))
    y_min = max(0.0, min(float(image_height), (y_center - box_height / 2) * image_height))
    x_max = max(0.0, min(float(image_width), (x_center + box_width / 2) * image_width))
    y_max = max(0.0, min(float(image_height), (y_center + box_height / 2) * image_height))

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def convert_wio_reef_fish_to_coco():
    """
    Convert the YOLO labels of the authors' train/valid/test folders to one COCO file
    with the source classes as categories (id = YOLO class id + 1), copying the images
    with the dataset prefix.
    """
    if source_annotations_path.exists():
        print(f"Source-class COCO file already exists: {source_annotations_path}")
        return

    images_path.mkdir(parents=True, exist_ok=True)
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": class_id + 1, "name": name}
            for class_id, name in enumerate(SOURCE_CLASSES)
        ],
    }
    box_counts = Counter()

    source_images = sorted(
        path
        for split in SOURCE_SPLITS
        for path in (source_root / split / "images").iterdir()
        if path.suffix.lower() == ".jpg"
    )
    print(f"Found {len(source_images)} images")

    for image_id, source_image_path in tqdm(
        enumerate(source_images, start=1),
        total=len(source_images),
        desc="Converting YOLO to COCO",
    ):
        label_path = (
            source_image_path.parent.parent / "labels" / f"{source_image_path.stem}.txt"
        )
        if not label_path.exists():
            raise FileNotFoundError(f"No label file for {source_image_path}")

        with Image.open(source_image_path) as image:
            image_width, image_height = image.size

        file_name = f"{DATASET_SHORTNAME}_{source_image_path.name}"
        shutil.copy2(source_image_path, images_path / file_name)
        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": image_width,
                "height": image_height,
            }
        )

        for class_id, x_center, y_center, box_width, box_height in parse_yolo_label_file(
            label_path
        ):
            bbox = yolo_box_to_coco_box(
                x_center, y_center, box_width, box_height, image_width, image_height
            )
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(f"Box collapsed to zero area in {source_image_path.name}")

            box_counts[SOURCE_CLASSES[class_id]] += 1
            coco_data["annotations"].append(
                {
                    "id": len(coco_data["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )

    # The box counts in annotations_summary.csv confirm the class order
    mismatched = {
        name: (count, box_counts[name])
        for name, count in load_expected_box_counts().items()
        if box_counts[name] != count
    }
    if mismatched:
        raise ValueError(f"Box counts differ from annotations_summary.csv: {mismatched}")

    partial = source_annotations_path.with_name(source_annotations_path.name + ".part")
    with open(partial, "w") as f:
        json.dump(coco_data, f)
    partial.rename(source_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    convert_wio_reef_fish_to_coco()
    map_annotations_to_fish_and_non_fish(
        source_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )


def dataset_splitting():
    video_locations = load_video_locations()
    unknown_videos = set()

    def should_the_image_be_included_in_train_set(image_filename: str) -> bool:
        video_id = get_video_id(image_filename)
        location = video_locations.get(video_id)
        if location is None:
            # Not in the authors' split files; see the module docstring
            unknown_videos.add(video_id)
            return True
        return location not in VALIDATION_LOCATIONS

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
    print(f"Videos without a location (put in train): {sorted(unknown_videos)}")


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
