"""
KSO Mussel Farm BRUV (epibenthic fauna in low-trophic aquaculture, Swedish west coast)
Source: https://doi.org/10.5281/zenodo.14713926 (coolblue_LTA_model.zip, folder model_dataset)
Split logic: By sampling day. Each day has one farm and one control recording (8 recordings on 4 days,
    6 Mar, 12 Apr, 17 Apr and 4 May 2024), and frames of one recording look alike, so whole days go
    to train or val. 4 May (239 frames, 26%) -> val; the other days -> train. 4 May is the only day
    besides 12 Apr with fish and crabs together (6 Mar is almost only gobies, 12 Apr holds every
    hermit crab, 17 Apr only small snails).
Category mapping:
  fish: Gobiidae sp, Pleuronectiformes sp
  non-fish: Hyas araneus (spider crab), Paguridae sp (hermit crab), Portunoidea sp (swimming crabs)
  discard: Gastropoda sp (pale snails of 2-3 cm crawling on the mat; size doubtful)

910 frames (640x360, resized by Roboflow) from 30-minute recordings of a downward-facing GoPro over a
black mat with a bait box, at a blue-mussel farm and a control site (COOL BLUE project). The human
boxes (Roboflow YOLO export; one label is a polygon, converted to its box) miss about a third of the
animals over 3 cm, mostly gobies in busy frames and animals on the algae beside the mat. Only the
original human boxes are used.
Boxes are clipped to the image; boxes 1 pixel or less wide or high are dropped. Frames without
boxes are kept. File names drop the Roboflow suffix ("_jpg.rf.<hash>").
"""

import hashlib
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image

from datasets.settings import Settings
from datasets.utils import (
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "kso_mussel_bruv"
ZIP_URL = "https://zenodo.org/api/records/14713926/files/coolblue_LTA_model.zip/content"
ZIP_MD5 = "0c1cc691513213c1594732ec7802cf86"

# Classes of the Roboflow export, in label index order
SOURCE_CLASSES = ["Gastropoda sp", "Gobiidae sp", "Hyas araneus", "Paguridae sp", "Pleuronectiformes sp",
                  "Portunoidea sp"]
# Every source category must be listed here explicitly; unlisted categories are discarded.
CATEGORIES_FILTER = {
    "Gobiidae sp": "fish",  # gobies
    "Pleuronectiformes sp": "fish",  # flatfish
    "Hyas araneus": "non-fish",  # spider crab
    "Paguridae sp": "non-fish",  # hermit crabs
    "Portunoidea sp": "non-fish",  # swimming and shore crabs
}
# Discarded: Gastropoda sp (small pale snails, about 2-3 cm: size doubtful)

VALIDATION_DAYS = {"4_5"}  # 4 May 2024
MIN_BOX_SIZE = 1  # pixels
FILE_NAME = re.compile(r"^(?P<site>control|farm)_(?P<day>\d+_\d+)_frame_(?P<frame>\d+)_jpg\.rf\.[0-9a-f]+\.jpg$")

settings = Settings()

raw_data_dir = settings.raw_dir / DATASET_SHORTNAME
zip_path = raw_data_dir / "coolblue_LTA_model.zip"
source_dir = raw_data_dir / "model_dataset"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
source_annotations_path = processing_dir / "kso_mussel_bruv_source_classes_coco.json"
mapped_annotations_path = processing_dir / settings.coco_file_name


def md5sum(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_data():
    """Download coolblue_LTA_model.zip from Zenodo and extract the frames and labels (not the model or videos)."""
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    extracted_marker = raw_data_dir / ".extracted"
    if extracted_marker.exists():
        print(f"KSO data already extracted at {source_dir}")
        return
    download_file_resumable(ZIP_URL, zip_path)
    actual = md5sum(zip_path)
    if actual != ZIP_MD5:
        raise ValueError(f"MD5 mismatch for {zip_path}: expected {ZIP_MD5}, got {actual}. Delete it and rerun.")
    print("Extracting model_dataset/{train,valid,test}...")
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist()
                   if re.match(r"^model_dataset/(train|valid|test)/(images|labels)/", m)]
        z.extractall(raw_data_dir, members)
    extracted_marker.touch()


def yolo_box(values: list[float], width: int, height: int) -> list[float] | None:
    """[x, y, w, h] in pixels of a YOLO box (cx cy w h) or polygon (x y pairs), clipped to the image."""
    if len(values) == 4:
        cx, cy, w, h = values
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    else:
        xs, ys = values[0::2], values[1::2]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    x0, y0 = max(0.0, x0) * width, max(0.0, y0) * height
    x1, y1 = min(1.0, x1) * width, min(1.0, y1) * height
    if x1 - x0 <= MIN_BOX_SIZE or y1 - y0 <= MIN_BOX_SIZE:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def convert_kso_to_coco():
    """One COCO file with every frame (copied with the dataset prefix) and its human boxes."""
    if source_annotations_path.exists():
        print(f"Source-class COCO file already exists: {source_annotations_path}")
        return

    category_ids = {name: i for i, name in enumerate(SOURCE_CLASSES, start=1)}
    coco = {"images": [], "annotations": [],
            "categories": [{"id": cid, "name": name} for name, cid in category_ids.items()]}
    images_path.mkdir(parents=True, exist_ok=True)
    stats, seen = Counter(), set()
    frames = sorted((path, split) for split in ("train", "valid", "test")
                    for path in (source_dir / split / "images").glob("*.jpg"))
    for image_id, (path, split) in enumerate(frames, start=1):
        parts = FILE_NAME.match(path.name)
        original = f"{parts['site']}_{parts['day']}_frame_{parts['frame']}.jpg"
        if original in seen:
            raise ValueError(f"Frame {original} appears twice")
        seen.add(original)
        with Image.open(path) as image:
            width, height = image.size
        file_name = f"{DATASET_SHORTNAME}_{original}"
        shutil.copy2(path, images_path / file_name)
        coco["images"].append({"id": image_id, "file_name": file_name, "width": width, "height": height,
                               "site": parts["site"], "day": parts["day"], "frame": int(parts["frame"]),
                               "recording": f"{parts['site']}_{parts['day']}", "roboflow_split": split})

        for line in (source_dir / split / "labels" / (path.stem + ".txt")).read_text().splitlines():
            values = line.split()
            if not values:
                continue
            bbox = yolo_box([float(v) for v in values[1:]], width, height)
            if bbox is None:
                stats["human boxes dropped as too small"] += 1
                continue
            stats["polygon labels"] += len(values) > 5
            coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id,
                                        "category_id": category_ids[SOURCE_CLASSES[int(values[0])]],
                                        "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})

    print(f"Frames: {len(coco['images'])}, boxes: {len(coco['annotations'])}, {dict(stats)}")
    partial = source_annotations_path.with_name(source_annotations_path.name + ".part")
    partial.write_text(json.dumps(coco))
    partial.rename(source_annotations_path)


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)
    convert_kso_to_coco()
    map_annotations_to_fish_and_non_fish(source_annotations_path, CATEGORIES_FILTER, mapped_annotations_path)


def dataset_splitting():
    with open(mapped_annotations_path) as f:
        day = {image["file_name"]: image["day"] for image in json.load(f)["images"]}
    should_the_image_be_included_in_train_set = lambda image_path: day[Path(image_path).name] not in VALIDATION_DAYS

    train_dataset_path = settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    val_dataset_path = settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
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
