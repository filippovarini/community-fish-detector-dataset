"""
EyeSea Underwater Video (PNNL DataHub)
Source: https://data.pnnl.gov/group/nodes/dataset/12978
Paper: Xu & Matzner (2018), Underwater Fish Detection using Deep Learning for Water Power
    Applications, CSCI 2018, doi:10.1109/CSCI46756.2018.00067

Frames (PNG) and Pascal VOC labels from video recorded at three water power sites. The
download is one 87 GB tar that holds a second tar (eyesea.tar) with:
    train_x/, train_y/                       authors' training frames / labels (all sites)
    {voith,wells,orpc}_test_x/, *_test_y/    authors' per-site test frames / labels
    videos/, keras-yolo3-master/             source videos and the authors' model: skipped
The site is encoded in the filename:
    Voith_<date>_..., voith_...          Voith Hydro tidal turbine, Fall of Warness (EMEC), Scotland
    WellsDam_...                         Wells Dam fish ladder window, Columbia River, Washington
    <YYYYMMDD>_<HHMMSS>..., orpc_...     ORPC RivGen turbine, Igiugig, Kvichak River, Alaska
Voith frames are a 2x2 mosaic of the turbine's camera views (one panel often "NO VIDEO").
Frames that have a label file but no boxes are kept as empty (negative) images; frames
without a label file are skipped.

Split logic: By site. Val = every Voith Hydro frame (from both the authors' train and test
    folders); train = Wells Dam + Igiugig. The authors' split is temporal within a single
    recording for Voith and Wells, so it is ignored.
Category mapping:
  fish: "fish" (the only class)
"""

import json
import os
import re
import shutil
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from datasets.settings import Settings
from datasets.utils import (
    add_dataset_shortname_prefix_to_image_names,
    download_file_resumable,
    map_annotations_to_fish_and_non_fish,
    save_preview_image,
    split_coco_dataset_into_train_validation,
)

DATASET_SHORTNAME = "eyesea"
DATA_URL = (
    "https://g-83fdd0.1beed.03c0.data.globus.org/cartuids/"
    "12978_d072f8e31a3fede18c0b6a6d98d50042.tar"
)
CATEGORIES_FILTER = {
    "fish": "fish",
}
# Folders of the inner tar to extract: (images folder, labels folder)
SPLIT_FOLDERS = [
    ("train_x", "train_y"),
    ("voith_test_x", "voith_test_y"),
    ("wells_test_x", "wells_test_y"),
    ("orpc_test_x", "orpc_test_y"),
]
SITE_PATTERNS = {
    "voith": re.compile(r"^(Voith_|voith_)"),  # incl. voith_Fish_with_operational_turbine_2
    "wells": re.compile(r"^WellsDam_"),
    "orpc": re.compile(r"^(\d{8}_\d{6}|orpc_)"),  # incl. orpc_adults_then_smolt
}
VAL_SITES = {"voith"}

settings = Settings()

raw_download_path = settings.raw_dir / DATASET_SHORTNAME
outer_tar_path = raw_download_path / "eyesea_pnnl_12978.tar"
raw_data_dir = raw_download_path / "eyesea"
extraction_done_marker = raw_download_path / ".extracted"
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
coco_images_path = processing_dir / settings.images_folder_name
coco_annotations_path = processing_dir / settings.coco_file_name
mapped_annotations_path = processing_dir / "eyesea_mapped_annotations.json"


def site_of(name: str) -> str | None:
    for site, pattern in SITE_PATTERNS.items():
        if pattern.match(name):
            return site
    return None


def download_data():
    if extraction_done_marker.exists():
        print(f"Data already downloaded and extracted: {raw_data_dir}")
        return

    raw_download_path.mkdir(parents=True, exist_ok=True)
    download_file_resumable(DATA_URL, outer_tar_path)

    # The outer tar holds eyesea.tar; stream it and keep only the frame/label folders
    wanted = tuple(f"eyesea/{folder}/" for pair in SPLIT_FOLDERS for folder in pair)
    print(f"Extracting {', '.join(wanted)} from {outer_tar_path}")
    with tarfile.open(outer_tar_path, "r") as outer:
        inner_member = next(m for m in outer if m.name.endswith("/eyesea.tar"))
        inner_file = outer.extractfile(inner_member)
        with tarfile.open(fileobj=inner_file, mode="r|") as inner:
            n_extracted = 0
            for member in tqdm(inner, desc="inner tar members"):
                name = member.name
                if not member.isfile() or not name.startswith(wanted):
                    continue
                if Path(name).name.startswith("._"):  # macOS metadata files
                    continue
                inner.extract(member, raw_download_path, filter="data")
                n_extracted += 1
    print(f"Extracted {n_extracted} files to {raw_data_dir}")

    extraction_done_marker.touch()
    print(f"Removing {outer_tar_path}")
    outer_tar_path.unlink()


def link_or_copy(source: Path, target: Path):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def voc_to_coco():
    """Convert the VOC labels of all extracted folders into one COCO dataset."""
    if coco_annotations_path.exists():
        print(f"COCO dataset already exists: {coco_annotations_path}")
        return

    coco_images_path.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    class_names = {}
    n_missing_image = 0
    n_unknown_site = 0
    n_bad_box = 0
    seen_names = set()

    for images_folder, labels_folder in SPLIT_FOLDERS:
        label_paths = sorted((raw_data_dir / labels_folder).glob("*.xml"))
        n_images_on_disk = len(list((raw_data_dir / images_folder).glob("*.png")))
        print(f"{labels_folder}: {len(label_paths)} label files, {images_folder}: {n_images_on_disk} images")

        for label_path in tqdm(label_paths, desc=labels_folder):
            image_path = raw_data_dir / images_folder / f"{label_path.stem}.png"
            if not image_path.exists():
                n_missing_image += 1
                continue
            if site_of(image_path.name) is None:
                n_unknown_site += 1
                print(f"Unknown site, skipped: {image_path.name}")
                continue
            if image_path.name in seen_names:
                raise ValueError(f"Duplicate frame name across folders: {image_path.name}")
            seen_names.add(image_path.name)

            with Image.open(image_path) as im:
                width, height = im.size
            link_or_copy(image_path, coco_images_path / image_path.name)
            image_id = len(images) + 1
            images.append({
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            })

            root = ET.parse(label_path).getroot()
            for obj in root.findall("object"):
                class_name = obj.findtext("name").strip()
                category_id = class_names.setdefault(class_name, len(class_names) + 1)
                box = obj.find("bndbox")
                x1 = max(0.0, float(box.findtext("xmin")))
                y1 = max(0.0, float(box.findtext("ymin")))
                x2 = min(float(width), float(box.findtext("xmax")))
                y2 = min(float(height), float(box.findtext("ymax")))
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    n_bad_box += 1
                    continue
                annotations.append({
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": n} for n, i in class_names.items()],
    }
    with open(coco_annotations_path, "w") as f:
        json.dump(coco, f, indent=2)

    print(f"COCO annotations saved to {coco_annotations_path}")
    print(f"Source classes: {class_names}")
    print(f"Number of images: {len(images)}")
    print(f"Number of annotations: {len(annotations)}")
    print(f"Skipped: {n_missing_image} labels without image, {n_unknown_site} unknown site, "
          f"{n_bad_box} degenerate boxes")


def processing():
    processing_dir.mkdir(parents=True, exist_ok=True)

    voc_to_coco()

    map_annotations_to_fish_and_non_fish(
        coco_annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )

    add_dataset_shortname_prefix_to_image_names(
        images_path=coco_images_path,
        annotations_path=mapped_annotations_path,
        dataset_shortname=DATASET_SHORTNAME,
    )


def dataset_splitting():
    def should_the_image_be_included_in_train_set(image_path: str) -> bool:
        name = Path(image_path).name.removeprefix(f"{DATASET_SHORTNAME}_")
        return site_of(name) not in VAL_SITES

    train_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    )
    val_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    )
    train_dataset_path.mkdir(parents=True)
    val_dataset_path.mkdir(parents=True)

    split_coco_dataset_into_train_validation(
        coco_images_path,
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
    save_preview_image(coco_images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    dataset_splitting()


if __name__ == "__main__":
    main()
