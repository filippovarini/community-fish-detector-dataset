"""
Marine Detect Dataset (FishInv + MegaFauna)
Source: https://github.com/Orange-OpenSource/marine-detect
  FishInv: https://stpubtenakanclyw.blob.core.windows.net/marine-detect/FishInv-dataset.zip
  MegaFauna: https://stpubtenakanclyw.blob.core.windows.net/marine-detect/MegaFauna-dataset.zip
Split logic: By original split. An image goes to val if it is in valid/test in either
  source dataset, otherwise to train.
Category mapping:
  fish: fish, bolbometopon_muricatum, chaetodontidae, cheilinus_undulatus,
    cromileptes_altivelis, haemulidae, lutjanidae, muraenidae, scaridae, serranidae,
    shark, ray
  non-fish: turtle, urchin, giant_clam, sea_cucumber, crown_of_thorns, lobster
  discard: none

The two datasets are merged into one, as many images (byte-identical, same filename)
appear in both with annotations divided between them: FishInv labels fish and
invertebrates, MegaFauna labels sharks, rays and turtles. Annotations of shared images
are combined into a single image entry.

Labels are YOLO format (class x_center y_center width height, normalised to [0, 1]).
Images from the OzFish dataset (filenames containing _L.MP4., _R.MP4., _L.avi.,
_R.avi.) are skipped, as OzFish is processed separately.
Images without annotations are kept.
Some camera images carry an EXIF orientation tag, while the labels are relative to the
stored (unrotated) pixels, so the tag is removed when copying.
"""

import json
import shutil
from pathlib import Path

from PIL import Image

from datasets.settings import Settings
from datasets.utils import (
    download_and_extract,
    map_annotations_to_fish_and_non_fish,
    split_coco_dataset_into_train_validation,
    add_dataset_shortname_prefix_to_image_names,
    remove_dataset_shortname_prefix_from_image_filename,
    save_preview_image,
)


DATASET_SHORTNAME = "marine_detect"
CATEGORIES_FILTER = {
    "fish": "fish",
    "bolbometopon_muricatum": "fish",
    "chaetodontidae": "fish",
    "cheilinus_undulatus": "fish",
    "cromileptes_altivelis": "fish",
    "haemulidae": "fish",
    "lutjanidae": "fish",
    "muraenidae": "fish",
    "scaridae": "fish",
    "serranidae": "fish",
    "shark": "fish",
    "ray": "fish",
    "turtle": "non-fish",
    "urchin": "non-fish",
    "giant_clam": "non-fish",
    "sea_cucumber": "non-fish",
    "crown_of_thorns": "non-fish",
    "lobster": "non-fish",
}

# Source datasets: (download url, root folder inside the zip, YOLO class names)
SOURCES = {
    "fishinv": (
        "https://stpubtenakanclyw.blob.core.windows.net/marine-detect/FishInv-dataset.zip",
        "notebooks/datasets/FishInvSplit",
        [
            "bolbometopon_muricatum", "chaetodontidae", "cheilinus_undulatus",
            "cromileptes_altivelis", "fish", "haemulidae", "lutjanidae", "muraenidae",
            "scaridae", "serranidae", "urchin", "giant_clam", "sea_cucumber",
            "crown_of_thorns", "lobster",
        ],
    ),
    "megafauna": (
        "https://stpubtenakanclyw.blob.core.windows.net/marine-detect/MegaFauna-dataset.zip",
        "notebooks/datasets/MegaFaunaSplit",
        ["ray", "shark", "turtle"],
    ),
}
SOURCE_SPLITS = ["train", "valid", "test"]
OZFISH_PATTERNS = ["_L.MP4.", "_R.MP4.", "_L.avi.", "_R.avi."]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
EXIF_ORIENTATION_TAG = 274

settings = Settings()

raw_dir = settings.raw_dir / DATASET_SHORTNAME
processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
images_path = processing_dir / settings.images_folder_name
annotations_path = processing_dir / "annotations.json"
mapped_annotations_path = processing_dir / "annotations_coco_mapped.json"


def download_data():
    for source, (url, root, _) in SOURCES.items():
        source_dir = raw_dir / source
        if (source_dir / root).exists():
            print(f"{source} already downloaded")
            continue
        source_dir.mkdir(parents=True, exist_ok=True)
        download_and_extract(source_dir, url, source)


def read_yolo_labels(label_path: Path, class_names: list[str], width: int, height: int):
    """Reads a YOLO label file into (class name, COCO bbox in pixels) tuples."""
    if not label_path.exists():
        return []
    labels = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, box_width, box_height = map(float, parts)
        # A few source labels are degenerate (zero width or height)
        if box_width <= 0 or box_height <= 0:
            continue
        bbox = [
            (x_center - box_width / 2) * width,
            (y_center - box_height / 2) * height,
            box_width * width,
            box_height * height,
        ]
        labels.append((class_names[int(class_id)], bbox))
    return labels


def copy_image_without_exif_orientation(source: Path, destination: Path) -> bool:
    """
    Copies an image, removing any EXIF orientation tag. Labels are relative to the
    stored pixels, but readers that apply EXIF orientation (OpenCV, supervision,
    most training loaders) would otherwise rotate the image away from its boxes.
    Returns True if the orientation tag was removed.
    """
    with Image.open(source) as img:
        exif = img.getexif()
        if exif.get(EXIF_ORIENTATION_TAG, 1) == 1:
            shutil.copy2(source, destination)
            return False
        del exif[EXIF_ORIENTATION_TAG]
        img.save(destination, exif=exif, quality=95)
        return True


def build_merged_coco_dataset():
    """
    Merges FishInv and MegaFauna into a single COCO dataset, combining the
    annotations of images present in both. Copies images to images_path.
    Each image records the source splits it came from in "original_splits".
    """
    if annotations_path.exists():
        print(f"Merged annotations already exist at {annotations_path}")
        return

    images_path.mkdir(parents=True, exist_ok=True)
    category_names = list(CATEGORIES_FILTER)
    category_name_to_id = {name: i + 1 for i, name in enumerate(category_names)}

    images = {}  # filename -> COCO image entry
    annotations = []
    ozfish_skipped = 0
    exif_rotated = 0

    for source, (_, root, class_names) in SOURCES.items():
        for split in SOURCE_SPLITS:
            split_dir = raw_dir / source / root / split
            for image_file in sorted((split_dir / "images").iterdir()):
                if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if any(pattern in image_file.name for pattern in OZFISH_PATTERNS):
                    ozfish_skipped += 1
                    continue

                if image_file.name not in images:
                    exif_rotated += copy_image_without_exif_orientation(
                        image_file, images_path / image_file.name
                    )
                    with Image.open(image_file) as img:
                        width, height = img.size
                    images[image_file.name] = {
                        "id": len(images) + 1,
                        "file_name": image_file.name,
                        "width": width,
                        "height": height,
                        "original_splits": [],
                    }
                image = images[image_file.name]
                image["original_splits"].append(f"{source}/{split}")

                label_path = split_dir / "labels" / f"{image_file.stem}.txt"
                for class_name, bbox in read_yolo_labels(
                    label_path, class_names, image["width"], image["height"]
                ):
                    annotations.append(
                        {
                            "id": len(annotations) + 1,
                            "image_id": image["id"],
                            "category_id": category_name_to_id[class_name],
                            "bbox": bbox,
                            "area": bbox[2] * bbox[3],
                            "iscrowd": 0,
                        }
                    )

    shared = sum(len(i["original_splits"]) > 1 for i in images.values())
    print(
        f"Merged {len(images)} images ({shared} in both datasets), "
        f"{len(annotations)} annotations, skipped {ozfish_skipped} OzFish images, "
        f"removed EXIF orientation from {exif_rotated} images"
    )

    coco_data = {
        "images": list(images.values()),
        "annotations": annotations,
        "categories": [
            {"id": category_name_to_id[name], "name": name} for name in category_names
        ],
    }
    with open(annotations_path, "w") as f:
        json.dump(coco_data, f)


def get_train_image_filenames() -> set[str]:
    """Images in valid/test of either source dataset go to val, the rest to train."""
    with open(annotations_path) as f:
        coco_data = json.load(f)
    return {
        image["file_name"]
        for image in coco_data["images"]
        if all(s.endswith("/train") for s in image["original_splits"])
    }


def main():
    # 1. DOWNLOAD
    download_data()

    # 2. PROCESS
    build_merged_coco_dataset()
    map_annotations_to_fish_and_non_fish(
        annotations_path, CATEGORIES_FILTER, mapped_annotations_path
    )
    add_dataset_shortname_prefix_to_image_names(
        images_path, mapped_annotations_path, DATASET_SHORTNAME
    )

    # 3. PREVIEW
    save_preview_image(images_path, mapped_annotations_path, DATASET_SHORTNAME)

    # 4. SPLIT
    train_image_filenames = get_train_image_filenames()
    should_the_image_be_included_in_train_set = (
        lambda image_filename: remove_dataset_shortname_prefix_from_image_filename(
            image_filename, DATASET_SHORTNAME
        )
        in train_image_filenames
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


if __name__ == "__main__":
    main()
