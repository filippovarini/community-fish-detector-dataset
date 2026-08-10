"""
OBSEA Dataset

Source: Zenodo record 14888328, obsea_dataset_v4.1.zip
https://doi.org/10.5281/zenodo.14888328

Expects obsea_dataset_v4.1.zip to be at:

    [repo root]/data/raw/obsea/obsea_dataset_v4.1.zip

The zip file contains a folder called "obsea_dataset_v4.1"; if necessary, this script
will extract that folder such that the final folder structure looks like:

    [repo root]/data/raw/obsea/obsea_dataset_v4.1/images
    [repo root]/data/raw/obsea/obsea_dataset_v4.1/labels

* Fish and fish-like source classes are mapped into the repo's single "fish" category.
* Zero-fish/background images are kept in the COCO images list with no annotations.
* The validation split is a based on filenames that indicate unique deployments
"""

import json
import os
import shutil
import zipfile

from tqdm import tqdm
from pathlib import Path

from PIL import Image

from datasets.settings import Settings
from datasets.utils.images import add_dataset_shortname_prefix_to_image_names
from datasets.utils.split import split_coco_dataset_into_train_validation


DATASET_SHORTNAME = "obsea"
SOURCE_ZIP_NAME = "obsea_dataset_v4.1.zip"
SOURCE_ZIP_MD5 = "7a2012d7a39fb42d155b29fc67d8753f"
SOURCE_FOLDER_NAME = "obsea_dataset_v4.1"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Clear non-fish source classes in obsea_dataset.json's YOLO class order.
NON_FISH_CLASS_IDS = {
    4,  # Asteroidea
    9,  # Chrysaora hysoscella
    13,  # Cranc
    15,  # Delphinus (dead)
    31,  # Diver
    34,  # Gateropodae
    46,  # Octopus vulgaris
    48,  # Posidonia oceanica (shot)
    49,  # Rhizostoma pulmo
}

EXPECTED_NON_FISH_CLASS_NAMES = {
    4: "Asteroidea",
    9: "Chrysaora hysoscella",
    13: "Cranc",
    15: "Delphinus (dead)",
    31: "Diver",
    34: "Gateropodae",
    46: "Octopus vulgaris",
    48: "Posidonia oceanica (shot)",
    49: "Rhizostoma pulmo",
}

# Deployment IDs assigned to the validation set
VALIDATION_GROUPS = {"AIPC608UW_10_167", "C4k0193"}

settings = Settings()


def load_source_class_names(source_root: Path) -> list[str]:
    """
    Load the list of class names from obsea_dataset.json
    """
    metadata_path = source_root / "obsea_dataset.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if not isinstance(metadata, dict):
        raise ValueError(f"Expected {metadata_path} to contain class-name counts")

    class_names = list(metadata.keys())
    for class_id, expected_name in EXPECTED_NON_FISH_CLASS_NAMES.items():
        actual_name = class_names[class_id]
        if actual_name != expected_name:
            raise ValueError(
                f"Unexpected OBSEA class mapping for class {class_id}: "
                f"expected {expected_name!r}, found {actual_name!r}"
            )

    return class_names


def list_images(source_root: Path) -> dict[str, Path]:
    """
    Return all images in [source_root] (non-recursive).
    """
    images = {}
    for image_path in sorted((source_root / "images").iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images[image_path.stem] = image_path
    return images


def list_labels(source_root: Path) -> dict[str, Path]:
    """
    Return all .txt files in [source_root] (non-recursive).
    """
    return {
        label_path.stem: label_path
        for label_path in sorted((source_root / "labels").glob("*.txt"))
        if label_path.is_file()
    }


def parse_yolo_label_file(
    label_path: Path,
    class_count: int,
) -> list[tuple[int, float, float, float, float]]:
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

            try:
                class_id = int(fields[0])
                x_center, y_center, box_width, box_height = [
                    float(value) for value in fields[1:]
                ]
            except ValueError as exc:
                raise ValueError(
                    f"Malformed YOLO values at {label_path}:{line_number}: {line}"
                ) from exc

            if not 0 <= class_id < class_count:
                raise ValueError(
                    f"Class id {class_id} outside [0, {class_count - 1}] at "
                    f"{label_path}:{line_number}"
                )

            if not (
                (0 <= x_center <= 1)
                and (0 <= y_center <= 1)
                and (0 < box_width <= 1)
                and (0 < box_height <= 1)
            ):
                raise ValueError(
                    f"YOLO coordinates outside normalized bounds at "
                    f"{label_path}:{line_number}: {line}"
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
    Convert a YOLO box (x,y,w,h) to COCO format (x1,y2,x2,y2)
    """
    x_min = (x_center - box_width / 2) * image_width
    y_min = (y_center - box_height / 2) * image_height
    x_max = (x_center + box_width / 2) * image_width
    y_max = (y_center + box_height / 2) * image_height

    x_min = max(0.0, min(float(image_width), x_min))
    y_min = max(0.0, min(float(image_height), y_min))
    x_max = max(0.0, min(float(image_width), x_max))
    y_max = max(0.0, min(float(image_height), y_max))

    return [x_min, y_min, x_max - x_min, y_max - y_min]


def get_source_group(image_stem_or_filename: str) -> str:
    """
    Extract an OBSEA deployment ID from a filename, returns
    "unknown" if it can't find a known deployment ID.
    """
    stem = Path(image_stem_or_filename).stem
    if stem.startswith(f"{DATASET_SHORTNAME}_"):
        stem = stem[len(f"{DATASET_SHORTNAME}_") :]

    if "AIPC608UW_10_167" in stem:
        return "AIPC608UW_10_167"
    if "IPC608_8BC7_166" in stem:
        return "IPC608_8BC7_166"
    if "IPC608_8B64_166" in stem:
        return "IPC608_8B64_166"
    if "IPC608_8B64_165" in stem:
        return "IPC608_8B64_165"
    if "C4k0193" in stem:
        return "C4k0193"
    if "Mero" in stem or "Morena" in stem:
        return "Mero/Morena video"
    if "Video" in stem:
        return "Video/other"
    return "unknown"


def copy_source_images_to_processing(
    source_images: dict[str, Path],
    coco_images_path: Path,
) -> None:
    """
    Copy all the files in the list [source_images] to the
    folder [coco_images_path].
    """
    coco_images_path.mkdir(parents=True, exist_ok=True)
    source_image_filenames = list(source_images.values())
    print('Copying {} files to {}'.format(
        len(source_image_filenames), str(coco_images_path)))
    for image_path in tqdm(source_image_filenames):
        shutil.copy2(image_path, coco_images_path / image_path.name)


def convert_obsea_to_coco(
    source_root: Path,
    coco_images_path: Path,
    coco_annotations_path: Path,
) -> dict:
    """
    The main function in this module, converts the entire OBSEA dataset from YOLO
    to COCO.
    """
    class_names = load_source_class_names(source_root)
    source_images = list_images(source_root)
    source_labels = list_labels(source_root)

    print("Read {} class names, enumerated {} images, enumerated {} labels".format(
        len(class_names),len(source_images),len(source_labels)))
    assert (len(class_names) > 0) and (len(source_images) > 0) and (len(source_labels) > 0)

    missing_labels = sorted(set(source_images) - set(source_labels))
    missing_images = sorted(set(source_labels) - set(source_images))
    if missing_labels or missing_images:
        raise ValueError(
            "OBSEA image/label mismatch. "
            f"Images without labels: {len(missing_labels)}. "
            f"Labels without images: {len(missing_images)}."
        )

    copy_source_images_to_processing(source_images, coco_images_path)

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": Settings.coco_categories,
    }

    source_yolo_rows = 0
    filtered_rows = 0
    kept_rows = 0
    zero_fish_images = 0
    image_id_by_stem = {}

    annotation_id = 1

    print("Converting data from YOLO to COCO")

    for image_id, (image_stem, source_image_path) in tqdm(enumerate(
        source_images.items(), start=1), total=len(source_images)):
        with Image.open(source_image_path) as image:
            image_width, image_height = image.size

        image_id_by_stem[image_stem] = image_id
        coco_data["images"].append(
            {
                "id": image_id,
                "file_name": source_image_path.name,
                "width": image_width,
                "height": image_height,
            }
        )

        label_rows = parse_yolo_label_file(source_labels[image_stem], len(class_names))
        source_yolo_rows += len(label_rows)
        image_kept_rows = 0

        for class_id, x_center, y_center, box_width, box_height in label_rows:
            if class_id in NON_FISH_CLASS_IDS:
                filtered_rows += 1
                continue

            bbox = yolo_box_to_coco_box(
                x_center,
                y_center,
                box_width,
                box_height,
                image_width,
                image_height,
            )
            if bbox[2] <= 0 or bbox[3] <= 0:
                raise ValueError(
                    f"Kept OBSEA box collapsed to zero area for {source_image_path.name}"
                )

            coco_data["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id_by_stem[image_stem],
                    "category_id": Settings.coco_category_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            kept_rows += 1
            image_kept_rows += 1

        if image_kept_rows == 0:
            zero_fish_images += 1

    with open(coco_annotations_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    return {
        "source_image_count": len(source_images),
        "source_label_count": len(source_labels),
        "source_yolo_rows": source_yolo_rows,
        "filtered_rows": filtered_rows,
        "kept_fish_annotations": kept_rows,
        "zero_fish_images": zero_fish_images,
        "coco_annotations_path": coco_annotations_path,
    }


def count_images_with_annotations(coco_data: dict) -> int:
    image_ids_with_annotations = {
        annotation["image_id"] for annotation in coco_data["annotations"]
    }
    return sum(
        1 for image in coco_data["images"] if image["id"] in image_ids_with_annotations
    )


def load_coco_summary(coco_path: Path) -> dict:
    with open(coco_path, "r") as f:
        coco_data = json.load(f)
    return {
        "images": len(coco_data["images"]),
        "images_with_fish_boxes": count_images_with_annotations(coco_data),
        "annotations": len(coco_data["annotations"]),
    }


def try_save_preview_image(
    coco_images_path: Path,
    coco_annotations_path: Path,
) -> None:
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        from datasets.utils.visualization import save_preview_image
    except ModuleNotFoundError as exc:
        print(
            "Skipping only preview image generation because "
            f"{exc.name} is not installed; dataset conversion will continue."
        )
        return

    save_preview_image(coco_images_path, coco_annotations_path, DATASET_SHORTNAME)


def main():

    source_root = settings.raw_dir / DATASET_SHORTNAME / SOURCE_FOLDER_NAME

    if (not source_root.is_dir()):
        print(f"Source folder not found at {source_root}, checking for zipfile")
        source_zip = settings.raw_dir / DATASET_SHORTNAME / SOURCE_ZIP_NAME
        if (not source_zip.is_file()):
            raise ValueError(f"Source zipfile not found at {source_zip}")
        with zipfile.ZipFile(source_zip, 'r') as zipf:
            zipf.extractall(settings.raw_dir / DATASET_SHORTNAME)
        if (not source_root.is_dir()):
            raise ValueError(f"Extracted zipfile {source_zip}, but folder {source_root} not found")

    print(f"Using OBSEA source folder: {source_root}")

    processing_dir = settings.intermediate_dir / DATASET_SHORTNAME
    processing_dir.mkdir(parents=True, exist_ok=True)

    coco_images_path = processing_dir / settings.images_folder_name
    coco_annotations_path = processing_dir / settings.coco_file_name

    if coco_images_path.exists():
        shutil.rmtree(coco_images_path)
    if coco_annotations_path.exists():
        coco_annotations_path.unlink()

    summary = convert_obsea_to_coco(
        source_root=source_root,
        coco_images_path=coco_images_path,
        coco_annotations_path=coco_annotations_path,
    )

    add_dataset_shortname_prefix_to_image_names(
        coco_images_path,
        coco_annotations_path,
        DATASET_SHORTNAME,
    )

    try_save_preview_image(coco_images_path, coco_annotations_path)

    def should_the_image_be_included_in_train_set(image_filename: str) -> bool:
        return get_source_group(image_filename) not in VALIDATION_GROUPS

    train_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.train_dataset_suffix}"
    )
    val_dataset_path = (
        settings.processed_dir / f"{DATASET_SHORTNAME}{settings.val_dataset_suffix}"
    )

    if train_dataset_path.exists():
        shutil.rmtree(train_dataset_path)
    if val_dataset_path.exists():
        shutil.rmtree(val_dataset_path)

    train_dataset_path.mkdir(parents=True, exist_ok=True)
    val_dataset_path.mkdir(parents=True, exist_ok=True)

    split_coco_dataset_into_train_validation(
        coco_images_path,
        coco_annotations_path,
        train_dataset_path,
        val_dataset_path,
        should_the_image_be_included_in_train_set,
    )

    train_summary = load_coco_summary(train_dataset_path / settings.coco_file_name)
    val_summary = load_coco_summary(val_dataset_path / settings.coco_file_name)

    print("OBSEA source summary:")
    print(f"  - source images: {summary['source_image_count']}")
    print(f"  - source labels: {summary['source_label_count']}")
    print(f"  - original YOLO rows: {summary['source_yolo_rows']}")
    print(f"  - filtered clear non-fish rows: {summary['filtered_rows']}")
    print(f"  - kept fish/fish-like annotations: {summary['kept_fish_annotations']}")
    print(f"  - zero-fish/background images kept: {summary['zero_fish_images']}")
    print("OBSEA split summary:")
    print(
        f"  - training: {train_summary['images']} images, "
        f"{train_summary['images_with_fish_boxes']} images with fish boxes, "
        f"{train_summary['annotations']} fish annotations"
    )
    print(
        f"  - validation: {val_summary['images']} images, "
        f"{val_summary['images_with_fish_boxes']} images with fish boxes, "
        f"{val_summary['annotations']} fish annotations"
    )


if __name__ == "__main__":
    main()
